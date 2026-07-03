"use client";

/**
 * `useXRay` — drive one `/ws/xray` generation run and accumulate the streamed
 * messages into React state the UI can render incrementally (token-by-token).
 *
 * Lifecycle: `run(prompt)` opens a fresh WebSocket, sends the prompt, and folds
 * each incoming message into state via {@link reduce}. A new `run` supersedes
 * any in-flight one (the old socket is closed and its late messages ignored).
 * The socket is closed once the run terminates (`done` or `error`) and on
 * unmount, so we never leak connections.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  WS_URL,
  type DeviceName,
  type DoneData,
  type MetaData,
  type ServerMessage,
  type StepData,
} from "@/lib/xray-protocol";

export type XRayStatus =
  | "idle"
  | "connecting"
  | "streaming"
  | "done"
  | "error";

export interface XRayState {
  status: XRayStatus;
  /** The prompt of the current/last run. */
  prompt: string | null;
  meta: MetaData | null;
  promptTokens: string[] | null;
  promptIds: number[] | null;
  /** Per-layer attention over the prompt, mean over heads, shape (layers, P, P). */
  promptAttention: number[][][] | null;
  /** Generated tokens in order, accumulated as they stream. */
  steps: StepData[];
  done: DoneData | null;
  error: string | null;
}

const INITIAL: XRayState = {
  status: "idle",
  prompt: null,
  meta: null,
  promptTokens: null,
  promptIds: null,
  promptAttention: null,
  steps: [],
  done: null,
  error: null,
};

/** Fold one server message into the accumulating run state. */
function reduce(state: XRayState, msg: ServerMessage): XRayState {
  switch (msg.type) {
    case "meta":
      return { ...state, meta: msg.data };
    case "tokens":
      return {
        ...state,
        promptTokens: msg.data.tokens,
        promptIds: msg.data.token_ids,
      };
    case "prompt_attention":
      return { ...state, promptAttention: msg.data.attention };
    case "step": {
      // Fast path: the server streams steps in order, so almost every message
      // is the next one — just append (O(1) amortized, no re-sort). A long
      // reasoning trace is hundreds of these messages; falling back to a full
      // filter+sort on every one made per-token cost grow with trace length
      // for no reason, since arrival order is already correct in practice.
      const last = state.steps[state.steps.length - 1];
      if (!last || msg.data.step === last.step + 1) {
        return { ...state, steps: [...state.steps, msg.data] };
      }
      // Rare fallback for genuinely out-of-order/duplicate arrival.
      const steps = [
        ...state.steps.filter((s) => s.step !== msg.data.step),
        msg.data,
      ].sort((a, b) => a.step - b.step);
      return { ...state, steps };
    }
    case "done":
      return { ...state, done: msg.data, status: "done" };
    case "error":
      return { ...state, status: "error", error: msg.data.message };
    default:
      return state;
  }
}

export interface UseXRay extends XRayState {
  run: (
    prompt: string,
    thinking?: boolean,
    device?: DeviceName,
    maxTokens?: number,
  ) => void;
}

export function useXRay(): UseXRay {
  const [state, setState] = useState<XRayState>(INITIAL);
  const socketRef = useRef<WebSocket | null>(null);

  // Close any open socket when the component using the hook unmounts.
  useEffect(() => {
    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, []);

  const run = useCallback(
    (
      prompt: string,
      thinking: boolean = true,
      device?: DeviceName,
      maxTokens?: number,
    ) => {
      const trimmed = prompt.trim();
      if (!trimmed) return;

      // Supersede any in-flight run. Nulling the ref first makes the old
      // socket's handlers no-ops (they guard on `socketRef.current === ws`).
      socketRef.current?.close();

      setState({ ...INITIAL, status: "connecting", prompt: trimmed });

      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;

      ws.onopen = () => {
        if (socketRef.current !== ws) return;
        ws.send(
          JSON.stringify({ prompt: trimmed, thinking, device, max_tokens: maxTokens }),
        );
        setState((s) => ({ ...s, status: "streaming" }));
      };

      ws.onmessage = (event) => {
        if (socketRef.current !== ws) return;
        let msg: ServerMessage;
        try {
          msg = JSON.parse(event.data as string);
        } catch {
          return; // ignore unparseable frames
        }
        setState((s) => reduce(s, msg));

        // A run ends on done or error; close the socket and stop listening.
        if (msg.type === "done" || msg.type === "error") {
          socketRef.current = null;
          ws.close();
        }
      };

      ws.onerror = () => {
        if (socketRef.current !== ws) return;
        setState((s) => ({
          ...s,
          status: "error",
          error: s.error ?? "Connection error. Is the backend running?",
        }));
      };

      ws.onclose = () => {
        // Only meaningful for the active socket closing unexpectedly mid-run;
        // a clean end nulled the ref already, so this guard short-circuits.
        if (socketRef.current !== ws) return;
        socketRef.current = null;
        setState((s) =>
          s.status === "streaming" || s.status === "connecting"
            ? {
                ...s,
                status: "error",
                error: s.error ?? "Connection closed before completion.",
              }
            : s,
        );
      };
    },
    [],
  );

  return { ...state, run };
}
