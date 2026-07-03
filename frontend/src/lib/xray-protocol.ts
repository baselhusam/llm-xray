/**
 * TypeScript mirror of the `/ws/xray` wire protocol.
 *
 * Tightly coupled to `backend/app/serializer.py` — when the message shapes
 * change there, change them here too (see CLAUDE.md "WebSocket protocol").
 *
 * The server streams a greedy autoregressive generation, in order, per run:
 *   1. one `meta` message (layer/head counts + model label)
 *   2. one `tokens` message (the tokenized prompt)
 *   3. one `prompt_attention` message (per-layer attention over the prompt)
 *   4. one `step` message per generated token (the model "thinking" out loud)
 *   5. one `done` message (full continuation + stop reason)
 * An out-of-band `error` message can arrive at any point instead.
 */

/** WebSocket endpoint. Override per-env with `NEXT_PUBLIC_XRAY_WS_URL`. */
export const WS_URL =
  process.env.NEXT_PUBLIC_XRAY_WS_URL ?? "ws://127.0.0.1:8000/ws/xray";

/** A logit-lens top-k prediction (also reused for a step's runner-up picks). */
export interface TopPrediction {
  token: string;
  token_id: number;
  prob: number;
}

export interface MetaData {
  num_layers: number;
  num_heads: number;
  /** Whether the model is reasoning (<think>…</think>) before answering. */
  thinking: boolean;
  model_label: string;
}

export interface TokensData {
  tokens: string[];
  token_ids: number[];
}

export interface PromptAttentionData {
  /** Per-layer attention over the prompt, mean over heads, shape (layers, P, P). */
  attention: number[][][];
}

/** One transformer layer's logit-lens top-k at a generation step's decision. */
export interface LayerTopK {
  layer_idx: number;
  top_predictions: TopPrediction[];
}

/** Which span of the generation a token belongs to. */
export type Phase = "think" | "answer";

/** One generated token: the model's commitment plus why it made it. */
export interface StepData {
  step: number;
  token: string;
  token_id: number;
  /** Probability the model assigned to the chosen token. */
  prob: number;
  /** Shannon entropy (nats) of the final next-token distribution. */
  entropy: number;
  /** Reasoning trace vs. the answer. */
  phase: Phase;
  /** Per-layer logit-lens trajectory — watch the prediction form with depth. */
  trajectory: LayerTopK[];
  /** The query row that produced this token, mean over heads, shape (layers, key_len). */
  attention_row: number[][];
}

export interface DoneData {
  generated_text: string;
  num_steps: number;
  stop_reason: string; // "eos" | "sentence" | "max_tokens" | "time_budget"
}

export interface ErrorData {
  message: string;
}

export type ServerMessage =
  | { type: "meta"; data: MetaData }
  | { type: "tokens"; data: TokensData }
  | { type: "prompt_attention"; data: PromptAttentionData }
  | { type: "step"; data: StepData }
  | { type: "done"; data: DoneData }
  | { type: "error"; data: ErrorData };

/** Fallback layer count (the live value comes from the `meta` message). */
export const NUM_LAYERS = 28;

/** Fallback model name for UI/branding (the live value comes from `meta`). */
export const MODEL_LABEL = "Qwen3-1.7B";

/** Human-readable copy for each stop reason (for the done banner). */
export const STOP_REASON_LABEL: Record<string, string> = {
  eos: "hit end-of-text",
  sentence: "finished a sentence",
  max_tokens: "reached the length cap",
  time_budget: "hit the time budget",
};
