"use client";

/**
 * `XRayProgress` — the streaming status line shown while a generation is in
 * flight. Reports where the run is: connecting, running the prompt's forward
 * pass, or how many tokens have been generated so far. Generation length isn't
 * known ahead of time (it stops at a sentence/EOS), so the bar tracks tokens
 * against the cap as a soft ceiling.
 */

import { Progress, ProgressLabel } from "@/components/ui/progress";
import type { XRayStatus } from "@/hooks/useXRay";

/**
 * Soft ceiling for the progress bar — matches the backend's MAX_NEW_TOKENS /
 * MAX_NEW_TOKENS_THINKING (`xray_engine.py`), which are both 1024. Most runs
 * stop well before this (sentence-end/EOS), so the bar rarely fills — that's
 * expected for a soft ceiling, not a bug.
 */
const TOKEN_CAP = 1024;

interface XRayProgressProps {
  status: XRayStatus;
  /** Number of tokens generated so far. */
  done: number;
}

export function XRayProgress({ status, done }: XRayProgressProps) {
  const message =
    status === "connecting"
      ? "Connecting to the model…"
      : done === 0
        ? "Reading the prompt…"
        : `Generating — ${done} token${done === 1 ? "" : "s"} so far`;

  return (
    <Progress
      value={Math.min(done, TOKEN_CAP)}
      max={TOKEN_CAP}
      className="flex-col items-stretch gap-2"
    >
      <div className="flex items-baseline justify-between gap-2">
        <ProgressLabel className="text-muted-foreground">{message}</ProgressLabel>
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {done} tok
        </span>
      </div>
    </Progress>
  );
}
