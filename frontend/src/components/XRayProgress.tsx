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
import { DEFAULT_MAX_TOKENS } from "@/lib/xray-protocol";

interface XRayProgressProps {
  status: XRayStatus;
  /** Number of tokens generated so far. */
  done: number;
  /**
   * The run's length cap (`meta.max_tokens`) — a soft ceiling: most runs stop
   * well before it (sentence-end/EOS), so the bar rarely fills. Falls back to
   * the default cap until `meta` lands.
   */
  cap?: number;
}

export function XRayProgress({ status, done, cap }: XRayProgressProps) {
  const tokenCap = cap ?? DEFAULT_MAX_TOKENS;
  const message =
    status === "connecting"
      ? "Connecting to the model…"
      : done === 0
        ? "Reading the prompt…"
        : `Generating — ${done} token${done === 1 ? "" : "s"} so far`;

  return (
    <Progress
      value={Math.min(done, tokenCap)}
      max={tokenCap}
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
