"use client";

/**
 * `ReplayControls` — play the finished generation back like a video. Greedy
 * decoding is deterministic and every step is already in state, so "replay" is
 * just a timer driving the pinned step; every instrument re-animates in sync.
 * Space toggles play, ←/→ scrub (wired globally in `XRayApp`).
 */

import { cn } from "@/lib/utils";

export const REPLAY_SPEEDS = [1, 2, 4];
/** Base per-token cadence at 1× (ms). */
export const REPLAY_TICK_MS = 260;

interface ReplayControlsProps {
  playing: boolean;
  speed: number;
  disabled: boolean;
  onToggle: () => void;
  onCycleSpeed: () => void;
}

export function ReplayControls({
  playing,
  speed,
  disabled,
  onToggle,
  onCycleSpeed,
}: ReplayControlsProps) {
  return (
    <div className="flex items-center gap-0.5 rounded-md border border-border bg-muted/40 p-0.5">
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-pressed={playing}
        title={
          playing
            ? "Pause replay (space)"
            : "Replay the generation (space · ←/→ scrub)"
        }
        className={cn(
          "cursor-pointer rounded px-2 py-0.5 font-mono text-[11px] transition-colors",
          playing
            ? "bg-primary text-primary-foreground"
            : "text-muted-foreground hover:bg-accent/10 hover:text-foreground",
          disabled && "cursor-not-allowed opacity-40",
        )}
      >
        {playing ? "❚❚" : "▶ replay"}
      </button>
      <button
        type="button"
        onClick={onCycleSpeed}
        disabled={disabled}
        title="Replay speed"
        className={cn(
          "cursor-pointer rounded px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground transition-colors hover:bg-accent/10 hover:text-foreground",
          disabled && "cursor-not-allowed opacity-40",
        )}
      >
        {speed}×
      </button>
    </div>
  );
}
