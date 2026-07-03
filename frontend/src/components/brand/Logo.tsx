/**
 * Brand identity components — the "Specimen X" crosshair monogram and its
 * lockups, straight from the LLM X-Ray brand guidelines.
 *
 * The mark is three parts: registration crop-marks (a scan bed / viewfinder, at
 * 34% opacity), the X of "X-Ray" (two rounded 45° strokes), and a single focal
 * node — the only place color is allowed inside the mark, the point the
 * instrument is looking at.
 *
 * `ink` defaults to `currentColor` so the crosshair inherits surrounding text
 * color (works on light or dark); `accent` defaults to the themed vermilion.
 */

import { cn } from "@/lib/utils";

interface MarkProps {
  className?: string;
  /** Color of the crop-marks + X. Defaults to currentColor. */
  ink?: string;
  /** Color of the focal node. Defaults to the themed accent (vermilion). */
  accent?: string;
  title?: string;
}

export function XRayMark({
  className,
  ink = "currentColor",
  accent = "var(--primary)",
  title = "LLM X-Ray",
}: MarkProps) {
  return (
    <svg
      viewBox="0 0 64 64"
      className={className}
      role="img"
      aria-label={title}
      fill="none"
    >
      {/* A — registration crop-marks */}
      <g
        stroke={ink}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.34}
      >
        <polyline points="14,22 14,14 22,14" />
        <polyline points="50,22 50,14 42,14" />
        <polyline points="14,42 14,50 22,50" />
        <polyline points="50,42 50,50 42,50" />
      </g>
      {/* B — the X */}
      <g stroke={ink} strokeWidth={5} strokeLinecap="round">
        <line x1="21" y1="21" x2="43" y2="43" />
        <line x1="43" y1="21" x2="21" y2="43" />
      </g>
      {/* C — focal node */}
      <circle cx="32" cy="32" r="3.6" fill={accent} />
    </svg>
  );
}

/** The "LLM X-Ray" wordmark, with the X carried in the accent. */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn("font-sans font-bold tracking-tight", className)}>
      LLM <span className="text-primary">X</span>-Ray
    </span>
  );
}

interface LockupProps {
  className?: string;
  /** Tailwind size utility classes for the mark (e.g. "size-9"). */
  markClassName?: string;
  /** Tailwind text-size for the wordmark. */
  wordClassName?: string;
  /** Optional mono sub-label under the wordmark (e.g. the model id). */
  sublabel?: string;
}

/** Horizontal lockup: the bare crosshair mark + wordmark (+ optional mono sub-label). */
export function Lockup({
  className,
  markClassName = "size-10",
  wordClassName = "text-sm",
  sublabel,
}: LockupProps) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <XRayMark className={cn("shrink-0 text-foreground", markClassName)} />
      <div className="flex min-w-0 flex-col leading-tight">
        <Wordmark className={wordClassName} />
        {sublabel && (
          <span className="truncate font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            {sublabel}
          </span>
        )}
      </div>
    </div>
  );
}
