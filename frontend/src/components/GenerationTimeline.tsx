"use client";

/**
 * `GenerationTimeline` — the spine of the instrument. The question is shown as
 * a muted lead-in; each *generated* token is an interactive chip underlined by
 * one of two encodings:
 *
 *   • certainty — probability of the chosen token (vermilion sure → ash unsure)
 *   • depth     — the layer where the logit lens locked onto the emitted token
 *                 (vermilion = knew instantly, ash = needed the whole network)
 *
 * Reasoning tokens (`<think>…</think>`) group into a dimmed block topped by an
 * entropy sparkline (spikes = hesitation mid-reasoning; click to jump there).
 * Attention arcs are drawn *over* the text from the selected token back to the
 * generated tokens that drove it (raw attention at the focused layer).
 * Hovering any chip highlights the same token in every other view; clicking a
 * chip rewinds every view to that decision (clicking it again unpins). A caret
 * blinks while streaming.
 */

import { memo, useCallback, useLayoutEffect, useRef, useState } from "react";

import { firstLockLayer } from "@/lib/steps";
import { displayToken, isStructuralToken } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import { VIZ, confidenceColor, depthColor, pct } from "@/lib/viz";
import type { StepData } from "@/lib/xray-protocol";

export type TimelineColorMode = "certainty" | "depth";

/** An attention arc target: a prior generated step and its influence in [0,1]. */
export interface TimelineArc {
  step: number;
  w: number;
}

interface GenerationTimelineProps {
  promptText: string;
  steps: StepData[];
  selectedStep: number;
  onSelect: (step: number) => void;
  streaming: boolean;
  colorMode: TimelineColorMode;
  numLayers: number;
  hoveredToken: string | null;
  onHoverToken: (token: string | null) => void;
  /** Arcs from the selected token back into the generation (weights ∈ [0,1]). */
  arcs: TimelineArc[];
}

/**
 * Memoized so a long trace (hundreds of chips) doesn't re-render every chip on
 * every streamed token — only the (at most two) chips whose `active`/`hovered`
 * status actually changed re-render. `step` keeps a stable identity once
 * appended (see `useXRay`'s reducer), so this comparison is cheap and correct.
 */
const TokenChip = memo(function TokenChip({
  step,
  active,
  hovered,
  colorMode,
  numLayers,
  onSelect,
  onHover,
  registerChip,
}: {
  step: StepData;
  active: boolean;
  hovered: boolean;
  colorMode: TimelineColorMode;
  numLayers: number;
  onSelect: (s: number) => void;
  onHover: (token: string | null) => void;
  registerChip: (step: number, el: HTMLButtonElement | null) => void;
}) {
  if (isStructuralToken(step.token)) return null;
  const lock = firstLockLayer(step, numLayers);
  const underline =
    colorMode === "depth" ? depthColor(lock, numLayers) : confidenceColor(step.prob);
  return (
    <button
      ref={(el) => registerChip(step.step, el)}
      type="button"
      onClick={() => onSelect(step.step)}
      onMouseEnter={() => onHover(step.token)}
      onMouseLeave={() => onHover(null)}
      aria-pressed={active}
      title={`step ${step.step} · ${step.phase} · chosen ${pct(step.prob)} · entropy ${step.entropy.toFixed(2)} nats · locked at L${lock}`}
      className={cn(
        "relative -mx-px cursor-pointer whitespace-pre rounded px-0.5",
        "transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
        // Direct hover gets the strong treatment (CSS :hover); the cross-view
        // same-token echo (`hovered`) is a soft tint only, so hovering "the"
        // doesn't flash rings across dozens of chips.
        active
          ? "bg-primary/35 text-foreground"
          : "text-foreground/90 hover:bg-accent/20",
        hovered && !active && "bg-accent/10",
      )}
      // The certainty/depth underline and the pinned ring are both box-shadows,
      // so they must live in one inline value (a Tailwind ring class would be
      // overridden by the inline style).
      style={{
        boxShadow: `inset 0 -2.5px 0 0 ${underline}${
          active ? ", 0 0 0 1px rgba(240,101,59,0.6)" : ""
        }`,
      }}
    >
      {displayToken(step.token)}
    </button>
  );
});

/**
 * Entropy across the reasoning trace — the model's "heartbeat" while it
 * thinks. Spikes are hesitation; click anywhere to jump to that step.
 */
function EntropySparkline({
  steps,
  selectedStep,
  onSelect,
}: {
  steps: StepData[];
  selectedStep: number;
  onSelect: (step: number) => void;
}) {
  const n = steps.length;
  if (n < 2) return null;
  const W = 560;
  const H = 30;
  const PAD_X = 2;
  const PAD_Y = 3;
  const maxE = steps.reduce((m, s) => Math.max(m, s.entropy), 0) || 1;
  const x = (i: number) => PAD_X + (i / (n - 1)) * (W - 2 * PAD_X);
  const y = (e: number) => H - PAD_Y - (e / maxE) * (H - 2 * PAD_Y);
  const line = steps
    .map((s, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(s.entropy).toFixed(1)}`)
    .join(" ");
  const hitW = (W - 2 * PAD_X) / (n - 1);
  const selIdx = steps.findIndex((s) => s.step === selectedStep);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="mb-1.5 block w-full"
      role="img"
      aria-label="Entropy across the reasoning trace — spikes are hesitation"
    >
      <path
        d={`${line} L ${x(n - 1).toFixed(1)} ${H - PAD_Y} L ${x(0).toFixed(1)} ${H - PAD_Y} Z`}
        fill="rgba(217,154,69,0.08)"
      />
      <path d={line} fill="none" stroke={VIZ.confMid} strokeWidth={1} strokeOpacity={0.85} />
      {selIdx >= 0 && (
        <circle cx={x(selIdx)} cy={y(steps[selIdx].entropy)} r={2.6} fill={VIZ.accent} />
      )}
      {steps.map((s, i) => (
        <rect
          key={s.step}
          x={x(i) - hitW / 2}
          y={0}
          width={hitW}
          height={H}
          fill="transparent"
          className="cursor-pointer"
          onClick={() => onSelect(s.step)}
        >
          <title>{`step ${s.step} · entropy ${s.entropy.toFixed(2)} nats`}</title>
        </rect>
      ))}
    </svg>
  );
}

/**
 * Curved attention arcs drawn over the wrapped text: selected chip → the
 * generated tokens that drove it (vermilion). Chip positions come from
 * live DOM measurement, re-taken on layout/resize, so the arcs survive
 * word-wrap. Arcs fade in (see `arc-fade` in globals.css) keyed on the
 * source step, so scrubbing swaps them smoothly instead of popping.
 */
function ArcOverlay({
  containerRef,
  chips,
  sourceStep,
  arcs,
  layoutKey,
}: {
  containerRef: React.RefObject<HTMLDivElement | null>;
  chips: React.RefObject<Map<number, HTMLButtonElement>>;
  sourceStep: number;
  arcs: TimelineArc[];
  layoutKey: string;
}) {
  const [paths, setPaths] = useState<{ d: string; t: number }[]>([]);
  const [resizeTick, setResizeTick] = useState(0);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setResizeTick((n) => n + 1));
    ro.observe(el);
    return () => ro.disconnect();
  }, [containerRef]);

  useLayoutEffect(() => {
    const container = containerRef.current;
    const src = chips.current?.get(sourceStep);
    if (!container || !src) {
      setPaths([]);
      return;
    }
    const c = container.getBoundingClientRect();
    const s = src.getBoundingClientRect();
    const sx = s.left + s.width / 2 - c.left;
    const sy = s.top - c.top + 1;
    const out: { d: string; t: number }[] = [];

    for (const a of arcs) {
      if (a.step === sourceStep) continue;
      const target = chips.current?.get(a.step);
      if (!target) continue;
      const r = target.getBoundingClientRect();
      const tx = r.left + r.width / 2 - c.left;
      const ty = r.top - c.top + 1;
      const my = Math.min(sy, ty) - 10 - 26 * a.w;
      out.push({
        d: `M ${sx} ${sy} Q ${(sx + tx) / 2} ${my} ${tx} ${ty}`,
        t: a.w,
      });
    }
    setPaths(out);
  }, [containerRef, chips, sourceStep, arcs, resizeTick, layoutKey]);

  if (paths.length === 0) return null;
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full overflow-visible" aria-hidden>
      {paths.map((p, i) => (
        <path
          key={`${sourceStep}-${i}`}
          d={p.d}
          className="arc-fade"
          fill="none"
          stroke={VIZ.accent}
          strokeWidth={0.75 + 2.25 * p.t}
          strokeOpacity={0.18 + 0.5 * p.t}
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}

export function GenerationTimeline({
  promptText,
  steps,
  selectedStep,
  onSelect,
  streaming,
  colorMode,
  numLayers,
  hoveredToken,
  onHoverToken,
  arcs,
}: GenerationTimelineProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chipsRef = useRef(new Map<number, HTMLButtonElement>());
  // Stable identity — passed as a prop into the memoized `TokenChip`, so
  // rebuilding it every render would defeat the memoization.
  const registerChip = useCallback((step: number, el: HTMLButtonElement | null) => {
    if (el) chipsRef.current.set(step, el);
    else chipsRef.current.delete(step);
  }, []);

  const think = steps.filter((s) => s.phase === "think");
  const answer = steps.filter((s) => s.phase === "answer");
  // Structural markers (<think>/</think>) are counted in `think` but render as
  // no chip (see TokenChip) — exclude them so the count matches what's visible.
  const contentThink = think.filter((s) => !isStructuralToken(s.token));

  const chip = (s: StepData) => (
    <TokenChip
      key={s.step}
      step={s}
      active={s.step === selectedStep}
      hovered={hoveredToken !== null && s.token === hoveredToken}
      colorMode={colorMode}
      numLayers={numLayers}
      onSelect={onSelect}
      onHover={onHoverToken}
      registerChip={registerChip}
    />
  );

  return (
    <div ref={containerRef} className="relative flex flex-col gap-3">
      <p className="font-mono text-[13px] text-muted-foreground/70">
        <span className="text-muted-foreground/50">Q&nbsp;</span>
        {promptText}
      </p>

      {think.length > 0 && (
        <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
          <div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70">
            <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/60" aria-hidden />
            reasoning · {contentThink.length} tokens
          </div>
          <EntropySparkline steps={contentThink} selectedStep={selectedStep} onSelect={onSelect} />
          <div className="flex flex-wrap items-baseline gap-y-1 font-mono text-[13px] leading-relaxed text-muted-foreground">
            {think.map(chip)}
            {streaming && answer.length === 0 && <Caret />}
          </div>
        </div>
      )}

      {(answer.length > 0 || think.length === 0) && (
        <div className="flex flex-wrap items-baseline gap-y-1.5 font-mono text-[15px] leading-loose">
          {think.length > 0 && (
            <span className="mr-1.5 mt-0.5 self-start rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wider text-accent">
              answer
            </span>
          )}
          {(think.length > 0 ? answer : steps).map(chip)}
          {streaming && (think.length === 0 || answer.length > 0) && <Caret />}
        </div>
      )}

      <ArcOverlay
        containerRef={containerRef}
        chips={chipsRef}
        sourceStep={selectedStep}
        arcs={arcs}
        layoutKey={`${steps.length}-${colorMode}`}
      />
    </div>
  );
}

function Caret() {
  return (
    <span
      className="ml-0.5 inline-block h-[1.1em] w-[2px] translate-y-[0.18em] animate-pulse bg-accent"
      aria-hidden
    />
  );
}
