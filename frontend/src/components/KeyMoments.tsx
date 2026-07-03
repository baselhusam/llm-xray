"use client";

/**
 * `KeyMoments` — the generation's near-misses, surfaced as clickable chips.
 * Each is a step where the runner-up almost won ("almost said X"); clicking
 * jumps every instrument to that decision. This is the antidote to a
 * 200-token reasoning trace: the interesting clicks come to you.
 */

import { displayToken } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import { pct } from "@/lib/viz";
import type { KeyMoment } from "@/lib/steps";

interface KeyMomentsProps {
  moments: KeyMoment[];
  selectedStep: number;
  onSelect: (step: number) => void;
}

export function KeyMoments({ moments, selectedStep, onSelect }: KeyMomentsProps) {
  if (moments.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground/70">
        key moments
      </span>
      {moments.map((m) => (
        <button
          key={m.step}
          type="button"
          onClick={() => onSelect(m.step)}
          aria-pressed={m.step === selectedStep}
          title={`step ${m.step} · chose ${pct(m.chosenProb)} over ${pct(m.runnerUpProb)}`}
          className={cn(
            "rounded-full border px-2.5 py-0.5 font-mono text-[11px] transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
            m.step === selectedStep
              ? "border-accent/60 bg-accent/15 text-foreground"
              : "border-border bg-muted/30 text-muted-foreground hover:border-accent/40 hover:text-foreground",
          )}
        >
          <span className="whitespace-pre text-foreground">{displayToken(m.chosen)}</span>
          <span className="text-muted-foreground/70"> not </span>
          <span className="whitespace-pre text-accent">{displayToken(m.runnerUp)}</span>
          <span className="text-muted-foreground/60"> {pct(m.runnerUpProb, 0)}</span>
        </button>
      ))}
    </div>
  );
}
