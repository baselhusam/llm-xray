"use client";

/**
 * `ArchitectureFlow` — the centerpiece: the model's *actual* topology with the
 * live forward pass flowing through it, token by token.
 *
 * Qwen3 is decoder-only — no encoder. So this draws the honest thing: a token
 * embedding at the bottom, the stack of N decoder blocks rising upward (the
 * residual stream), a final RMSNorm + unembedding (lm_head) at the top, and the
 * emitted token. Each block is lit by *real* data — its logit-lens top-1 guess
 * and confidence (vermilion once that guess locks onto the token the model finally
 * chose). A pulse sweeps up the stack each time a new token is produced.
 *
 * Click a block to expand its internal circuit on the right: RMSNorm → masked
 * self-attention (GQA) → +residual → RMSNorm → SwiGLU MLP → +residual. The
 * attention node shows which context tokens that block actually attended to
 * (from `attention_row`); the block output shows its logit-lens guess.
 *
 * Honesty note: the engine hooks each block's *output* (the residual stream),
 * so the two sublayer boxes are drawn for structure while the live numbers bind
 * at block granularity — surfaced in the caption so it's never misleading.
 */

import { motion } from "framer-motion";

import { isAttnSink } from "@/lib/attention";
import { displayToken } from "@/lib/tokens";
import { cn } from "@/lib/utils";
import { VIZ, confidenceColor, pct } from "@/lib/viz";
import type { LayerTopK } from "@/lib/xray-protocol";

interface ArchitectureFlowProps {
  /** Per-layer logit-lens for the selected step (ascending layer_idx). */
  trajectory: LayerTopK[];
  /** Per-layer attention rows for the selected step, shape (layers, key_len). */
  attentionRow: number[][];
  /** Context tokens aligned to the attention columns. */
  contextTokens: string[];
  /** The token the model actually committed to this step. */
  finalTokenId: number;
  emittedToken: string;
  emittedProb: number;
  /** The token whose forward pass this represents (last context token). */
  inputToken: string;
  numLayers: number;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  /** True when the selected layer is auto-following the first-lock block. */
  autoFollow: boolean;
  streaming: boolean;
  /** Changes per generated token — re-triggers the rising pulse. */
  stepKey: number;
  phase: "think" | "answer";
}

interface BlockRow {
  layer: number;
  token: string;
  prob: number;
  locked: boolean;
}

/** Faint node used for the embedding / unembedding caps and circuit boxes. */
function CapNode({
  label,
  children,
  tone = "muted",
}: {
  label: string;
  children?: React.ReactNode;
  tone?: "muted" | "primary" | "accent";
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-3 rounded-lg border px-3 py-2",
        tone === "primary" && "border-primary/40 bg-primary/10",
        tone === "accent" && "border-accent/40 bg-accent/10",
        tone === "muted" && "border-border/70 bg-card/40",
      )}
    >
      <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      {children}
    </div>
  );
}

function TokenPill({ token, tone }: { token: string; tone: "accent" | "muted" }) {
  return (
    <span
      className={cn(
        "whitespace-pre rounded-md border px-2 py-0.5 font-mono text-xs",
        tone === "accent"
          ? "border-accent/50 bg-accent/15 text-accent"
          : "border-border bg-muted/40 text-foreground",
      )}
    >
      {displayToken(token)}
    </span>
  );
}

/** One labeled stage in a block's internal circuit. */
function Step({ children, emphasis }: { children: React.ReactNode; emphasis?: boolean }) {
  return (
    <div
      className={cn(
        "rounded-md border px-3 py-1.5 text-center text-[11px]",
        emphasis
          ? "border-primary/40 bg-primary/10 text-foreground"
          : "border-border/70 bg-card/40 text-muted-foreground",
      )}
    >
      {children}
    </div>
  );
}

function Wire() {
  return <div className="mx-auto h-3 w-px bg-border" aria-hidden />;
}

function Add() {
  return (
    <div className="mx-auto flex size-6 items-center justify-center rounded-full border border-accent/50 bg-accent/10 text-xs text-accent">
      ⊕
    </div>
  );
}

/** The expanded internal circuit of one decoder block. */
function BlockDetail({
  layer,
  row,
  attended,
  autoFollow,
}: {
  layer: number;
  row: BlockRow | undefined;
  attended: { token: string; weight: number; rel: number }[];
  autoFollow: boolean;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <h3 className="flex items-baseline gap-2 font-mono text-sm font-semibold text-foreground">
          Decoder block L{layer}
          {autoFollow && (
            <span className="rounded bg-accent/15 px-1.5 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-accent">
              auto · lock point
            </span>
          )}
        </h3>
        {row && (
          <span className="font-mono text-[11px] text-muted-foreground">
            guess{" "}
            <span className={cn("whitespace-pre", row.locked ? "text-accent" : "text-foreground")}>
              {displayToken(row.token)}
            </span>{" "}
            {pct(row.prob, 0)}
          </span>
        )}
      </div>

      <Step>hidden state in →</Step>
      <Wire />
      <Step>RMSNorm</Step>
      <Wire />
      <Step emphasis>masked self-attention · GQA</Step>

      {/* Real attention readout for this block */}
      <div className="rounded-md border border-border/60 bg-background/40 px-3 py-2">
        <div className="mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
          attended to
        </div>
        {attended.length > 0 ? (
          <div className="flex flex-col gap-1">
            {attended.map((a, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-16 shrink-0 truncate whitespace-pre font-mono text-[11px] text-foreground">
                  {displayToken(a.token)}
                </span>
                <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted/40">
                  <span
                    className="block h-full rounded-full"
                    style={{ width: `${Math.max(4, Math.round(a.rel * 100))}%`, background: VIZ.weight }}
                  />
                </span>
                <span className="w-9 shrink-0 text-right font-mono text-[10px] text-muted-foreground">
                  {pct(a.weight, 0)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <span className="font-mono text-[11px] text-muted-foreground">—</span>
        )}
      </div>

      <Add />
      <div className="text-center text-[10px] text-muted-foreground">+ residual</div>
      <Wire />
      <Step>RMSNorm</Step>
      <Wire />
      <Step emphasis>SwiGLU MLP</Step>
      <Add />
      <div className="text-center text-[10px] text-muted-foreground">+ residual</div>
      <Wire />
      <Step>hidden state out →</Step>

      <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground/80">
        Sublayer boxes are structural; live values bind to this block&apos;s output
        (the residual stream, via the logit lens) and its attention.
      </p>
    </div>
  );
}

export function ArchitectureFlow({
  trajectory,
  attentionRow,
  contextTokens,
  finalTokenId,
  emittedToken,
  emittedProb,
  inputToken,
  numLayers,
  selectedLayer,
  onSelectLayer,
  autoFollow,
  streaming,
  stepKey,
  phase,
}: ArchitectureFlowProps) {
  const rows: BlockRow[] = trajectory.map((t) => {
    const top1 = t.top_predictions[0];
    return {
      layer: t.layer_idx,
      token: top1?.token ?? "",
      prob: top1?.prob ?? 0,
      locked: top1?.token_id === finalTokenId,
    };
  });
  const firstLock = rows.find((r) => r.locked)?.layer ?? -1;
  // Top → bottom in the DOM = output → input, so the stack visibly rises.
  const topDown = [...rows].reverse();

  const selRow = rows.find((r) => r.layer === selectedLayer);
  const selAttn = attentionRow[selectedLayer] ?? [];
  // Exclude the first-token attention sink + chat scaffolding — otherwise this
  // readout is dominated by `<|im_start|>` and tells you nothing about content.
  const ranked = selAttn
    .map((weight, i) => ({ weight, token: contextTokens[i] ?? "", i }))
    .filter((a) => a.token !== "" && !isAttnSink(a.token, a.i))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 4);
  // Bars are scaled relative to the top content token so they stay legible;
  // the percentage label keeps the true attention weight.
  const topW = ranked[0]?.weight || 1;
  const attended = ranked.map((a) => ({ ...a, rel: a.weight / topW }));

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)]">
      {/* Left: the stack */}
      <div className="flex flex-col gap-2">
        {/* Output cap */}
        <CapNode label="unembedding · lm_head" tone="accent">
          <span className="flex items-center gap-2">
            <TokenPill token={emittedToken} tone="accent" />
            <span className="font-mono text-[11px] text-accent">{pct(emittedProb, 0)}</span>
          </span>
        </CapNode>
        <div className="text-center text-[10px] text-muted-foreground">final RMSNorm ↑</div>

        {/* The decoder stack */}
        <div className="relative">
          {/* Rising pulse — sweeps up once per produced token. */}
          <motion.div
            key={stepKey}
            aria-hidden
            className="pointer-events-none absolute inset-x-0 z-0 h-14 rounded-full"
            style={{
              background:
                "linear-gradient(to top, transparent, color-mix(in oklch, var(--accent) 26%, transparent), transparent)",
            }}
            initial={{ top: "100%", opacity: streaming ? 0.9 : 0.5 }}
            animate={{ top: "-3.5rem", opacity: 0 }}
            transition={{ duration: 0.95, ease: "easeOut" }}
          />

          <div className="thin-scroll relative z-10 flex max-h-[460px] flex-col gap-px overflow-y-auto pr-1">
            {topDown.map((r) => {
              const active = r.layer === selectedLayer;
              const barColor = r.locked ? VIZ.accent : confidenceColor(r.prob);
              return (
                <button
                  key={r.layer}
                  type="button"
                  onClick={() => onSelectLayer(r.layer)}
                  aria-pressed={active}
                  className={cn(
                    "group flex items-center gap-2 rounded-md px-2 py-1 text-left transition-colors",
                    active ? "bg-white/[0.07] ring-1 ring-primary/40" : "hover:bg-white/[0.04]",
                  )}
                >
                  <span
                    className={cn(
                      "w-8 shrink-0 font-mono text-[10px]",
                      active ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    L{r.layer}
                  </span>
                  {/* Block glyph with attn/mlp sublayer ticks */}
                  <span
                    className="flex h-4 w-10 shrink-0 items-center justify-center gap-0.5 rounded-sm border"
                    style={{
                      borderColor: r.locked ? "color-mix(in oklch, var(--accent) 50%, transparent)" : "var(--border)",
                      background: `color-mix(in oklch, ${barColor} 22%, transparent)`,
                    }}
                    aria-hidden
                  >
                    <span className="h-2 w-1 rounded-[1px]" style={{ background: barColor }} />
                    <span className="h-2 w-1 rounded-[1px] opacity-60" style={{ background: barColor }} />
                  </span>
                  <span
                    className={cn(
                      "w-20 shrink-0 truncate whitespace-pre font-mono text-[11px]",
                      r.locked ? "text-accent" : "text-foreground",
                      r.layer === firstLock && "font-semibold",
                    )}
                  >
                    {displayToken(r.token)}
                  </span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: VIZ.track }}>
                    <span
                      className="block h-full rounded-full transition-[width] duration-300"
                      style={{ width: `${Math.max(2, Math.round(r.prob * 100))}%`, background: barColor }}
                    />
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Input cap */}
        <div className="text-center text-[10px] text-muted-foreground">↑ token embedding</div>
        <CapNode label={`input token · ${phase === "think" ? "reasoning" : "answer"}`} tone="muted">
          <TokenPill token={inputToken} tone="muted" />
        </CapNode>
      </div>

      {/* Right: block detail */}
      <div className="rounded-xl border border-border/60 bg-background/30 p-4">
        <BlockDetail layer={selectedLayer} row={selRow} attended={attended} autoFollow={autoFollow} />
        <p className="mt-3 border-t border-border/50 pt-3 text-[11px] text-muted-foreground">
          {numLayers} decoder blocks · GQA self-attention · SwiGLU MLP · RMSNorm.
          {autoFollow
            ? " Auto-following the lock point; click any block to pin it."
            : " Pinned — click the active block again to resume auto-follow."}
        </p>
      </div>
    </div>
  );
}
