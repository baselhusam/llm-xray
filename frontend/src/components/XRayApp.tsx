"use client";

/**
 * `XRayApp` — the dashboard shell. A persistent control sidebar (prompt, thinking
 * toggle, examples, history) drives one chat-templated generation; the main
 * canvas streams the instrument views as the model answers.
 *
 * Cross-view state: which generated token is selected (`selectedStep` — follows
 * the head of the stream until pinned), which layer is focused (set by clicking
 * a trajectory row, drives raw attention), the attention mode (rollout vs raw),
 * and whether thinking is requested. Scrubbing the timeline / confidence curve
 * rewinds every view to that token's decision. D3 owns SVG interiors; React owns
 * state and layout.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { AppSidebar } from "@/components/AppSidebar";
import { ArchitectureFlow } from "@/components/ArchitectureFlow";
import { AttentionContext } from "@/components/AttentionContext";
import { ConfidenceCurve } from "@/components/ConfidenceCurve";
import { ExamplePrompts } from "@/components/ExamplePrompts";
import {
  GenerationTimeline,
  type TimelineArc,
  type TimelineColorMode,
} from "@/components/GenerationTimeline";
import { HowItWorks } from "@/components/HowItWorks";
import { KeyMoments } from "@/components/KeyMoments";
import { LensHeatmap } from "@/components/LensHeatmap";
import { LensTrajectory } from "@/components/LensTrajectory";
import {
  REPLAY_SPEEDS,
  REPLAY_TICK_MS,
  ReplayControls,
} from "@/components/ReplayControls";
import { ShareSection } from "@/components/ShareSection";
import { StepDistribution } from "@/components/StepDistribution";
import { ViewTabs, type XRayView } from "@/components/ViewTabs";
import { VizSkeleton } from "@/components/VizSkeleton";
import { XRayProgress } from "@/components/XRayProgress";
import { useDevices } from "@/hooks/useDevices";
import { useMediaQuery, DESKTOP_QUERY } from "@/hooks/useMediaQuery";
import { useXRay } from "@/hooks/useXRay";
import { contextTokens, rolloutRow, stepAttentionRow } from "@/lib/attention";
import { firstLockLayer, keyMoments } from "@/lib/steps";
import { cn } from "@/lib/utils";
import { displayToken, isContentToken, isStructuralToken } from "@/lib/tokens";
import { pct } from "@/lib/viz";
import {
  DEFAULT_MAX_TOKENS,
  MODEL_LABEL,
  NUM_LAYERS,
  STOP_REASON_LABEL,
  type DeviceName,
  type TopPrediction,
} from "@/lib/xray-protocol";

const fadeUp = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.25, ease: "easeOut" as const },
};

const HISTORY_CAP = 12;

type AttnMode = "rollout" | "raw";

function Panel({
  title,
  hint,
  children,
  className,
  hero,
  actions,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
  hero?: boolean;
  actions?: React.ReactNode;
}) {
  return (
    <section
      className={cn(
        "flex flex-col gap-3 rounded-xl p-4 sm:p-5",
        hero ? "panel-ring" : "surface-panel border border-border/60",
        className,
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="flex items-center gap-2 text-[13px] font-medium">
          <span
            className={cn(
              "h-1.5 w-1.5 rounded-full",
              hero ? "bg-primary" : "bg-muted-foreground/50",
            )}
            aria-hidden
          />
          {title}
        </h2>
        {actions ?? (hint && (
          <span className="font-mono text-[10px] text-muted-foreground">{hint}</span>
        ))}
      </div>
      {children}
    </section>
  );
}

export function XRayApp({ initialPrompt }: { initialPrompt?: string }) {
  const xray = useXRay();
  const isDesktop = useMediaQuery(DESKTOP_QUERY);
  const devices = useDevices();

  const [thinking, setThinking] = useState(true);
  const [maxTokens, setMaxTokens] = useState(DEFAULT_MAX_TOKENS);
  const [device, setDevice] = useState<DeviceName>("cpu");
  // Seed the pill from whatever the backend auto-selected (cuda > mps > cpu)
  // once `/api/devices` resolves — but only once, so it doesn't clobber a
  // choice the user already made while the fetch was in flight.
  const deviceSeeded = useRef(false);
  useEffect(() => {
    if (devices.current && !deviceSeeded.current) {
      deviceSeeded.current = true;
      setDevice(devices.current);
    }
  }, [devices]);
  const [pinnedStep, setPinnedStep] = useState<number | null>(null);
  // null = auto-follow the first-lock layer; a number = a layer the user pinned.
  const [layerPinned, setLayerPinned] = useState<number | null>(null);
  const [attnMode, setAttnMode] = useState<AttnMode>("rollout");
  const [view, setView] = useState<XRayView>("trajectory");
  const [history, setHistory] = useState<string[]>([]);
  // Timeline underline encoding: chosen-token probability vs. lock depth.
  const [colorMode, setColorMode] = useState<TimelineColorMode>("certainty");
  // Cross-view token highlight (raw token string, matched everywhere).
  const [hoveredToken, setHoveredToken] = useState<string | null>(null);
  // Replay: a timer drives `pinnedStep` through the finished generation.
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(REPLAY_SPEEDS[0]);

  const handleSubmit = (prompt: string) => {
    setPinnedStep(null);
    setLayerPinned(null);
    setAttnMode("rollout");
    setView("trajectory");
    setPlaying(false);
    setHoveredToken(null);
    setHistory((h) => [prompt, ...h.filter((p) => p !== prompt)].slice(0, HISTORY_CAP));
    xray.run(prompt, thinking, device, maxTokens);
  };

  // Clicking a layer pins it; clicking the already-active one returns to auto.
  const handleSelectLayer = (layer: number) =>
    setLayerPinned((p) => (p === layer ? null : layer));

  // Auto-run a prompt arriving from a shared `/?prompt=…` link, exactly once.
  const autoRan = useRef(false);
  useEffect(() => {
    if (initialPrompt && !autoRan.current) {
      autoRan.current = true;
      handleSubmit(initialPrompt);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPrompt]);

  const busy = xray.status === "connecting" || xray.status === "streaming";
  const numLayers = xray.meta?.num_layers ?? NUM_LAYERS;
  const modelLabel = xray.meta?.model_label ?? MODEL_LABEL;

  const steps = xray.steps;
  const hasGen = steps.length > 0;
  const lastStep = steps.length - 1;
  const selectedStep = hasGen ? Math.min(pinnedStep ?? lastStep, lastStep) : 0;
  const current = hasGen ? steps[selectedStep] : null;

  // Scrubbing by hand (chip / curve / key-moment click / arrow keys) always
  // interrupts an in-flight replay. Stable identity (useCallback) so it
  // doesn't defeat memoization on the hundreds of chips that receive it.
  const scrubTo = useCallback((step: number) => {
    setPlaying(false);
    setPinnedStep(step);
  }, []);

  // Timeline chips toggle: clicking the already-pinned token unpins it
  // (back to auto-follow), clicking any other pins that one. KeyMoments /
  // ConfidenceCurve / sparkline keep plain `scrubTo` — jumping is their job.
  const toggleStep = useCallback((step: number) => {
    setPlaying(false);
    setPinnedStep((p) => (p === step ? null : step));
  }, []);

  // Clicking anywhere that doesn't operate on the pinned token deselects it
  // entirely (arcs and highlights included). The surfaces that legitimately
  // explore the pinned token — the generation spine, the instrument panels,
  // the architecture flow — are marked `data-keep-pin`; everything else
  // (canvas whitespace, header, answer, sidebar, page margin) clears the pin.
  // Only listens while something is pinned.
  useEffect(() => {
    if (pinnedStep === null) return;
    const onPointerDown = (e: MouseEvent) => {
      const el = e.target instanceof Element ? e.target : null;
      if (!el?.closest("[data-keep-pin]")) {
        setPlaying(false);
        setPinnedStep(null);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, [pinnedStep]);

  const togglePlay = () => {
    if (!hasGen || busy) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    // Restart from the top when unpinned or already at the end.
    if (pinnedStep === null || pinnedStep >= lastStep) setPinnedStep(0);
    setPlaying(true);
  };
  const cycleSpeed = () =>
    setSpeed((s) => REPLAY_SPEEDS[(REPLAY_SPEEDS.indexOf(s) + 1) % REPLAY_SPEEDS.length]);

  // Latest values for the replay ticker and the global keyboard handler
  // (synced in an effect, not during render — see conventions).
  const liveCtx = useRef({ hasGen, busy, selectedStep, lastStep, togglePlay });
  useEffect(() => {
    liveCtx.current = { hasGen, busy, selectedStep, lastStep, togglePlay };
  });

  // Replay ticker — greedy decoding is deterministic, so stepping the pinned
  // step through the stored generation replays the run exactly. Stops itself
  // at the end of the generation. (`playing` can't outlive a run: a new submit
  // resets it before `busy` flips.)
  useEffect(() => {
    if (!playing || busy || !hasGen) return;
    const id = window.setInterval(() => {
      const ctx = liveCtx.current;
      if (ctx.selectedStep >= ctx.lastStep) {
        setPlaying(false);
        return;
      }
      setPinnedStep(ctx.selectedStep + 1);
    }, REPLAY_TICK_MS / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, busy, hasGen]);
  // Global keyboard scrubbing: ←/→ step, space toggles replay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLElement && t.closest("input, textarea, select, [contenteditable=true]"))
        return;
      const ctx = liveCtx.current;
      if (!ctx.hasGen) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        const next = ctx.selectedStep + (e.key === "ArrowLeft" ? -1 : 1);
        setPinnedStep(Math.max(0, Math.min(ctx.lastStep, next)));
      } else if (e.key === "Escape") {
        setPlaying(false);
        setPinnedStep(null);
      } else if (e.key === " ") {
        // Let space keep activating a focused button.
        if (t instanceof HTMLElement && t.closest("button")) return;
        e.preventDefault();
        ctx.togglePlay();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // The "current block": the shallowest layer whose logit-lens top-1 first locks
  // onto the token the model actually emitted — the depth where the decision
  // crystallizes. The block detail auto-follows it per token (recomputed as you
  // scrub / as the stream advances) unless the user pins a layer by clicking.
  const firstLock = useMemo(
    () => (current ? firstLockLayer(current, numLayers) : numLayers - 1),
    [current, numLayers],
  );
  const effLayer = Math.min(layerPinned ?? firstLock, numLayers - 1);

  // The model's actual next-token distribution == the final layer's logit lens.
  const distribution: TopPrediction[] = useMemo(
    () => (current ? current.trajectory[current.trajectory.length - 1]?.top_predictions ?? [] : []),
    [current],
  );
  const runnerUp = distribution[1] ?? null;

  const ctxTokens = useMemo(
    () => (hasGen ? contextTokens(xray.promptTokens ?? [], steps, selectedStep) : []),
    [hasGen, xray.promptTokens, steps, selectedStep],
  );

  // Rollout is O(L·S³); compute it lazily for the selected step, and skip the
  // heavy work while the unpinned head is still streaming (it changes every tick).
  // Deliberately NOT keyed on `steps.length`/`steps` identity: once a step is
  // pinned, `selectedStep` alone determines the result (its context is fixed),
  // so re-running this on every later token that streams in — an O(L·S³) pass
  // over hundreds of tokens, tens of times a second — would peg the main thread
  // and freeze the UI for the rest of a long reasoning trace.
  const rollout = useMemo(() => {
    if (attnMode !== "rollout" || !current || !xray.promptAttention) return null;
    if (busy && pinnedStep === null) return null;
    return rolloutRow(xray.promptAttention, steps, selectedStep);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attnMode, current, xray.promptAttention, selectedStep, busy, pinnedStep]);

  const rawWeights = useMemo(
    () => (current ? stepAttentionRow(current, effLayer) : []),
    [current, effLayer],
  );
  const attnWeights = rollout ?? rawWeights;
  const showingRollout = attnMode === "rollout" && rollout !== null;

  // The generation's near-misses — the steps worth clicking.
  const moments = useMemo(() => keyMoments(steps), [steps]);

  const P = xray.promptTokens?.length ?? 0;

  // Attention arcs over the timeline: the selected token's strongest influences
  // among *generated* tokens (context indices P+j ↔ steps[j]), weights
  // normalized to the strongest target. Deliberately fed from the RAW attention
  // row at the effective layer — never rollout, whose Π(0.5A+0.5I) product
  // mathematically piles mass onto the earliest positions and made every arc
  // point at the first few reasoning tokens regardless of content. Punctuation/
  // whitespace-only tokens are skipped too (secondary sinks, no meaning).
  const timelineArcs = useMemo<TimelineArc[]>(() => {
    if (!current) return [];
    const gen: { step: number; w: number }[] = [];
    for (let j = 0; j < selectedStep; j++) {
      const w = rawWeights[P + j];
      if (w === undefined) break;
      const tok = steps[j].token;
      if (isStructuralToken(tok) || !isContentToken(tok)) continue;
      gen.push({ step: steps[j].step, w });
    }
    let max = 0;
    for (const g of gen) if (g.w > max) max = g.w;
    if (max <= 0) return [];
    return gen
      .filter((g) => g.w >= 0.15 * max)
      .sort((a, b) => b.w - a.w)
      .slice(0, 5)
      .map((g) => ({ step: g.step, w: g.w / max }));
  }, [current, rawWeights, P, selectedStep, steps]);

  const showViz = hasGen || busy;

  const trajectoryBody = current ? (
    <LensTrajectory
      trajectory={current.trajectory}
      finalTokenId={current.token_id}
      selectedLayer={effLayer}
      onSelectLayer={handleSelectLayer}
    />
  ) : (
    <VizSkeleton variant="trajectory" />
  );

  const heatmapBody = current ? (
    <LensHeatmap
      trajectory={current.trajectory}
      chosenTokenId={current.token_id}
      hoveredToken={hoveredToken}
      onHoverToken={setHoveredToken}
    />
  ) : (
    <VizSkeleton variant="heatmap" />
  );

  const distributionBody = current ? (
    <div className="flex flex-col gap-3">
      <StepDistribution
        predictions={distribution}
        chosenTokenId={current.token_id}
        hoveredToken={hoveredToken}
        onHoverToken={setHoveredToken}
      />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted-foreground">
        <span>
          entropy <span className="text-foreground">{current.entropy.toFixed(2)}</span> nats
        </span>
        {runnerUp && (
          <span>
            almost{" "}
            <span className="whitespace-pre text-accent">{displayToken(runnerUp.token)}</span>{" "}
            {pct(runnerUp.prob)}
          </span>
        )}
      </div>
    </div>
  ) : (
    <VizSkeleton variant="distribution" />
  );

  const attentionBody = current ? (
    <div className="flex flex-col gap-2">
      <AttentionContext
        tokens={ctxTokens}
        weights={attnWeights}
        producedToken={current.token}
        hoveredToken={hoveredToken}
        onHoverToken={setHoveredToken}
      />
      <p className="font-mono text-[10px] leading-relaxed text-muted-foreground/70">
        Excludes the first-token attention sink &amp; chat-template scaffolding
        (shown muted) — they absorb most attention but carry no content.
      </p>
    </div>
  ) : (
    <VizSkeleton variant="attention" />
  );

  const attnActions = (
    <div className="flex items-center gap-0.5 rounded-md border border-border bg-muted/40 p-0.5">
      {(["rollout", "raw"] as const).map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={attnMode === m}
          onClick={() => setAttnMode(m)}
          className={cn(
            "rounded px-2 py-0.5 font-mono text-[11px] transition-colors",
            attnMode === m
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {m}
        </button>
      ))}
    </div>
  );

  const PANELS: Record<
    XRayView,
    { title: string; hint?: string; actions?: React.ReactNode; body: React.ReactNode }
  > = {
    trajectory: {
      title: "Logit-lens trajectory",
      hint: `${numLayers} layers`,
      body: trajectoryBody,
    },
    heatmap: {
      title: "Logit-lens heatmap",
      hint: "layer × candidate · prob",
      body: heatmapBody,
    },
    distribution: {
      title: "Next-token candidates",
      hint: "what it considered",
      body: distributionBody,
    },
    attention: {
      title: "Attention attribution",
      actions: attnActions,
      hint: showingRollout
        ? "rollout · all layers"
        : attnMode === "rollout"
          ? "raw (rollout after generation)"
          : `raw · layer ${effLayer}`,
      body: attentionBody,
    },
  };

  const answerText = steps.filter((s) => s.phase === "answer").map((s) => s.token).join("");
  const thinkCount = steps.filter((s) => s.phase === "think").length;

  // A compact status descriptor for the canvas header.
  const phaseNow = hasGen ? steps[lastStep].phase : null;
  const statusText =
    xray.status === "connecting"
      ? "Connecting…"
      : xray.status === "streaming"
        ? phaseNow === "think"
          ? "Reasoning…"
          : "Answering…"
        : xray.done
          ? STOP_REASON_LABEL[xray.done.stop_reason] ?? "Done"
          : "Ready";

  return (
    <div className="lg:flex lg:h-screen lg:overflow-hidden">
      {/* Sidebar — sticky full-height rail on desktop, stacked header on mobile. */}
      <aside className="surface-sidebar thin-scroll border-b border-border/60 lg:h-screen lg:w-[300px] lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-b-0">
        <AppSidebar
          modelLabel={modelLabel}
          busy={busy}
          initialPrompt={initialPrompt}
          thinking={thinking}
          onToggleThinking={() => setThinking((t) => !t)}
          maxTokens={maxTokens}
          onMaxTokensChange={setMaxTokens}
          device={device}
          availableDevices={devices.available}
          onDeviceChange={setDevice}
          onSubmit={handleSubmit}
          history={history}
          onPickHistory={handleSubmit}
        />
      </aside>

      {/* Canvas — independently scrolling instrument area. */}
      <div className="thin-scroll min-w-0 flex-1 lg:h-screen lg:overflow-y-auto">
        <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-5 py-6 sm:px-8 sm:py-8">
          {/* Canvas header */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/60 pb-4">
            <div className="flex min-w-0 flex-col gap-0.5">
              <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground/80">
                {showViz ? "Now analyzing" : "Live forward pass"}
              </span>
              <span className="truncate text-sm font-medium text-foreground">
                {xray.prompt || "Type a prompt to begin"}
              </span>
            </div>
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium",
                busy
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border bg-card/40 text-muted-foreground",
              )}
            >
              {busy && <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-accent" aria-hidden />}
              {statusText}
              {hasGen && (
                <span className="font-mono text-muted-foreground/80">· {steps.length} tok</span>
              )}
            </span>
          </div>

          {/* Error */}
          <AnimatePresence>
            {xray.status === "error" && xray.error && (
              <motion.div
                key="error"
                {...fadeUp}
                role="alert"
                className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive"
              >
                {xray.error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {!showViz && (
            <motion.div
              key="empty"
              {...fadeUp}
              className="flex flex-col items-center gap-6 py-16 text-center sm:py-24"
            >
              <h1 className="bg-gradient-to-b from-foreground to-primary bg-clip-text text-4xl font-bold tracking-tight text-transparent sm:text-5xl">
                Watch an AI think.
              </h1>
              <p className="max-w-md text-balance text-sm text-muted-foreground sm:text-base">
                Ask the model a question and watch it reason token by token — the
                alternatives it weighed and how each prediction forms across the
                network&apos;s depth. Then share the x-ray.
              </p>
              <ExamplePrompts onPick={handleSubmit} busy={busy} className="max-w-lg" />
            </motion.div>
          )}

          {/* Progress */}
          <AnimatePresence>
            {busy && (
              <motion.div key="progress" {...fadeUp}>
                <XRayProgress
                  status={xray.status}
                  done={steps.length}
                  cap={xray.meta?.max_tokens}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Generation spine */}
          {showViz && xray.prompt && (
            <motion.section
              key="spine"
              {...fadeUp}
              data-keep-pin
              className="surface-panel flex flex-col gap-4 rounded-xl border border-border/60 p-4 sm:p-5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground/80">
                  Generation
                  <span className="ml-2 font-normal normal-case tracking-normal text-muted-foreground/60">
                    click any token · ←/→ scrub · space replay
                  </span>
                </h2>
                {hasGen && (
                  <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center gap-0.5 rounded-md border border-border bg-muted/40 p-0.5">
                      {(["certainty", "depth"] as const).map((m) => (
                        <button
                          key={m}
                          type="button"
                          aria-pressed={colorMode === m}
                          onClick={() => setColorMode(m)}
                          title={
                            m === "depth"
                              ? "Underline = layer where the decision locked (hot = knew early)"
                              : "Underline = probability of the chosen token"
                          }
                          className={cn(
                            "rounded px-2 py-0.5 font-mono text-[11px] transition-colors",
                            colorMode === m
                              ? "bg-primary text-primary-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                    <ReplayControls
                      playing={playing}
                      speed={speed}
                      disabled={busy}
                      onToggle={togglePlay}
                      onCycleSpeed={cycleSpeed}
                    />
                    <span className="font-mono text-[10px] text-muted-foreground">
                      step {selectedStep + 1} / {steps.length}
                    </span>
                  </div>
                )}
              </div>
              {moments.length > 0 && (
                <KeyMoments moments={moments} selectedStep={selectedStep} onSelect={scrubTo} />
              )}
              <GenerationTimeline
                promptText={xray.prompt}
                steps={steps}
                selectedStep={selectedStep}
                onSelect={toggleStep}
                streaming={busy}
                colorMode={colorMode}
                numLayers={numLayers}
                hoveredToken={hoveredToken}
                onHoverToken={setHoveredToken}
                arcs={timelineArcs}
              />
              {hasGen && (
                <div className="border-t border-border/50 pt-3">
                  <ConfidenceCurve steps={steps} selectedStep={selectedStep} onSelect={scrubTo} />
                </div>
              )}
            </motion.section>
          )}

          {/* Architecture hero — the model's topology with the live pass flowing through it */}
          {showViz && (
            <motion.div key="architecture" {...fadeUp} data-keep-pin>
              <Panel
                hero
                title="Architecture · live forward pass"
                hint={`decoder-only · ${numLayers} blocks`}
              >
                {current ? (
                  <ArchitectureFlow
                    trajectory={current.trajectory}
                    attentionRow={current.attention_row}
                    contextTokens={ctxTokens}
                    finalTokenId={current.token_id}
                    emittedToken={current.token}
                    emittedProb={current.prob}
                    inputToken={ctxTokens[ctxTokens.length - 1] ?? ""}
                    numLayers={numLayers}
                    selectedLayer={effLayer}
                    onSelectLayer={handleSelectLayer}
                    autoFollow={layerPinned === null}
                    streaming={busy}
                    stepKey={selectedStep}
                    phase={current.phase}
                  />
                ) : (
                  <VizSkeleton variant="trajectory" />
                )}
              </Panel>
            </motion.div>
          )}

          {/* Instrument views */}
          {showViz &&
            (isDesktop ? (
              <motion.div key="viz-desktop" {...fadeUp} data-keep-pin className="flex flex-col gap-5">
                <div className="grid items-start gap-5 lg:grid-cols-2">
                  <Panel
                    hero
                    title={PANELS.trajectory.title}
                    hint={PANELS.trajectory.hint}
                    className="lg:row-span-2"
                  >
                    {PANELS.trajectory.body}
                  </Panel>
                  <Panel title={PANELS.distribution.title} hint={PANELS.distribution.hint}>
                    {PANELS.distribution.body}
                  </Panel>
                  <Panel
                    title={PANELS.attention.title}
                    hint={PANELS.attention.hint}
                    actions={PANELS.attention.actions}
                  >
                    {PANELS.attention.body}
                  </Panel>
                </div>
                <Panel title={PANELS.heatmap.title} hint={PANELS.heatmap.hint}>
                  {PANELS.heatmap.body}
                </Panel>
              </motion.div>
            ) : (
              <motion.div key="viz-mobile" {...fadeUp} data-keep-pin className="flex flex-col gap-3">
                <ViewTabs value={view} onChange={setView} />
                <AnimatePresence mode="wait">
                  <motion.div
                    key={view}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                  >
                    <Panel
                      title={PANELS[view].title}
                      hint={PANELS[view].hint}
                      actions={PANELS[view].actions}
                    >
                      {PANELS[view].body}
                    </Panel>
                  </motion.div>
                </AnimatePresence>
              </motion.div>
            ))}

          {/* Answer — appears live with the first answer-phase token and keeps
              filling as the model writes; stats land on done. */}
          <AnimatePresence>
            {(xray.done || answerText.trim().length > 0) && (
              <motion.section
                key="done"
                {...fadeUp}
                className="glow-primary flex flex-col gap-3 rounded-xl bg-card/60 p-5"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground/80">
                    Answer
                  </span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {xray.done ? (
                      <>
                        {thinkCount > 0 && `${thinkCount} reasoning · `}
                        {xray.done.num_steps} tokens ·{" "}
                        {STOP_REASON_LABEL[xray.done.stop_reason] ?? xray.done.stop_reason}
                      </>
                    ) : (
                      "streaming…"
                    )}
                  </span>
                </div>
                <p className="text-lg leading-relaxed text-foreground">
                  {answerText.trim() || (xray.done ? xray.done.generated_text.trim() : "")}
                  {busy && (
                    <span
                      className="ml-1 inline-block h-[1.05em] w-[2px] translate-y-[0.15em] animate-pulse bg-accent"
                      aria-hidden
                    />
                  )}
                </p>
              </motion.section>
            )}
          </AnimatePresence>

          {/* Share */}
          {xray.done && xray.prompt && hasGen && (
            <motion.div key="share" {...fadeUp}>
              <ShareSection
                prompt={xray.prompt}
                modelLabel={modelLabel}
                answerText={(answerText.trim() || xray.done.generated_text).slice(0, 320)}
                steps={steps}
              />
            </motion.div>
          )}

          <div id="how-it-works" className="scroll-mt-6">
            <HowItWorks />
          </div>
        </main>
      </div>
    </div>
  );
}
