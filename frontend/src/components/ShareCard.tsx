"use client";

/**
 * `ShareCard` — the standalone, branded artifact captured to PNG for sharing.
 *
 * It must read well with zero context (someone scrolling a feed), so it leads
 * with what the instrument *found*: a real per-token confidence curve over the
 * whole generation (reasoning region dimmed, answer region lit), the run's
 * stats, and the single hardest decision the model faced. Colors are explicit
 * brand hex (not theme vars) so the rasterized PNG is faithful regardless of
 * where html-to-image computes styles. The figure is plain inline SVG built
 * from the streamed steps — deterministic markup, no D3, rasterizes exactly.
 *
 * Rendered off-screen by `ShareSection`; `forwardRef` exposes the node to capture.
 */

import { forwardRef } from "react";

import { keyMoments } from "@/lib/steps";
import { displayToken, isStructuralToken } from "@/lib/tokens";
import { SITE_HOST } from "@/lib/share";
import type { StepData } from "@/lib/xray-protocol";

// Brand palette (on dark).
const INK = "#16181d";
const PAPER = "#f1eee7";
const VERMILION = "#f0653b";
const AMBER = "#d99a45";
const SANS = "var(--font-hanken), system-ui, sans-serif";
const MONO = "var(--font-spline), monospace";

const DIM = "rgba(241,238,231,0.5)";
const FAINT = "rgba(241,238,231,0.35)";
const HAIRLINE = "rgba(241,238,231,0.1)";

/** The crosshair monogram, inlined bare so the PNG capture is self-contained. */
function Mark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden>
      <g
        stroke={PAPER}
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
      <g stroke={PAPER} strokeWidth={5} strokeLinecap="round">
        <line x1="21" y1="21" x2="43" y2="43" />
        <line x1="43" y1="21" x2="21" y2="43" />
      </g>
      <circle cx="32" cy="32" r="3.6" fill={VERMILION} />
    </svg>
  );
}

function clip(s: string, n: number): string {
  const t = s.trim();
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

/**
 * The card's hero figure: chosen-token probability per generated token, as a
 * plain SVG polyline. The reasoning span is shaded and labeled; the lowest-
 * confidence steps get amber markers (the model's hesitations).
 */
function ConfidenceFigure({ steps }: { steps: StepData[] }) {
  const W = 544;
  const H = 110;
  const PAD_X = 4;
  const PAD_T = 8;
  const PAD_B = 6;

  const n = steps.length;
  if (n < 2) return null;
  const x = (i: number) => PAD_X + (i / (n - 1)) * (W - 2 * PAD_X);
  const y = (p: number) => PAD_T + (1 - p) * (H - PAD_T - PAD_B);

  const line = steps
    .map((s, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(s.prob).toFixed(1)}`)
    .join(" ");
  const area = `${line} L ${x(n - 1).toFixed(1)} ${H - PAD_B} L ${x(0).toFixed(1)} ${H - PAD_B} Z`;

  // Reasoning span (contiguous prefix in a thinking run).
  const lastThink = steps.reduce((m, s, i) => (s.phase === "think" ? i : m), -1);

  // The ≤3 deepest hesitations, marked in amber.
  const dips = steps
    .map((s, i) => ({ i, prob: s.prob }))
    .sort((a, b) => a.prob - b.prob)
    .slice(0, 3);

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
      {/* 100% / 50% / 0% gridlines */}
      {[1, 0.5, 0].map((p) => (
        <line
          key={p}
          x1={PAD_X}
          x2={W - PAD_X}
          y1={y(p)}
          y2={y(p)}
          stroke={HAIRLINE}
          strokeWidth={1}
        />
      ))}
      {/* reasoning region, dimmed */}
      {lastThink > 0 && (
        <rect
          x={x(0)}
          y={PAD_T - 4}
          width={x(lastThink) - x(0)}
          height={H - PAD_T - PAD_B + 8}
          fill="rgba(241,238,231,0.045)"
        />
      )}
      <path d={area} fill="rgba(240,101,59,0.10)" />
      <path d={line} fill="none" stroke={VERMILION} strokeWidth={1.6} strokeLinejoin="round" />
      {dips.map((d) => (
        <circle key={d.i} cx={x(d.i)} cy={y(d.prob)} r={3} fill={AMBER} />
      ))}
    </svg>
  );
}

interface ShareCardProps {
  prompt: string;
  modelLabel: string;
  answerText: string;
  steps: StepData[];
}

export const ShareCard = forwardRef<HTMLDivElement, ShareCardProps>(
  function ShareCard({ prompt, modelLabel, answerText, steps }, ref) {
    // Structural markers (<think>/</think>) render no chip in the app; keep the
    // card's numbers consistent with what the timeline shows.
    const content = steps.filter((s) => !isStructuralToken(s.token));
    const thinkCount = content.filter((s) => s.phase === "think").length;
    const answerCount = content.length - thinkCount;
    const avgConf =
      content.length > 0
        ? content.reduce((sum, s) => sum + s.prob, 0) / content.length
        : 0;
    const hardest = keyMoments(steps, 1)[0] ?? null;

    const stats: { label: string; value: string }[] = [
      { label: "tokens", value: String(content.length) },
      ...(thinkCount > 0
        ? [
            { label: "reasoning", value: String(thinkCount) },
            { label: "answer", value: String(answerCount) },
          ]
        : []),
      { label: "avg confidence", value: `${(avgConf * 100).toFixed(0)}%` },
    ];

    return (
      <div
        ref={ref}
        style={{
          width: 600,
          padding: 28,
          backgroundColor: INK,
          backgroundImage:
            "radial-gradient(120% 80% at 100% 0%, rgba(240,101,59,0.16), transparent 62%)",
          color: PAPER,
          fontFamily: SANS,
          border: "1px solid rgba(241,238,231,0.1)",
          borderRadius: 18,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {/* header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Mark size={32} />
            <span style={{ fontFamily: SANS, fontSize: 19, fontWeight: 700, letterSpacing: "-0.4px" }}>
              LLM <span style={{ color: VERMILION }}>X</span>-Ray
            </span>
          </span>
          <span
            style={{
              fontFamily: MONO,
              fontSize: 11,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: DIM,
            }}
          >
            {modelLabel} · live forward pass
          </span>
        </div>

        {/* question + one-line answer excerpt */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            fontFamily: MONO,
            fontSize: 16,
            lineHeight: 1.4,
            wordBreak: "break-word",
          }}
        >
          <div style={{ color: PAPER }}>
            <span style={{ color: VERMILION, fontWeight: 700 }}>Q&nbsp;&nbsp;</span>
            {clip(prompt, 90)}
          </div>
          <div style={{ color: DIM, fontSize: 14 }}>
            <span style={{ color: VERMILION, fontWeight: 700 }}>A&nbsp;&nbsp;</span>
            {clip(answerText, 110)}
          </div>
        </div>

        {/* the finding — confidence per generated token */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontFamily: MONO,
              fontSize: 10,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: FAINT,
            }}
          >
            <span>confidence per token</span>
            {thinkCount > 0 && <span>shaded = reasoning · dots = hesitation</span>}
          </div>
          <ConfidenceFigure steps={content} />
        </div>

        {/* run stats + the hardest decision */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ display: "flex", gap: 10 }}>
            {stats.map((s) => (
              <div
                key={s.label}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                  padding: "8px 12px",
                  borderRadius: 10,
                  border: `1px solid ${HAIRLINE}`,
                  backgroundColor: "rgba(241,238,231,0.04)",
                }}
              >
                <span style={{ fontFamily: MONO, fontSize: 17, fontWeight: 600, color: PAPER }}>
                  {s.value}
                </span>
                <span
                  style={{
                    fontFamily: MONO,
                    fontSize: 9.5,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    color: FAINT,
                  }}
                >
                  {s.label}
                </span>
              </div>
            ))}
          </div>
          {hardest && (
            <div style={{ fontFamily: MONO, fontSize: 13, color: DIM }}>
              hardest call · said{" "}
              <span style={{ color: PAPER, whiteSpace: "pre" }}>
                “{displayToken(hardest.chosen)}”
              </span>{" "}
              {(hardest.chosenProb * 100).toFixed(0)}%, almost{" "}
              <span style={{ color: AMBER, whiteSpace: "pre" }}>
                “{displayToken(hardest.runnerUp)}”
              </span>{" "}
              {(hardest.runnerUpProb * 100).toFixed(0)}%
            </div>
          )}
        </div>

        {/* footer watermark */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingTop: 14,
            borderTop: `1px solid ${HAIRLINE}`,
            fontSize: 12,
            color: FAINT,
          }}
        >
          <span>Type a prompt. Watch an AI think — token by token.</span>
          <span style={{ color: VERMILION, fontWeight: 600 }}>{SITE_HOST}</span>
        </div>
      </div>
    );
  },
);
