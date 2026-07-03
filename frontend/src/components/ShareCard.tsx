"use client";

/**
 * `ShareCard` — the standalone, branded artifact captured to PNG for sharing.
 *
 * It must read well with zero context (someone scrolling a feed), so it carries
 * the full identity: the crosshair mark + wordmark, the prompt, the model's
 * continuation, the final prediction + the alternative it rejected, and a URL
 * watermark. Colors are explicit brand hex (not theme vars) so the rasterized
 * PNG is faithful regardless of where html-to-image computes styles. No D3 here
 * — a faithful text card on brand ink with the single vermilion accent.
 *
 * Rendered off-screen by `XRayApp`; `forwardRef` exposes the node to capture.
 */

import { forwardRef } from "react";

import { displayToken } from "@/lib/tokens";
import { SITE_HOST } from "@/lib/share";
import type { TopPrediction } from "@/lib/xray-protocol";

// Brand palette (on dark).
const INK = "#16181d";
const PAPER = "#f1eee7";
const VERMILION = "#f0653b";
const SANS = "var(--font-hanken), system-ui, sans-serif";
const MONO = "var(--font-spline), monospace";

/** The crosshair monogram, inlined so the PNG capture is self-contained. */
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

interface ShareCardProps {
  prompt: string;
  modelLabel: string;
  generatedText: string;
  finalToken: string;
  finalProb: number;
  runnerUp: TopPrediction | null;
}

export const ShareCard = forwardRef<HTMLDivElement, ShareCardProps>(
  function ShareCard(
    { prompt, modelLabel, generatedText, finalToken, finalProb, runnerUp },
    ref,
  ) {
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
          gap: 18,
        }}
      >
        {/* header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Mark size={30} />
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
              color: "rgba(241,238,231,0.5)",
            }}
          >
            {modelLabel} · token-by-token
          </span>
        </div>

        {/* question + answer */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
            fontFamily: MONO,
            fontSize: 18,
            lineHeight: 1.4,
            wordBreak: "break-word",
          }}
        >
          <div style={{ color: "rgba(241,238,231,0.55)" }}>
            <span style={{ color: VERMILION, fontWeight: 700 }}>Q&nbsp;&nbsp;</span>
            {prompt}
          </div>
          <div style={{ color: PAPER }}>
            <span style={{ color: VERMILION, fontWeight: 700 }}>A&nbsp;&nbsp;</span>
            {generatedText}
          </div>
        </div>

        {/* prediction + the rejected alternative */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
            fontSize: 15,
          }}
        >
          <span style={{ color: "rgba(241,238,231,0.6)" }}>last token →</span>
          <span
            style={{
              fontFamily: MONO,
              fontSize: 16,
              fontWeight: 600,
              padding: "4px 12px",
              borderRadius: 8,
              backgroundColor: VERMILION,
              color: INK,
              whiteSpace: "pre",
            }}
          >
            {displayToken(finalToken)}
          </span>
          <span style={{ fontFamily: MONO, color: "rgba(241,238,231,0.6)" }}>
            {(finalProb * 100).toFixed(1)}%
          </span>
          {runnerUp && (
            <span style={{ fontFamily: MONO, fontSize: 13, color: "rgba(241,238,231,0.45)" }}>
              almost “{displayToken(runnerUp.token)}” {(runnerUp.prob * 100).toFixed(1)}%
            </span>
          )}
        </div>

        {/* footer watermark */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingTop: 14,
            borderTop: "1px solid rgba(241,238,231,0.1)",
            fontSize: 12,
            color: "rgba(241,238,231,0.45)",
          }}
        >
          <span>Type a prompt. Watch an AI think.</span>
          <span style={{ color: VERMILION, fontWeight: 600 }}>{SITE_HOST}</span>
        </div>
      </div>
    );
  },
);
