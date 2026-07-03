/**
 * `/api/og` — the server-rendered share image for links.
 *
 * When a shared `/?prompt=…` link is unfurled (X, LinkedIn, Slack…), the page's
 * `generateMetadata` points the OG/Twitter image here with the prompt (and, if
 * known, the predicted token + probability) in the query. We render a branded
 * card with `next/og` (Satori → PNG). Satori only supports flexbox + a CSS
 * subset, so this is a text card on brand ink with the single vermilion accent;
 * the crosshair mark is inlined as a data-URI image (the rich D3 belongs to the
 * client PNG export). It just has to look good when someone scrolls past it.
 */

import { ImageResponse } from "next/og";

import { displayToken } from "@/lib/tokens";
import { SITE_HOST } from "@/lib/share";
import { MODEL_LABEL } from "@/lib/xray-protocol";

export const contentType = "image/png";
export const size = { width: 1200, height: 630 };

const INK = "#16181d";
const PAPER = "#f1eee7";
const VERMILION = "#f0653b";

// The crosshair monogram as a data URI — Satori renders <img> reliably, inline
// <svg> less so. Paper crop-marks + X, vermilion focal node.
const MARK_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64" fill="none"><g stroke="${PAPER}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"><polyline points="14,22 14,14 22,14"/><polyline points="50,22 50,14 42,14"/><polyline points="14,42 14,50 22,50"/><polyline points="50,42 50,50 42,50"/></g><g stroke="${PAPER}" stroke-width="5" stroke-linecap="round"><line x1="21" y1="21" x2="43" y2="43"/><line x1="43" y1="21" x2="21" y2="43"/></g><circle cx="32" cy="32" r="3.6" fill="${VERMILION}"/></svg>`;
const MARK_URI = `data:image/svg+xml;base64,${Buffer.from(MARK_SVG).toString("base64")}`;

export function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const prompt = (searchParams.get("prompt") ?? "").slice(0, 160).trim();
  const token = searchParams.get("token");
  const probRaw = searchParams.get("prob");
  const prob = probRaw ? Number(probRaw) : null;

  const headline = prompt || "Type a prompt. Watch an AI think.";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 72,
          backgroundColor: INK,
          backgroundImage:
            "radial-gradient(1000px 620px at 100% 0%, rgba(240,101,59,0.2), transparent)",
          color: PAPER,
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={MARK_URI} width={56} height={56} alt="" />
            <div style={{ display: "flex", fontSize: 34, fontWeight: 700, letterSpacing: -1 }}>
              LLM&nbsp;<span style={{ display: "flex", color: VERMILION }}>X</span>-Ray
            </div>
          </div>
          <div style={{ display: "flex", fontSize: 24, color: "rgba(241,238,231,0.5)" }}>
            {MODEL_LABEL} · token-by-token
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
          <div style={{ display: "flex", fontSize: 28, letterSpacing: 4, textTransform: "uppercase", color: VERMILION }}>
            Prompt
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 64,
              fontWeight: 600,
              lineHeight: 1.15,
              maxWidth: 1056,
            }}
          >
            “{headline}”
          </div>
          {prompt && token ? (
            <div style={{ display: "flex", alignItems: "center", gap: 20, fontSize: 36 }}>
              <span style={{ display: "flex", color: "rgba(241,238,231,0.55)" }}>next →</span>
              <span
                style={{
                  display: "flex",
                  padding: "8px 24px",
                  borderRadius: 14,
                  backgroundColor: VERMILION,
                  color: INK,
                  fontWeight: 700,
                }}
              >
                {displayToken(token)}
              </span>
              {prob !== null && !Number.isNaN(prob) ? (
                <span style={{ display: "flex", color: "rgba(241,238,231,0.55)" }}>
                  {(prob * 100).toFixed(1)}%
                </span>
              ) : null}
            </div>
          ) : null}
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            paddingTop: 28,
            borderTop: "1px solid rgba(241,238,231,0.12)",
            fontSize: 26,
            color: "rgba(241,238,231,0.5)",
          }}
        >
          <span style={{ display: "flex" }}>Model interpretability, made visible.</span>
          <span style={{ display: "flex", color: VERMILION, fontWeight: 700 }}>{SITE_HOST}</span>
        </div>
      </div>
    ),
    size,
  );
}
