"use client";

/**
 * `AttentionContext` — honest attention, demoted to a supporting role. For the
 * selected generated token it shows *which context tokens drove it*, as a strip
 * of cells (one per prior token) tinted by influence. Two modes:
 *
 *   • rollout — attention rolled out across all layers (Abnar & Zuidema): the
 *     defensible "what actually drove this token" attribution. Layer/head agnostic.
 *   • raw     — the raw attention row at one layer/head. Useful but, on its own,
 *     a weak explanation — hence not the default.
 *
 * The strongest contributors are labeled; the rest stay unlabeled to keep the
 * strip legible as the context grows.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { contentMax, isAttnSink } from "@/lib/attention";
import { displayToken } from "@/lib/tokens";
import { MONO, VIZ, weightColor } from "@/lib/viz";

interface AttentionContextProps {
  tokens: string[]; // the context tokens (prompt + earlier generated)
  weights: number[]; // influence per context token, same length as tokens
  producedToken: string;
  /** Cross-view token highlight (matched by raw token string). */
  hoveredToken?: string | null;
  onHoverToken?: (token: string | null) => void;
}

const VB_W = 560;
const STRIP_Y = 10;
const STRIP_H = 30;
const LABEL_AREA = 64;
const MAX_LABELS = 8;

export function AttentionContext({
  tokens,
  weights,
  producedToken,
  hoveredToken = null,
  onHoverToken,
}: AttentionContextProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const hoverRef = useRef(onHoverToken);
  useEffect(() => {
    hoverRef.current = onHoverToken;
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const S = Math.min(tokens.length, weights.length);
    if (S === 0) {
      svg.selectAll("*").remove();
      return;
    }

    const height = STRIP_Y + STRIP_H + LABEL_AREA;
    const innerW = VB_W - 8;
    const cw = innerW / S;
    // Scale over content tokens only — the first-token attention sink + chat
    // scaffolding otherwise hog the scale and flatten every content cell to black.
    const excluded = (i: number) => isAttnSink(tokens[i], i);
    const max = contentMax(tokens.slice(0, S), weights.slice(0, S));
    svg.attr("viewBox", `0 0 ${VB_W} ${height}`);

    // Label only the strongest *content* contributors so the strip stays readable.
    const ranked = weights
      .slice(0, S)
      .map((w, i) => ({ i, w }))
      .filter((d) => !excluded(d.i))
      .sort((a, b) => b.w - a.w);
    const labeled = new Set(ranked.slice(0, MAX_LABELS).map((d) => d.i));

    const data = d3.range(S).map((i) => ({ i, w: weights[i], token: tokens[i] }));
    const t = svg.transition().duration(380).ease(d3.easeCubicOut);

    const ensure = (cls: string) => {
      let g = svg.select<SVGGElement>(`g.${cls}`);
      if (g.empty()) g = svg.append("g").attr("class", cls);
      return g;
    };

    // cells
    ensure("cells")
      .selectAll<SVGRectElement, { i: number; w: number; token: string }>("rect")
      .data(data, (d) => d.i)
      .join((enter) =>
        enter
          .append("rect")
          .attr("y", STRIP_Y)
          .attr("height", STRIP_H)
          .attr("rx", 1)
          .attr("stroke", VIZ.surface)
          .attr("stroke-width", 0.5),
      )
      .attr("x", (d) => 4 + d.i * cw)
      .attr("width", Math.max(1, cw - 1))
      .on("mouseenter", (_e, d) => hoverRef.current?.(d.token))
      .on("mouseleave", () => hoverRef.current?.(null))
      // @ts-expect-error d3 transition typing
      .transition(t)
      // Excluded cells (sink / scaffolding) are drawn as a flat muted swatch —
      // present and visible, but not pretending to be a real content weight.
      .attr("fill", (d) => (excluded(d.i) ? VIZ.track : weightColor(d.w / max)));

    // labels (rotated) for the strongest contributors
    ensure("labels")
      .selectAll<SVGTextElement, { i: number; token: string }>("text")
      .data(data, (d) => d.i)
      .join("text")
      .attr("font-size", 10)
      .attr("font-family", MONO)
      .attr("text-anchor", "end")
      .attr(
        "transform",
        (d) => `translate(${4 + d.i * cw + cw / 2},${STRIP_Y + STRIP_H + 8}) rotate(-50)`,
      )
      .attr("fill", (d) => (labeled.has(d.i) ? VIZ.text : "transparent"))
      .text((d) => displayToken(d.token));

    // produced token caption at the right
    ensure("caption")
      .selectAll<SVGTextElement, number>("text")
      .data([0])
      .join("text")
      .attr("x", VB_W - 4)
      .attr("y", STRIP_Y - 1)
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("font-family", MONO)
      .attr("fill", VIZ.accent)
      .text(`→ ${displayToken(producedToken)}`);
  }, [tokens, weights, producedToken]);

  // Cross-view highlight — a cheap stroke-only pass, deliberately separate from
  // the draw effect above so hovering never restarts its transitions.
  useEffect(() => {
    d3.select(svgRef.current)
      .select("g.cells")
      .selectAll<SVGRectElement, { token: string }>("rect")
      .attr("stroke", (d) =>
        hoveredToken !== null && d.token === hoveredToken ? VIZ.accent : VIZ.surface,
      )
      .attr("stroke-width", (d) =>
        hoveredToken !== null && d.token === hoveredToken ? 1.5 : 0.5,
      );
  }, [hoveredToken, tokens, weights]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      role="img"
      aria-label="Attention attribution: which context tokens drove the selected generated token"
    />
  );
}
