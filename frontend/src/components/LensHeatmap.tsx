"use client";

/**
 * `LensHeatmap` — the dense view. For the selected generated token it lays out
 * every layer (rows, input → output downward) against the candidate tokens the
 * logit lens surfaces (columns), with cell brightness = probability. You watch
 * the probability mass migrate and concentrate as depth increases — early layers
 * spread thinly across many guesses, late layers commit (the chosen token's
 * column, outlined vermilion, lights up at the bottom).
 *
 * Columns are the union of every layer's top-k, ranked by their peak probability
 * across depth, capped so the grid stays legible.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { displayToken } from "@/lib/tokens";
import { MONO, VIZ, weightColor } from "@/lib/viz";
import type { LayerTopK } from "@/lib/xray-protocol";

interface LensHeatmapProps {
  trajectory: LayerTopK[];
  chosenTokenId: number;
  /** Cross-view token highlight (matched by raw token string). */
  hoveredToken?: string | null;
  onHoverToken?: (token: string | null) => void;
}

const MAX_COLS = 10;
const ROW_H = 13;
const LABEL_W = 26;
const TOP_PAD = 58;

export function LensHeatmap({
  trajectory,
  chosenTokenId,
  hoveredToken = null,
  onHoverToken,
}: LensHeatmapProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const hoverRef = useRef(onHoverToken);
  useEffect(() => {
    hoverRef.current = onHoverToken;
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const L = trajectory.length;
    if (L === 0) {
      svg.selectAll("*").remove();
      return;
    }

    // Candidate columns: union of all layers' top-k, ranked by peak probability.
    const peak = new Map<number, { token: string; prob: number }>();
    for (const layer of trajectory) {
      for (const p of layer.top_predictions) {
        const cur = peak.get(p.token_id);
        if (!cur || p.prob > cur.prob) peak.set(p.token_id, { token: p.token, prob: p.prob });
      }
    }
    const cols = [...peak.entries()]
      .map(([token_id, v]) => ({ token_id, ...v }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, MAX_COLS);
    const colIndex = new Map(cols.map((c, i) => [c.token_id, i]));
    const C = cols.length;

    // Per-layer prob lookup.
    const cells: { l: number; c: number; w: number; token: string }[] = [];
    let maxW = 0;
    trajectory.forEach((layer, li) => {
      for (const p of layer.top_predictions) {
        const ci = colIndex.get(p.token_id);
        if (ci === undefined) continue;
        cells.push({ l: li, c: ci, w: p.prob, token: p.token });
        if (p.prob > maxW) maxW = p.prob;
      }
    });
    maxW = maxW || 1;

    const cw = 30;
    const gridW = C * cw;
    const gridH = L * ROW_H;
    const VB_W = LABEL_W + gridW + 8;
    const VB_H = TOP_PAD + gridH + 6;
    svg.attr("viewBox", `0 0 ${VB_W} ${VB_H}`);

    const chosenCol = colIndex.get(chosenTokenId);
    const t = svg.transition().duration(360).ease(d3.easeCubicOut);
    const ensure = (cls: string) => {
      let g = svg.select<SVGGElement>(`g.${cls}`);
      if (g.empty()) g = svg.append("g").attr("class", cls);
      return g;
    };

    // cells
    ensure("cells")
      .selectAll<SVGRectElement, { l: number; c: number; w: number; token: string }>("rect")
      .data(cells, (d) => `${d.l}-${d.c}`)
      .join((enter) =>
        enter
          .append("rect")
          .attr("x", (d) => LABEL_W + d.c * cw)
          .attr("y", (d) => TOP_PAD + d.l * ROW_H)
          .attr("width", cw - 1)
          .attr("height", ROW_H - 1)
          .attr("rx", 1.5),
      )
      .attr("x", (d) => LABEL_W + d.c * cw)
      .attr("y", (d) => TOP_PAD + d.l * ROW_H)
      .attr("width", cw - 1)
      .attr("height", ROW_H - 1)
      .on("mouseenter", (_e, d) => hoverRef.current?.(d.token))
      .on("mouseleave", () => hoverRef.current?.(null))
      // @ts-expect-error d3 transition typing
      .transition(t)
      .attr("fill", (d) =>
        d.c === chosenCol ? d3.interpolateRgb("#3a201a", VIZ.accent)(d.w / maxW) : weightColor(d.w / maxW),
      );

    // chosen-column outline
    svg.select("rect.chosen").remove();
    if (chosenCol !== undefined) {
      svg
        .append("rect")
        .attr("class", "chosen")
        .attr("x", LABEL_W + chosenCol * cw - 0.5)
        .attr("y", TOP_PAD)
        .attr("width", cw)
        .attr("height", gridH)
        .attr("fill", "none")
        .attr("stroke", VIZ.accent)
        .attr("stroke-width", 1)
        .attr("pointer-events", "none");
    }

    // column headers (candidate tokens, rotated)
    ensure("cols")
      .selectAll<SVGTextElement, (typeof cols)[number]>("text")
      .data(cols, (d) => d.token_id)
      .join("text")
      .attr("transform", (_d, i) => `translate(${LABEL_W + i * cw + cw / 2},${TOP_PAD - 6}) rotate(-45)`)
      .attr("font-size", 10)
      .attr("font-family", MONO)
      .attr("fill", (d) => (d.token_id === chosenTokenId ? VIZ.accent : VIZ.text))
      .style("cursor", "default")
      .on("mouseenter", (_e, d) => hoverRef.current?.(d.token))
      .on("mouseleave", () => hoverRef.current?.(null))
      .text((d) => displayToken(d.token));

    // row labels (every 4th layer + last)
    ensure("rows")
      .selectAll<SVGTextElement, number>("text")
      .data(d3.range(L).filter((i) => i % 4 === 0 || i === L - 1))
      .join("text")
      .attr("x", LABEL_W - 5)
      .attr("y", (i) => TOP_PAD + i * ROW_H + ROW_H / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("font-size", 8)
      .attr("font-family", MONO)
      .attr("fill", VIZ.textDim)
      .text((i) => `L${i}`);
  }, [trajectory, chosenTokenId]);

  // Cross-view highlight — stroke matching cells only, separate from the draw
  // effect so hovering never restarts its transitions.
  useEffect(() => {
    d3.select(svgRef.current)
      .select("g.cells")
      .selectAll<SVGRectElement, { token: string }>("rect")
      .attr("stroke", (d) =>
        hoveredToken !== null && d.token === hoveredToken ? VIZ.accent : "none",
      )
      .attr("stroke-width", 1);
  }, [hoveredToken, trajectory]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      style={{ maxHeight: 460 }}
      role="img"
      aria-label="Logit-lens heatmap: candidate token probability at each layer"
    />
  );
}
