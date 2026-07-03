"use client";

/**
 * `StepDistribution` — for the selected generated token, the model's actual
 * next-token distribution (the final layer's logit lens): the top candidates
 * with their real probabilities on a fixed [0,1] scale. The chosen token glows
 * vermilion; the rejected runner-up is called out ("almost: X"). Bars are bound by
 * rank so they morph in place as you scrub the timeline.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { displayToken } from "@/lib/tokens";
import { MONO, VIZ, confidenceColor, pct } from "@/lib/viz";
import type { TopPrediction } from "@/lib/xray-protocol";

interface StepDistributionProps {
  predictions: TopPrediction[];
  chosenTokenId: number;
  /** Cross-view token highlight (matched by raw token string). */
  hoveredToken?: string | null;
  onHoverToken?: (token: string | null) => void;
}

const VB_W = 380;
const ROW_H = 30;
const LABEL_X = 96;
const VALUE_W = 50;

export function StepDistribution({
  predictions,
  chosenTokenId,
  hoveredToken = null,
  onHoverToken,
}: StepDistributionProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const hoverRef = useRef(onHoverToken);
  useEffect(() => {
    hoverRef.current = onHoverToken;
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const n = predictions.length;
    if (n === 0) {
      svg.selectAll("*").remove();
      return;
    }

    const height = n * ROW_H + 8;
    const barX = LABEL_X + 8;
    const barMax = VB_W - barX - VALUE_W;
    const x = d3.scaleLinear().domain([0, 1]).range([0, barMax]);
    const y = (rank: number) => 8 + rank * ROW_H;
    const barH = ROW_H - 12;
    svg.attr("viewBox", `0 0 ${VB_W} ${height}`);

    const rows = svg
      .selectAll<SVGGElement, TopPrediction>("g.row")
      .data(predictions, (_d, i) => i)
      .join((enter) => {
        const g = enter.append("g").attr("class", "row");
        g.append("rect").attr("class", "track");
        g.append("rect").attr("class", "bar");
        g.append("text").attr("class", "label");
        g.append("text").attr("class", "value");
        return g;
      })
      .attr("transform", (_d, i) => `translate(0,${y(i)})`)
      .on("mouseenter", (_e, d) => hoverRef.current?.(d.token))
      .on("mouseleave", () => hoverRef.current?.(null));

    const t = svg.transition().duration(400).ease(d3.easeCubicOut);

    rows
      .select<SVGRectElement>("rect.track")
      .attr("x", barX)
      .attr("y", 0)
      .attr("width", barMax)
      .attr("height", barH)
      .attr("rx", 3)
      .attr("fill", VIZ.track);

    rows
      .select<SVGRectElement>("rect.bar")
      .attr("x", barX)
      .attr("y", 0)
      .attr("height", barH)
      .attr("rx", 3)
      // @ts-expect-error d3 transition typing on selection
      .transition(t)
      .attr("width", (d) => Math.max(0, x(d.prob)))
      .attr("fill", (d) =>
        d.token_id === chosenTokenId ? VIZ.accent : confidenceColor(d.prob),
      );

    rows
      .select<SVGTextElement>("text.label")
      .attr("x", LABEL_X)
      .attr("y", barH / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("font-size", 13)
      .attr("font-family", MONO)
      .attr("fill", (d) => (d.token_id === chosenTokenId ? VIZ.accent : VIZ.text))
      .text((d) => displayToken(d.token));

    rows
      .select<SVGTextElement>("text.value")
      .attr("x", VB_W - VALUE_W + 6)
      .attr("y", barH / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "start")
      .attr("font-size", 12)
      .attr("font-family", MONO)
      .attr("fill", VIZ.textDim)
      .text((d) => pct(d.prob));
  }, [predictions, chosenTokenId]);

  // Cross-view highlight — tint the row track only, separate from the draw
  // effect so hovering never restarts the bar transitions.
  useEffect(() => {
    d3.select(svgRef.current)
      .selectAll<SVGGElement, TopPrediction>("g.row")
      .select<SVGRectElement>("rect.track")
      .attr("fill", (d) =>
        hoveredToken !== null && d.token === hoveredToken
          ? "rgba(240,101,59,0.12)"
          : VIZ.track,
      );
  }, [hoveredToken, predictions]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      role="img"
      aria-label="Next-token distribution: top candidate tokens and their probabilities"
    />
  );
}
