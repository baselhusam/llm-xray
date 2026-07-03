"use client";

/**
 * `ConfidenceCurve` — the model's certainty across the whole generation. Each
 * generated token is a point; the line is the probability it assigned to the
 * token it chose. Peaks are where it was sure, dips are where it was groping
 * between options. The selected step is ringed; click any point to jump there.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { MONO, VIZ, confidenceColor } from "@/lib/viz";
import type { StepData } from "@/lib/xray-protocol";

interface ConfidenceCurveProps {
  steps: StepData[];
  selectedStep: number;
  onSelect: (step: number) => void;
}

const VB_W = 560;
const VB_H = 120;
const M = { top: 12, right: 12, bottom: 18, left: 28 };

export function ConfidenceCurve({ steps, selectedStep, onSelect }: ConfidenceCurveProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const onSelectRef = useRef(onSelect);
  useEffect(() => {
    onSelectRef.current = onSelect;
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    if (steps.length === 0) {
      svg.selectAll("*").remove();
      return;
    }
    svg.attr("viewBox", `0 0 ${VB_W} ${VB_H}`);

    const innerW = VB_W - M.left - M.right;
    const innerH = VB_H - M.top - M.bottom;
    const n = steps.length;
    const x = d3
      .scaleLinear()
      .domain([0, Math.max(1, n - 1)])
      .range([M.left, M.left + innerW]);
    const y = d3.scaleLinear().domain([0, 1]).range([M.top + innerH, M.top]);

    const ensure = (cls: string, tag = "g") => {
      let s = svg.select<SVGGElement>(`${tag}.${cls}`);
      if (s.empty()) s = svg.append(tag).attr("class", cls) as never;
      return s;
    };

    // gridlines at 0 / 0.5 / 1
    const gridG = ensure("grid");
    gridG
      .selectAll<SVGLineElement, number>("line")
      .data([0, 0.5, 1])
      .join("line")
      .attr("x1", M.left)
      .attr("x2", M.left + innerW)
      .attr("y1", (d) => y(d))
      .attr("y2", (d) => y(d))
      .attr("stroke", VIZ.grid)
      .attr("stroke-width", 1);
    gridG
      .selectAll<SVGTextElement, number>("text")
      .data([0, 0.5, 1])
      .join("text")
      .attr("x", M.left - 6)
      .attr("y", (d) => y(d))
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("font-size", 9)
      .attr("font-family", MONO)
      .attr("fill", VIZ.textDim)
      .text((d) => `${d * 100 | 0}%`);

    const t = svg.transition().duration(360).ease(d3.easeCubicOut);

    const line = d3
      .line<StepData>()
      .x((_d, i) => x(i))
      .y((d) => y(d.prob))
      .curve(d3.curveMonotoneX);
    const area = d3
      .area<StepData>()
      .x((_d, i) => x(i))
      .y0(M.top + innerH)
      .y1((d) => y(d.prob))
      .curve(d3.curveMonotoneX);

    ensure("area", "path")
      .datum(steps)
      // @ts-expect-error d3 transition typing
      .transition(t)
      .attr("d", area)
      .attr("fill", "rgba(240,101,59,0.10)");

    ensure("line", "path")
      .datum(steps)
      .attr("fill", "none")
      .attr("stroke", VIZ.accent)
      .attr("stroke-width", 1.5)
      // @ts-expect-error d3 transition typing
      .transition(t)
      .attr("d", line);

    ensure("dots")
      .selectAll<SVGCircleElement, StepData>("circle")
      .data(steps, (d) => d.step)
      .join("circle")
      .attr("cx", (_d, i) => x(i))
      .attr("r", (d) => (d.step === selectedStep ? 4.5 : 2.5))
      .style("cursor", "pointer")
      .attr("stroke", VIZ.surface)
      .attr("stroke-width", (d) => (d.step === selectedStep ? 2 : 0))
      .attr("fill", (d) => confidenceColor(d.prob))
      .on("click", (_e, d) => onSelectRef.current(d.step))
      // @ts-expect-error d3 transition typing
      .transition(t)
      .attr("cy", (d) => y(d.prob));
  }, [steps, selectedStep]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      role="img"
      aria-label="Confidence across the generation: probability of each chosen token"
    />
  );
}
