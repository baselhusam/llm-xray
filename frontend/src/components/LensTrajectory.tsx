"use client";

/**
 * `LensTrajectory` — the hero. For the selected generated token, project every
 * layer's hidden state through the unembedding (logit lens) and show its top-1
 * guess + confidence as you descend the stack. Reading top → bottom you watch
 * the prediction *form*: early layers guess noise, then the answer emerges and
 * locks in (rows whose guess matches the final token glow vermilion). Clicking a row
 * focuses that layer for the raw-attention view.
 */

import { useEffect, useRef } from "react";
import * as d3 from "d3";

import { displayToken } from "@/lib/tokens";
import { MONO, VIZ, confidenceColor, pct } from "@/lib/viz";
import type { LayerTopK } from "@/lib/xray-protocol";

interface LensTrajectoryProps {
  trajectory: LayerTopK[];
  finalTokenId: number;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

const VB_W = 360;
const ROW_H = 15;
const PAD_Y = 6;
const TOKEN_X = 120;
const BAR_X0 = 128;
const BAR_X1 = 312;

interface Row {
  layer: number;
  token: string;
  prob: number;
  locked: boolean;
}

export function LensTrajectory({
  trajectory,
  finalTokenId,
  selectedLayer,
  onSelectLayer,
}: LensTrajectoryProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const onSelectRef = useRef(onSelectLayer);
  useEffect(() => {
    onSelectRef.current = onSelectLayer;
  });

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const L = trajectory.length;
    if (L === 0) {
      svg.selectAll("*").remove();
      return;
    }

    const rows: Row[] = trajectory.map((t) => {
      const top1 = t.top_predictions[0];
      return {
        layer: t.layer_idx,
        token: top1?.token ?? "",
        prob: top1?.prob ?? 0,
        locked: top1?.token_id === finalTokenId,
      };
    });
    const firstLock = rows.find((r) => r.locked)?.layer ?? -1;

    const height = L * ROW_H + PAD_Y * 2;
    const barW = BAR_X1 - BAR_X0;
    const y = (i: number) => PAD_Y + i * ROW_H;
    const x = d3.scaleLinear().domain([0, 1]).range([0, barW]);
    svg.attr("viewBox", `0 0 ${VB_W} ${height}`);

    const t = svg.transition().duration(420).ease(d3.easeCubicOut);

    const g = svg
      .selectAll<SVGGElement, Row>("g.row")
      .data(rows, (d) => d.layer)
      .join((enter) => {
        const row = enter.append("g").attr("class", "row").style("cursor", "pointer");
        row.append("rect").attr("class", "hit");
        row.append("rect").attr("class", "track");
        row.append("rect").attr("class", "bar");
        row.append("text").attr("class", "idx");
        row.append("text").attr("class", "tok");
        row.append("text").attr("class", "val");
        return row;
      })
      .attr("transform", (_d, i) => `translate(0,${y(i)})`)
      .on("click", (_e, d) => onSelectRef.current(d.layer));

    g.select<SVGRectElement>("rect.hit")
      .attr("x", 0)
      .attr("y", -1)
      .attr("width", VB_W)
      .attr("height", ROW_H)
      .attr("rx", 3)
      .attr("fill", (d) => (d.layer === selectedLayer ? "rgba(255,255,255,0.07)" : "transparent"));

    g.select<SVGRectElement>("rect.track")
      .attr("x", BAR_X0)
      .attr("y", 2)
      .attr("width", barW)
      .attr("height", ROW_H - 6)
      .attr("rx", 2)
      .attr("fill", VIZ.track);

    g.select<SVGRectElement>("rect.bar")
      .attr("x", BAR_X0)
      .attr("y", 2)
      .attr("height", ROW_H - 6)
      .attr("rx", 2)
      // @ts-expect-error d3 transition typing on selection
      .transition(t)
      .attr("width", (d) => Math.max(0, x(d.prob)))
      .attr("fill", (d) => (d.locked ? VIZ.accent : confidenceColor(d.prob)));

    g.select<SVGTextElement>("text.idx")
      .attr("x", 4)
      .attr("y", ROW_H / 2)
      .attr("dy", "0.32em")
      .attr("font-size", 9)
      .attr("font-family", MONO)
      .attr("fill", (d) => (d.layer === selectedLayer ? VIZ.text : VIZ.textDim))
      .text((d) => `L${d.layer}`);

    g.select<SVGTextElement>("text.tok")
      .attr("x", TOKEN_X)
      .attr("y", ROW_H / 2)
      .attr("dy", "0.32em")
      .attr("text-anchor", "end")
      .attr("font-size", 11)
      .attr("font-family", MONO)
      .attr("fill", (d) => (d.locked ? VIZ.accent : VIZ.text))
      .attr("font-weight", (d) => (d.layer === firstLock ? 600 : 400))
      .text((d) => displayToken(d.token));

    g.select<SVGTextElement>("text.val")
      .attr("x", BAR_X1 + 6)
      .attr("y", ROW_H / 2)
      .attr("dy", "0.32em")
      .attr("font-size", 9)
      .attr("font-family", MONO)
      .attr("fill", VIZ.textDim)
      .text((d) => pct(d.prob, 0));
  }, [trajectory, finalTokenId, selectedLayer]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      style={{ maxHeight: 520 }}
      role="img"
      aria-label="Logit-lens trajectory: each layer's top-1 next-token guess and confidence, from input to output"
    />
  );
}
