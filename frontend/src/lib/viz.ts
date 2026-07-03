/**
 * Shared visual language for the X-Ray instrument — the brand identity applied
 * to the data viz.
 *
 * Colors are explicit hex (not theme CSS vars) on purpose: D3 sets them as SVG
 * attributes and html-to-image rasterizes them, so we never want to depend on
 * CSS-var resolution timing. Following the brand: one warm vermilion accent does
 * all the signalling (what the instrument is looking at — the locked/chosen/
 * certain state), and everything structural is ink & paper. No second hue, no
 * rainbows — the restraint is the brand.
 */

import * as d3 from "d3";

export const VIZ = {
  text: "#f1eee7", // paper
  textDim: "#9a958c", // muted mono
  grid: "rgba(241,238,231,0.07)",
  track: "rgba(241,238,231,0.06)",
  surface: "#1b1c1f", // ink soft — stroke between cells
  accent: "#f0653b", // vermilion (on dark) — the focal signal / locked-in
  accentDeep: "#d9542b", // vermilion (on light)
  weight: "#c8bca4", // warm parchment — structural attention weight (neutral)
  // Confidence "thermal" stops — a warm ramp analogous to the brand vermilion.
  confLow: "#6f6a61", // ash / taupe — groping, unsure
  confMid: "#d99a45", // warm amber — warming up
  confHigh: "#f0653b", // vermilion — locked in
} as const;

const clamp01 = (t: number) => Math.max(0, Math.min(1, t));

// Structural / attention weight: faint warm ink → warm parchment. Neutral on
// purpose, so the warm confidence ramp stays the thing that "means" something.
const weightRamp = d3.interpolateRgb("#2b2824", VIZ.weight);
export function weightColor(t: number): string {
  return weightRamp(clamp01(t));
}

// Certainty as a restrained warm "thermal" gradient: ash (unsure) → amber →
// vermilion (locked in). All warm/analogous to the accent — confidence reads as
// heat without becoming a rainbow or fighting the ink-and-paper brand.
const confScale = d3
  .scaleLinear<string>()
  .domain([0, 0.5, 1])
  .range([VIZ.confLow, VIZ.confMid, VIZ.confHigh])
  .interpolate(d3.interpolateRgb)
  .clamp(true);
export function confidenceColor(p: number): string {
  return confScale(clamp01(p));
}

/**
 * Decision depth on the same warm ramp: a *shallow* lock (the logit lens
 * matched the emitted token early — the model "knew" instantly) reads hot
 * vermilion; a *deep* lock (it needed the whole network) reads ash. Reusing
 * the confidence ramp keeps heat meaning "sure" in both encodings.
 */
export function depthColor(lockLayer: number, numLayers: number): string {
  return confScale(1 - lockLayer / Math.max(1, numLayers - 1));
}

export const pct = (p: number, digits = 1) => `${(p * 100).toFixed(digits)}%`;

/** Normalized certainty in [0,1] from Shannon entropy (nats), given vocab size. */
export function certainty(entropy: number, vocab = 151936): number {
  return clamp01(1 - entropy / Math.log(vocab));
}

export const MONO = "var(--font-spline), monospace";
