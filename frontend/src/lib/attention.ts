/**
 * Helpers for turning the streamed generation attention into the rows the
 * attention-attribution view draws.
 *
 * Wire shapes (see `xray-protocol.ts`) — both are **mean over heads** (the
 * backend averages heads to keep long reasoning traces cheap to stream):
 *   promptAttention   : (layers, P, P)        — attention over the prompt
 *   step.attention_row: (layers, P+step)      — the query row that produced the
 *                       generated token at this step
 *
 * A generated token at step `t` was predicted at position `P+t-1` while
 * attending over `P+t` context tokens (prompt + earlier generated tokens).
 */

import type { StepData } from "@/lib/xray-protocol";

/**
 * A single generated token's raw attention over its context, at one layer
 * (already mean over heads). Length = `P + step.step`.
 */
export function stepAttentionRow(step: StepData, layer: number): number[] {
  return step.attention_row[layer] ?? [];
}

/**
 * Attention **rollout** (Abnar & Zuidema, 2020) for the selected step, computed
 * client-side on demand. Composes residual-augmented, renormalized per-layer
 * attention across all layers; the returned row is each context token's
 * influence on the selected generated token. Length = `P + step`.
 *
 * Only the *last row* of the L-layer rollout product is ever needed, so it's
 * tracked as a single row vector walked through the layers in reverse
 * (`v ← v·A_l`) rather than materializing the full S×S product at each layer.
 * That's an O(L·S²) walk instead of O(L·S³) matrix-matrix multiplies — the
 * difference between milliseconds and tens of seconds on a several-hundred-
 * token reasoning trace, which used to peg the main thread on every scrub.
 */
export function rolloutRow(
  promptAttention: number[][][],
  steps: StepData[],
  upto: number,
): number[] {
  if (!promptAttention?.length) return [];
  const P = promptAttention[0].length;
  const n = Math.max(0, Math.min(upto, steps.length - 1));
  const S = P + n;
  const L = promptAttention.length;

  const aug = new Float64Array(S * S);
  let v = new Float64Array(S);
  let next = new Float64Array(S);
  v[S - 1] = 1;

  for (let l = L - 1; l >= 0; l--) {
    // Build this layer's (S×S) causal mean-attention matrix.
    // Rows 0..P-1 from the prompt block; P..S-1 from generated step rows.
    aug.fill(0);
    for (let q = 0; q < P; q++) {
      const row = promptAttention[l][q];
      for (let j = 0; j <= q && j < S; j++) aug[q * S + j] = row[j];
    }
    for (let s = 1; s <= n; s++) {
      const row = steps[s]?.attention_row[l];
      if (!row) continue;
      const q = P + s - 1;
      const len = Math.min(row.length, S);
      for (let j = 0; j < len; j++) aug[q * S + j] = row[j];
    }
    // Residual augment (0.5·A + 0.5·I), renormalize rows.
    for (let i = 0; i < S; i++) {
      let sum = 0;
      for (let j = 0; j < S; j++) {
        const val = 0.5 * aug[i * S + j] + (i === j ? 0.5 : 0);
        aug[i * S + j] = val;
        sum += val;
      }
      if (sum > 0) for (let j = 0; j < S; j++) aug[i * S + j] /= sum;
    }
    // v ← v·aug (row-vector × matrix): the next layer's contribution to the
    // one row we care about.
    next.fill(0);
    for (let i = 0; i < S; i++) {
      const vi = v[i];
      if (vi === 0) continue;
      const base = i * S;
      for (let j = 0; j < S; j++) next[j] += vi * aug[base + j];
    }
    [v, next] = [next, v];
  }

  return Array.from(v);
}

/** The context tokens a step attended over: prompt + earlier generated tokens. */
export function contextTokens(
  promptTokens: string[],
  steps: StepData[],
  step: number,
): string[] {
  return [...promptTokens, ...steps.slice(0, step).map((s) => s.token)];
}

export function arrayMax(a: number[]): number {
  let max = 0;
  for (const v of a) if (v > max) max = v;
  return max || 1;
}

/**
 * Chat-template scaffolding tokens. They carry no content meaning, yet — together
 * with the first-token **attention sink** — they swallow almost all attention
 * (rollout over a chat-templated prompt collapses to ~100% on `<|im_start|>`).
 * We exclude them from attention *attribution* so what's left is the influence
 * over real content tokens; the cells stay visible but uncounted (honest, not
 * hidden). See `isAttnSink`.
 */
const STRUCTURAL_TOKENS = new Set([
  "<|im_start|>",
  "<|im_end|>",
  "<|endoftext|>",
  "<think>",
  "</think>",
]);

/**
 * True for tokens that should be excluded from attention attribution: the
 * first-position attention sink (`index === 0`, the canonical sink in
 * decoder-only transformers) and any chat-template special token.
 */
export function isAttnSink(token: string, index: number): boolean {
  return index === 0 || STRUCTURAL_TOKENS.has(token.trim());
}

/**
 * Max attention weight over **content** tokens only (sink/scaffolding excluded),
 * for normalizing the attribution color/bar scale so the sink doesn't flatten
 * every content cell to black.
 */
export function contentMax(tokens: string[], weights: number[]): number {
  const S = Math.min(tokens.length, weights.length);
  let max = 0;
  for (let i = 0; i < S; i++) {
    if (isAttnSink(tokens[i], i)) continue;
    if (weights[i] > max) max = weights[i];
  }
  return max || 1;
}
