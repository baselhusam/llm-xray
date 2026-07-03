/**
 * Byte-level BPE tokens are decoded to real strings, so leading spaces
 * and newlines are significant. Render them with visible markers so chips,
 * axes, and prediction labels read faithfully (and don't collapse to "").
 */
export function displayToken(token: string): string {
  return token.replace(/ /g, "·").replace(/\n/g, "⏎");
}

/**
 * Structural reasoning markers that can appear *in a generation*. They delimit
 * the thinking trace but carry no content — views render them as phase
 * dividers, not token chips, and analytics (key moments, arcs) skip them.
 */
const STRUCTURAL = new Set(["<think>", "</think>"]);

export function isStructuralToken(token: string): boolean {
  return STRUCTURAL.has(token.trim());
}
