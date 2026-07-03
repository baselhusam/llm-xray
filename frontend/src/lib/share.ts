/**
 * Share constants + URL helpers. Pure (no DOM), so this is safe to import from
 * both server (metadata / OG route) and client (ShareCard / export). The PNG
 * capture itself lives in `export-card.ts`, which is client-only.
 */

/** Canonical site origin; override per-env with `NEXT_PUBLIC_SITE_URL`. */
export const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

/** Bare host shown on the card watermark (no scheme). */
export const SITE_HOST = SITE_URL.replace(/^https?:\/\//, "");

/**
 * A link that reopens the app pre-loaded with this prompt (closes the loop).
 * Carrying the prediction too lets the unfurled OG card show it without re-running.
 */
export function buildShareUrl(
  prompt: string,
  pred?: { token: string; prob: number },
): string {
  const u = new URL(SITE_URL);
  u.searchParams.set("prompt", prompt);
  if (pred) {
    u.searchParams.set("token", pred.token);
    u.searchParams.set("prob", pred.prob.toFixed(4));
  }
  return u.toString();
}

/** URL of the dynamically generated OG image for a given run. */
export function buildOgImageUrl(params: {
  prompt: string;
  token?: string;
  prob?: number;
}): string {
  const u = new URL("/api/og", SITE_URL);
  u.searchParams.set("prompt", params.prompt);
  if (params.token !== undefined) u.searchParams.set("token", params.token);
  if (params.prob !== undefined) u.searchParams.set("prob", params.prob.toFixed(4));
  return u.toString();
}
