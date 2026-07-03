"use client";

/**
 * `export-card` — rasterize a DOM node (the {@link ShareCard}) to a PNG via
 * html-to-image, then download it or copy it to the clipboard. Client-only:
 * html-to-image touches `document`/`window`, so don't import this on the server.
 *
 * We capture at 2× pixel ratio for crisp output and force the card's dark
 * background so transparent corners don't bleed white on light surfaces.
 */

import { toBlob, toPng } from "html-to-image";

const CARD_BG = "#0a0a0c";

interface CaptureOptions {
  /** Settle delay (ms) so D3's enter transition finishes before capture. */
  settleMs?: number;
}

/** Wait one paint + an optional settle window before grabbing the node. */
async function settle(ms: number): Promise<void> {
  await new Promise((r) => requestAnimationFrame(() => r(null)));
  if (ms > 0) await new Promise((r) => setTimeout(r, ms));
}

function pngOptions(node: HTMLElement) {
  return {
    pixelRatio: 2,
    backgroundColor: CARD_BG,
    width: node.offsetWidth,
    height: node.offsetHeight,
    cacheBust: true,
  };
}

/** Capture `node` and trigger a PNG download named after the prompt. */
export async function downloadCardPng(
  node: HTMLElement,
  filename: string,
  { settleMs = 0 }: CaptureOptions = {},
): Promise<void> {
  await settle(settleMs);
  const dataUrl = await toPng(node, pngOptions(node));
  const link = document.createElement("a");
  link.download = filename;
  link.href = dataUrl;
  link.click();
}

/**
 * Capture `node` and write the PNG to the clipboard. Returns `false` when the
 * browser doesn't support image clipboard writes (caller can fall back to
 * download), `true` on success.
 */
export async function copyCardPng(
  node: HTMLElement,
  { settleMs = 0 }: CaptureOptions = {},
): Promise<boolean> {
  if (
    typeof ClipboardItem === "undefined" ||
    !navigator.clipboard?.write
  ) {
    return false;
  }
  await settle(settleMs);
  const blob = await toBlob(node, pngOptions(node));
  if (!blob) return false;
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
  return true;
}
