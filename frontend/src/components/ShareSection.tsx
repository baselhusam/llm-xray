"use client";

/**
 * `ShareSection` — the viral surface: the export buttons plus the off-screen
 * {@link ShareCard} they capture. Rendered by `XRayApp` once a run finishes.
 *
 * The card is mounted off-screen (not `display:none`, which would stop it
 * painting and break capture) and carries the run's findings — the confidence
 * curve, stats, and the hardest decision. Buttons rasterize it to PNG to
 * download or copy. (No "copy link": the app runs on-prem, so URLs wouldn't
 * resolve for anyone else.)
 */

import { useRef, useState } from "react";
import { Check, Copy, Download } from "lucide-react";

import { ShareCard } from "@/components/ShareCard";
import { Button } from "@/components/ui/button";
import { copyCardPng, downloadCardPng } from "@/lib/export-card";
import type { StepData } from "@/lib/xray-protocol";

interface ShareSectionProps {
  prompt: string;
  modelLabel: string;
  answerText: string;
  steps: StepData[];
}

type Flash = "downloaded" | "copied" | null;

// The card is static (no D3); a short settle is plenty for fonts/layout.
const SETTLE_MS = 120;

function slugify(prompt: string): string {
  const base = prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40);
  return `${base || "prompt"}-xray.png`;
}

export function ShareSection({ prompt, modelLabel, answerText, steps }: ShareSectionProps) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState<"download" | "copy" | null>(null);
  const [flash, setFlash] = useState<Flash>(null);

  const showFlash = (f: Flash) => {
    setFlash(f);
    setTimeout(() => setFlash((cur) => (cur === f ? null : cur)), 1800);
  };

  const onDownload = async () => {
    if (!cardRef.current || busy) return;
    setBusy("download");
    try {
      await downloadCardPng(cardRef.current, slugify(prompt), { settleMs: SETTLE_MS });
      showFlash("downloaded");
    } catch (e) {
      console.error("card download failed", e);
    } finally {
      setBusy(null);
    }
  };

  const onCopyImage = async () => {
    if (!cardRef.current || busy) return;
    setBusy("copy");
    try {
      const ok = await copyCardPng(cardRef.current, { settleMs: SETTLE_MS });
      if (ok) showFlash("copied");
      else await downloadCardPng(cardRef.current, slugify(prompt), { settleMs: 0 });
    } catch (e) {
      console.error("card copy failed", e);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-medium text-muted-foreground">
        Share this X-ray
      </h2>
      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onDownload} disabled={busy !== null} className="h-9">
          {flash === "downloaded" ? (
            <Check className="size-4" />
          ) : (
            <Download className="size-4" />
          )}
          {busy === "download" ? "Rendering…" : "Download card"}
        </Button>
        <Button
          onClick={onCopyImage}
          disabled={busy !== null}
          variant="secondary"
          className="h-9"
        >
          {flash === "copied" ? (
            <Check className="size-4" />
          ) : (
            <Copy className="size-4" />
          )}
          {busy === "copy" ? "Rendering…" : "Copy image"}
        </Button>
      </div>

      {/* Off-screen capture target — painted but visually removed. */}
      <div
        aria-hidden
        style={{
          position: "fixed",
          top: 0,
          left: -99999,
          pointerEvents: "none",
          opacity: 1,
        }}
      >
        <ShareCard
          ref={cardRef}
          prompt={prompt}
          modelLabel={modelLabel}
          answerText={answerText}
          steps={steps}
        />
      </div>
    </section>
  );
}
