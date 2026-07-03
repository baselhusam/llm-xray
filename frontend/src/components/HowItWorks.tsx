/**
 * `HowItWorks` — the landing-page explainer. A short pipeline line plus three
 * cards, one per view (trajectory / candidates / attention), so a visitor knows
 * what they're looking at before (or after) they run a prompt. Static.
 */

import { MODEL_LABEL, NUM_LAYERS } from "@/lib/xray-protocol";

const PIPELINE = [
  "your prompt",
  "pick a token",
  `read all ${NUM_LAYERS} layers`,
  "append + repeat",
  "streamed live",
] as const;

const VIEWS = [
  {
    title: "Logit-lens trajectory",
    body: `Project every layer's hidden state through the unembedding to see ${MODEL_LABEL}'s running guess. Read top to bottom and watch the prediction form — noise in early layers, locking onto the answer (vermilion) with depth.`,
  },
  {
    title: "Next-token candidates",
    body: "The real distribution behind each chosen token: what it picked, how sure it was, and the runner-up it rejected — the “it was 51% sure, almost said something else” moment.",
  },
  {
    title: "Attention attribution",
    body: "Which earlier tokens drove this one. Attention rolled out across all layers gives an honest read; flip to raw to inspect a single layer or head.",
  },
] as const;

export function HowItWorks() {
  return (
    <section className="flex flex-col gap-5 border-t border-border pt-10">
      <div className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold tracking-tight">How it works</h2>
        <p className="max-w-2xl text-sm text-muted-foreground">
          {MODEL_LABEL} answers your prompt one token at a time on a real
          forward pass. For every token, PyTorch hooks capture each layer&apos;s hidden
          state and attention, rolled out and streamed back live — nothing pre-baked.
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-1.5 font-mono text-xs text-muted-foreground">
          {PIPELINE.map((step, i) => (
            <span key={step} className="flex items-center gap-1.5">
              <span className="inline-flex items-center gap-1.5 rounded-md bg-card/70 px-2.5 py-1.5 ring-1 ring-border">
                <span className="text-[10px] font-semibold text-primary">{i + 1}</span>
                {step}
              </span>
              {i < PIPELINE.length - 1 && <span className="text-primary/50">→</span>}
            </span>
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        {VIEWS.map((v) => (
          <div
            key={v.title}
            className="surface-panel flex flex-col gap-2 rounded-xl border border-border/60 p-5 transition-colors hover:border-primary/40"
          >
            <h3 className="flex items-center gap-2 text-sm font-medium">
              <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
              {v.title}
            </h3>
            <p className="text-xs leading-relaxed text-muted-foreground">{v.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
