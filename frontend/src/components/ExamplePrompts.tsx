"use client";

/**
 * `ExamplePrompts` — the "try it now" strip: a curated set of short prompts that
 * read interestingly through the model. Clicking one runs it immediately, so a
 * first-time visitor never faces a blank field. Disabled while a run streams.
 *
 * `variant="row"` (default) wraps pill chips horizontally (empty-state hero);
 * `variant="stack"` lays out full-width rows for the narrow sidebar.
 */

import { cn } from "@/lib/utils";

interface ExamplePromptsProps {
  onPick: (prompt: string) => void;
  busy: boolean;
  variant?: "row" | "stack";
  className?: string;
}

/** Short prompts that read interestingly as the model reasons + answers. */
export const EXAMPLES = [
  "Why is the sky blue?",
  "What is the capital of France?",
  "Is 17 a prime number?",
  "Write a haiku about autumn.",
  "What comes after Tuesday?",
  "Explain gravity to a child.",
] as const;

export function ExamplePrompts({
  onPick,
  busy,
  variant = "row",
  className,
}: ExamplePromptsProps) {
  if (variant === "stack") {
    return (
      <div className={cn("flex flex-col gap-1.5", className)}>
        {EXAMPLES.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={busy}
            onClick={() => onPick(prompt)}
            className="truncate rounded-lg border border-border/70 bg-card/40 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
          >
            {prompt}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className={cn("flex flex-wrap items-center justify-center gap-2", className)}>
      {EXAMPLES.map((prompt) => (
        <button
          key={prompt}
          type="button"
          disabled={busy}
          onClick={() => onPick(prompt)}
          className="rounded-full border border-border bg-card/50 px-3 py-1.5 text-xs text-muted-foreground transition-all hover:-translate-y-px hover:border-primary/40 hover:bg-primary/10 hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
