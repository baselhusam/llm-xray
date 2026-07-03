"use client";

/**
 * `PromptInput` — the prompt field + run button that kicks off an X-Ray.
 *
 * Controlled locally; submitting (Enter or the button) calls `onSubmit`. The
 * field is disabled while a run is in flight so a socket can't be flooded.
 */

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
  /** True while a run is streaming — disables input + relabels the button. */
  busy: boolean;
  /** Seeds the field — e.g. a prompt arriving from a shared link. */
  initialValue?: string;
  /** "inline" = field + button on one row; "stacked" = button on its own row (sidebar). */
  layout?: "inline" | "stacked";
  autoFocus?: boolean;
}

const DEFAULT_PROMPT = "Why is the sky blue?";

export function PromptInput({
  onSubmit,
  busy,
  initialValue,
  layout = "inline",
  autoFocus = true,
}: PromptInputProps) {
  const [value, setValue] = useState(initialValue ?? DEFAULT_PROMPT);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || busy) return;
    onSubmit(trimmed);
  };

  const stacked = layout === "stacked";

  const button = (
    <Button
      type="submit"
      disabled={busy || !value.trim()}
      className={cn("h-11 gap-2 px-5 font-medium", stacked && "w-full")}
    >
      {busy ? (
        <>
          <span className="pulse-dot h-2 w-2 rounded-full bg-current" aria-hidden />
          Running…
        </>
      ) : (
        "X-Ray it"
      )}
    </Button>
  );

  return (
    <form
      className={cn("flex w-full", stacked ? "flex-col gap-2" : "items-center gap-2")}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="group flex w-full items-center gap-2 rounded-xl border border-border bg-background/50 p-1.5 transition-colors focus-within:border-primary/50 focus-within:shadow-[0_0_0_3px_color-mix(in_oklch,var(--primary)_14%,transparent)]">
        <span className="pl-3 font-mono text-sm text-muted-foreground select-none" aria-hidden>
          {">"}
        </span>
        <Input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask the model anything…"
          disabled={busy}
          aria-label="Prompt"
          className="h-11 flex-1 border-0 bg-transparent px-1 text-base shadow-none focus-visible:ring-0 dark:bg-transparent"
          autoFocus={autoFocus}
        />
        {!stacked && button}
      </div>
      {stacked && button}
    </form>
  );
}
