"use client";

/**
 * `AppSidebar` — the persistent control rail of the dashboard shell. Holds the
 * brand mark, the prompt + thinking toggle (the only inputs), curated examples,
 * and this session's run history. Presentational: all state lives in `XRayApp`
 * and arrives via props. On desktop it's a sticky, full-height scroll column;
 * on mobile `XRayApp` renders it as a normal stacked block at the top.
 */

import { Brain } from "lucide-react";

import { Lockup } from "@/components/brand/Logo";
import { ExamplePrompts } from "@/components/ExamplePrompts";
import { PromptInput } from "@/components/PromptInput";
import { cn } from "@/lib/utils";

interface AppSidebarProps {
  modelLabel: string;
  busy: boolean;
  initialPrompt?: string;
  thinking: boolean;
  onToggleThinking: () => void;
  onSubmit: (prompt: string) => void;
  history: string[];
  onPickHistory: (prompt: string) => void;
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground/80">
      {children}
    </span>
  );
}

export function AppSidebar({
  modelLabel,
  busy,
  initialPrompt,
  thinking,
  onToggleThinking,
  onSubmit,
  history,
  onPickHistory,
}: AppSidebarProps) {
  return (
    <div className="flex h-full flex-col gap-7 p-5">
      {/* Brand */}
      <Lockup markClassName="size-6" wordClassName="text-sm" sublabel={modelLabel} />

      {/* Prompt */}
      <div className="flex flex-col gap-3">
        <SectionLabel>Prompt</SectionLabel>
        <PromptInput
          onSubmit={onSubmit}
          busy={busy}
          initialValue={initialPrompt}
          layout="stacked"
          autoFocus={false}
        />
        <button
          type="button"
          role="switch"
          aria-checked={thinking}
          disabled={busy}
          onClick={onToggleThinking}
          className={cn(
            "inline-flex items-center justify-between rounded-lg border px-3 py-2 text-xs font-medium transition-colors disabled:opacity-50",
            thinking
              ? "border-accent/40 bg-accent/10 text-accent"
              : "border-border bg-card/30 text-muted-foreground hover:text-foreground",
          )}
        >
          <span className="inline-flex items-center gap-1.5">
            <Brain className="size-3.5" />
            Reasoning trace
          </span>
          <span className="font-mono text-[10px] uppercase tracking-wider">
            {thinking ? "on" : "off"}
          </span>
        </button>
      </div>

      {/* Examples */}
      <div className="flex flex-col gap-3">
        <SectionLabel>Examples</SectionLabel>
        <ExamplePrompts onPick={onSubmit} busy={busy} variant="stack" />
      </div>

      {/* History */}
      {history.length > 0 && (
        <div className="flex min-h-0 flex-col gap-3">
          <SectionLabel>History</SectionLabel>
          <div className="thin-scroll flex max-h-48 flex-col gap-1 overflow-y-auto pr-1">
            {history.map((prompt, i) => (
              <button
                key={`${prompt}-${i}`}
                type="button"
                disabled={busy}
                onClick={() => onPickHistory(prompt)}
                className="truncate rounded-md px-2.5 py-1.5 text-left text-xs text-muted-foreground transition-colors hover:bg-card/50 hover:text-foreground disabled:pointer-events-none disabled:opacity-50"
                title={prompt}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-auto hidden border-t border-border/60 pt-4 lg:block">
        <a
          href="#how-it-works"
          className="text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          How it works ↓
        </a>
      </div>
    </div>
  );
}
