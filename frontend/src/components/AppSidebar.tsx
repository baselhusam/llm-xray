"use client";

/**
 * `AppSidebar` — the persistent control rail of the dashboard shell. Holds the
 * brand mark, the run inputs (prompt, thinking toggle, max-tokens cap, device
 * pill), curated examples, and this session's run history. Presentational: all state lives in `XRayApp`
 * and arrives via props. On desktop it's a sticky, full-height scroll column;
 * on mobile `XRayApp` renders it as a normal stacked block at the top.
 */

import { useState } from "react";

import { Brain, ChevronsUpDown, Hash } from "lucide-react";

import { Lockup } from "@/components/brand/Logo";
import { DeviceSelector } from "@/components/DeviceSelector";
import { ExamplePrompts } from "@/components/ExamplePrompts";
import { PromptInput } from "@/components/PromptInput";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import {
  DEFAULT_MAX_TOKENS,
  MAX_TOKENS_LIMIT,
  type DeviceName,
} from "@/lib/xray-protocol";

interface AppSidebarProps {
  modelLabel: string;
  busy: boolean;
  initialPrompt?: string;
  thinking: boolean;
  onToggleThinking: () => void;
  maxTokens: number;
  onMaxTokensChange: (maxTokens: number) => void;
  device: DeviceName;
  availableDevices: DeviceName[];
  onDeviceChange: (device: DeviceName) => void;
  onSubmit: (prompt: string) => void;
  history: string[];
  onPickHistory: (prompt: string) => void;
}

/** Curated length caps (multiples of 128, doubling to the backend's limit). */
const MAX_TOKENS_PRESETS = [128, 256, 512, 1024, 2048, 4096];

/**
 * Length-cap picker: a popover menu of preset caps plus a free-typed custom
 * value. Custom edits live in a local draft so partial typing (empty field,
 * mid-edit numbers) doesn't clamp under the user's fingers; the value commits
 * on Enter/Set, clamped to what the backend accepts (1–MAX_TOKENS_LIMIT).
 */
function MaxTokensField({
  value,
  onChange,
  disabled,
}: {
  value: number;
  onChange: (value: number) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(String(value));

  const commit = (next: number) => {
    const clamped = Math.min(Math.max(next, 1), MAX_TOKENS_LIMIT);
    setDraft(String(clamped));
    onChange(clamped);
    setOpen(false);
  };

  const commitDraft = () => {
    const parsed = Number.parseInt(draft, 10);
    commit(Number.isNaN(parsed) ? DEFAULT_MAX_TOKENS : parsed);
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) setDraft(String(value)); // drop a stale, uncommitted draft
      }}
    >
      <PopoverTrigger
        disabled={disabled}
        aria-label="Maximum generated tokens"
        className={cn(
          "flex items-center justify-between gap-2 rounded-lg border border-border bg-card/30 px-3 py-2 text-xs font-medium text-muted-foreground transition-colors disabled:opacity-50",
          "hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
        )}
      >
        <span className="inline-flex items-center gap-1.5">
          <Hash className="size-3.5" />
          Max tokens
        </span>
        <span className="inline-flex items-center gap-1 font-mono text-[11px] tabular-nums text-foreground">
          {value}
          <ChevronsUpDown className="size-3 text-muted-foreground" />
        </span>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={6} className="w-56 gap-2 p-2">
        <div className="grid grid-cols-3 gap-1">
          {MAX_TOKENS_PRESETS.map((preset) => {
            const active = preset === value;
            return (
              <button
                key={preset}
                type="button"
                onClick={() => commit(preset)}
                className={cn(
                  "cursor-pointer rounded-md px-2 py-1.5 font-mono text-[11px] tabular-nums transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                {preset}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1.5 border-t border-border/60 pt-2">
          <input
            type="number"
            inputMode="numeric"
            min={1}
            max={MAX_TOKENS_LIMIT}
            value={draft}
            placeholder={`1–${MAX_TOKENS_LIMIT}`}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitDraft();
            }}
            className="h-7 w-full min-w-0 rounded-md border border-border bg-transparent px-2 font-mono text-[11px] tabular-nums text-foreground outline-none placeholder:text-muted-foreground/60 focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-label="Custom maximum generated tokens"
          />
          <button
            type="button"
            onClick={commitDraft}
            className="h-7 cursor-pointer rounded-md bg-muted/60 px-2.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            Set
          </button>
        </div>
      </PopoverContent>
    </Popover>
  );
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
  maxTokens,
  onMaxTokensChange,
  device,
  availableDevices,
  onDeviceChange,
  onSubmit,
  history,
  onPickHistory,
}: AppSidebarProps) {
  return (
    <div className="flex h-full flex-col gap-7 p-5">
      {/* Brand */}
      <Lockup markClassName="size-9" wordClassName="text-sm" sublabel={modelLabel} />

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
        <MaxTokensField
          value={maxTokens}
          onChange={onMaxTokensChange}
          disabled={busy}
        />
        <DeviceSelector
          value={device}
          available={availableDevices}
          onChange={onDeviceChange}
          disabled={busy}
        />
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
