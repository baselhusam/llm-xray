"use client";

/**
 * `ViewTabs` — segmented control that switches which visualization is shown on
 * mobile (desktop shows them all split-screen, so this is hidden there). The
 * active pill is a single Framer Motion layer animated between options via a
 * shared `layoutId`, so the highlight slides rather than snaps.
 */

import { motion } from "framer-motion";

import { cn } from "@/lib/utils";

export type XRayView = "trajectory" | "heatmap" | "distribution" | "attention";

const VIEWS: { id: XRayView; label: string }[] = [
  { id: "trajectory", label: "Trajectory" },
  { id: "heatmap", label: "Heatmap" },
  { id: "distribution", label: "Candidates" },
  { id: "attention", label: "Attention" },
];

interface ViewTabsProps {
  value: XRayView;
  onChange: (view: XRayView) => void;
  className?: string;
}

export function ViewTabs({ value, onChange, className }: ViewTabsProps) {
  return (
    <div
      role="tablist"
      aria-label="Visualization"
      className={cn(
        "flex items-center gap-1 rounded-lg border border-border bg-muted/40 p-1",
        className,
      )}
    >
      {VIEWS.map((view) => {
        const active = value === view.id;
        return (
          <button
            key={view.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(view.id)}
            className={cn(
              "relative flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="view-tab-pill"
                className="absolute inset-0 rounded-md bg-primary"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative z-10">{view.label}</span>
          </button>
        );
      })}
    </div>
  );
}
