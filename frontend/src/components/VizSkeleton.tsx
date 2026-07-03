"use client";

/**
 * `VizSkeleton` — placeholder bodies for the visualization panels, shown while a
 * run is connecting / before its first `step` lands. Each variant roughly traces
 * the shape of the real view so the layout doesn't jump when data arrives.
 */

import { Skeleton } from "@/components/ui/skeleton";
import { NUM_LAYERS } from "@/lib/xray-protocol";

export type VizSkeletonVariant = "trajectory" | "heatmap" | "distribution" | "attention";

// Row/column counts mirror the real views (one row per layer in
// LensTrajectory/LensHeatmap; LensHeatmap's MAX_COLS is 10) so the panel
// doesn't visibly resize the moment real data lands.
const HEATMAP_COLS = 10;

export function VizSkeleton({ variant }: { variant: VizSkeletonVariant }) {
  if (variant === "heatmap") {
    return (
      <div
        className="grid w-full gap-1 p-1"
        style={{ gridTemplateColumns: `repeat(${HEATMAP_COLS}, 1fr)` }}
      >
        {Array.from({ length: NUM_LAYERS * HEATMAP_COLS }, (_, i) => (
          <Skeleton key={i} className="h-3 rounded-sm" />
        ))}
      </div>
    );
  }

  if (variant === "trajectory") {
    return (
      <div className="flex flex-col gap-1.5 py-1">
        {Array.from({ length: NUM_LAYERS }, (_, i) => (
          <div key={i} className="flex items-center gap-2">
            <Skeleton className="h-3 w-7 shrink-0" />
            <Skeleton className="h-3 w-16 shrink-0" />
            <Skeleton className="h-3 flex-1" style={{ maxWidth: `${30 + ((i * 37) % 60)}%` }} />
          </div>
        ))}
      </div>
    );
  }

  if (variant === "attention") {
    return (
      <div className="flex flex-col gap-3 py-2">
        <div className="flex gap-0.5">
          {Array.from({ length: 18 }, (_, i) => (
            <Skeleton key={i} className="h-7 flex-1" />
          ))}
        </div>
        <Skeleton className="h-3 w-1/2" />
      </div>
    );
  }

  // distribution — ranked bars
  return (
    <div className="flex flex-col gap-3 py-2">
      {Array.from({ length: 5 }, (_, i) => (
        <div key={i} className="flex items-center gap-3">
          <Skeleton className="h-4 w-20 shrink-0" />
          <Skeleton className="h-5 flex-1" style={{ maxWidth: `${90 - i * 16}%` }} />
        </div>
      ))}
    </div>
  );
}
