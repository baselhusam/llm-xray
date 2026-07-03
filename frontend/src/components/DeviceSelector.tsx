"use client";

/**
 * `DeviceSelector` — segmented pill to pick which device runs inference
 * (CPU always available; CUDA/MPS shown only if the backend reports them via
 * `/api/devices`). Mirrors `ViewTabs`' sliding-pill pattern (shared Framer
 * Motion `layoutId`). Selecting a device is sent with the next run's WS
 * message (`useXRay.run`); the backend switches the model onto it in place.
 */

import { motion } from "framer-motion";

import { cn } from "@/lib/utils";
import type { DeviceName } from "@/lib/xray-protocol";

const LABELS: Record<DeviceName, string> = {
  cpu: "CPU",
  cuda: "CUDA",
  mps: "MPS",
};

interface DeviceSelectorProps {
  value: DeviceName;
  available: DeviceName[];
  onChange: (device: DeviceName) => void;
  disabled?: boolean;
  className?: string;
}

export function DeviceSelector({
  value,
  available,
  onChange,
  disabled,
  className,
}: DeviceSelectorProps) {
  // Nothing to choose between (cpu-only, or the devices fetch hasn't resolved
  // yet) — don't render a pointless single-option control.
  if (available.length <= 1) return null;

  return (
    <div
      role="tablist"
      aria-label="Inference device"
      className={cn(
        "flex items-center gap-1 rounded-lg border border-border bg-muted/40 p-1",
        className,
      )}
    >
      {available.map((device) => {
        const active = value === device;
        return (
          <button
            key={device}
            type="button"
            role="tab"
            aria-selected={active}
            disabled={disabled}
            onClick={() => onChange(device)}
            className={cn(
              "relative flex-1 rounded-md px-2.5 py-1.5 font-mono text-[11px] font-medium transition-colors disabled:opacity-50",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="device-pill"
                className="absolute inset-0 rounded-md bg-primary"
                transition={{ type: "spring", stiffness: 500, damping: 38 }}
              />
            )}
            <span className="relative z-10">{LABELS[device]}</span>
          </button>
        );
      })}
    </div>
  );
}
