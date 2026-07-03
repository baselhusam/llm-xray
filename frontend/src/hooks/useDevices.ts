"use client";

/**
 * `useDevices` — fetches which inference devices the backend can run on
 * (`GET /api/devices`) so the CPU/CUDA/MPS pill only ever shows real options,
 * seeded with whatever the backend auto-selected at startup.
 */

import { useEffect, useState } from "react";

import { API_BASE_URL, type DeviceName } from "@/lib/xray-protocol";

export interface UseDevices {
  available: DeviceName[];
  /** The backend's active device once known; null until the fetch resolves. */
  current: DeviceName | null;
}

export function useDevices(): UseDevices {
  const [state, setState] = useState<UseDevices>({ available: ["cpu"], current: null });

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE_URL}/api/devices`)
      .then((res) => res.json())
      .then((data: { available: DeviceName[]; current: DeviceName }) => {
        if (cancelled) return;
        setState({ available: data.available, current: data.current });
      })
      .catch(() => {
        // Backend unreachable — keep the cpu-only fallback, no pill for
        // devices we can't confirm exist.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
