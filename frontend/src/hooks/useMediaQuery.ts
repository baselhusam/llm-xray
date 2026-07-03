"use client";

/**
 * `useMediaQuery` — subscribe to a CSS media query and re-render on changes.
 *
 * SSR-safe: returns `false` on the server and the first client render (no
 * `window`), then settles to the real match in a layout effect before paint,
 * so it never causes a hydration mismatch flash for content that only mounts
 * after data arrives. Used to pick split-screen (desktop) vs. tabbed (mobile).
 */

import { useEffect, useLayoutEffect, useState } from "react";

// useLayoutEffect warns when it runs during SSR (no DOM to measure); fall back
// to useEffect there since "use client" components still render once on the
// server. This is what actually makes the settle happen before paint on the
// client, per the docstring above.
const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useIsomorphicLayoutEffect(() => {
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange(); // sync to the current value on mount / query change
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);

  return matches;
}

/** Tailwind's `lg` breakpoint — the desktop/mobile split point for the app. */
export const DESKTOP_QUERY = "(min-width: 1024px)";
