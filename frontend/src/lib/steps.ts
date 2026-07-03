/**
 * Derived per-step analytics shared across views.
 *
 *   • `firstLockLayer` — the depth where a decision crystallized: the
 *     shallowest layer whose logit-lens top-1 already matches the token the
 *     model actually emitted. Drives the auto-followed layer focus and the
 *     timeline's "depth" coloring.
 *   • `keyMoments` — the near-miss decisions: steps where the runner-up came
 *     closest to winning. These are the interesting clicks in a 200-token
 *     reasoning trace, surfaced as chips so nobody has to scrub for them.
 */

import { isStructuralToken } from "@/lib/tokens";
import type { StepData } from "@/lib/xray-protocol";

/** Shallowest layer whose logit-lens top-1 matches the emitted token. */
export function firstLockLayer(step: StepData, numLayers: number): number {
  const hit = step.trajectory.find(
    (t) => t.top_predictions[0]?.token_id === step.token_id,
  );
  return hit ? hit.layer_idx : numLayers - 1;
}

export interface KeyMoment {
  step: number;
  chosen: string;
  chosenProb: number;
  runnerUp: string;
  runnerUpProb: number;
  /** p(chosen) − p(runnerUp); small = the model nearly said something else. */
  margin: number;
}

// Below this the "almost" token is noise, not a real contender.
const MIN_RUNNER_UP_PROB = 0.05;

/** The `max` steps with the smallest win margin, returned in step order. */
export function keyMoments(steps: StepData[], max = 4): KeyMoment[] {
  const moments: KeyMoment[] = [];
  for (const s of steps) {
    if (isStructuralToken(s.token)) continue;
    const final = s.trajectory[s.trajectory.length - 1]?.top_predictions ?? [];
    const chosen = final[0];
    const runnerUp = final[1];
    if (!chosen || !runnerUp || runnerUp.prob < MIN_RUNNER_UP_PROB) continue;
    moments.push({
      step: s.step,
      chosen: s.token,
      chosenProb: chosen.prob,
      runnerUp: runnerUp.token,
      runnerUpProb: runnerUp.prob,
      margin: chosen.prob - runnerUp.prob,
    });
  }
  return moments
    .sort((a, b) => a.margin - b.margin)
    .slice(0, max)
    .sort((a, b) => a.step - b.step);
}
