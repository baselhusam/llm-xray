"""Regression tests for the hook engine's core invariants.

Codifies the manual "Verify:" checks described in CLAUDE.md's milestone notes
(row-stochastic attention, rollout math moved client-side, batched logit lens
== per-layer loop, phase tagging, generation stop conditions) as real pytest
assertions instead of one-off manual runs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from app.xray_engine import XRayHookEngine

PROMPT = "The capital of France is"


def _assert_row_stochastic(attn: np.ndarray, atol: float = 1e-4) -> None:
    """Every row of a (..., key_len) attention array should sum to ~1."""
    sums = attn.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=atol), sums


class TestPromptAttention:
    def test_prompt_attention_is_row_stochastic(self, engine: XRayHookEngine):
        session = engine.generate(PROMPT, thinking=False)
        capture, _first = session.prime()

        assert capture.attention.shape[0] == capture.num_layers
        assert capture.attention.shape[1] == capture.attention.shape[2] == len(
            capture.tokens
        )
        _assert_row_stochastic(capture.attention)


class TestGenerationSteps:
    def test_step_attention_row_is_row_stochastic(self, engine: XRayHookEngine):
        session = engine.generate(PROMPT, thinking=False)
        _capture, first = session.prime()
        assert first is not None
        _assert_row_stochastic(first.attention_row)

        second = session.advance()
        if second is not None:
            _assert_row_stochastic(second.attention_row)

    def test_step_fields_are_well_formed(self, engine: XRayHookEngine):
        session = engine.generate(PROMPT, thinking=False)
        _capture, first = session.prime()
        assert first is not None
        assert 0.0 <= first.prob <= 1.0
        assert first.entropy >= 0.0
        assert first.phase == "answer"  # thinking off → no reasoning phase at all
        assert len(first.trajectory) == _capture.num_layers
        # Trajectory layers are ascending and each top-1 pick is a valid probability.
        for i, layer in enumerate(first.trajectory):
            assert layer.layer_idx == i
            assert 0.0 <= layer.top_predictions[0].prob <= 1.0

    def test_thinking_off_never_emits_think_phase(self, engine: XRayHookEngine):
        session = engine.generate(PROMPT, thinking=False)
        capture, first = session.prime()
        assert capture.thinking is False
        steps = [first] if first is not None else []
        while session.stop_reason is None:
            step = session.advance()
            if step is None:
                break
            steps.append(step)
        assert all(s.phase == "answer" for s in steps)

    def test_generation_terminates(self, engine: XRayHookEngine, monkeypatch):
        # Cap the run short so a slow/rambling continuation can't stall the suite;
        # what's under test is that *some* stop condition fires, not which one.
        monkeypatch.setattr("app.xray_engine.MAX_NEW_TOKENS", 32)
        session = engine.generate(PROMPT, thinking=False)
        _capture, first = session.prime()
        steps = [first] if first is not None else []
        while session.stop_reason is None and len(steps) < 40:
            step = session.advance()
            if step is None:
                break
            steps.append(step)

        done = session.done()
        assert done.stop_reason in {"eos", "sentence", "max_tokens", "time_budget"}
        assert done.num_steps == len(steps)
        assert done.generated_text  # produced *some* text


class TestBatchedLogitLens:
    def test_batched_lens_matches_per_layer_loop(self, engine: XRayHookEngine):
        """`_lens` projects all layers in one matmul — verify it agrees with
        running final_norm + lm_head one layer at a time (the pre-M10 approach)."""
        adapter = engine._xray._require_adapter()
        torch.manual_seed(0)
        hidden_size = adapter.final_norm.weight.shape[-1]
        num_layers = adapter.num_layers
        last_vecs = torch.randn(num_layers, hidden_size)

        batched_probs, batched_top = engine._lens(last_vecs, adapter, top_k=5)

        for i in range(num_layers):
            logits_i = adapter.lm_head(adapter.final_norm(last_vecs[i : i + 1]))
            probs_i = torch.softmax(logits_i, dim=-1)[0]
            assert torch.allclose(batched_probs[i], probs_i, atol=1e-5), i
            assert batched_top.indices[i, 0].item() == int(torch.argmax(probs_i))


class TestHookCleanup:
    def test_hooks_do_not_leak_across_runs(self, engine: XRayHookEngine):
        blocks = engine._xray._require_adapter().layers
        baseline = [len(b._forward_hooks) for b in blocks]

        session = engine.generate(PROMPT, thinking=False)
        session.prime()
        session.advance()

        after = [len(b._forward_hooks) for b in blocks]
        assert after == baseline
