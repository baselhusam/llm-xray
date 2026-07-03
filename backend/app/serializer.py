"""Serialize X-Ray captures into the WebSocket wire protocol.

Tensors are serialized with **orjson** (``OPT_SERIALIZE_NUMPY``) — it encodes
numpy arrays directly and is ~10x faster than stdlib ``json`` on these payloads.
By contract everything reaching this module is already detached, on CPU, and
numpy (the engine does that); we never serialize grad-tracking tensors.

This module and the frontend ``useXRay`` hook are tightly coupled — the message
shapes here mirror the protocol documented in CLAUDE.md. Change them together.
"""

from __future__ import annotations

from typing import Any

import orjson

from app.xray_engine import (
    GenerationDone,
    GenerationStep,
    PromptCapture,
    TopPrediction,
)

_DUMP_OPTS = orjson.OPT_SERIALIZE_NUMPY


def _dumps(payload: dict[str, Any]) -> str:
    """orjson → str (WebSocket text frames carry JSON; clients JSON.parse it)."""
    return orjson.dumps(payload, option=_DUMP_OPTS).decode("utf-8")


def _predictions(preds: list[TopPrediction]) -> list[dict[str, Any]]:
    return [
        {"token": p.token, "token_id": p.token_id, "prob": p.prob} for p in preds
    ]


def tokens_message(tokens: list[str], token_ids: list[int]) -> str:
    """First message of a run: the tokenized prompt."""
    return _dumps(
        {"type": "tokens", "data": {"tokens": tokens, "token_ids": token_ids}}
    )


def error_message(message: str) -> str:
    """Out-of-band error frame (bad input, timeout, internal failure)."""
    return _dumps({"type": "error", "data": {"message": message}})


# --- generation protocol (M10) ---------------------------------------------
# A generation run streams: meta → tokens → prompt_attention → step×N → done.
# This module and the frontend ``useXRay`` hook are tightly coupled — change the
# shapes together.


def meta_message(capture: PromptCapture, model_label: str) -> str:
    """Architecture facts the frontend uses as the source of truth for sizing."""
    return _dumps(
        {
            "type": "meta",
            "data": {
                "num_layers": capture.num_layers,
                "num_heads": capture.num_heads,
                "thinking": capture.thinking,
                "model_label": model_label,
            },
        }
    )


def prompt_attention_message(capture: PromptCapture) -> str:
    """The prompt's per-layer attention block (mean over heads), shape (layers, P, P)."""
    return _dumps(
        {"type": "prompt_attention", "data": {"attention": capture.attention}}
    )


def step_message(step: GenerationStep) -> str:
    """One generated token: chosen token + phase + entropy + trajectory + attention."""
    return _dumps(
        {
            "type": "step",
            "data": {
                "step": step.step,
                "token": step.token,
                "token_id": step.token_id,
                "prob": step.prob,
                "entropy": step.entropy,
                "phase": step.phase,
                "trajectory": [
                    {
                        "layer_idx": layer.layer_idx,
                        "top_predictions": _predictions(layer.top_predictions),
                    }
                    for layer in step.trajectory
                ],
                "attention_row": step.attention_row,  # numpy (layers, key_len), mean heads
            },
        }
    )


def done_message(done: GenerationDone) -> str:
    """Terminal frame: full continuation + how the run stopped."""
    return _dumps(
        {
            "type": "done",
            "data": {
                "generated_text": done.generated_text,
                "num_steps": done.num_steps,
                "stop_reason": done.stop_reason,
            },
        }
    )
