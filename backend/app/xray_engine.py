"""The X-Ray hook engine — the heart of the product.

Registers a PyTorch forward hook on every decoder block (to capture the
residual-stream hidden state) and reads per-layer attention from the forward
pass's ``output_attentions``, then applies the **logit lens**: each layer's
hidden state is projected through the model's final norm and the unembedding
(``lm_head``) to reveal what the model "believes" the next token is at that
depth. Predictions sharpen toward later layers.

The entry point is :meth:`XRayHookEngine.generate` — **greedy autoregressive
generation**: drives the model token-by-token with a KV cache, yielding
per-step captures so the frontend can *watch the model think* as text streams.
Each step carries the chosen token, its probability + Shannon entropy, the
28-layer logit-lens trajectory, the per-head attention row that produced it,
and an **attention rollout** row (an honest "what drove this token"
attribution).

Module paths are resolved generically via :class:`~app.model.ModelAdapter`, so
this engine is model-agnostic (GPT-2, Qwen3, …). Hooks are always removed after
use (see :meth:`_hooks`) so nothing leaks across requests. The model must be
loaded with eager attention — SDPA/Flash kernels never expose the attention
weights (see ``XRayModel.load``).
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from app.model import ModelAdapter, XRayModel

# Generation budget / stop tuning. The model is fed a chat-templated prompt and
# answers; with thinking on it first emits a <think>…</think> reasoning trace.
# Reasoning is long, so it gets a bigger cap + budget; the answer alone is short.
MAX_NEW_TOKENS = 1024  # answer-only (thinking off)
MAX_NEW_TOKENS_THINKING = 1024  # reasoning trace + answer
MIN_ANSWER_TOKENS = 6  # don't sentence-stop the answer on an early "."
# Time budgets sized so the 1024-token cap is actually reachable (~0.13s/token
# on CPU ≈ 135s for a full run); they remain a safety valve, not the usual stop.
GEN_TIME_BUDGET_S = 180.0
GEN_TIME_BUDGET_THINKING_S = 300.0
_SENTENCE_END = re.compile(r"[.!?\n]")


@dataclass
class TopPrediction:
    """One entry of a layer's logit-lens top-k for the decision position."""

    token: str
    token_id: int
    prob: float


# --- Generation captures ---------------------------------------------------


@dataclass
class LayerTopK:
    """A single layer's logit-lens top-k at one generation step's decision position."""

    layer_idx: int
    top_predictions: list[TopPrediction]


@dataclass
class PromptCapture:
    """Captured from the initial prompt forward pass (seeds the views)."""

    tokens: list[str]
    token_ids: list[int]
    num_layers: int
    num_heads: int
    thinking: bool
    # Per-layer attention over the prompt, mean over heads, shape (layers, P, P).
    # Mean-over-heads keeps the payload tractable for long reasoning traces; the
    # frontend rolls it out across layers on demand for the selected step.
    attention: np.ndarray


@dataclass
class GenerationStep:
    """Everything captured when the model commits to one generated token."""

    step: int
    token: str
    token_id: int
    prob: float
    entropy: float  # Shannon entropy (nats) of the final-layer next-token dist.
    phase: str  # "think" | "answer"
    # Per-layer logit-lens top-k at this token's decision position (the depth
    # trajectory: watch the prediction sharpen from layer 0 → last).
    trajectory: list[LayerTopK]
    # The query row that produced this token, mean over heads, shape
    # (layers, key_len). key_len == number of context tokens it attended over.
    attention_row: np.ndarray


@dataclass
class GenerationDone:
    """Terminal marker for a generation run."""

    generated_text: str
    num_steps: int
    stop_reason: str  # "eos" | "sentence" | "max_tokens" | "time_budget"


def _entropy(probs: torch.Tensor) -> float:
    """Shannon entropy (nats) of a 1-D probability vector."""
    p = probs[probs > 0]
    return float(-(p * torch.log(p)).sum().item())


class XRayHookEngine:
    """Runs the model with forward hooks capturing per-layer attention + hidden states.

    One engine instance wraps the shared :class:`XRayModel`. Hooks are registered
    fresh per forward pass and torn down again (see :meth:`_hooks`) — no state is
    carried between runs at the engine level (a :class:`GenerationSession` holds
    the per-run cache while generating).
    """

    def __init__(self, model: XRayModel):
        self._xray = model

    def generate(
        self, prompt: str, thinking: bool = True, top_k: int = 5
    ) -> "GenerationSession":
        """Create a step-wise generation session for ``prompt``.

        The prompt is wrapped in the model's chat template so the model *answers*
        it; with ``thinking`` on (and the model supports it) it first emits a
        ``<think>…</think>`` reasoning trace. The session exposes
        :meth:`GenerationSession.prime` (prompt pass → capture + first token) and
        :meth:`GenerationSession.advance` (one more token, or ``None`` at a stop),
        so each token can be streamed as it lands.
        """
        return GenerationSession(self, prompt, thinking=thinking, top_k=top_k)

    # --- shared internals -------------------------------------------------

    @contextmanager
    def _hooks(self) -> Iterator[list[torch.Tensor]]:
        """Register residual-stream hooks on every decoder block for one pass.

        Yields a list (indexed by layer) that the hooks fill with each block's
        output hidden state ``(1, q_len, H)``. Hooks are removed unconditionally
        on exit. The attention weights come from the forward pass's
        ``output_attentions`` instead (its per-module return signature varies by
        architecture), so only hidden states are hooked here.
        """
        blocks = self._xray._require_adapter().layers
        n = len(blocks)
        hidden_states: list[torch.Tensor | None] = [None] * n
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def block_hook(i: int):
            def fn(_module, _inp, out):
                # Decoder blocks return a tuple (GPT-2) or a bare tensor (Qwen3);
                # out[0] is the residual stream in either case.
                hidden_states[i] = out[0] if isinstance(out, tuple) else out

            return fn

        try:
            for i, block in enumerate(blocks):
                handles.append(block.register_forward_hook(block_hook(i)))
            yield hidden_states  # type: ignore[misc]
        finally:
            for h in handles:
                h.remove()

        missing = [i for i, v in enumerate(hidden_states) if v is None]
        if missing:
            raise RuntimeError(
                f"Hook capture incomplete — hidden_states missing {missing}. "
                "Is the model loaded with eager attention?"
            )

    @staticmethod
    def _require_attentions(outputs, n: int) -> list[torch.Tensor]:
        attentions = list(outputs.attentions) if outputs.attentions is not None else []
        if len(attentions) != n:
            raise RuntimeError(
                f"Expected {n} attention layers, got {len(attentions)}. "
                "Is the model loaded with eager attention?"
            )
        return attentions

    @staticmethod
    def _lens(
        last_vecs: torch.Tensor, adapter: ModelAdapter, top_k: int
    ) -> tuple[torch.Tensor, torch.return_types.topk]:
        """Batched logit lens over per-layer last-position vectors.

        ``last_vecs`` is ``(L, H)``. At Qwen3's 151,936 vocab the unembedding
        dominates cost, so we run final_norm + lm_head **once** over all layers
        rather than 28 separate matmuls. Returns ``(probs (L, vocab), topk)``.
        """
        logits = adapter.lm_head(adapter.final_norm(last_vecs))  # (L, vocab)
        probs = torch.softmax(logits, dim=-1)
        return probs, torch.topk(probs, top_k, dim=-1)

    @staticmethod
    def _decode_topk(tokenizer, top, layer_idx: int) -> list[TopPrediction]:
        ids = top.indices[layer_idx].tolist()
        vals = top.values[layer_idx].tolist()
        return [
            TopPrediction(token=tokenizer.decode([int(tid)]), token_id=int(tid), prob=float(p))
            for tid, p in zip(ids, vals)
        ]


class GenerationSession:
    """Holds the per-run state (KV cache + assembled attention) for one generation.

    Drive it with :meth:`prime` once, then :meth:`advance` until it returns
    ``None``. All torch work happens inside these calls so the caller can run
    each on a worker thread and stream the result between calls.
    """

    def __init__(
        self,
        engine: XRayHookEngine,
        prompt: str,
        thinking: bool = True,
        top_k: int = 5,
    ):
        self._engine = engine
        self._xray = engine._xray
        self.prompt = prompt
        self.top_k = top_k

        self._tokenizer, self._model = self._xray._require_loaded()
        self._adapter = self._xray._require_adapter()
        self._device = self._xray.device

        # Resolve the thinking markers; only enable thinking if the tokenizer
        # actually has them and a chat template exists (keeps this model-agnostic).
        self._think_open = self._special_id("<think>")
        self._think_close = self._special_id("</think>")
        has_template = getattr(self._tokenizer, "chat_template", None) is not None
        self.thinking = bool(thinking and has_template and self._think_open is not None)
        self._has_template = has_template
        self.max_new_tokens = MAX_NEW_TOKENS_THINKING if self.thinking else MAX_NEW_TOKENS
        self._time_budget = (
            GEN_TIME_BUDGET_THINKING_S if self.thinking else GEN_TIME_BUDGET_S
        )

        self._past = None  # transformers Cache, carried across steps
        self._next_token_id: int | None = None  # token to feed into the next pass
        self._generated_ids: list[int] = []
        self._step = 0
        self._answer_tokens = 0
        self._in_think = self.thinking  # the model emits <think> first when on
        self._start_time = 0.0
        self.stop_reason: str | None = None

    def _special_id(self, token: str) -> int | None:
        try:
            tid = self._tokenizer.convert_tokens_to_ids(token)
        except Exception:  # noqa: BLE001
            return None
        unk = getattr(self._tokenizer, "unk_token_id", None)
        return tid if isinstance(tid, int) and tid >= 0 and tid != unk else None

    def _prompt_ids(self) -> list[int]:
        """Chat-template the prompt so the model answers it (raw encode otherwise).

        We render the template to a string then encode it — ``apply_chat_template``
        with ``tokenize=True`` returns a tokenizers ``Encoding`` here, not a plain
        ``list[int]``, so go through text (the chat special tokens round-trip to
        their ids on encode).
        """
        if self._has_template:
            text = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": self.prompt}],
                add_generation_prompt=True,
                enable_thinking=self.thinking,
                tokenize=False,
            )
            return self._tokenizer.encode(text)
        return self._tokenizer.encode(self.prompt)

    @torch.no_grad()
    def prime(self) -> tuple[PromptCapture, GenerationStep | None]:
        """Run the prompt pass; return the capture + the first token (step 0)."""
        self._start_time = time.monotonic()
        token_ids = self._prompt_ids()
        input_ids = torch.tensor([token_ids], device=self._device)

        with self._engine._hooks() as hidden_states:
            outputs = self._model(input_ids, output_attentions=True, use_cache=True)
        n = len(hidden_states)
        attentions = self._engine._require_attentions(outputs, n)
        self._past = outputs.past_key_values

        # Mean over heads → (layers, P, P). Mean keeps the payload small for long
        # reasoning traces; the frontend rolls it out across layers on demand.
        prompt_attn = np.stack(
            [attentions[l][0].mean(0).detach().cpu().numpy() for l in range(n)]
        )  # (L, P, P)
        num_heads = attentions[0].shape[1]

        capture = PromptCapture(
            tokens=[self._tokenizer.decode([t]) for t in token_ids],
            token_ids=list(token_ids),
            num_layers=n,
            num_heads=int(num_heads),
            thinking=self.thinking,
            attention=np.ascontiguousarray(prompt_attn),
        )

        # Step 0: prediction at the prompt's last position; its decision row is
        # the prompt's final query row (mean over heads), length P.
        last_vecs = torch.stack([hs[0, -1, :] for hs in hidden_states])  # (L, H)
        step = self._build_step(last_vecs, np.ascontiguousarray(prompt_attn[:, -1, :]))
        return capture, step

    @torch.no_grad()
    def advance(self) -> GenerationStep | None:
        """Generate one more token, or return ``None`` once a stop condition fires."""
        if self.stop_reason is not None or self._next_token_id is None:
            return None

        input_ids = torch.tensor([[self._next_token_id]], device=self._device)
        with self._engine._hooks() as hidden_states:
            outputs = self._model(
                input_ids,
                past_key_values=self._past,
                output_attentions=True,
                use_cache=True,
            )
        n = len(hidden_states)
        attentions = self._engine._require_attentions(outputs, n)
        self._past = outputs.past_key_values

        # (layers, key_len): the new query row over all context keys, mean heads.
        step_attn_row = np.stack(
            [attentions[l][0, :, 0, :].mean(0).detach().cpu().numpy() for l in range(n)]
        )
        last_vecs = torch.stack([hs[0, -1, :] for hs in hidden_states])  # (L, H)
        return self._build_step(last_vecs, np.ascontiguousarray(step_attn_row))

    # --- internals --------------------------------------------------------

    def _build_step(
        self, hidden_last: torch.Tensor, step_attn_row: np.ndarray
    ) -> GenerationStep | None:
        """Assemble a :class:`GenerationStep`, or ``None`` if the model emits EOS.

        ``step_attn_row`` is ``(layers, key_len)`` (mean over heads) for this
        token's decision position. Updates the rolling cache and stop state.
        """
        probs, top = self._engine._lens(hidden_last, self._adapter, self.top_k)
        final = probs[-1]
        token_id = int(torch.argmax(final))

        eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is not None and token_id == eos:
            self.stop_reason = "eos"
            return None  # don't surface the end-of-turn marker as a step

        # Phase: the <think>…</think> span (inclusive) is reasoning, the rest is
        # the answer. The closing marker still belongs to the thinking phase.
        phase = "think" if self._in_think else "answer"
        if token_id == self._think_close:
            self._in_think = False
        if phase == "answer":
            self._answer_tokens += 1

        trajectory = [
            LayerTopK(
                layer_idx=l,
                top_predictions=self._engine._decode_topk(self._tokenizer, top, l),
            )
            for l in range(probs.shape[0])
        ]

        step = GenerationStep(
            step=self._step,
            token=self._tokenizer.decode([token_id]),
            token_id=token_id,
            prob=float(final[token_id]),
            entropy=_entropy(final),
            phase=phase,
            trajectory=trajectory,
            attention_row=step_attn_row,
        )

        self._generated_ids.append(token_id)
        self._next_token_id = token_id
        self._step += 1
        self._maybe_stop(step)
        return step

    def _maybe_stop(self, step: GenerationStep) -> None:
        # Never sentence-stop inside the reasoning trace (it has many sentences);
        # only finish on a sentence once we're a few tokens into the answer.
        if (
            step.phase == "answer"
            and self._answer_tokens >= MIN_ANSWER_TOKENS
            and _SENTENCE_END.search(step.token)
        ):
            self.stop_reason = "sentence"
        elif self._step >= self.max_new_tokens:
            self.stop_reason = "max_tokens"
        elif time.monotonic() - self._start_time > self._time_budget:
            self.stop_reason = "time_budget"

    def done(self) -> GenerationDone:
        return GenerationDone(
            generated_text=self._tokenizer.decode(
                self._generated_ids, skip_special_tokens=True
            ),
            num_steps=self._step,
            stop_reason=self.stop_reason or "max_tokens",
        )
