"""Model + tokenizer loading and basic inference.

The model is loaded once at process startup (see lifespan in main.py) and reused
across requests. Runs on CPU by design — see CLAUDE.md scope guardrails.

The engine is **model-agnostic**: loading goes through ``AutoModelForCausalLM`` and
the architecture-specific module paths (decoder blocks, final norm, unembedding) are
resolved generically by :class:`ModelAdapter`, so swapping ``MODEL_NAME`` between e.g.
``Qwen/Qwen3-0.6B-Base`` and ``gpt2`` needs no engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

MODEL_NAME = "Qwen/Qwen3-1.7B"  # 2.03B params, 28 layers, 16 query heads, vocab 151936


@dataclass
class TokenizedPrompt:
    tokens: list[str]
    token_ids: list[int]


@dataclass
class NextTokenPrediction:
    token: str
    token_id: int
    prob: float


@dataclass
class ModelAdapter:
    """Architecture-agnostic handles into a causal LM.

    Resolves the few modules the X-Ray engine needs without naming any
    architecture-specific attribute. Works for GPT-2 (``transformer.h`` /
    ``ln_f``) and Qwen3 / Llama-style models (``model.layers`` / ``norm``).
    """

    layers: torch.nn.ModuleList  # the decoder blocks (residual-stream hooks)
    final_norm: torch.nn.Module  # final norm applied before the unembedding
    lm_head: torch.nn.Module  # unembedding projection to vocab logits

    @classmethod
    def resolve(cls, model: PreTrainedModel) -> "ModelAdapter":
        decoder = model.get_decoder()  # GPT-2: .transformer ; Qwen3: .model
        layers = getattr(decoder, "layers", None)
        if layers is None:
            layers = decoder.h  # GPT-2 names the block list ``h``
        final_norm = getattr(decoder, "norm", None)
        if final_norm is None:
            final_norm = decoder.ln_f  # GPT-2 names the final norm ``ln_f``
        return cls(layers=layers, final_norm=final_norm, lm_head=model.get_output_embeddings())

    @property
    def num_layers(self) -> int:
        return len(self.layers)


class XRayModel:
    """Holds the model + tokenizer and offers tokenization and a single-step
    next-token prediction, plus a :class:`ModelAdapter` for the hook engine."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model: PreTrainedModel | None = None
        self.adapter: ModelAdapter | None = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Eager attention is required: SDPA/Flash kernels never materialize the
        # attention probabilities the X-Ray is built around (output_attentions
        # returns None under those backends). float32 on CPU (bf16 is slow /
        # partially unsupported for CPU ops). See M2 notes in CLAUDE.md.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, attn_implementation="eager", dtype=torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
        self.adapter = ModelAdapter.resolve(self.model)

    def _require_loaded(self) -> tuple[object, PreTrainedModel]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load() at startup.")
        return self.tokenizer, self.model

    def _require_adapter(self) -> ModelAdapter:
        if self.adapter is None:
            raise RuntimeError("Model not loaded. Call load() at startup.")
        return self.adapter

    @property
    def num_layers(self) -> int:
        return self._require_adapter().num_layers

    def tokenize(self, prompt: str) -> TokenizedPrompt:
        tokenizer, _ = self._require_loaded()
        token_ids = tokenizer.encode(prompt)
        # Per-token display strings (GPT-2 uses byte-level BPE; leading spaces
        # render as the literal token text after decoding each id individually).
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        return TokenizedPrompt(tokens=tokens, token_ids=token_ids)

    @torch.no_grad()
    def predict_next(self, prompt: str) -> NextTokenPrediction:
        tokenizer, model = self._require_loaded()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        logits = model(input_ids).logits  # (1, seq, vocab)
        last_logits = logits[0, -1]
        probs = torch.softmax(last_logits, dim=-1)
        top_id = int(torch.argmax(probs).item())
        return NextTokenPrediction(
            token=tokenizer.decode([top_id]),
            token_id=top_id,
            prob=float(probs[top_id].item()),
        )


# Singleton, populated by the FastAPI lifespan handler at startup.
xray_model = XRayModel()
