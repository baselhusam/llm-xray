# LLM X-Ray

**Type a prompt. Watch an AI think. Share the X-ray.**

LLM X-Ray is an interactive web tool that runs your prompt through a small LLM
(Qwen3-1.7B) with **greedy autoregressive generation** on CPU and streams what
happens inside, token by token — attention patterns, per-layer logit-lens
predictions, and token probabilities — as live, animated graphics. The model
answers your question (with an optional `<think>…</think>` reasoning trace
first), so you watch it actually think, not just autocomplete. Think MRI scan,
but for a language model. Every run produces a shareable card.

> No pre-baked demos: the visuals are driven by **your** prompt, captured from a live
> PyTorch forward pass and streamed to the browser token by token.

---

## What you see

| View | What it shows |
|---|---|
| **Architecture flow** (hero) | The model's actual decoder-only topology — token embedding → 28 decoder blocks → final RMSNorm + unembedding — lit up by the live pass. Click a block to expand its internal circuit (RMSNorm → GQA attention → +residual → RMSNorm → SwiGLU MLP → +residual) and see what it attended to. |
| **Generation timeline + confidence curve** | The generated tokens as a running transcript — reasoning trace (dimmed) separate from the answer — each underlined by how confident the model was. Click any token to rewind every other view to that decision. |
| **Logit-lens trajectory** | The model's best next-token guess read off *every* layer for the selected token — watch the prediction sharpen from noise in early layers to a locked-in answer at the top. |
| **Logit-lens heatmap** | Layers × candidate tokens, cell brightness = probability — watch probability mass concentrate with depth. |
| **Next-token candidates** | The real distribution behind the chosen token: what it picked, how sure it was, and the runner-up it almost said instead. |
| **Attention attribution** | Which earlier tokens actually drove a decision, via attention rollout (honest, cross-layer) or raw per-layer attention. |
| **Shareable card** | One-click PNG of the current Q&A (branded) + dynamic OG images so shared links unfurl with the prediction. |

---

## Architecture

Two apps, streaming between them over a WebSocket:

```
┌─────────────────────────────┐         WebSocket          ┌──────────────────────────────┐
│  frontend/  (Next.js 16)     │  ── prompt + thinking ───▶ │  backend/  (FastAPI)          │
│                              │                            │                               │
│  useXRay hook ─▶ XRayApp     │  ◀── meta ─────────────────│  XRayModel (Qwen3-1.7B, CPU)  │
│       │                      │  ◀── tokens ────────────── │  GenerationSession            │
│       ▼                      │  ◀── prompt_attention ──── │   ├ chat-template + <think>    │
│  D3 views (architecture flow │  ◀── step × N (per token) ─│   ├ KV-cache autoregressive    │
│  / trajectory / heatmap /    │  ◀── done ──────────────── │   │   generation                │
│  timeline / attention)       │                            │   ├ forward hooks: hidden state│
│  + ShareCard                 │                            │   ├ output_attentions (mean     │
│                              │                            │   │   over heads, streamed)     │
│                              │                            │   └ batched logit lens + rollout│
└─────────────────────────────┘                            └──────────────────────────────┘
        Tailwind v4 · Framer Motion · shadcn/ui                 PyTorch · HF Transformers · orjson
```

**Flow:** prompt → WS → the prompt is wrapped in the model's chat template so it
*answers* → **greedy autoregressive generation** (a prompt pass primes a KV cache,
then one token per step) with hooks capturing per-layer hidden states + attention
at each step → batched logit lens (all layers' last-position vectors through
`final_norm` + `lm_head` in one matmul) → stream JSON per generated token →
React renders incrementally as the model writes. Attention rollout (the "what
drove this token" attribution) is computed **client-side**, lazily, for the
selected step only.

The core is the **hook engine** (`backend/app/xray_engine.py`): it registers forward
hooks on every decoder block to capture the residual stream and reads attention weights
from the forward pass's `output_attentions`, then projects each layer's hidden state to a
top-k prediction. Architecture-specific module paths are resolved by `ModelAdapter`
(`model.py`), so the engine is **model-agnostic** — swap `MODEL_NAME` (e.g. back to
`gpt2` or `Qwen/Qwen3-0.6B-Base`) with no engine changes (a non-chat model just falls
back to raw continuation). Tensors are detached to CPU numpy and serialized with
**orjson**.

### WebSocket protocol

Endpoint `/ws/xray`. Client sends `{"prompt": "...", "thinking": bool}` (or a bare
string; thinking defaults on). The server streams, in order:

1. `{"type": "meta", "data": {"num_layers", "num_heads", "thinking", "model_label"}}`
2. `{"type": "tokens", "data": {"tokens", "token_ids"}}` — the chat-templated prompt tokens
3. `{"type": "prompt_attention", "data": {"attention"}}` — `(layers, P, P)`, mean over heads
4. one `{"type": "step", "data": {"step", "token", "token_id", "prob", "entropy", "phase", "trajectory", "attention_row"}}` **per generated token**
5. `{"type": "done", "data": {"generated_text", "num_steps", "stop_reason"}}`

An out-of-band `{"type": "error", "data": {"message"}}` can replace the stream at any point.

---

## Tech stack

- **Backend** — FastAPI · PyTorch · HuggingFace Transformers (Qwen3-1.7B, 2.03B params, 28 layers, 16 query heads, CPU) · orjson · Python 3.11
- **Frontend** — Next.js 16 (App Router, TS, `src/`) · Tailwind v4 · Framer Motion · shadcn/ui · D3.js
- **Sharing** — html-to-image (client PNG) · `next/og` (server OG cards)

Qwen3-1.7B on CPU is a deliberate choice — it runs free on a small box, understands
and answers real questions (with an optional reasoning trace), and per-token latency
stays around ~0.13s (a full thinking run is ~15–25s). The engine is model-agnostic,
so swapping `MODEL_NAME` to something smaller (e.g. `Qwen/Qwen3-0.6B-Base`) or back to
`gpt2` works with no engine changes.

---

## Run it locally

You'll need **Python 3.11** and **Node 18+**. Run the two apps in separate terminals.

### 1. Backend (port 8000)

```bash
cd backend
.venv/bin/pip install -r requirements.txt   # first run only (downloads torch + Qwen3-1.7B)
.venv/bin/uvicorn app.main:app --reload      # add --port N if 8000 is taken
```

Health check: `curl http://127.0.0.1:8000/api/health` → `{"status":"healthy","model_loaded":true}`

> The Qwen3-1.7B weights (~7 GB in float32) download from HuggingFace on first model
> load and are cached afterward, so the first startup is slower than the rest.

### 2. Frontend (port 3000)

```bash
cd frontend
npm install        # first run only
npm run dev        # http://localhost:3000
```

Open <http://localhost:3000>, type a prompt (or click an example), and watch.

### Configuration

Both default to localhost, so no env file is needed for local dev. Override per-env:

| Variable | App | Default | Purpose |
|---|---|---|---|
| `NEXT_PUBLIC_XRAY_WS_URL` | frontend | `ws://127.0.0.1:8000/ws/xray` | backend WebSocket endpoint |
| `NEXT_PUBLIC_SITE_URL` | frontend | `http://localhost:3000` | absolute URLs for share links + OG images |
| `CORS_ALLOW_ORIGINS` | backend | `http://localhost:3000,http://127.0.0.1:3000` | comma-separated origins allowed to hit the API/WS |

If you run the backend on a non-default port, set `NEXT_PUBLIC_XRAY_WS_URL` to match.

### Useful commands

```bash
# frontend
npm run build        # production build
npm run lint         # eslint

# backend
.venv/bin/uvicorn app.main:app --reload
```

---

## Project layout

```
backend/
  app/
    main.py          FastAPI app + /ws/xray WebSocket + /api health/generate
    model.py         XRayModel + ModelAdapter — loads the model once (eager attn)
    xray_engine.py   XRayHookEngine + GenerationSession — KV-cache generation + logit lens (the core)
    serializer.py    orjson tensor serialization for the WS protocol
    schemas.py       REST request/response shapes
frontend/
  src/
    app/             Next.js routes (page, layout, api/og OG-card route)
    components/      XRayApp shell + D3/instrument views (architecture flow, trajectory,
                      heatmap, timeline, confidence curve, attention, distribution) + share
    hooks/           useXRay (WebSocket), useMediaQuery
    lib/             attention/rollout math, tokens, share URLs, protocol types
```

---

## Screenshots & demo

> _Add a screenshot of the architecture flow view and a short demo GIF here._
>
> - `docs/architecture-flow.png` — the live forward pass hero view
> - `docs/demo.gif` — a full run, prompt → streamed generation → share card

---

## Why Qwen3-1.7B?

It's small enough to run free on CPU (~0.13s/token), but modern and instruction-tuned
enough to actually *answer* — including an optional `<think>…</think>` reasoning trace
— rather than just autocomplete, and its logit lens sharpens into coherent words across
28 layers (where GPT-2 small often stayed noisy). The engine is model-agnostic (a
`ModelAdapter` resolves architecture-specific module paths; chat template + `<think>`
markers are resolved at runtime with raw-encode/no-thinking fallbacks), so `MODEL_NAME`
can swap to a smaller Qwen3 variant, `gpt2`, or another causal LM without engine
changes. Larger models and side-by-side multi-model comparison are out of scope for v1
— see `CLAUDE.md` for the roadmap and scope guardrails.
