"""FastAPI entrypoint for LLM X-Ray.

Loads GPT-2 small once at startup and exposes:
- ``/api/generate`` — tokens + next-token prediction (M1, sanity REST endpoint)
- ``/ws/xray`` — streams the full X-Ray (tokens → per-layer → output) over a
  WebSocket (M3), driven by the hook engine.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import serializer
from app.model import available_devices, xray_model
from app.schemas import GenerateRequest, GenerateResponse, NextToken
from app.xray_engine import MAX_NEW_TOKENS_LIMIT, XRayHookEngine

logger = logging.getLogger("llmxray")

# Hard ceiling on a single forward pass (the prompt pass, or one generated
# token). A token is well under a second on CPU; anything near this means
# something is wrong, so we cut the run rather than hang the socket. The overall
# generation length/time is bounded inside the engine (MAX_NEW_TOKENS /
# GEN_TIME_BUDGET_S), not here.
RUN_TIMEOUT_S = 15.0
# Bound prompt length defensively (mirrors the REST schema's max_length).
MAX_PROMPT_CHARS = 512

# One engine instance, created at startup once the model is loaded.
engine: XRayHookEngine | None = None

# Serializes device switches + generation runs across all sockets. A device
# switch moves the *shared* model in place (XRayModel.set_device), so two runs
# racing on different devices (or a switch landing mid-generation) would hand
# the hook engine tensors on one device and params on another. One global lock
# is cheap here — this is a single-model local tool, not a multi-tenant server.
_inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once on startup; reused across all requests.
    xray_model.load()
    global engine
    engine = XRayHookEngine(xray_model)
    yield


app = FastAPI(title="LLM X-Ray", version="0.1.0", lifespan=lifespan)

# Frontend origins allowed to call the API/WS. Defaults to the local dev
# server; override in deployment via CORS_ALLOW_ORIGINS (comma-separated).
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "LLM X-Ray", "status": "ok"}


@app.get("/api/health")
def health():
    return {"status": "healthy", "model_loaded": xray_model.model is not None}


@app.get("/api/devices")
def devices():
    """Devices this machine can run inference on, plus the one currently active
    (frontend uses this to render the CPU/CUDA/MPS pill and its initial value)."""
    return {"available": available_devices(), "current": str(xray_model.device)}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    tokenized = xray_model.tokenize(req.prompt)
    pred = xray_model.predict_next(req.prompt)
    return GenerateResponse(
        prompt=req.prompt,
        tokens=tokenized.tokens,
        token_ids=tokenized.token_ids,
        next_token=NextToken(token=pred.token, token_id=pred.token_id, prob=pred.prob),
    )


def _parse_prompt(raw: str) -> tuple[str, bool, str | None, int | None]:
    """Accept a bare prompt string or
    ``{"prompt": "...", "thinking": bool, "device": "cpu"|"cuda"|"mps",
    "max_tokens": int}`` JSON.

    Returns ``(prompt, thinking, device, max_tokens)``; ``device`` is ``None``
    when the client didn't send one (leave the current device as-is), and
    ``max_tokens`` is ``None`` when omitted (use the engine's default cap).
    Raises ``ValueError`` with a client-facing message on anything unusable.
    """
    prompt = raw
    thinking = True
    device = None
    max_tokens = None
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            payload = orjson.loads(stripped)
        except orjson.JSONDecodeError as exc:
            raise ValueError("Malformed JSON message.") from exc
        if not isinstance(payload, dict) or "prompt" not in payload:
            raise ValueError("JSON message must contain a 'prompt' field.")
        prompt = payload["prompt"]
        thinking = bool(payload.get("thinking", True))
        raw_device = payload.get("device")
        if raw_device is not None:
            if not isinstance(raw_device, str):
                raise ValueError("'device' must be a string.")
            device = raw_device
        raw_max = payload.get("max_tokens")
        if raw_max is not None:
            if not isinstance(raw_max, int) or isinstance(raw_max, bool):
                raise ValueError("'max_tokens' must be an integer.")
            if not 1 <= raw_max <= MAX_NEW_TOKENS_LIMIT:
                raise ValueError(
                    f"'max_tokens' must be between 1 and {MAX_NEW_TOKENS_LIMIT}."
                )
            max_tokens = raw_max
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds {MAX_PROMPT_CHARS} characters.")
    return prompt, thinking, device, max_tokens


@app.websocket("/ws/xray")
async def ws_xray(websocket: WebSocket):
    """Stream a generation X-Ray per prompt: meta → tokens → prompt_attention →
    step×N → done.

    Runs are serial per socket: we ``await`` each run before reading the next
    message, so a client can't trigger overlapping inferences on one connection.
    Each forward pass (prompt + every generated token) is dispatched to a worker
    thread so the CPU-bound work never blocks the event loop, and each token is
    streamed as it lands so the client *watches* the model generate.
    """
    assert engine is not None, "engine not initialized (lifespan did not run)"
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                prompt, thinking, device, max_tokens = _parse_prompt(raw)
            except ValueError as exc:
                await websocket.send_text(serializer.error_message(str(exc)))
                continue

            await _run_and_stream(websocket, prompt, thinking, device, max_tokens)
    except WebSocketDisconnect:
        logger.info("ws/xray client disconnected")


async def _step(coro_fn):
    """Run one blocking forward pass on a worker thread with a per-pass timeout."""
    return await asyncio.wait_for(asyncio.to_thread(coro_fn), timeout=RUN_TIMEOUT_S)


async def _run_and_stream(
    websocket: WebSocket,
    prompt: str,
    thinking: bool,
    device: str | None,
    max_tokens: int | None,
) -> None:
    """Drive one greedy generation and stream its messages token-by-token."""
    model_label = xray_model.model_name.split("/")[-1]

    async with _inference_lock:
        if device is not None:
            await asyncio.to_thread(xray_model.set_device, device)
        await _run_locked(websocket, prompt, thinking, model_label, max_tokens)


async def _run_locked(
    websocket: WebSocket,
    prompt: str,
    thinking: bool,
    model_label: str,
    max_tokens: int | None,
) -> None:
    session = engine.generate(prompt, thinking=thinking, max_new_tokens=max_tokens)

    try:
        capture, first = await _step(session.prime)
        await websocket.send_text(
            serializer.meta_message(
                capture, model_label, str(xray_model.device), session.max_new_tokens
            )
        )
        await websocket.send_text(
            serializer.tokens_message(capture.tokens, capture.token_ids)
        )
        await websocket.send_text(serializer.prompt_attention_message(capture))
        if first is not None:
            await websocket.send_text(serializer.step_message(first))

        while session.stop_reason is None:
            step = await _step(session.advance)
            if step is None:
                break
            await websocket.send_text(serializer.step_message(step))

        await websocket.send_text(serializer.done_message(session.done()))
    except asyncio.TimeoutError:
        await websocket.send_text(serializer.error_message("Inference timed out."))
    except WebSocketDisconnect:
        raise  # handled by the caller; stop streaming to a gone client
    except Exception:  # noqa: BLE001 — surface a clean error, log the detail.
        logger.exception("ws/xray generation failed")
        await websocket.send_text(
            serializer.error_message("Internal error during generation.")
        )
