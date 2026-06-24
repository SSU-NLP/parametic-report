import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from parametic_studio.catalog import available_models

app = FastAPI()

SESSION = None        # default/back-compat single kernel (tests set this)
SESSIONS: dict = {}   # model_id -> ModelSession (multi-model registry)
_locks: dict = {}     # model_id -> asyncio.Lock (serialize concurrent loads of the same model)


async def _ensure(model_id):
    if model_id in SESSIONS:
        return SESSIONS[model_id]
    lock = _locks.setdefault(model_id, asyncio.Lock())
    async with lock:
        if model_id not in SESSIONS:
            from parametic_studio.kernel.model_session import ModelSession
            # off the event loop: a blocking from_pretrained here would freeze all other models.
            SESSIONS[model_id] = await asyncio.to_thread(ModelSession.from_pretrained, model_id)
    return SESSIONS[model_id]


def _loaded(msg):
    mid = msg.get("model")
    return SESSIONS.get(mid, SESSION) if mid is not None else SESSION


async def _run_generation(websocket, msg):
    model = msg.get("model")
    session = await _ensure(model) if model is not None else SESSION
    max_tokens = msg.get("max_tokens", 256)
    probes = msg.get("probes", ["attention"])
    stopped = asyncio.Event()

    async def listen_stop():
        try:
            while True:
                m = await websocket.receive_json()
                if m.get("type") == "stop":
                    session.stop()
                    stopped.set()
                    return
        except WebSocketDisconnect:
            return

    listener = asyncio.create_task(listen_stop())
    count = 0
    for ev in session.generate_text(msg["prompt"], max_tokens, probes=probes):
        await websocket.send_json(
            {"type": "token", "model": model, "step": ev["step"], "token_id": ev["token_id"], "text": ev["text"]}
        )
        for row in ev.get("attn_rows", []):  # prefill emits many rows; decode emits one
            await websocket.send_json({"type": "attention", "model": model, "step": ev["step"],
                                       "shape": list(row.shape), "data": row.cpu().tolist()})
        if "act" in ev:
            await websocket.send_json({"type": "activation", "model": model, "step": ev["step"],
                                       "shape": list(ev["act"].shape), "data": ev["act"].cpu().tolist()})
        if "logit" in ev:
            await websocket.send_json({"type": "logitlens", "model": model, "step": ev["step"], "layers": ev["logit"]})
        count += 1

    listener.cancel()
    try:
        await listener
    except (asyncio.CancelledError, WebSocketDisconnect):
        pass

    reason = "stopped" if stopped.is_set() else "eos" if count < max_tokens else "max_tokens"
    await websocket.send_json({"type": "done", "model": model, "reason": reason})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            msg = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        t = msg.get("type")
        if t == "generate":
            await _run_generation(websocket, msg)
        elif t == "catalog":
            await websocket.send_json({"type": "catalog", "models": available_models()})
        elif t == "open":
            await websocket.send_json({"type": "loading", "model": msg["model"]})
            await _ensure(msg["model"])  # non-blocking (thread); other models keep streaming
            await websocket.send_json({"type": "opened", "model": msg["model"]})
        elif t == "close":
            s = SESSIONS.pop(msg["model"], None)
            _locks.pop(msg["model"], None)
            if s is not None and hasattr(s, "close"):
                s.close()  # remove hooks → frees the model
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            await websocket.send_json({"type": "closed", "model": msg["model"]})
        elif t == "drilldown":
            ph = _loaded(msg).drilldown(msg["layer"])
            await websocket.send_json({"type": "perhead", "model": msg.get("model"), "layer": msg["layer"],
                                       "shape": list(ph.shape), "data": ph.cpu().tolist()})
        elif t == "spot":
            spot = _loaded(msg).compute_spot(msg["examples"])
            await websocket.send_json({"type": "spotmap", "model": msg.get("model"), **spot})


def serve(model_id, host="127.0.0.1", port=8000):
    import uvicorn

    from parametic_studio.kernel.model_session import ModelSession

    global SESSION
    SESSION = ModelSession.from_pretrained(model_id)
    SESSIONS[model_id] = SESSION  # default model is addressable by id too
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    serve(os.environ.get("PARAMETIC_STUDIO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
