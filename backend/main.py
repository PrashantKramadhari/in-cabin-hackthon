"""CabinSense backend: starts all sensor nodes + fusion engine, streams the
fused cabin state over websocket, and accepts control commands (scenario
injection, mitigation confirm/dismiss). Serves the React HMI at /."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import fusion
import scenarios
from bus import bus
from sensors import radar, audio, vehicle, vibration

# Try to load the live audio node; fall back to synthetic if deps absent
try:
    from sensors import audio_yamnet as _audio_live
    _USE_LIVE_AUDIO = True
except Exception:
    _USE_LIVE_AUDIO = False

# Try to load the vision node; graceful fallback if mediapipe/ultralytics absent
try:
    from sensors import vision as _vision
    _USE_VISION = True
except Exception:
    _USE_VISION = False

app = FastAPI(title="CabinSense")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

_demo_running = False

# Each step: (scenario_name, hold_seconds, mitigation_ids_to_auto_confirm)
DEMO_SCRIPT = [
    ("idle",              3,  []),
    ("child_crying_rear", 5,  ["comfort_audio"]),
    ("driver_stress",     5,  ["persona_calm"]),
    ("driver_tired",      5,  ["persona_alert"]),
    ("seatbelt_misuse",   4,  []),
    ("pothole_object",    6,  ["secure_object"]),
    ("child_left_behind", 5,  []),
    ("idle",              2,  []),
]


async def _run_demo() -> None:
    global _demo_running
    _demo_running = True
    try:
        for scene, hold, auto_confirm in DEMO_SCRIPT:
            scenarios.apply(scene)
            await asyncio.sleep(2)           # let sensors settle
            for mid in auto_confirm:
                fusion.confirm(mid)
            await asyncio.sleep(hold)
    finally:
        _demo_running = False


@app.on_event("startup")
async def _startup() -> None:
    audio_node = _audio_live.run if _USE_LIVE_AUDIO else audio.run
    nodes = [radar.run, audio_node, vehicle.run, vibration.run, fusion.run]
    if _USE_VISION:
        nodes.append(_vision.run)
    for node in nodes:
        asyncio.create_task(node())


@app.get("/api/caps")
async def caps() -> dict:
    return {"live_audio": _USE_LIVE_AUDIO, "live_vision": _USE_VISION}


@app.get("/api/audio-mode")
async def audio_mode() -> dict:
    return {"live_yamnet": _USE_LIVE_AUDIO}


@app.get("/api/scenarios")
async def list_scenarios() -> dict:
    return {"scenarios": scenarios.SCENARIOS, "current": scenarios.world.scenario}


@app.post("/api/demo/play")
async def demo_play() -> dict:
    global _demo_running
    if _demo_running:
        return {"status": "already_running"}
    asyncio.create_task(_run_demo())
    return {"status": "started", "steps": len(DEMO_SCRIPT)}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    q = bus.subscribe("fused")

    async def pump() -> None:
        while True:
            await sock.send_text(json.dumps(await q.get()))

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            cmd = msg.get("cmd")
            if cmd == "scenario":
                scenarios.apply(msg["name"])
            elif cmd == "confirm":
                fusion.confirm(msg["id"])
            elif cmd == "dismiss":
                fusion.dismiss(msg["id"])
            elif cmd == "audio_chunk" and _USE_LIVE_AUDIO:
                samples = msg.get("data", [])
                if samples and not _audio_live.chunk_queue.full():
                    await _audio_live.chunk_queue.put(samples)
            elif cmd == "vision_frame" and _USE_VISION:
                b64 = msg.get("data", "")
                if b64 and not _vision.frame_queue.full():
                    await _vision.frame_queue.put(b64)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/deck")
async def deck() -> FileResponse:
    return FileResponse(FRONTEND / "deck.html")


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
