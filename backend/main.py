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

# CLAP zero-shot takes priority; then BabyNet → AST → synthetic
try:
    from sensors import audio_clap as _audio_live  # type: ignore[assignment]
    _USE_LIVE_AUDIO = True
    _AUDIO_NODE_NAME = "clap_zs"
except Exception as _e:
    print(f"[main] CLAP zero-shot unavailable ({_e}), trying BabyNet …")
    try:
        from sensors import baby_net as _audio_live  # type: ignore[assignment]
        _USE_LIVE_AUDIO = True
        _AUDIO_NODE_NAME = "baby_net"
    except Exception as _e2:
        print(f"[main] BabyNet unavailable ({_e2}), trying AST …")
        try:
            from sensors import audio_yamnet as _audio_live  # type: ignore[assignment]
            _USE_LIVE_AUDIO = True
            _AUDIO_NODE_NAME = "ast"
        except Exception:
            _USE_LIVE_AUDIO = False
            _AUDIO_NODE_NAME = "synthetic"

# Vision: probe-load at startup (Qwen CPU/GPU → MediaPipe fallback)
_vision = None
_USE_VISION = False
_USE_QWEN_VISION = False
_VISION_NODE_NAME = "none"


def _init_vision_sync() -> None:
    """Try Qwen first, then MediaPipe; sets module globals from probe result."""
    global _vision, _USE_VISION, _USE_QWEN_VISION, _VISION_NODE_NAME
    import torch
    from sensors import vision_status as vstat

    try:
        from sensors import qwen_vision as qv
        can_try_qwen = torch.cuda.is_available() or qv.weights_available()
        if not can_try_qwen:
            print(
                "[main] Qwen skipped on CPU — no weights at backend/models/qwen2-vl-2b "
                "and no HF cache (use MediaPipe or download weights)"
            )
        elif qv.try_load():
            _vision = qv
            _USE_VISION = True
            _USE_QWEN_VISION = True
            _VISION_NODE_NAME = qv.NODE_NAME
            print(f"[main] Vision ready: {qv.NODE_NAME} on {qv.DEVICE}")
            return
        else:
            print(f"[main] QwenVision not loaded: {qv.LOAD_ERROR}")
    except Exception as exc:
        print(f"[main] QwenVision unavailable ({exc})")

    try:
        from sensors import vision as mpv
        if mpv.try_load():
            _vision = mpv
            _USE_VISION = True
            _USE_QWEN_VISION = False
            _VISION_NODE_NAME = mpv.NODE_NAME
            print(f"[main] Vision ready: {mpv.NODE_NAME} (fallback)")
            return
        print(f"[main] MediaPipe vision not loaded: {mpv.LOAD_ERROR}")
    except Exception as exc:
        print(f"[main] MediaPipe vision unavailable ({exc})")

    vstat.set_failed(node="none", error="no vision backend available", backend="none")
    _vision = None
    _USE_VISION = False
    _USE_QWEN_VISION = False
    _VISION_NODE_NAME = "none"
    print("[main] No vision backend available")

app = FastAPI(title="CabinSense")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
VIDEO_DIR = Path(__file__).resolve().parent / "video"

# serve bundled JS libs before any routes so /static/* is always available
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

# serve video files from backend/video/
if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

_demo_stop = False

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


async def _interruptible_sleep(seconds: float) -> bool:
    """Sleep in short ticks so _demo_stop is checked every 0.3 s.
    Returns True if the sleep was interrupted."""
    tick = 0.3
    elapsed = 0.0
    while elapsed < seconds:
        if _demo_stop:
            return True
        await asyncio.sleep(min(tick, seconds - elapsed))
        elapsed += tick
    return _demo_stop


async def _run_demo() -> None:
    global _demo_stop
    _demo_stop = False
    scenarios.world.demo_running = True
    try:
        for scene, hold, auto_confirm in DEMO_SCRIPT:
            if _demo_stop:
                break
            scenarios.apply(scene)
            if await _interruptible_sleep(2.0):   # settle; exits within 0.3 s of stop
                break
            for mid in auto_confirm:
                fusion.confirm(mid)
            if await _interruptible_sleep(hold):   # hold; exits within 0.3 s of stop
                break
    finally:
        scenarios.world.demo_running = False
        _demo_stop = False


@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_vision_sync)
    audio_node = _audio_live.run if _USE_LIVE_AUDIO else audio.run
    nodes = [radar.run, audio_node, vehicle.run, vibration.run, fusion.run]
    if _USE_VISION and _vision is not None:
        nodes.append(_vision.run)
    for node in nodes:
        asyncio.create_task(node())


_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


@app.get("/api/videos")
async def list_videos() -> dict:
    """Return sorted list of video filenames available in backend/video/."""
    if not VIDEO_DIR.exists():
        return {"videos": []}
    files = sorted(
        f.name for f in VIDEO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in _VIDEO_EXTS
    )
    return {"videos": files}


@app.get("/api/vision-thresholds")
async def get_vision_thresholds() -> dict:
    from sensors.vision_runtime import snapshot
    return {"thresholds": snapshot()}


@app.put("/api/vision-thresholds")
async def put_vision_thresholds(body: dict) -> dict:
    from sensors.vision_runtime import update
    return {"thresholds": update(body.get("thresholds", body))}


@app.get("/api/caps")
async def caps() -> dict:
    from config import audio as acfg
    from sensors import vision_status as vstat
    out = {
        "live_audio":    _USE_LIVE_AUDIO,
        "audio_node":    _AUDIO_NODE_NAME,
        "audio_chunk_samples": acfg.chunk_samples,
        "audio_chunk_sec": round(acfg.chunk_samples / 16_000, 2),
    }
    out.update(vstat.snapshot())
    from sensors.vision_runtime import snapshot as vision_thresh_snapshot
    out["vision_thresholds"] = vision_thresh_snapshot()
    # legacy keys — now driven by actual load state
    if not out.get("vision_ready"):
        out["qwen_vision"] = False
    return out


@app.get("/api/audio-mode")
async def audio_mode() -> dict:
    return {"live_yamnet": _USE_LIVE_AUDIO}


_UI_LAYOUT_PATH = Path(__file__).resolve().parent / "data" / "ui_layout.json"


@app.get("/api/ui-layout")
async def get_ui_layout() -> dict:
    """Return saved HMI panel layout (positions, sizes, visibility)."""
    if _UI_LAYOUT_PATH.is_file():
        try:
            return json.loads(_UI_LAYOUT_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


@app.put("/api/ui-layout")
async def put_ui_layout(body: dict) -> dict:
    """Persist HMI panel layout across server restarts and browsers."""
    _UI_LAYOUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _UI_LAYOUT_PATH.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"status": "saved"}


@app.get("/api/scenarios")
async def list_scenarios() -> dict:
    return {"scenarios": scenarios.SCENARIOS, "current": scenarios.world.scenario}


@app.post("/api/demo/play")
async def demo_play() -> dict:
    if scenarios.world.demo_running:
        return {"status": "already_running"}
    asyncio.create_task(_run_demo())
    return {"status": "started", "steps": len(DEMO_SCRIPT)}


@app.post("/api/demo/stop")
async def demo_stop_endpoint() -> dict:
    global _demo_stop
    _demo_stop = True
    return {"status": "stopping"}


def _sync_audio_from_seats() -> None:
    """Derive world audio state from the loudest per-seat audio event."""
    priority = {"crying": 5, "barking": 4, "shouting": 3, "talking": 2, "happy": 1, "none": 0}
    best_label = "none"
    best_conf = 0.0
    best_pri = 0
    for occ in scenarios.world.seats.values():
        if not occ.occupied:
            continue
        pri = priority.get(occ.audio_event, 0)
        if pri > best_pri:
            best_pri = pri
            best_label = occ.audio_event
            # Minimum 0.70 so configured events always clear fusion threshold
            best_conf = round(min(1.0, max(0.70, 0.80 + occ.distress * 0.15)), 2)
    scenarios.world.audio_label = best_label
    scenarios.world.audio_conf = best_conf


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    q = bus.subscribe("fused")

    async def pump() -> None:
        while True:
            data = await q.get()
            try:
                await sock.send_text(json.dumps(data))
            except (TypeError, ValueError) as exc:
                # Non-serialisable value slipped through — log and skip
                import traceback
                print(f"[pump] JSON error: {exc}\n{traceback.format_exc()}")

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await sock.receive_text())
            cmd = msg.get("cmd")
            if cmd == "scenario":
                _demo_stop = True          # cancel auto-play if running
                scenarios.apply(msg["name"])
            elif cmd == "configure_seat":
                seat_id = msg.get("seat")
                if seat_id in scenarios.world.seats:
                    occ = scenarios.world.seats[seat_id]
                    ov = scenarios.world.seat_overrides.setdefault(seat_id, {})
                    if "occupied" in msg:
                        occ.occupied = bool(msg["occupied"])
                        ov["occupied"] = occ.occupied
                        if not occ.occupied:
                            occ.audio_event = "none"
                            scenarios.world.seat_overrides.pop(seat_id, None)
                    if "kind" in msg:
                        if occ.kind != msg["kind"]:
                            occ.audio_event = "none"
                            ov.pop("audio_event", None)
                        occ.kind = msg["kind"]
                        ov["kind"] = occ.kind
                    if "buckled" in msg:
                        occ.buckled = bool(msg["buckled"])
                        ov["buckled"] = occ.buckled
                    if "distress" in msg:
                        occ.distress = float(msg["distress"])
                        ov["distress"] = occ.distress
                    if "audio_event" in msg:
                        occ.audio_event = msg["audio_event"]
                        ov["audio_event"] = occ.audio_event
                    if "heart_rate_bpm" in msg:
                        v = msg["heart_rate_bpm"]
                        occ.heart_rate_bpm = float(v) if v is not None else None
                        ov["heart_rate_bpm"] = occ.heart_rate_bpm
                    if "respiration_rpm" in msg:
                        v = msg["respiration_rpm"]
                        occ.respiration_rpm = float(v) if v is not None else None
                        ov["respiration_rpm"] = occ.respiration_rpm
                    if "emotion" in msg:
                        occ.emotion = msg["emotion"]
                        ov["emotion"] = occ.emotion
                    if seat_id == "driver":
                        scenarios.world.driver_emotion = occ.emotion
                        if occ.heart_rate_bpm is not None:
                            scenarios.world.driver_hr = occ.heart_rate_bpm
                        if occ.respiration_rpm is not None:
                            scenarios.world.driver_resp = occ.respiration_rpm
                    _sync_audio_from_seats()
            elif cmd == "configure_vehicle":
                w = scenarios.world
                if "speed_kmh" in msg:
                    w.speed_kmh = float(msg["speed_kmh"])
                if "visibility" in msg:
                    w.visibility = msg["visibility"]
                if "pothole_ahead_m" in msg:
                    v = msg["pothole_ahead_m"]
                    w.pothole_ahead_m = float(v) if v is not None else None
            elif cmd == "configure_vibration":
                w = scenarios.world
                if "override" in msg:
                    w.vib_override = bool(msg["override"])
                if "road_quality" in msg:
                    w.vib_road_quality = msg["road_quality"]
                if "rms" in msg:
                    w.vib_rms = float(msg["rms"])
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
            elif cmd == "reset_vision":
                if _vision is not None and hasattr(_vision, "reset_cache"):
                    _vision.reset_cache()
                fusion.reset_vision()
                scenarios.world.seat_overrides.clear()
                scenarios.apply("idle")
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
