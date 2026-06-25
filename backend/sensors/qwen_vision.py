"""Qwen2-VL-2B-Instruct local vision node — all-seat occupant analysis.

GPU: 4-bit quant (~2.5 GB VRAM).  CPU: float32 fallback (slow, no GPU required).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from bus import bus
from sensors import vision_status as vstat

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=8)

_HUB_ID   = "Qwen/Qwen2-VL-2B-Instruct"
_LOCAL    = Path(__file__).resolve().parent.parent / "models" / "qwen2-vl-2b"
_MODEL_ID = str(_LOCAL) if _LOCAL.exists() else _HUB_ID
_model: Any     = None
_processor: Any = None
_device: str    = "cpu"

NODE_NAME = "qwen2vl"
READY = False
LOAD_ERROR: str | None = None
DEVICE = "cpu"

_SEAT_IDS = ["driver", "front_passenger", "rear_left", "rear_middle", "rear_right"]

_PROMPT = (
    "You are analyzing an in-vehicle cabin camera image. "
    "Identify ALL visible occupants and map each one to the correct seat.\n\n"
    "SEAT LAYOUT (5 seats):\n"
    "  driver          — front row, LEFT side (next to left window/door)\n"
    "  front_passenger — front row, RIGHT side (next to right window/door)\n"
    "  rear_left       — rear row, LEFT side (next to left window/door)\n"
    "  rear_middle     — rear row, CENTER (flanked by rear_left and rear_right, no side window)\n"
    "  rear_right      — rear row, RIGHT side (next to right window/door)\n\n"
    "CLASSIFICATION RULES:\n"
    "  kind='adult'  — teenager or older; full adult body proportions, adult face features.\n"
    "  kind='child'  — clearly young child, noticeably smaller than an adult, under ~10 years old.\n"
    "  kind='infant' — baby or toddler, requires car seat.\n"
    "  kind='pet'    — dog or cat visible.\n"
    "  kind='unknown'— person present but age/type not clearly visible.\n"
    "  IMPORTANT: default to adult unless the person is CLEARLY and OBVIOUSLY a young child.\n"
    "  Do NOT call someone a child just because they appear smaller due to distance or camera angle.\n\n"
    "For each seat output exactly these keys: "
    "occupied (bool), kind (adult|child|infant|pet|unknown), "
    "emotion (calm|happy|stressed|tired|distressed), buckled (bool), "
    "objects (array of strings — loose/unsecured items ON the seat: "
    "box, bag, laptop, bottle, phone, toy, groceries; empty [] if none).\n"
    "Set occupied=false only if the seat is clearly empty or fully out of frame.\n"
    "Reply with ONLY valid JSON, no markdown:\n"
    '{"driver":{...},"front_passenger":{...},"rear_left":{...},"rear_middle":{...},"rear_right":{...}}'
)

_EMPTY_SEAT = {"occupied": False, "kind": "unknown", "emotion": "calm", "buckled": False, "objects": []}


def weights_available() -> bool:
    """True if local weights or a HuggingFace cache snapshot exists."""
    if _LOCAL.is_dir() and any(_LOCAL.iterdir()):
        return True
    hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2-VL-2B-Instruct"
    snaps = hub / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


def _load_model() -> None:
    global _model, _processor, _device, NODE_NAME, READY, LOAD_ERROR
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    use_cuda = torch.cuda.is_available()
    _device = "cuda" if use_cuda else "cpu"
    NODE_NAME = "qwen2vl_gpu" if use_cuda else "qwen2vl_cpu"

    print(f"[QwenVision] Loading {_MODEL_ID} on {_device} …")
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)

    if use_cuda:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            quantization_config=quant,
            device_map="cuda",
            torch_dtype=torch.float16,
        )
    else:
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        _model.to("cpu")

    _model.eval()
    READY = True
    LOAD_ERROR = None
    globals()["DEVICE"] = _device
    vstat.set_ready(node=NODE_NAME, device=_device, backend="qwen")
    print(f"[QwenVision] Model ready ({NODE_NAME})")


def try_load() -> bool:
    """Eager load for startup probe. Returns True if weights are ready."""
    global LOAD_ERROR
    if READY and _model is not None:
        return True
    try:
        _load_model()
        return True
    except Exception as exc:
        LOAD_ERROR = str(exc)
        READY = False
        vstat.set_failed(node="qwen2vl", error=LOAD_ERROR, backend="qwen")
        print(f"[QwenVision] load failed: {exc}")
        return False


def _safe_seat(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return dict(_EMPTY_SEAT)
    kind = str(raw.get("kind", "unknown"))
    # Trust occupied flag directly; only force-true for infant/pet since those
    # are easy to miss in the occupied field but hard to misidentify as a kind.
    occupied = bool(raw.get("occupied", False)) or kind in ("infant", "pet")
    raw_objs = raw.get("objects", [])
    objects  = [str(o) for o in raw_objs if isinstance(o, str)] if isinstance(raw_objs, list) else []
    return {
        "occupied": occupied,
        "kind":     kind,
        "emotion":  str(raw.get("emotion", "calm")),
        "buckled":  bool(raw.get("buckled", False)),
        "objects":  objects,
    }


def _infer(b64: str) -> dict[str, dict]:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    img_bytes = base64.b64decode(b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  _PROMPT},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
    response   = _processor.decode(new_tokens[0], skip_special_tokens=True).strip()

    print(f"[QwenVision] raw response: {response[:120]}")

    # Extract first JSON object from response
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        print("[QwenVision] no JSON found in response")
        return {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as e:
        print(f"[QwenVision] JSON parse error: {e}  raw={m.group()[:200]}")
        return {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

    seats = {sid: _safe_seat(data.get(sid)) for sid in _SEAT_IDS}
    occupied = {sid: s for sid, s in seats.items() if s["occupied"]}
    print(f"[QwenVision] detected: {occupied if occupied else 'nobody'}")
    return seats


_last_seen: dict[str, dict] = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}
_empty_streak: dict[str, int] = {sid: 0 for sid in _SEAT_IDS}
_CLEAR_AFTER = 5   # consecutive empty detections before a seat is cleared


def reset_cache() -> None:
    """Clear the sticky per-seat cache — call when video source changes."""
    for sid in _SEAT_IDS:
        _last_seen[sid] = dict(_EMPTY_SEAT)
        _empty_streak[sid] = 0


async def run() -> None:
    loop = asyncio.get_event_loop()
    if not READY:
        ok = await loop.run_in_executor(None, try_load)
        if not ok:
            print("[QwenVision] run() exiting — model not loaded")
            return

    while True:
        b64 = await frame_queue.get()
        try:
            seats = await loop.run_in_executor(None, _infer, b64)
        except Exception as exc:
            print(f"[QwenVision] inference error: {exc}")
            seats = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

        # Sticky per-seat cache: keep previous value until confident it changed
        for sid in _SEAT_IDS:
            if seats[sid].get("occupied"):
                _last_seen[sid] = seats[sid]
                _empty_streak[sid] = 0
            else:
                _empty_streak[sid] += 1
                if _empty_streak[sid] >= _CLEAR_AFTER:
                    _last_seen[sid] = dict(_EMPTY_SEAT)
                # else: keep last known occupied state

        # Publish all-seat topic
        await bus.publish("vision_all_seats", {"ts": time.time(), **_last_seen})

        # Publish legacy vision_driver topic for backward compat with fusion rules
        drv = seats.get("driver", _EMPTY_SEAT)
        emotion = drv.get("emotion", "calm")
        await bus.publish("vision_driver", {
            "ts":           time.time(),
            "face_detected": drv.get("occupied", False),
            "ear":           0.30,           # not computed by Qwen
            "mouth_ratio":   0.10,
            "drowsy":        emotion == "tired",
            "emotion":       emotion,
        })
        # vision_objects: not provided by this node — keeps last value in fusion
