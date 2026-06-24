"""Qwen2-VL-2B-Instruct local vision node — all-seat occupant analysis.

Replaces vision.py when available.
Model:   Qwen/Qwen2-VL-2B-Instruct  (4-bit quant → ~2.5 GB VRAM)
Latency: ~100–150 ms on RTX 3080

Publishes to two bus topics:
  "vision_driver"    — legacy compat: {face_detected, ear, drowsy, emotion}
  "vision_all_seats" — {driver:{…}, front_passenger:{…}, rear_left:{…}, rear_right:{…}}

Each seat dict:
  occupied  bool
  kind      "adult"|"child"|"infant"|"unknown"
  emotion   "calm"|"happy"|"stressed"|"tired"|"distressed"
  buckled   bool
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import time
from typing import Any

import torch

from bus import bus

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)

_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_model: Any     = None
_processor: Any = None

_SEAT_IDS = ["driver", "front_passenger", "rear_left", "rear_right"]

_PROMPT = (
    "You are analyzing an in-vehicle cabin camera image. "
    "Identify occupants in all 4 seat positions: driver (front-left), "
    "front_passenger (front-right), rear_left, rear_right. "
    "For each seat return a JSON object with exactly these keys: "
    "occupied (bool), kind (adult|child|infant|unknown), "
    "emotion (calm|happy|stressed|tired|distressed), buckled (bool). "
    "If a seat is not clearly visible or empty set occupied=false. "
    "Reply with ONLY a valid JSON object — no markdown, no explanation:\n"
    '{"driver":{...},"front_passenger":{...},"rear_left":{...},"rear_right":{...}}'
)

_EMPTY_SEAT = {"occupied": False, "kind": "unknown", "emotion": "calm", "buckled": False}


def _load_model() -> None:
    global _model, _processor
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    print(f"[QwenVision] Loading {_MODEL_ID} (4-bit) …")
    _processor = AutoProcessor.from_pretrained(_MODEL_ID)
    _model = Qwen2VLForConditionalGeneration.from_pretrained(
        _MODEL_ID,
        quantization_config=quant,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    _model.eval()
    print("[QwenVision] Model ready")


def _safe_seat(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return dict(_EMPTY_SEAT)
    return {
        "occupied": bool(raw.get("occupied", False)),
        "kind":     str(raw.get("kind", "unknown")),
        "emotion":  str(raw.get("emotion", "calm")),
        "buckled":  bool(raw.get("buckled", False)),
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
    ).to("cuda")

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

    # Extract first JSON object from response
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

    return {sid: _safe_seat(data.get(sid)) for sid in _SEAT_IDS}


async def run() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)

    while True:
        b64 = await frame_queue.get()
        try:
            seats = await loop.run_in_executor(None, _infer, b64)
        except Exception as exc:
            print(f"[QwenVision] inference error: {exc}")
            seats = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}

        # Publish all-seat topic
        await bus.publish("vision_all_seats", {"ts": time.time(), **seats})

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
