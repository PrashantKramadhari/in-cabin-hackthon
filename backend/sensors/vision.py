"""Vision node — MediaPipe Tasks FaceLandmarker + YOLOv8n.

Browser streams webcam JPEG frames over websocket as:
  {cmd: "vision_frame", data: "<base64-jpeg>"}

This node:
  1. Decodes the JPEG with OpenCV
  2. Runs MediaPipe FaceLandmarker (Tasks API, v0.10+) → EAR-based drowsiness
     + heuristic emotion from mouth/eye geometry
  3. Runs YOLOv8n → detects loose objects on seats (bottle, bag, laptop, etc.)

Publishes to two bus topics consumed by fusion.py:
  "vision_driver"  — {face_detected, ear, drowsy, emotion, mouth_ratio}
  "vision_objects" — {detections: [{label, confidence, box}]}
"""
from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bus import bus

_landmarker: Any = None
_yolo: Any = None

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)

EAR_THRESH = 0.20
MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task")

# MediaPipe 478-landmark indices
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
_MOUTH_TOP = 13
_MOUTH_BOT = 14
_MOUTH_L   = 78
_MOUTH_R   = 308


def _load_models() -> None:
    global _landmarker, _yolo
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as mpv
    from ultralytics import YOLO

    opts = mpv.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = mpv.FaceLandmarker.create_from_options(opts)
    _yolo = YOLO("yolov8n.pt")


def _ear(lm, indices, w, h) -> float:
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    ho = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * ho + 1e-6))


def _run_face(bgr: np.ndarray) -> dict:
    import mediapipe as mp
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker.detect(mp_img)

    if not result.face_landmarks:
        return {"face_detected": False}

    lm = result.face_landmarks[0]

    left_ear  = _ear(lm, _LEFT_EYE,  w, h)
    right_ear = _ear(lm, _RIGHT_EYE, w, h)
    avg_ear   = round((left_ear + right_ear) / 2, 3)

    top   = np.array([lm[_MOUTH_TOP].x * w, lm[_MOUTH_TOP].y * h])
    bot   = np.array([lm[_MOUTH_BOT].x * w, lm[_MOUTH_BOT].y * h])
    left  = np.array([lm[_MOUTH_L].x  * w, lm[_MOUTH_L].y  * h])
    right = np.array([lm[_MOUTH_R].x  * w, lm[_MOUTH_R].y  * h])
    mouth_r = round(float(np.linalg.norm(top - bot) /
                          (np.linalg.norm(left - right) + 1e-6)), 3)

    drowsy = avg_ear < EAR_THRESH
    if drowsy:
        emotion = "tired"
    elif mouth_r > 0.5:
        emotion = "stressed"
    elif avg_ear > 0.32:
        emotion = "happy"
    else:
        emotion = "calm"

    return {"face_detected": True, "ear": avg_ear,
            "mouth_ratio": mouth_r, "drowsy": drowsy, "emotion": emotion}


def _run_yolo(bgr: np.ndarray) -> list[dict]:
    results = _yolo(bgr, verbose=False, conf=0.35)[0]
    out = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = _yolo.names[cls_id]
        conf   = round(float(box.conf[0]), 2)
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
        out.append({"label": name, "confidence": conf, "box": [x1, y1, x2, y2]})
    return out


def _process(b64: str) -> tuple[dict, list[dict]]:
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"face_detected": False}, []
    return _run_face(bgr), _run_yolo(bgr)


async def run() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_models)

    while True:
        b64 = await frame_queue.get()
        face, objs = await loop.run_in_executor(None, _process, b64)
        await bus.publish("vision_driver",  {"ts": time.time(), **face})
        await bus.publish("vision_objects", {"ts": time.time(), "detections": objs})
