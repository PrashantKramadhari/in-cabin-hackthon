"""Shared vision-backend status — updated by whichever node loads successfully."""
from __future__ import annotations

NODE_NAME: str = "none"
DEVICE: str | None = None
READY: bool = False
ERROR: str | None = None
BACKEND: str = "none"  # qwen | mediapipe
YOLO_READY: bool = False
EMOTION_READY: bool = False


def set_ready(*, node: str, device: str | None, backend: str,
              yolo: bool = False, emotion: bool = False) -> None:
    global NODE_NAME, DEVICE, READY, ERROR, BACKEND, YOLO_READY, EMOTION_READY
    NODE_NAME = node
    DEVICE = device
    READY = True
    ERROR = None
    BACKEND = backend
    YOLO_READY = yolo
    EMOTION_READY = emotion


def set_failed(*, node: str, error: str, backend: str) -> None:
    global NODE_NAME, DEVICE, READY, ERROR, BACKEND, YOLO_READY, EMOTION_READY
    NODE_NAME = node
    DEVICE = None
    READY = False
    ERROR = error
    BACKEND = backend
    YOLO_READY = False
    EMOTION_READY = False


def snapshot() -> dict:
    return {
        "vision_node": NODE_NAME,
        "vision_ready": READY,
        "vision_device": DEVICE,
        "vision_status": "ready" if READY else (ERROR or "not loaded"),
        "vision_backend": BACKEND,
        "yolo_ready": YOLO_READY,
        "emotion_ready": EMOTION_READY,
        "live_vision": READY,
        "qwen_vision": READY and BACKEND == "qwen",
    }
