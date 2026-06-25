"""Shared vision-backend status — updated by whichever node loads successfully."""
from __future__ import annotations

NODE_NAME: str = "none"
DEVICE: str | None = None
READY: bool = False
ERROR: str | None = None
BACKEND: str = "none"  # qwen | mediapipe


def set_ready(*, node: str, device: str | None, backend: str) -> None:
    global NODE_NAME, DEVICE, READY, ERROR, BACKEND
    NODE_NAME = node
    DEVICE = device
    READY = True
    ERROR = None
    BACKEND = backend


def set_failed(*, node: str, error: str, backend: str) -> None:
    global NODE_NAME, DEVICE, READY, ERROR, BACKEND
    NODE_NAME = node
    DEVICE = None
    READY = False
    ERROR = error
    BACKEND = backend


def snapshot() -> dict:
    return {
        "vision_node": NODE_NAME,
        "vision_ready": READY,
        "vision_device": DEVICE,
        "vision_status": "ready" if READY else (ERROR or "not loaded"),
        "vision_backend": BACKEND,
        "live_vision": READY,
        "qwen_vision": READY and BACKEND == "qwen",
    }
