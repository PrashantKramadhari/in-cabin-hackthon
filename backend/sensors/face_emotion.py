"""YuNet multi-face detection + HSEmotion ONNX — per-seat cabin emotions."""
from __future__ import annotations

import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from config import vision as vcfg
from sensors import vision_runtime as vrun

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_YUNET_PATH = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_HSE_CACHE = Path.home() / ".hsemotion"
_HSE_MODEL = "enet_b0_8_best_afew"
_HSE_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
    "models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx"
)

_SEAT_IDS = ["driver", "front_passenger", "rear_left", "rear_middle", "rear_right"]
_SEAT_ROIS: dict[str, dict[str, float]] = vcfg.seat_rois

_FER_TO_CABIN = {
    "Anger":    "stressed",
    "Contempt": "stressed",
    "Disgust":  "distressed",
    "Fear":     "distressed",
    "Happiness": "happy",
    "Neutral":  "calm",
    "Sadness":  "tired",
    "Surprise": "happy",
}
_IDX_TO_FER = {
    0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear",
    4: "Happiness", 5: "Neutral", 6: "Sadness", 7: "Surprise",
}

_yunet: Any = None
_ort_session: ort.InferenceSession | None = None
READY = False
LOAD_ERROR: str | None = None
_history: dict[str, deque[str]] = {
    sid: deque(maxlen=vcfg.emotion_smooth_frames) for sid in _SEAT_IDS
}


def _download_yunet() -> None:
    if _YUNET_PATH.is_file() and _YUNET_PATH.stat().st_size > 100_000:
        return
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[face_emotion] Downloading YuNet …")
    urllib.request.urlretrieve(_YUNET_URL, _YUNET_PATH)


def _download_hsemotion() -> Path:
    _HSE_CACHE.mkdir(parents=True, exist_ok=True)
    fpath = _HSE_CACHE / f"{_HSE_MODEL}.onnx"
    if not fpath.is_file() or fpath.stat().st_size < 100_000:
        print("[face_emotion] Downloading HSEmotion ONNX …")
        urllib.request.urlretrieve(_HSE_URL, fpath)
    return fpath


def try_load() -> bool:
    global _yunet, _ort_session, READY, LOAD_ERROR
    if READY and _yunet is not None and _ort_session is not None:
        return True
    try:
        _download_yunet()
        _yunet = cv2.FaceDetectorYN.create(
            str(_YUNET_PATH),
            "",
            (320, 320),
            vrun.get("yunet_score_thresh"),
            vcfg.yunet_nms_thresh,
            5000,
        )
        model_path = _download_hsemotion()
        _ort_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )
        READY = True
        LOAD_ERROR = None
        print("[face_emotion] YuNet + HSEmotion ready (multi-face)")
        return True
    except Exception as exc:
        LOAD_ERROR = str(exc)
        READY = False
        print(f"[face_emotion] load failed: {exc}")
        return False


def reset_cache() -> None:
    for sid in _SEAT_IDS:
        _history[sid].clear()


def _norm_box(box: list[float], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1 / w, y1 / h, x2 / w, y2 / h


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _seat_for_box(box: list[float], w: int, h: int) -> str | None:
    nb = _norm_box(box, w, h)
    cx, cy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
    best_sid, best_score = None, -1.0
    for sid in _SEAT_IDS:
        r = _SEAT_ROIS.get(sid)
        if not r:
            continue
        roi = (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])
        if roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]:
            score = _iou(nb, roi)
            if score > best_score:
                best_score, best_sid = score, sid
    return best_sid


def _preprocess_face(rgb: np.ndarray) -> np.ndarray:
    x = cv2.resize(rgb, (224, 224)) / 255.0
    x[..., 0] = (x[..., 0] - 0.485) / 0.229
    x[..., 1] = (x[..., 1] - 0.456) / 0.224
    x[..., 2] = (x[..., 2] - 0.406) / 0.225
    return x.transpose(2, 0, 1).astype("float32")[np.newaxis, ...]


def _classify_face(rgb: np.ndarray) -> tuple[str, str, float]:
    """Return (cabin_emotion, fer_label, confidence)."""
    assert _ort_session is not None
    logits = _ort_session.run(None, {"input": _preprocess_face(rgb)})[0][0]
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    pred = int(np.argmax(probs))
    fer = _IDX_TO_FER[pred]
    conf = float(probs[pred])
    cabin = _FER_TO_CABIN.get(fer, "calm")
    return cabin, fer, conf


def _smooth(seat_id: str, emotion: str) -> str:
    _history[seat_id].append(emotion)
    return Counter(_history[seat_id]).most_common(1)[0][0]


def analyze(bgr: np.ndarray) -> list[dict[str, Any]]:
    """Detect all faces, classify emotion, map to seat. Returns face detection dicts."""
    if not READY or _yunet is None or _ort_session is None:
        return []

    h, w = bgr.shape[:2]
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(bgr)
    if faces is None or len(faces) == 0:
        return []

    out: list[dict[str, Any]] = []
    for face in faces:
        x, y, bw, bh = map(int, face[:4])
        score = float(face[14]) if len(face) > 14 else 1.0
        if score < vrun.get("yunet_score_thresh"):
            continue
        x2, y2 = x + bw, y + bh
        seat = _seat_for_box([x, y, x2, y2], w, h)
        if not seat:
            continue

        pad = int(0.15 * max(bw, bh))
        crop = bgr[
            max(0, y - pad):min(h, y2 + pad),
            max(0, x - pad):min(w, x2 + pad),
        ]
        if crop.size == 0:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cabin, fer, conf = _classify_face(rgb)
        if conf < vrun.get("emotion_min_conf"):
            cabin = "calm"
        cabin = _smooth(seat, cabin)

        out.append({
            "seat": seat,
            "box": [x, y, x2, y2],
            "emotion": cabin,
            "fer_label": fer,
            "confidence": round(conf, 2),
            "face_score": round(score, 2),
        })
    return out
