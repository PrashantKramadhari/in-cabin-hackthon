"""Vision node — YuNet/HSEmotion multi-face + MediaPipe drowsiness + YOLOv8n objects."""
from __future__ import annotations

import asyncio
import base64
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bus import bus
from config import vision as vcfg
from sensors import face_emotion as fem
from sensors import vision_status as vstat
from sensors import vision_runtime as vrun
from sensors import vision_temporal as vtemp

_landmarker: Any = None
_yolo: Any = None

NODE_NAME = "yunet_hsemotion_yolo"
READY = False
YOLO_READY = False
EMOTION_READY = False
LOAD_ERROR: str | None = None
DEVICE = "cpu"

frame_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)

EAR_THRESH = vcfg.ear_drowsy_thresh
_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH  = str(_MODELS_DIR / "face_landmarker.task")
_YOLO_PATH  = str(_MODELS_DIR / "yolov8n.pt")

_SEAT_IDS = ["driver", "front_passenger", "rear_left", "rear_middle", "rear_right"]
_SEAT_ROIS: dict[str, dict[str, float]] = vcfg.seat_rois
_EMPTY_SEAT = {
    "occupied": False, "kind": "unknown", "emotion": "calm",
    "buckled": False, "objects": [],
}

_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]


def reset_cache() -> None:
    fem.reset_cache()
    vtemp.reset()


def _download_yolo() -> None:
    dest = Path(_YOLO_PATH)
    if dest.is_file():
        return
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[vision] Downloading yolov8n.pt …")
    from ultralytics import YOLO
    YOLO("yolov8n.pt")
    for src in (Path("yolov8n.pt"), Path.cwd() / "yolov8n.pt"):
        if src.is_file():
            shutil.move(str(src), str(dest))
            print(f"[vision] yolov8n.pt saved → {dest}")
            return
    hub = Path.home() / ".cache" / "ultralytics"
    for pt in hub.rglob("yolov8n.pt"):
        shutil.copy2(pt, dest)
        print(f"[vision] yolov8n.pt copied from cache → {dest}")
        return
    raise FileNotFoundError(
        "yolov8n.pt not found — run: cd backend && python download_models.py"
    )


def _load_models() -> None:
    global _landmarker, _yolo, READY, YOLO_READY, EMOTION_READY, LOAD_ERROR
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"face_landmarker.task missing at {MODEL_PATH}")
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

    _download_yolo()
    _yolo = YOLO(_YOLO_PATH)
    YOLO_READY = True

    EMOTION_READY = fem.try_load()

    READY = True
    LOAD_ERROR = None
    vstat.set_ready(
        node=NODE_NAME, device=DEVICE, backend="mediapipe",
        yolo=True, emotion=EMOTION_READY,
    )
    print(
        f"[vision] YuNet/HSEmotion: {'on' if EMOTION_READY else 'off'}, "
        "MediaPipe drowsy + YOLOv8n ready"
    )


def try_load() -> bool:
    global READY, LOAD_ERROR
    if READY and _landmarker is not None:
        return True
    try:
        _load_models()
        return True
    except Exception as exc:
        LOAD_ERROR = str(exc)
        READY = False
        vstat.set_failed(node="mediapipe", error=LOAD_ERROR, backend="mediapipe")
        print(f"[vision] load failed: {exc}")
        return False


def _ear(lm, indices, w, h) -> float:
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    ho = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * ho + 1e-6))


def _run_face(bgr: np.ndarray) -> dict:
    """MediaPipe driver landmarks — EAR/drowsiness only (emotion via HSEmotion)."""
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
    drowsy = avg_ear < EAR_THRESH
    drowsy = vtemp.smooth_drowsy(drowsy)

    return {
        "face_detected": True,
        "ear": avg_ear,
        "drowsy": drowsy,
        "emotion": "tired" if drowsy else "calm",
    }


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
    best_sid, best_iou = None, 0.0
    for sid in _SEAT_IDS:
        r = _SEAT_ROIS.get(sid)
        if not r:
            continue
        roi = (r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"])
        score = _iou(nb, roi)
        if score > best_iou:
            best_iou, best_sid = score, sid
    return best_sid if best_iou >= 0.05 else None


def _person_kind(box: list[float], seat_id: str, h: int) -> str:
    r = _SEAT_ROIS.get(seat_id, {})
    seat_h_px = r.get("h", 0.4) * h
    person_h = box[3] - box[1]
    if seat_h_px > 0 and person_h / seat_h_px < vcfg.child_bbox_height_ratio:
        return "child"
    return "adult"


def _run_yolo(bgr: np.ndarray) -> list[dict]:
    if _yolo is None:
        return []
    h, w = bgr.shape[:2]
    results = _yolo.predict(
        bgr, verbose=False, conf=vrun.get("yolo_conf_thresh"), iou=0.45,
    )[0]
    out: list[dict] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name   = _yolo.names[cls_id]
        conf   = round(float(box.conf[0]), 2)
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
        pixel_box = [x1, y1, x2, y2]
        seat = _seat_for_box(pixel_box, w, h)
        det: dict[str, Any] = {
            "label": name,
            "confidence": conf,
            "box": pixel_box,
        }
        if seat:
            det["seat"] = seat
        out.append(det)
    return out


def _build_all_seats(
    face: dict,
    detections: list[dict],
    face_hits: list[dict],
    h: int,
) -> dict[str, dict]:
    seats = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}
    person_labels = set(vcfg.yolo_person_labels)
    pet_labels    = set(vcfg.yolo_pet_labels)
    object_labels = set(vcfg.yolo_object_labels)

    for det in detections:
        label = det["label"]
        seat  = det.get("seat")
        if not seat or seat not in seats:
            continue
        occ = seats[seat]
        if label in person_labels:
            occ["occupied"] = True
            kind = _person_kind(det["box"], seat, h)
            if occ["kind"] == "unknown" or kind == "child":
                occ["kind"] = kind
        elif label in pet_labels:
            occ["occupied"] = True
            occ["kind"] = "pet"
        elif label in object_labels:
            if label not in occ["objects"]:
                occ["objects"].append(label)

    for fh in face_hits:
        seat = fh.get("seat")
        if not seat or seat not in seats:
            continue
        occ = seats[seat]
        occ["occupied"] = True
        if occ["kind"] == "unknown":
            occ["kind"] = "adult"
        occ["emotion"] = fh.get("emotion", "calm")

    drv = seats["driver"]
    if face.get("face_detected"):
        drv["occupied"] = True
        if drv["kind"] == "unknown":
            drv["kind"] = "adult"
        if face.get("drowsy"):
            drv["emotion"] = "tired"
        elif not face_hits and drv["emotion"] == "calm":
            drv["emotion"] = face.get("emotion", "calm")

    return seats


def _process(b64: str) -> tuple[dict, list[dict], dict[str, dict], list[dict]]:
    try:
        data = base64.b64decode(b64)
        arr  = np.frombuffer(data, np.uint8)
        bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            empty = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}
            return {"face_detected": False}, [], empty, []
        h = bgr.shape[0]
        face = _run_face(bgr)
        face_hits = fem.analyze(bgr) if EMOTION_READY else []
        objs = _run_yolo(bgr)
        seats = _build_all_seats(face, objs, face_hits, h)
        objs = vtemp.smooth_objects(objs)
        seats = vtemp.smooth_seats(seats)

        driver_emotion = seats["driver"].get("emotion", "calm")
        face["emotion"] = driver_emotion

        return face, objs, seats, face_hits
    except Exception as exc:
        print(f"[vision] process error: {exc}")
        empty = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}
        return {"face_detected": False}, [], empty, []


async def run() -> None:
    loop = asyncio.get_event_loop()
    if not READY:
        ok = await loop.run_in_executor(None, try_load)
        if not ok:
            print("[vision] run() exiting — models not loaded")
            return

    while True:
        b64 = await frame_queue.get()
        try:
            face, objs, seats, face_hits = await loop.run_in_executor(
                None, _process, b64,
            )
        except Exception as exc:
            print(f"[vision] run error: {exc}")
            face, objs = {"face_detected": False}, []
            seats = {sid: dict(_EMPTY_SEAT) for sid in _SEAT_IDS}
            face_hits = []

        face = {k: (bool(v) if isinstance(v, (bool,)) else
                    float(v) if hasattr(v, "__float__") else v)
                for k, v in face.items()}
        ts = time.time()
        await bus.publish("vision_driver",    {"ts": ts, **face})
        await bus.publish("vision_objects",   {"ts": ts, "detections": objs})
        await bus.publish("vision_all_seats", {"ts": ts, **seats})
        await bus.publish("vision_faces",     {"ts": ts, "faces": face_hits})
