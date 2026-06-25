"""Multi-frame temporal smoothing for vision detections (YOLO, seats, drowsy)."""
from __future__ import annotations

from collections import Counter, deque
from typing import Any

from config import vision as vcfg
from sensors import vision_runtime as vrun

_SEAT_IDS = ["driver", "front_passenger", "rear_left", "rear_middle", "rear_right"]
_WINDOW = vcfg.temporal_window_frames
_CLEAR = vcfg.temporal_clear_misses


def _min_hits() -> int:
    return int(vrun.get("temporal_min_hits"))

# (seat, label) → recent hit deque
_obj_hits: dict[tuple[str, str], deque[bool]] = {}
_obj_last: dict[tuple[str, str], dict] = {}
_obj_miss: dict[tuple[str, str], int] = {}

# seat → occupancy hit deque
_occ_hits: dict[str, deque[bool]] = {s: deque(maxlen=_WINDOW) for s in _SEAT_IDS}
_occ_stable: dict[str, bool] = {s: False for s in _SEAT_IDS}
# seat → object labels seen
_obj_label_hits: dict[tuple[str, str], deque[bool]] = {}
_obj_label_stable: dict[tuple[str, set[str]]] = {s: set() for s in _SEAT_IDS}
_kind_hist: dict[str, deque[str]] = {s: deque(maxlen=_WINDOW) for s in _SEAT_IDS}

_drowsy_hits: deque[bool] = deque(maxlen=_WINDOW)
_drowsy_stable = False


def reset() -> None:
    global _drowsy_stable
    _obj_hits.clear()
    _obj_last.clear()
    _obj_miss.clear()
    _obj_label_hits.clear()
    for s in _SEAT_IDS:
        _occ_hits[s].clear()
        _occ_stable[s] = False
        _obj_label_stable[s] = set()
        _kind_hist[s].clear()
    _drowsy_hits.clear()
    _drowsy_stable = False


def _stable_from_deque(hits: deque[bool]) -> bool:
    if not hits:
        return False
    return sum(hits) >= _min_hits()


def smooth_drowsy(raw_drowsy: bool) -> bool:
    global _drowsy_stable
    _drowsy_hits.append(raw_drowsy)
    if raw_drowsy:
        if _stable_from_deque(_drowsy_hits):
            _drowsy_stable = True
    else:
        miss = 0
        for v in reversed(_drowsy_hits):
            if v:
                break
            miss += 1
        if miss >= _CLEAR:
            _drowsy_stable = False
    return _drowsy_stable


def smooth_objects(raw: list[dict]) -> list[dict]:
    """Keep YOLO boxes that appear in min_hits of last window frames."""
    seen: set[tuple[str, str]] = set()
    for det in raw:
        seat = det.get("seat")
        if not seat:
            continue
        key = (seat, det["label"])
        seen.add(key)
        if key not in _obj_hits:
            _obj_hits[key] = deque(maxlen=_WINDOW)
            _obj_miss[key] = 0
        _obj_hits[key].append(True)
        _obj_miss[key] = 0
        _obj_last[key] = det

    for key in list(_obj_hits.keys()):
        if key not in seen:
            _obj_hits[key].append(False)
            _obj_miss[key] = _obj_miss.get(key, 0) + 1
            if _obj_miss[key] >= _CLEAR:
                _obj_hits.pop(key, None)
                _obj_last.pop(key, None)
                _obj_miss.pop(key, None)

    out: list[dict] = []
    for key, hits in _obj_hits.items():
        if sum(hits) >= _min_hits() and key in _obj_last:
            out.append(_obj_last[key])
    return out


def smooth_seats(raw: dict[str, dict]) -> dict[str, dict]:
    """Stabilise per-seat occupancy, kind, and object lists."""
    stable: dict[str, dict] = {}
    all_obj_labels: set[str] = set()
    for det_key in _obj_last:
        all_obj_labels.add(det_key[1])

    for sid in _SEAT_IDS:
        r = raw.get(sid, {})
        _occ_hits[sid].append(bool(r.get("occupied")))

        if _stable_from_deque(_occ_hits[sid]):
            _occ_stable[sid] = True
        elif not r.get("occupied"):
            miss = 0
            for v in reversed(_occ_hits[sid]):
                if v:
                    break
                miss += 1
            if miss >= _CLEAR:
                _occ_stable[sid] = False

        kind = str(r.get("kind", "unknown"))
        if r.get("occupied") and kind != "unknown":
            _kind_hist[sid].append(kind)

        seat = {
            "occupied": _occ_stable[sid],
            "kind": "unknown",
            "emotion": r.get("emotion", "calm"),
            "buckled": bool(r.get("buckled", False)),
            "objects": [],
        }

        if _kind_hist[sid]:
            seat["kind"] = Counter(_kind_hist[sid]).most_common(1)[0][0]
        elif _occ_stable[sid]:
            seat["kind"] = r.get("kind", "adult") or "adult"

        # Stabilise loose-object labels per seat
        stable_objs: set[str] = set()
        for label in all_obj_labels:
            key = (sid, label)
            if key not in _obj_label_hits:
                _obj_label_hits[key] = deque(maxlen=_WINDOW)
            present = label in (r.get("objects") or [])
            in_yolo = any(
                d.get("seat") == sid and d.get("label") == label
                for d in _obj_last.values()
            )
            _obj_label_hits[key].append(present or in_yolo)
            if sum(_obj_label_hits[key]) >= _min_hits():
                stable_objs.add(label)
            elif sum(_obj_label_hits[key]) == 0:
                _obj_label_hits.pop(key, None)
        seat["objects"] = sorted(stable_objs)
        if not _occ_stable[sid]:
            seat["kind"] = "unknown"
            seat["objects"] = []
            seat["emotion"] = "calm"
        stable[sid] = seat

    return stable
