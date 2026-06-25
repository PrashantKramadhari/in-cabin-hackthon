"""Runtime-adjustable vision detection thresholds (Pipeline Debug sliders)."""
from __future__ import annotations

from config import vision as vcfg

_KEYS = (
    "yolo_conf_thresh",
    "yunet_score_thresh",
    "emotion_min_conf",
    "display_conf_thresh",
    "temporal_min_hits",
)

_defaults: dict[str, float] = {
    "yolo_conf_thresh": float(vcfg.yolo_conf_thresh),
    "yunet_score_thresh": float(vcfg.yunet_score_thresh),
    "emotion_min_conf": float(vcfg.emotion_min_conf),
    "display_conf_thresh": 0.6,
    "temporal_min_hits": float(vcfg.temporal_min_hits),
}

_state: dict[str, float] = dict(_defaults)


def get(key: str) -> float:
    return float(_state.get(key, _defaults[key]))


def update(patch: dict) -> dict[str, float]:
    for key, val in patch.items():
        if key not in _defaults:
            continue
        if key == "temporal_min_hits":
            _state[key] = max(1.0, min(5.0, float(val)))
        else:
            _state[key] = max(0.05, min(0.95, float(val)))
    return snapshot()


def snapshot() -> dict[str, float]:
    return {k: round(_state.get(k, _defaults[k]), 3) for k in _KEYS}
