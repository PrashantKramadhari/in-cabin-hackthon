"""Load config.yaml and expose typed accessors."""
from __future__ import annotations
from pathlib import Path
import yaml

_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(_PATH) as _f:
    _raw = yaml.safe_load(_f)


def _get(path: str):
    """Dot-separated key lookup, e.g. 'fusion.score.pothole_score'."""
    node = _raw
    for key in path.split("."):
        node = node[key]
    return node


# ── Fusion ────────────────────────────────────────────────────────────────────
class _Score:
    driver_hr_default           = _get("fusion.score.driver_hr_default")
    driver_resp_default         = _get("fusion.score.driver_resp_default")
    driver_hr_high_thresh       = _get("fusion.score.driver_hr_high_thresh")
    driver_hr_high_multiplier   = _get("fusion.score.driver_hr_high_multiplier")
    driver_hr_high_max          = _get("fusion.score.driver_hr_high_max")
    driver_resp_high_thresh     = _get("fusion.score.driver_resp_high_thresh")
    driver_resp_high_multiplier = _get("fusion.score.driver_resp_high_multiplier")
    driver_resp_high_max        = _get("fusion.score.driver_resp_high_max")
    driver_hr_low_thresh        = _get("fusion.score.driver_hr_low_thresh")
    driver_hr_low_score         = _get("fusion.score.driver_hr_low_score")
    driver_stressed_score       = _get("fusion.score.driver_stressed_score")
    driver_fatigued_score       = _get("fusion.score.driver_fatigued_score")

    audio_conf_threshold        = _get("fusion.score.audio_conf_threshold")
    audio_crying_score          = _get("fusion.score.audio_crying_score")
    audio_shouting_score        = _get("fusion.score.audio_shouting_score")
    audio_rattle_score          = _get("fusion.score.audio_rattle_score")
    audio_talking_child_score   = _get("fusion.score.audio_talking_child_score")
    audio_talking_score         = _get("fusion.score.audio_talking_score")
    audio_happy_score           = _get("fusion.score.audio_happy_score")

    visibility_low_score        = _get("fusion.score.visibility_low_score")
    speed_high_thresh           = _get("fusion.score.speed_high_thresh")
    speed_high_score            = _get("fusion.score.speed_high_score")

    child_presence_score        = _get("fusion.score.child_presence_score")
    distress_threshold          = _get("fusion.score.distress_threshold")
    distress_multiplier         = _get("fusion.score.distress_multiplier")
    distress_max                = _get("fusion.score.distress_max")
    child_hr_thresh             = _get("fusion.score.child_hr_thresh")
    pet_hr_thresh               = _get("fusion.score.pet_hr_thresh")
    child_hr_multiplier         = _get("fusion.score.child_hr_multiplier")
    child_hr_max                = _get("fusion.score.child_hr_max")
    child_hr_critical_offset    = _get("fusion.score.child_hr_critical_offset")

    radar_child_hr_multiplier   = _get("fusion.score.radar_child_hr_multiplier")
    radar_child_hr_max          = _get("fusion.score.radar_child_hr_max")
    radar_resp_high_thresh      = _get("fusion.score.radar_resp_high_thresh")
    radar_resp_multiplier       = _get("fusion.score.radar_resp_multiplier")
    radar_resp_max              = _get("fusion.score.radar_resp_max")
    radar_motion_thresh         = _get("fusion.score.radar_motion_thresh")
    radar_motion_score          = _get("fusion.score.radar_motion_score")

    pothole_score               = _get("fusion.score.pothole_score")
    rough_road_score            = _get("fusion.score.rough_road_score")
    pothole_ahead_score         = _get("fusion.score.pothole_ahead_score")
    pothole_ahead_warn_m        = _get("fusion.score.pothole_ahead_warn_m")

    drowsy_score                = _get("fusion.score.drowsy_score")
    stressed_expression_score   = _get("fusion.score.stressed_expression_score")
    yolo_object_score           = _get("fusion.score.yolo_object_score")
    qwen_object_score_per_item  = _get("fusion.score.qwen_object_score_per_item")
    qwen_object_max_score       = _get("fusion.score.qwen_object_max_score")


class _Mitigations:
    pothole_warning_distance_m  = _get("fusion.mitigations.pothole_warning_distance_m")


class _Fusion:
    hz                          = _get("fusion.hz")
    grace_period_s              = _get("fusion.grace_period_s")
    score                       = _Score()
    mitigations                 = _Mitigations()


# ── Audio ─────────────────────────────────────────────────────────────────────
class _Audio:
    bus_conf_threshold          = _get("audio.bus_conf_threshold")
    ast_conf_threshold          = _get("audio.ast_conf_threshold")
    babynet_conf_threshold      = _get("audio.babynet_conf_threshold")
    ast_fallback_threshold      = _get("audio.ast_fallback_threshold")
    clap_model                  = _get("audio.clap_model")
    chunk_samples               = _get("audio.chunk_samples")
    clap_delay_chunks           = _get("audio.clap_delay_chunks")
    clap_smooth_chunks          = _get("audio.clap_smooth_chunks")
    clap_conf_threshold         = _get("audio.clap_conf_threshold")
    clap_margin_over_none        = _get("audio.clap_margin_over_none")
    clap_categories             = _get("audio.clap_categories")


# ── Vision ────────────────────────────────────────────────────────────────────
class _Vision:
    ear_drowsy_thresh           = _get("vision.ear_drowsy_thresh")
    ear_happy_thresh            = _get("vision.ear_happy_thresh")
    mouth_stressed_thresh       = _get("vision.mouth_stressed_thresh")
    yolo_conf_thresh            = _get("vision.yolo_conf_thresh")
    child_bbox_height_ratio     = _get("vision.child_bbox_height_ratio")
    seat_rois                   = _get("vision.seat_rois")
    yolo_person_labels          = _get("vision.yolo_person_labels")
    yolo_pet_labels             = _get("vision.yolo_pet_labels")
    yolo_object_labels          = _get("vision.yolo_object_labels")
    yunet_score_thresh          = _get("vision.yunet_score_thresh")
    yunet_nms_thresh            = _get("vision.yunet_nms_thresh")
    emotion_model               = _get("vision.emotion_model")
    emotion_smooth_frames       = _get("vision.emotion_smooth_frames")
    emotion_min_conf            = _get("vision.emotion_min_conf")
    temporal_window_frames      = _get("vision.temporal_window_frames")
    temporal_min_hits           = _get("vision.temporal_min_hits")
    temporal_clear_misses       = _get("vision.temporal_clear_misses")
    driver_drowsy_min_frames    = _get("vision.driver_drowsy_min_frames")


# ── Vibration ─────────────────────────────────────────────────────────────────
class _Vibration:
    hz                          = _get("vibration.hz")
    rms_rough                   = _get("vibration.rms_rough")
    rms_pothole                 = _get("vibration.rms_pothole")
    pothole_warn_m              = _get("vibration.pothole_warn_m")


# ── Sensors ───────────────────────────────────────────────────────────────────
class _Sensors:
    radar_hz                    = _get("sensors.radar_hz")
    vehicle_hz                  = _get("sensors.vehicle_hz")


# ── Public API ────────────────────────────────────────────────────────────────
fusion    = _Fusion()
audio     = _Audio()
vision    = _Vision()
vibration = _Vibration()
sensors   = _Sensors()
