"""Fusion / decision engine.

Subscribes to every sensor topic, keeps the latest frame from each, and at a
fixed cadence computes:
  1. a Cognitive-Load / Attention-Risk score (0..100), and
  2. a set of mitigations, each non-intrusive and confirm-first (idea #4).

Rule-based + light weighting here for transparency and <200ms latency; the same
interface accepts an ML scorer later. Mitigation confirmation state is preserved
across recomputes so the HMI 'Confirm' action sticks.
"""
from __future__ import annotations

import asyncio
import time

from bus import bus
from config import fusion as cfg
from world import world

# latest frame per modality
_latest: dict[str, dict] = {
    "radar": {}, "audio": {}, "vehicle": {},
    "vision_driver": {}, "vision_objects": {},
    "vibration": {}, "vision_all_seats": {},
}

# mitigation registry: id -> {..., status}
_mitigations: dict[str, dict] = {}

HZ = cfg.hz
_GRACE = cfg.grace_period_s
S = cfg.score   # shorthand
M = cfg.mitigations


def _driver_seat() -> dict | None:
    for s in _latest["radar"].get("seats", []):
        if s["seat"] == "driver":
            return s
    return None


def _audio_effective() -> dict:
    """Resolved audio label for fusion — classifier first, world/seat fallback second."""
    sensor = _latest["audio"]
    s_label = sensor.get("label", "none") or "none"
    s_conf = float(sensor.get("confidence", 0.0) or 0.0)
    if s_label not in ("none", "") and s_conf > 0.25:
        return {"label": s_label, "confidence": s_conf, "source": "classifier"}
    if world.audio_label not in ("none", "", None):
        return {
            "label": world.audio_label,
            "confidence": world.audio_conf,
            "source": "world",
        }
    return {"label": "none", "confidence": 0.0, "source": "none"}


def _effective_audio() -> tuple[str, float]:
    """Return (label, confidence) — sensor audio first, seat-config world fallback second."""
    eff = _audio_effective()
    return eff["label"], eff["confidence"]


def _cognitive_load() -> tuple[int, list[str]]:
    """Return (score 0..100, contributing factors)."""
    score = 0.0
    factors: list[str] = []

    # ── Driver vitals (radar micro-doppler) ──────────────────────────────
    d = _driver_seat()
    if d and d.get("occupied"):
        hr = d.get("heart_rate_bpm") or S.driver_hr_default
        resp = d.get("respiration_rpm") or S.driver_resp_default
        if hr > S.driver_hr_high_thresh:
            score += min(S.driver_hr_high_max, (hr - S.driver_hr_high_thresh) * S.driver_hr_high_multiplier)
            factors.append("elevated heart rate")
        if resp > S.driver_resp_high_thresh:
            score += min(S.driver_resp_high_max, (resp - S.driver_resp_high_thresh) * S.driver_resp_high_multiplier)
            factors.append("rapid breathing")
        if hr < S.driver_hr_low_thresh:
            score += S.driver_hr_low_score; factors.append("low arousal / fatigue")

    # ── Driver emotion (world state / seat config) ────────────────────────
    if world.driver_emotion == "stressed":
        score += S.driver_stressed_score; factors.append("driver stressed")
    elif world.driver_emotion == "tired":
        score += S.driver_fatigued_score; factors.append("driver fatigued")

    # ── Audio anomaly (sensor or seat-config fallback) ───────────────────
    audio_label, audio_conf = _effective_audio()
    _child_pet_seats = [s for s in _latest["radar"].get("seats", [])
                        if s.get("occupant") in ("child", "pet") and s.get("occupied")]
    _child_pet_seats = _child_pet_seats + _vision_child_pet_seats()
    if audio_label in ("crying", "barking", "animal") and audio_conf > S.audio_conf_threshold:
        score += S.audio_crying_score; factors.append(audio_label + " in cabin")
    elif audio_label == "shouting" and audio_conf > S.audio_conf_threshold:
        score += S.audio_shouting_score; factors.append("shouting in cabin")
    elif audio_label == "rattle" and audio_conf > S.audio_conf_threshold:
        score += S.audio_rattle_score; factors.append("rattling object")
    elif audio_label == "talking" and audio_conf > S.audio_conf_threshold:
        if _child_pet_seats:
            score += S.audio_talking_child_score; factors.append("baby babbling in cabin")
        else:
            score += S.audio_talking_score; factors.append("speech activity")
    elif audio_label == "happy" and audio_conf > S.audio_conf_threshold:
        score += S.audio_happy_score; factors.append("baby happy in cabin")

    # ── Vehicle context ──────────────────────────────────────────────────
    veh = _latest["vehicle"]
    if veh.get("visibility") in ("low", "rain", "fog"):
        score += S.visibility_low_score; factors.append("reduced visibility")
    if veh.get("speed_kmh", 0) > S.speed_high_thresh:
        score += S.speed_high_score; factors.append("high speed")

    # ── Child/pet: presence + distress + HR (world.seats — instant, no radar lag) ──
    _scored_hr_seats: set[str] = set()
    for sid, occ in world.seats.items():
        if sid == "driver" or not occ.occupied or occ.kind not in ("child", "pet"):
            continue
        score += S.child_presence_score
        factors.append(occ.kind + " in cabin (" + sid.replace("_", " ") + ")")
        if occ.distress > S.distress_threshold:
            score += min(S.distress_max, occ.distress * S.distress_multiplier)
            factors.append(occ.kind + " distress " + str(int(occ.distress * 100)) + "%")
        # HR — use manual override if set, otherwise auto-derived default
        hr_thresh = S.child_hr_thresh if occ.kind == "child" else S.pet_hr_thresh
        hr = occ.heart_rate_bpm  # None = auto (radar.py derives it)
        if hr is not None and hr > hr_thresh:
            over = hr - hr_thresh
            score += min(S.child_hr_max, over * S.child_hr_multiplier)
            factors.append(occ.kind + " elevated HR (" + str(int(hr)) + " bpm)")
            _scored_hr_seats.add(sid)

    # ── Rear occupant vitals + motion from radar (motion not in world.seats) ──
    for s in _latest["radar"].get("seats", []):
        if s.get("seat") == "driver":
            continue
        if not (s.get("occupant") in ("child", "pet") and s.get("occupied")):
            continue
        seat_id  = s.get("seat", "")
        occ_hr   = s.get("heart_rate_bpm") or 0
        occ_resp = s.get("respiration_rpm") or 0
        motion   = s.get("motion") or 0
        kind     = s["occupant"]
        hr_thresh = S.child_hr_thresh if kind == "child" else S.pet_hr_thresh
        # Only score HR from radar if world.seats didn't already score it
        if seat_id not in _scored_hr_seats and occ_hr > hr_thresh:
            score += min(S.radar_child_hr_max, (occ_hr - hr_thresh) * S.radar_child_hr_multiplier)
            factors.append(kind + " elevated HR (" + str(int(occ_hr)) + " bpm)")
        if occ_resp > S.radar_resp_high_thresh:
            score += min(S.radar_resp_max, (occ_resp - S.radar_resp_high_thresh) * S.radar_resp_multiplier)
            factors.append(kind + " rapid breathing")
        if motion > S.radar_motion_thresh:
            score += S.radar_motion_score; factors.append("agitated " + kind)

    # ── Road quality (IMU) ───────────────────────────────────────────────
    vib = _latest["vibration"]
    if vib.get("road_quality") == "pothole":
        score += S.pothole_score; factors.append("pothole / road shock (IMU)")
    elif vib.get("road_quality") == "rough":
        score += S.rough_road_score; factors.append("rough road (IMU)")
    if vib.get("pothole_ahead_m") is not None and vib["pothole_ahead_m"] < S.pothole_ahead_warn_m:
        score += S.pothole_ahead_score; factors.append("rough road ahead (IMU)")

    # ── Vision: drowsiness / stress ──────────────────────────────────────
    vd = _latest["vision_driver"]
    if vd.get("face_detected"):
        if vd.get("drowsy"):
            score += S.drowsy_score; factors.append("drowsy eyes (camera)")
        elif vd.get("emotion") == "stressed":
            score += S.stressed_expression_score; factors.append("stress expression (camera)")

    # ── Vision: loose object (YOLOv8n) ───────────────────────────────────
    for det in _latest["vision_objects"].get("detections", []):
        if det["label"] in ("backpack", "suitcase", "bottle", "cup", "book", "laptop"):
            score += S.yolo_object_score; factors.append(f"loose {det['label']} (camera)"); break

    # ── Vision: loose objects on seats (Qwen) ────────────────────────────
    for sid, sv in _latest.get("vision_all_seats", {}).items():
        if isinstance(sv, dict) and sv.get("objects"):
            obj_str = ", ".join(sv["objects"])
            score += min(S.qwen_object_max_score, len(sv["objects"]) * S.qwen_object_score_per_item)
            factors.append(f"unsecured object on {sid.replace('_', ' ')} ({obj_str})")
            break  # one factor entry is enough

    return int(min(100, score)), factors


def _proposed() -> list[dict]:
    """Compute which mitigations should currently be offered."""
    out: list[dict] = []
    audio_label, audio_conf = _effective_audio()
    veh = _latest["vehicle"]
    d = _driver_seat()

    # --- USE CASE 1: audio comfort, corroborated by occupancy in ANY seat ---
    # Merge radar seats + world.seats so mitigations fire immediately on manual config,
    # without waiting for the radar node to republish after a scenario/seat change.
    radar_child_pet = {
        s["seat"] for s in _latest["radar"].get("seats", [])
        if s.get("occupant") in ("child", "pet") and s.get("occupied")
    }
    world_child_pet = {
        sid for sid, occ in world.seats.items()
        if sid != "driver" and occ.occupied and occ.kind in ("child", "pet")
    }
    child_pet_seat_ids = radar_child_pet | world_child_pet
    child_pet_seats = [{"seat": sid} for sid in child_pet_seat_ids] + _vision_child_pet_seats()
    seat_names = ", ".join(
        sid.replace("_", " ").title() for sid in child_pet_seat_ids
    ) if child_pet_seat_ids else ""

    if audio_label in ("crying", "animal", "barking") and audio_conf > S.audio_conf_threshold \
            and child_pet_seats:
        who = "child" if audio_label in ("crying",) else "pet"
        out.append(dict(
            id="comfort_audio", title="Soothe cabin",
            usecase="Audio comfort",
            detail=f"{who.title()} distress detected in {seat_names}. "
                   "Lower media volume 30%, warm AC +1°C, soft cabin lighting.",
            severity="advisory", confirm=True))

    elif audio_label == "talking" and audio_conf > S.audio_conf_threshold and child_pet_seats:
        out.append(dict(
            id="baby_engagement", title="Engage baby",
            usecase="Audio comfort",
            detail=f"Baby babbling in {seat_names}. Play a nursery rhyme or "
                   "soft music to maintain a calm, stimulating environment.",
            severity="advisory", confirm=True))

    elif audio_label == "happy" and audio_conf > S.audio_conf_threshold and child_pet_seats:
        out.append(dict(
            id="baby_happy", title="Passenger content",
            usecase="Cabin monitoring",
            detail=f"Baby sounds happy in {seat_names}. Maintain current "
                   "cabin temperature, lighting and media volume.",
            severity="advisory", confirm=True))

    # --- Shouting / passenger distress ---
    if audio_label == "shouting" and audio_conf > S.audio_conf_threshold:
        out.append(dict(
            id="shouting_alert", title="Passenger distress",
            usecase="Audio comfort",
            detail="Shouting detected in cabin. Check passenger status and consider "
                   "pulling over if safe.",
            severity="warning", confirm=True))

    # --- USE CASE 2: driver persona tuning from vitals + emotion ---
    if d and d.get("occupied"):
        hr = d.get("heart_rate_bpm") or S.driver_hr_default
        if hr > S.driver_hr_high_thresh or world.driver_emotion == "stressed":
            out.append(dict(
                id="persona_calm", title="Calming persona",
                usecase="Persona tuning",
                detail="Driver stress detected (HR ↑, breathing ↑). "
                       "Switch to calm playlist, cool blue lighting, temp −1°C.",
                severity="advisory", confirm=True))
        elif hr < S.driver_hr_low_thresh or world.driver_emotion == "tired":
            out.append(dict(
                id="persona_alert", title="Alertness boost",
                usecase="Persona tuning",
                detail="Fatigue signs detected. Upbeat playlist, brighter "
                       "lighting, fresh-air burst, suggest a break.",
                severity="advisory", confirm=True))

    # --- USE CASE 3a: seatbelt misuse (world.seats is always current) ---
    for sid, occ in world.seats.items():
        if occ.occupied and not occ.buckled:
            out.append(dict(
                id="belt_" + sid, title="Seatbelt not engaged",
                usecase="Seatbelt safety",
                detail=sid.replace("_", " ").title() + " occupied but "
                       "belt not properly worn. Chime + visual reminder.",
                severity="warning", confirm=False))

    # --- USE CASE 3b: pre-emptive object securing before rough road ---
    if world.unsecured_object and veh.get("pothole_ahead_m") is not None:
        dist = veh["pothole_ahead_m"]
        if dist <= M.pothole_warning_distance_m:
            out.append(dict(
                id="secure_object", title="Secure loose item",
                usecase="Pothole-aware advisory",
                detail=f"Unsecured object detected, rough road in ~{int(dist)} m. "
                       "Advisory: secure item now to prevent displacement.",
                severity="warning", confirm=True))

    # --- USE CASE 2 corroboration: vision emotion overrides world state ---
    vd = _latest["vision_driver"]
    if vd.get("face_detected") and vd.get("emotion") in ("tired", "stressed"):
        cam_emotion = vd["emotion"]
        mit_id = "persona_calm" if cam_emotion == "stressed" else "persona_alert"
        if not any(m["id"] == mit_id for m in out):
            out.append(dict(
                id=mit_id,
                title="Calming persona" if cam_emotion == "stressed" else "Alertness boost",
                usecase="Persona tuning (camera)",
                detail=f"Camera detected {cam_emotion} expression. "
                       + ("Calm playlist, cool lighting, temp −1°C."
                          if cam_emotion == "stressed"
                          else "Upbeat playlist, bright lighting, fresh-air burst."),
                severity="advisory", confirm=True))

    # --- vision: unsecured object → pre-empt pothole advisory ---
    yolo_obj = next(
        (d for d in _latest["vision_objects"].get("detections", [])
         if d["label"] in ("backpack", "suitcase", "bottle", "cup", "book", "laptop")),
        None)
    if yolo_obj and veh.get("pothole_ahead_m") is not None:
        dist = veh["pothole_ahead_m"]
        if dist <= M.pothole_warning_distance_m and not any(m["id"] == "secure_object" for m in out):
            out.append(dict(
                id="secure_object_cam", title="Secure loose item (camera)",
                usecase="Pothole-aware advisory",
                detail=f"{yolo_obj['label'].title()} detected by camera, rough road "
                       f"in ~{int(dist)} m. Secure it now.",
                severity="warning", confirm=True))

    # --- Qwen: loose objects on seats → advisory / pothole pre-empt ---
    pothole_dist = veh.get("pothole_ahead_m")
    for sid, sv in _latest.get("vision_all_seats", {}).items():
        if not isinstance(sv, dict):
            continue
        objs = sv.get("objects", [])
        if not objs:
            continue
        obj_list   = ", ".join(objs)
        seat_label = sid.replace("_", " ").title()
        if pothole_dist is not None and pothole_dist <= M.pothole_warning_distance_m:
            out.append(dict(
                id="loose_obj_" + sid, title="Secure loose item",
                usecase="Pothole-aware advisory",
                detail=f"{obj_list.title()} on {seat_label} — rough road in ~{int(pothole_dist)} m. "
                       "Secure or stow the item now to prevent it becoming a projectile.",
                severity="warning", confirm=True))
        else:
            out.append(dict(
                id="loose_obj_" + sid, title="Unsecured item on seat",
                usecase="Object safety",
                detail=f"{obj_list.title()} detected on {seat_label}. "
                       "Could slide or fall during sudden braking or cornering.",
                severity="advisory", confirm=True))

    # --- Elevated child/pet HR mitigation (fires immediately from world.seats) ---
    for sid, occ in world.seats.items():
        if sid == "driver" or not occ.occupied or occ.kind not in ("child", "pet"):
            continue
        hr = occ.heart_rate_bpm
        if hr is None:
            continue
        hr_thresh = S.child_hr_thresh if occ.kind == "child" else S.pet_hr_thresh
        if hr > hr_thresh:
            seat_label = sid.replace("_", " ").title()
            severity = "critical" if hr > hr_thresh + S.child_hr_critical_offset else "warning"
            out.append(dict(
                id="hr_alert_" + sid,
                title=occ.kind.title() + " heartbeat elevated",
                usecase="Radar vital monitoring",
                detail=seat_label + " — " + occ.kind + " heart rate " + str(int(hr)) + " bpm "
                       + "(normal <" + str(hr_thresh) + "). "
                       + ("Immediate check recommended." if severity == "critical"
                          else "Monitor and adjust cabin comfort."),
                severity=severity, confirm=True))

    # --- Child/pet monitoring card (fires on manual config, no audio required) ---
    manual_child_pet = [
        (sid, occ) for sid, occ in world.seats.items()
        if sid != "driver" and occ.occupied and occ.kind in ("child", "pet")
    ]
    if manual_child_pet and not any(m["id"] in ("comfort_audio", "baby_engagement") for m in out):
        high_distress = any(occ.distress > 0.5 for _, occ in manual_child_pet)
        who_list = ", ".join(
            occ.kind.title() + " (" + sid.replace("_", " ").title() + ")"
            for sid, occ in manual_child_pet
        )
        out.append(dict(
            id="cabin_monitoring", title="Cabin monitoring active",
            usecase="Child / pet welfare",
            detail=who_list + " detected via radar. Vitals monitoring active. "
                   + ("Elevated distress — check cabin comfort." if high_distress
                      else "Adjust volume, AC or lighting as needed."),
            severity="warning" if high_distress else "advisory", confirm=False))

    # --- safety-critical: child left behind ---
    driver_present = bool(d and d.get("occupied"))
    rear_child = any(s.get("occupant") == "child" and s.get("occupied")
                     for s in _latest["radar"].get("seats", []))
    if rear_child and not driver_present:
        out.append(dict(
            id="child_left", title="CHILD PRESENCE ALERT",
            usecase="Child presence detection",
            detail="Child detected in rear seat with no driver present. "
                   "Escalate: horn + lights + owner notification.",
            severity="critical", confirm=False))
    return out




def _reconcile(proposed: list[dict]) -> list[dict]:
    """Merge proposed list with existing confirmation state.

    Confirmed/dismissed mitigations are kept for _GRACE seconds after their
    trigger condition clears, so a momentary radar flicker doesn't reset a
    user's explicit Apply/Dismiss choice.
    """
    now = time.time()
    seen = set()
    for m in proposed:
        seen.add(m["id"])
        prev = _mitigations.get(m["id"])
        if prev:
            m["status"] = prev["status"]
        else:
            m["status"] = "active" if not m["confirm"] else "proposed"
        _mitigations[m["id"]] = dict(m, _ts=now)

    to_del = []
    for k, m in _mitigations.items():
        if k not in seen:
            age = now - m.get("_ts", now)
            if m.get("status") in ("active", "dismissed") and age < _GRACE:
                continue  # grace period: keep user's choice visible
            to_del.append(k)
    for k in to_del:
        del _mitigations[k]

    return [{kk: vv for kk, vv in m.items() if kk != "_ts"}
            for m in _mitigations.values()]


def reset_vision() -> None:
    """Clear all cached vision data so a fresh video starts clean."""
    _latest["vision_all_seats"] = {}
    _latest["vision_driver"] = {}
    _latest["vision_objects"] = {}


def confirm(mitigation_id: str) -> None:
    if mitigation_id in _mitigations:
        _mitigations[mitigation_id]["status"] = "active"


def dismiss(mitigation_id: str) -> None:
    if mitigation_id in _mitigations:
        _mitigations[mitigation_id]["status"] = "dismissed"


def _norm_kind(k: str) -> str:
    """Normalise Qwen kind strings to world model kinds."""
    return "child" if k in ("child", "infant") else k  # infant → child


def _vision_child_pet_seats() -> list[dict]:
    """Seats where Qwen detected a child/infant/pet not already in world radar."""
    seen = {s["seat"] for s in _latest["radar"].get("seats", [])
            if s.get("occupant") in ("child", "pet") and s.get("occupied")}
    out = []
    for sid, sv in _latest.get("vision_all_seats", {}).items():
        if not isinstance(sv, dict) or not sv.get("occupied"):
            continue
        kind = _norm_kind(sv.get("kind", "unknown"))
        if kind in ("child", "pet") and sid not in seen:
            out.append({"seat": sid, "occupant": kind, "occupied": True})
    return out


def _build_seat_configs() -> dict:
    """Merge world seat state with Qwen per-seat vision data."""
    vs = _latest.get("vision_all_seats", {})
    configs = {}
    for sid, occ in world.seats.items():
        cfg = {
            "occupied":         occ.occupied,
            "kind":             occ.kind,
            "buckled":          occ.buckled,
            "distress":         round(occ.distress, 2),
            "audio_event":      occ.audio_event,
            "heart_rate_bpm":   occ.heart_rate_bpm,
            "respiration_rpm":  occ.respiration_rpm,
            "emotion":          occ.emotion,
        }
        seat_v = vs.get(sid)
        if isinstance(seat_v, dict) and seat_v.get("occupied"):
            cfg["occupied"] = True
            raw_kind = seat_v.get("kind", "")
            has_manual = sid in world.seat_overrides
            # Vision overrides kind unless user manually configured this seat
            if raw_kind and raw_kind != "unknown" and not has_manual:
                cfg["kind"] = _norm_kind(raw_kind)
            emotion = seat_v.get("emotion", cfg["emotion"])
            cfg["emotion"] = emotion
            # Derive audio_event from Qwen emotion when no manual override
            if not has_manual or "audio_event" not in world.seat_overrides.get(sid, {}):
                if emotion == "distressed":
                    cfg["audio_event"] = "crying"
                elif emotion == "stressed":
                    cfg["audio_event"] = "shouting"
                elif emotion in ("calm", "happy", "tired"):
                    cfg["audio_event"] = "none"
        configs[sid] = cfg
    return configs


async def _consume(topic: str) -> None:
    async for frame in bus.stream(topic):
        _latest[topic] = frame


async def run() -> None:
    for t in ("radar", "audio", "vehicle", "vibration",
              "vision_driver", "vision_objects", "vision_all_seats"):
        asyncio.create_task(_consume(t))
    while True:
        t0 = time.perf_counter()
        score, factors = _cognitive_load()
        mitigations = _reconcile(_proposed())
        state = dict(
            ts=time.time(),
            scenario=world.scenario,
            demo_running=world.demo_running,
            cognitive_load=score,
            factors=factors,
            radar=_latest["radar"],
            audio=_latest["audio"],
            audio_effective=_audio_effective(),
            vehicle=_latest["vehicle"],
            vision_driver=_latest["vision_driver"],
            vision_objects=_latest["vision_objects"],
            vibration=_latest["vibration"],
            mitigations=mitigations,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            # World-level config exposed so the HMI can reflect / edit it.
            # Qwen vision_all_seats overlays per-seat emotion when available.
            seat_configs=_build_seat_configs(),
            vision_all_seats=_latest.get("vision_all_seats", {}),
            driver_emotion=world.driver_emotion,
            vib_override=world.vib_override,
            vib_road_quality=world.vib_road_quality,
            vib_rms=round(world.vib_rms, 3),
            world_speed_kmh=world.speed_kmh,
            world_visibility=world.visibility,
            world_pothole_ahead_m=world.pothole_ahead_m,
        )
        await bus.publish("fused", state)
        await asyncio.sleep(1 / HZ)
