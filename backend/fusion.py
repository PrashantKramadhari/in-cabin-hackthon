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
from world import world

# latest frame per modality
_latest: dict[str, dict] = {
    "radar": {}, "audio": {}, "vehicle": {},
    "vision_driver": {}, "vision_objects": {},
    "vibration": {}, "vision_all_seats": {},
}

# mitigation registry: id -> {..., status}
_mitigations: dict[str, dict] = {}

HZ = 10


def _driver_seat() -> dict | None:
    for s in _latest["radar"].get("seats", []):
        if s["seat"] == "driver":
            return s
    return None


def _effective_audio() -> tuple[str, float]:
    """Return (label, confidence) — sensor audio first, seat-config world fallback second.

    The live YAMNet node reads the microphone and publishes to the bus regardless of
    world.audio_label, so scenario/seat-config audio state would otherwise be invisible
    to the fusion engine when live audio is active.  This fallback ensures that manually
    configured seat events (crying, barking, shouting…) always reach the cognitive-load
    formula, while real microphone detections take priority.
    """
    sensor = _latest["audio"]
    s_label = sensor.get("label", "none")
    s_conf = sensor.get("confidence", 0.0)
    if s_label not in ("none", "", None) and s_conf > 0.25:
        return s_label, s_conf
    # Fallback: audio state derived from per-seat config (set by _sync_audio_from_seats)
    if world.audio_label not in ("none", "", None):
        return world.audio_label, world.audio_conf
    return "none", 0.0


def _cognitive_load() -> tuple[int, list[str]]:
    """Return (score 0..100, contributing factors)."""
    score = 0.0
    factors: list[str] = []

    # ── Driver vitals (radar micro-doppler) ──────────────────────────────
    d = _driver_seat()
    if d and d.get("occupied"):
        hr = d.get("heart_rate_bpm") or 72
        resp = d.get("respiration_rpm") or 14
        if hr > 95:
            score += min(30, (hr - 95) * 1.5); factors.append("elevated heart rate")
        if resp > 20:
            score += min(15, (resp - 20) * 2); factors.append("rapid breathing")
        if hr < 60:
            score += 12; factors.append("low arousal / fatigue")

    # ── Driver emotion (world state / seat config) ────────────────────────
    if world.driver_emotion == "stressed":
        score += 10; factors.append("driver stressed")
    elif world.driver_emotion == "tired":
        score += 8; factors.append("driver fatigued")

    # ── Audio anomaly (sensor or seat-config fallback) ───────────────────
    audio_label, audio_conf = _effective_audio()
    _child_pet_seats = [s for s in _latest["radar"].get("seats", [])
                        if s.get("occupant") in ("child", "pet") and s.get("occupied")]
    if audio_label in ("crying", "barking", "animal") and audio_conf > 0.25:
        score += 25; factors.append(audio_label + " in cabin")
    elif audio_label == "shouting" and audio_conf > 0.25:
        score += 20; factors.append("shouting in cabin")
    elif audio_label == "rattle" and audio_conf > 0.25:
        score += 10; factors.append("rattling object")
    elif audio_label == "talking" and audio_conf > 0.25:
        if _child_pet_seats:
            score += 10; factors.append("baby babbling in cabin")
        else:
            score += 5; factors.append("speech activity")
    elif audio_label == "happy" and audio_conf > 0.25:
        score += 2; factors.append("baby happy in cabin")

    # ── Vehicle context ──────────────────────────────────────────────────
    veh = _latest["vehicle"]
    if veh.get("visibility") in ("low", "rain", "fog"):
        score += 12; factors.append("reduced visibility")
    if veh.get("speed_kmh", 0) > 100:
        score += 8; factors.append("high speed")

    # ── Agitated rear occupants (count all, not just first) ──────────────
    for s in _latest["radar"].get("seats", []):
        if s.get("seat") == "driver":
            continue
        if s.get("occupant") in ("child", "pet") and s.get("occupied") and (s.get("motion") or 0) > 0.4:
            score += 10; factors.append("agitated " + s["occupant"])

    # ── Road quality (IMU) ───────────────────────────────────────────────
    vib = _latest["vibration"]
    if vib.get("road_quality") == "pothole":
        score += 18; factors.append("pothole / road shock (IMU)")
    elif vib.get("road_quality") == "rough":
        score += 8; factors.append("rough road (IMU)")
    if vib.get("pothole_ahead_m") is not None and vib["pothole_ahead_m"] < 80:
        score += 5; factors.append("rough road ahead (IMU)")

    # ── Vision: drowsiness / stress ──────────────────────────────────────
    vd = _latest["vision_driver"]
    if vd.get("face_detected"):
        if vd.get("drowsy"):
            score += 15; factors.append("drowsy eyes (camera)")
        elif vd.get("emotion") == "stressed":
            score += 10; factors.append("stress expression (camera)")

    # ── Vision: loose object ──────────────────────────────────────────────
    for det in _latest["vision_objects"].get("detections", []):
        if det["label"] in ("backpack", "suitcase", "bottle", "cup", "book", "laptop"):
            score += 6; factors.append(f"loose {det['label']} (camera)"); break

    return int(min(100, score)), factors


def _proposed() -> list[dict]:
    """Compute which mitigations should currently be offered."""
    out: list[dict] = []
    audio_label, audio_conf = _effective_audio()
    veh = _latest["vehicle"]
    d = _driver_seat()

    # --- USE CASE 1: audio comfort, corroborated by occupancy in ANY seat ---
    child_pet_seats = [
        s for s in _latest["radar"].get("seats", [])
        if s.get("occupant") in ("child", "pet") and s.get("occupied")
    ]
    seat_names = ", ".join(
        s["seat"].replace("_", " ").title() for s in child_pet_seats
    ) if child_pet_seats else ""

    if audio_label in ("crying", "animal", "barking") and audio_conf > 0.25 \
            and child_pet_seats:
        who = "child" if audio_label in ("crying",) else "pet"
        out.append(dict(
            id="comfort_audio", title="Soothe cabin",
            usecase="Audio comfort",
            detail=f"{who.title()} distress detected in {seat_names}. "
                   "Lower media volume 30%, warm AC +1°C, soft cabin lighting.",
            severity="advisory", confirm=True))

    elif audio_label == "talking" and audio_conf > 0.25 and child_pet_seats:
        out.append(dict(
            id="baby_engagement", title="Engage baby",
            usecase="Audio comfort",
            detail=f"Baby babbling in {seat_names}. Play a nursery rhyme or "
                   "soft music to maintain a calm, stimulating environment.",
            severity="advisory", confirm=True))

    elif audio_label == "happy" and audio_conf > 0.25 and child_pet_seats:
        out.append(dict(
            id="baby_happy", title="Passenger content",
            usecase="Cabin monitoring",
            detail=f"Baby sounds happy in {seat_names}. Maintain current "
                   "cabin temperature, lighting and media volume.",
            severity="advisory", confirm=True))

    # --- Shouting / passenger distress ---
    if audio_label == "shouting" and audio_conf > 0.25:
        out.append(dict(
            id="shouting_alert", title="Passenger distress",
            usecase="Audio comfort",
            detail="Shouting detected in cabin. Check passenger status and consider "
                   "pulling over if safe.",
            severity="warning", confirm=True))

    # --- USE CASE 2: driver persona tuning from vitals + emotion ---
    if d and d.get("occupied"):
        hr = d.get("heart_rate_bpm") or 72
        if hr > 95 or world.driver_emotion == "stressed":
            out.append(dict(
                id="persona_calm", title="Calming persona",
                usecase="Persona tuning",
                detail="Driver stress detected (HR ↑, breathing ↑). "
                       "Switch to calm playlist, cool blue lighting, temp −1°C.",
                severity="advisory", confirm=True))
        elif hr < 60 or world.driver_emotion == "tired":
            out.append(dict(
                id="persona_alert", title="Alertness boost",
                usecase="Persona tuning",
                detail="Fatigue signs detected. Upbeat playlist, brighter "
                       "lighting, fresh-air burst, suggest a break.",
                severity="advisory", confirm=True))

    # --- USE CASE 3a: seatbelt misuse ---
    for s in _latest["radar"].get("seats", []):
        if s.get("occupied") and not s.get("buckled"):
            out.append(dict(
                id=f"belt_{s['seat']}", title="Seatbelt not engaged",
                usecase="Seatbelt safety",
                detail=f"{s['seat'].replace('_', ' ').title()} occupied but "
                       "belt not properly worn. Chime + visual reminder.",
                severity="warning", confirm=False))

    # --- USE CASE 3b: pre-emptive object securing before rough road ---
    if world.unsecured_object and veh.get("pothole_ahead_m") is not None:
        dist = veh["pothole_ahead_m"]
        if dist <= 120:
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
        if dist <= 120 and not any(m["id"] == "secure_object" for m in out):
            out.append(dict(
                id="secure_object_cam", title="Secure loose item (camera)",
                usecase="Pothole-aware advisory",
                detail=f"{yolo_obj['label'].title()} detected by camera, rough road "
                       f"in ~{int(dist)} m. Secure it now.",
                severity="warning", confirm=True))

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


_GRACE = 6.0  # seconds to keep confirmed/dismissed mitigations after condition clears


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


def confirm(mitigation_id: str) -> None:
    if mitigation_id in _mitigations:
        _mitigations[mitigation_id]["status"] = "active"


def dismiss(mitigation_id: str) -> None:
    if mitigation_id in _mitigations:
        _mitigations[mitigation_id]["status"] = "dismissed"


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
            # Qwen sees someone here — enrich emotion + kind if unknown
            cfg["emotion"] = seat_v.get("emotion", cfg["emotion"])
            if cfg["kind"] == "unknown" and seat_v.get("kind"):
                cfg["kind"] = seat_v["kind"]
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
            vehicle=_latest["vehicle"],
            vision_driver=_latest["vision_driver"],
            vision_objects=_latest["vision_objects"],
            vibration=_latest["vibration"],
            mitigations=mitigations,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            # World-level config exposed so the HMI can reflect / edit it.
            # Qwen vision_all_seats overlays per-seat emotion when available.
            seat_configs=_build_seat_configs(),
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
