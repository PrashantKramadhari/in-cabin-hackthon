"""Scripted demo scenarios. Each sets the world ground truth so all sensor
nodes render a coherent situation. Triggered from the HMI for a clean demo."""
from __future__ import annotations

from world import World, Occupant, world


def _reset(w: World) -> None:
    w.seats = {
        "driver": Occupant(occupied=True, kind="adult", buckled=True),
        "front_passenger": Occupant(),
        "rear_left": Occupant(),
        "rear_right": Occupant(),
    }
    w.driver_hr = 72.0
    w.driver_resp = 14.0
    w.driver_emotion = "calm"
    w.audio_label = "none"
    w.audio_conf = 0.0
    w.speed_kmh = 60.0
    w.pothole_ahead_m = None
    w.visibility = "good"
    w.unsecured_object = False
    w.object_motion = 0.0


def apply(name: str) -> None:
    w = world
    _reset(w)
    w.scenario = name

    if name == "idle":
        return

    if name == "child_crying_rear":
        # USE CASE 1: audio anomaly corroborated by radar occupancy + vitals
        w.seats["rear_right"] = Occupant(
            occupied=True, kind="child", buckled=True, distress=0.85)
        w.audio_label = "crying"
        w.audio_conf = 0.92

    elif name == "pet_agitated":
        w.seats["rear_left"] = Occupant(
            occupied=True, kind="pet", buckled=False, distress=0.7)
        w.audio_label = "animal"
        w.audio_conf = 0.8

    elif name == "driver_stress":
        # USE CASE 2: radar vitals + emotion -> persona tuning
        w.driver_hr = 104.0
        w.driver_resp = 22.0
        w.driver_emotion = "stressed"
        w.visibility = "low"

    elif name == "driver_tired":
        w.driver_hr = 58.0
        w.driver_resp = 10.0
        w.driver_emotion = "tired"

    elif name == "seatbelt_misuse":
        # USE CASE 3a: occupied but not properly buckled
        w.seats["driver"] = Occupant(
            occupied=True, kind="adult", buckled=False)

    elif name == "pothole_object":
        # USE CASE 3b: unsecured object + rough road predicted ahead
        w.unsecured_object = True
        w.object_motion = 0.55
        w.pothole_ahead_m = 80.0
        w.speed_kmh = 75.0

    elif name == "child_left_behind":
        # safety-critical: rear child present, no driver
        w.seats["driver"] = Occupant(occupied=False)
        w.seats["rear_right"] = Occupant(
            occupied=True, kind="child", buckled=True, distress=0.4)
        w.speed_kmh = 0.0


SCENARIOS = [
    "idle",
    "child_crying_rear",
    "pet_agitated",
    "driver_stress",
    "driver_tired",
    "seatbelt_misuse",
    "pothole_object",
    "child_left_behind",
]
