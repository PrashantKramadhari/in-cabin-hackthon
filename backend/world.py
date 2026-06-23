"""Shared 'ground truth' cabin state that the synthetic sensor nodes read from.

Scenarios mutate this world; each sensor node then renders a realistic, noisy
stream consistent with it. This lets the demo inject situations on demand while
keeping every modality coherent (radar, audio, vibration all agree).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Occupant:
    occupied: bool = False
    kind: str = "unknown"      # adult | child | pet | unknown
    buckled: bool = False
    distress: float = 0.0      # 0..1 (crying child, agitated pet, ...)


@dataclass
class World:
    # per-seat occupants
    seats: dict[str, Occupant] = field(default_factory=lambda: {
        "driver": Occupant(occupied=True, kind="adult", buckled=True),
        "front_passenger": Occupant(),
        "rear_left": Occupant(),
        "rear_right": Occupant(),
    })
    # driver physiology (drives cognitive-load estimate)
    driver_hr: float = 72.0          # bpm baseline
    driver_resp: float = 14.0        # breaths/min baseline
    driver_emotion: str = "calm"     # calm | stressed | tired | happy

    # acoustic environment
    audio_label: str = "none"        # crying | animal | rattle | speech | none
    audio_conf: float = 0.0

    # vehicle / road
    speed_kmh: float = 60.0
    pothole_ahead_m: float | None = None
    visibility: str = "good"

    # loose objects
    unsecured_object: bool = False
    object_motion: float = 0.0       # 0..1

    scenario: str = "idle"


# single shared world
world = World()
