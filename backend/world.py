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
    kind: str = "unknown"       # adult | child | pet | unknown
    buckled: bool = False
    distress: float = 0.0       # 0..1 (crying child, agitated pet, ...)
    audio_event: str = "none"   # none | talking | shouting | crying | happy | barking
    heart_rate_bpm: float | None = None   # None = auto-derived from kind
    respiration_rpm: float | None = None  # None = auto-derived from kind
    emotion: str = "calm"       # calm | stressed | tired | happy (driver only used in fusion)


@dataclass
class World:
    # per-seat occupants
    seats: dict[str, Occupant] = field(default_factory=lambda: {
        "driver": Occupant(occupied=True, kind="adult", buckled=True),
        "front_passenger": Occupant(),
        "rear_left": Occupant(),
        "rear_middle": Occupant(),
        "rear_right": Occupant(),
    })
    # driver physiology (drives cognitive-load estimate)
    driver_hr: float = 72.0          # bpm baseline
    driver_resp: float = 14.0        # breaths/min baseline
    driver_emotion: str = "calm"     # calm | stressed | tired | happy

    # acoustic environment (synthesised from per-seat audio_events)
    audio_label: str = "none"
    audio_conf: float = 0.0

    # vehicle / road
    speed_kmh: float = 60.0
    pothole_ahead_m: float | None = None
    visibility: str = "good"

    # loose objects
    unsecured_object: bool = False
    object_motion: float = 0.0

    # vibration override (user-controlled via HMI)
    vib_override: bool = False
    vib_road_quality: str = "smooth"   # smooth | rough | pothole
    vib_rms: float = 0.05

    scenario: str = "idle"
    demo_running: bool = False
    seat_overrides: dict = field(default_factory=dict)


# single shared world
world = World()
