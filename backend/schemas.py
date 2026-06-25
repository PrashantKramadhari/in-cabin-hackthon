"""Shared event/state shapes passed over the bus.

Kept as plain dicts on the wire; these dataclasses document the contract and
provide constructors so nodes stay consistent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Literal

SeatId = Literal["driver", "front_passenger", "rear_left", "rear_middle", "rear_right"]


@dataclass
class SeatState:
    seat: str
    occupied: bool
    occupant: str = "unknown"          # adult | child | pet | unknown
    buckled: bool = False
    respiration_rpm: float | None = None   # breaths/min (radar micro-doppler)
    heart_rate_bpm: float | None = None    # bpm (radar micro-doppler)
    motion: float = 0.0                # 0..1 micro-motion / displacement energy


@dataclass
class RadarFrame:
    ts: float = field(default_factory=time.time)
    seats: list[SeatState] = field(default_factory=list)
    point_count: int = 0
    source: str = "radar"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AudioEvent:
    ts: float = field(default_factory=time.time)
    label: str = "none"                # crying | animal | rattle | speech | none
    confidence: float = 0.0
    source: str = "audio"
    distress_class: str = "none"       # kid | human | pet | vehicle | none (zero-shot)
    prompt: str = ""                 # winning CLAP text prompt (debug)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VehicleContext:
    ts: float = field(default_factory=time.time)
    speed_kmh: float = 0.0
    pothole_ahead_m: float | None = None   # distance to predicted rough patch
    visibility: str = "good"               # good | low
    source: str = "vehicle"

    def to_dict(self) -> dict:
        return asdict(self)
