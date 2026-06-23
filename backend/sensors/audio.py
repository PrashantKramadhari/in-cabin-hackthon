"""Audio anomaly node.

Synthetic for the demo (renders world.audio_label with jittered confidence).
To go live, replace `_event()` with a YAMNet inference loop over the mic:
classes of interest -> {Crying/sobbing, Cat, Dog, Rattle, Speech}.
The bus contract (AudioEvent) stays the same.
"""
from __future__ import annotations

import asyncio
import random

from bus import bus
from schemas import AudioEvent
from world import world

HZ = 5


def _event() -> AudioEvent:
    if world.audio_label == "none":
        return AudioEvent(label="none", confidence=0.0)
    conf = max(0.0, min(1.0, world.audio_conf + random.uniform(-0.05, 0.05)))
    return AudioEvent(label=world.audio_label, confidence=round(conf, 2))


async def run() -> None:
    while True:
        await bus.publish("audio", _event().to_dict())
        await asyncio.sleep(1 / HZ)
