"""Vehicle / vibration context node.

Provides speed, predicted rough-road distance (the 'pothole ahead' signal that
enables *pre-emptive* object-securing advisories), and visibility. In a vehicle
this fuses CAN speed, GPS + a road-roughness map, and IMU. Synthetic here.
"""
from __future__ import annotations

import asyncio

from bus import bus
from schemas import VehicleContext
from world import world

HZ = 5


def _ctx() -> VehicleContext:
    # close the distance to a predicted pothole over time for a live feel
    if world.pothole_ahead_m is not None and world.speed_kmh > 0:
        step = world.speed_kmh / 3.6 / HZ  # metres per tick
        world.pothole_ahead_m = max(0.0, world.pothole_ahead_m - step)
    return VehicleContext(
        speed_kmh=round(world.speed_kmh, 1),
        pothole_ahead_m=(None if world.pothole_ahead_m is None
                         else round(world.pothole_ahead_m, 1)),
        visibility=world.visibility,
    )


async def run() -> None:
    while True:
        await bus.publish("vehicle", _ctx().to_dict())
        await asyncio.sleep(1 / HZ)
