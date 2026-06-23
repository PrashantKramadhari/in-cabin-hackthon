"""Synthetic mmWave in-cabin radar node.

Renders a realistic radar frame from world ground truth: per-seat occupancy,
contactless vitals (respiration + heart rate via micro-doppler), and
micro-motion energy. In a production cabin this node would parse point clouds +
vital signs over UART from an IWR6843/AWR-class sensor; the contract on the bus
is identical, so only this file changes for real hardware.
"""
from __future__ import annotations

import asyncio
import random

from bus import bus
from schemas import RadarFrame, SeatState
from world import world

HZ = 10  # radar reporting rate


def _jitter(x: float, pct: float = 0.05) -> float:
    return round(x * (1 + random.uniform(-pct, pct)), 1)


def _frame() -> RadarFrame:
    seats: list[SeatState] = []
    points = 0
    for seat_id, occ in world.seats.items():
        if not occ.occupied:
            seats.append(SeatState(seat=seat_id, occupied=False))
            continue
        points += random.randint(12, 40)
        if seat_id == "driver":
            resp = _jitter(world.driver_resp)
            hr = _jitter(world.driver_hr)
        elif occ.kind == "child":
            # children breathe faster; distress raises it further
            resp = _jitter(24 + occ.distress * 12)
            hr = _jitter(100 + occ.distress * 25)
        elif occ.kind == "pet":
            resp = _jitter(30 + occ.distress * 20)
            hr = _jitter(110 + occ.distress * 30)
        else:
            resp = _jitter(15)
            hr = _jitter(75)
        motion = round(min(1.0, occ.distress * 0.6 +
                           (world.object_motion if seat_id == "driver" else 0) +
                           random.uniform(0, 0.08)), 2)
        seats.append(SeatState(
            seat=seat_id, occupied=True, occupant=occ.kind,
            buckled=occ.buckled, respiration_rpm=resp,
            heart_rate_bpm=hr, motion=motion))
    return RadarFrame(seats=seats, point_count=points)


async def run() -> None:
    while True:
        await bus.publish("radar", _frame().to_dict())
        await asyncio.sleep(1 / HZ)
