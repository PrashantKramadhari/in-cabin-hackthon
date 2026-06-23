"""IMU / vibration replay node.

Replays a 50 Hz accelerometer CSV (columns: timestamp_ms, ax, ay, az,
road_label) in real time. At each tick it:
  1. Computes a 0.5 s RMS of the z-axis (vertical shock) as road roughness.
  2. Classifies road quality: smooth / rough / pothole.
  3. When a rough segment is detected ahead (look-ahead window), sets
     world.pothole_ahead_m so the vehicle node counts it down to zero —
     enabling the *pre-emptive* object-securing advisory before the bump hits.

Publishes to "vibration" bus topic; fusion.py also subscribes to update the
cognitive-load score with real road context.

To use real hardware: swap the CSV reader for a serial read from an
IWR6843 IMU output or a phone sensor stream; keep the publish contract.
"""
from __future__ import annotations

import asyncio
import csv
import math
import pathlib
import time
from collections import deque

from bus import bus
from world import world

CSV_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "route_imu.csv"
HZ = 50
WINDOW = int(HZ * 0.5)      # 0.5 s RMS window
LOOKAHEAD_ROWS = HZ * 4     # 4 s ahead: how far we look for rough patches

RMS_ROUGH   = 0.18          # z-axis RMS threshold for "rough"
RMS_POTHOLE = 0.55          # z-axis RMS threshold for "pothole"
POTHOLE_WARN_M = 100.0      # distance (m) to start warning


def _load() -> list[dict]:
    rows = []
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "ts_ms": int(r["timestamp_ms"]),
                "ax": float(r["ax"]),
                "ay": float(r["ay"]),
                "az": float(r["az"]),
                "label": r["road_label"],
            })
    return rows


def _rms(buf: deque) -> float:
    if not buf:
        return 0.0
    g = 9.81
    return math.sqrt(sum((v - g) ** 2 for v in buf) / len(buf))


def _lookahead_rms(rows: list[dict], idx: int) -> float:
    end = min(idx + LOOKAHEAD_ROWS, len(rows))
    vals = [r["az"] for r in rows[idx:end]]
    if not vals:
        return 0.0
    g = 9.81
    return math.sqrt(sum((v - g) ** 2 for v in vals) / len(vals))


async def run() -> None:
    rows = _load()
    buf: deque[float] = deque(maxlen=WINDOW)
    speed = world.speed_kmh or 60.0        # m/s reference for distance estimate
    idx = 0

    while True:
        loop_start = time.perf_counter()

        if idx >= len(rows):
            idx = 0             # loop the route

        row = rows[idx]
        buf.append(row["az"])
        current_rms = _rms(buf)

        # classify current road quality
        if current_rms >= RMS_POTHOLE:
            quality = "pothole"
        elif current_rms >= RMS_ROUGH:
            quality = "rough"
        else:
            quality = "smooth"

        # look ahead to warn before the rough patch arrives
        ahead_rms = _lookahead_rms(rows, idx)
        if ahead_rms >= RMS_ROUGH:
            # estimate distance: lookahead window duration × current speed
            lookahead_s = LOOKAHEAD_ROWS / HZ
            speed_ms = (world.speed_kmh or 60) / 3.6
            dist_m = round(lookahead_s * speed_ms, 1)
            if world.pothole_ahead_m is None or world.pothole_ahead_m > dist_m:
                world.pothole_ahead_m = dist_m
        elif quality == "smooth" and world.pothole_ahead_m is not None and world.pothole_ahead_m < 5:
            world.pothole_ahead_m = None  # clear after we've passed

        await bus.publish("vibration", {
            "ts": time.time(),
            "ax": row["ax"], "ay": row["ay"], "az": row["az"],
            "rms_z": round(current_rms, 4),
            "road_quality": quality,
            "pothole_ahead_m": world.pothole_ahead_m,
            "source": "imu_replay",
        })

        idx += 1
        # pace to real time (50 Hz)
        elapsed = time.perf_counter() - loop_start
        await asyncio.sleep(max(0, 1 / HZ - elapsed))
