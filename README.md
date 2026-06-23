# CabinSense — In-Cabin Disturbance Intelligence

An intelligent in-cabin anomaly detection & mitigation layer that sits alongside
ADAS/DMS. It fuses **multi-modal sensing** (radar, audio, vision, vibration,
vehicle context) into a single **Cognitive-Load / Attention-Risk score** and
drives **non-intrusive, confirm-first** cabin mitigations.

> One pipeline, many use cases:
> **sense → detect anomaly → fuse with context → adaptive feedback.**

## Architecture

```
 SENSING NODES            FUSION / DECISION ENGINE          FEEDBACK / HMI
  Radar (mmWave)  ─┐
  Mic   (audio)   ─┤      anomaly detectors                 Cabin HMI panel
  Cam   (vision)  ─┼─bus─▶  + context fuser           ─────▶ ANC / volume / AC
  IMU   (vibration)┤        → Cognitive-Load score          lighting / playlist
  Vehicle / GPS   ─┘        → chosen mitigation             warnings (confirm-first)
```

## MVP use cases (all through one engine)
1. **Audio comfort** — crying/animal detected (mic) + corroborated by radar
   occupancy/vitals → lower volume, adjust AC.
2. **Driver persona tuning** — radar vitals (HR/breathing) + face emotion →
   playlist / lighting / temperature.
3. **Seatbelt misuse + unsecured object + pothole-aware advisory** — seat
   occupancy vs. buckle mismatch; object micro-motion; vibration-ahead → advise
   securing objects *before* the pothole.

Every mitigation is **confirm-first** (idea #4): non-intrusive, driver stays in control.

## Run (backend)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://localhost:8000/  (health) and ws://localhost:8000/ws (stream)
```

## Layout
```
backend/
  bus.py        # async pub/sub message bus
  schemas.py    # shared event/state dataclasses
  fusion.py     # cognitive-load score + mitigation rules
  sensors/
    radar.py    # synthetic mmWave radar node (occupancy + vitals)
  main.py       # FastAPI + websocket + scenario injection
frontend/       # React Cabin HMI (Day 1 PM)
```
