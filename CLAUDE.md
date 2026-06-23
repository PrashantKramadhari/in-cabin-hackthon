# CabinSense — Project Plan & Guide

> **CabinSense**: an in-cabin disturbance-intelligence layer that sits beside
> ADAS/DMS. It fuses multi-modal sensing into a single **Cognitive-Load /
> Attention-Risk** score and drives **non-intrusive, confirm-first** cabin
> mitigations. Hackathon build — 2 days.

## 1. The winning framing
The 5 raw ideas are **not** 5 features — they are use cases riding on **one
pipeline**:

```
sense (multi-modal) → detect anomaly → fuse with context → adaptive feedback
```

This single-platform story directly answers the PS "critical gap" (drivers face
continuous, unmonitored micro-distractions with no intelligent support) and
scores on Innovation, Driver-Centric design, Feasibility, Safety, and Business
value at once.

## 2. Architecture
```
 SENSING NODES            FUSION / DECISION ENGINE          FEEDBACK / HMI
  Radar (mmWave)  ─┐
  Mic   (audio)   ─┤      anomaly detectors                 Cabin HMI panel
  Cam   (vision)  ─┼─bus─▶  + context fuser           ─────▶ ANC / volume / AC
  IMU   (vibration)┤        → Cognitive-Load score          lighting / playlist
  Vehicle / GPS   ─┘        → chosen mitigation             warnings (confirm-first)
```
The edge-node + bus + decision-engine topology mirrors a software-defined
vehicle, so the same design maps onto embedded automotive hardware (scalability
+ feasibility points). Fusion latency measured at <1 ms (goal: <200 ms).

## 3. MVP scope (3 use cases, one engine) — radar as a first-class sensor
1. **Audio comfort** — crying/animal (mic) corroborated by **radar occupancy +
   vitals** → lower volume, adjust AC, soft lighting.
2. **Persona tuning** — **radar contactless vitals** (HR/breathing) + emotion →
   calming or alerting playlist/lighting/temperature.
3. **Seatbelt misuse + unsecured object + pothole-aware advisory** — radar
   occupancy-vs-buckle mismatch; object micro-motion; rough-road-ahead →
   advise securing items *before* the pothole.
Bonus safety: **child-presence / left-behind** alert (radar) — high-impact, cheap.
Idea #4 ("ask before acting") = the **confirm-first UX** on every mitigation.

### Why radar
Production-relevant (TI IWR6843/AWR class), privacy-preserving (no camera for
sensitive detections), and the only sensor giving contactless driver
vitals — a *direct* cognitive-load signal, which is literally the PS title.

## 4. Repo layout
```
backend/
  bus.py          async pub/sub message bus (drops oldest -> bounded latency)
  schemas.py      RadarFrame / AudioEvent / VehicleContext contracts
  world.py        shared 'ground truth' cabin state
  scenarios.py    scripted demo situations that mutate the world
  sensors/
    radar.py      synthetic mmWave node: occupancy + vitals + micro-motion
    audio.py      audio-event node (synthetic; YAMNet hook documented)
    vehicle.py    speed / pothole-ahead / visibility node
  fusion.py       cognitive-load score + mitigation rules + confirm state
  main.py         FastAPI: websocket stream + scenario/confirm commands + HMI
frontend/
  index.html      React Cabin HMI (CDN React, zero-build) — gauge, cabin map,
                  radar vitals, acoustic panel, mitigation cards, scenarios
```

## 5. Run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
# open http://localhost:8000
```
Click a scenario button → watch the cabin map, cognitive-load gauge, and
mitigation cards react. "Apply"/"Dismiss" exercise the confirm-first flow.

## 6. Two-day timeline
**Day 1 (DONE)** — bus, websocket server, radar/audio/vehicle nodes, fusion
engine (cognitive-load score + mitigations + confirm state), React HMI, all 7
demo scenarios verified end-to-end.

**Day 2 (TODO)**
- [ ] Replace synthetic audio with **YAMNet** live mic inference (crying/animal/rattle).
- [ ] Add **vision node**: MediaPipe face (emotion/drowsiness) + YOLOv8n
      (seatbelt/box) — optional, graceful-degrade if camera absent.
- [ ] **Vibration replay**: drive `pothole_ahead` from a recorded IMU/route CSV.
- [ ] **Demo timeline button**: auto-play the full story (idle → crying →
      stress → pothole → seatbelt) hands-free for judging.
- [ ] **Pitch deck** (deliverables 5.1–5.3): architecture slide, the
      one-pipeline framing, radar differentiation, safety + business value.
- [ ] HMI polish: small animations, ANC/“applied” confirmation toast.

## 7. Mapping to PS scoring
- **Innovation** — contactless radar vitals as a cognitive-load signal + one
  unifying fusion engine + *pre-emptive* object-securing.
- **Driver-centric** — confirm-first, non-intrusive; nothing auto-acts without
  consent.
- **Feasibility / cost** — reuses existing/near-term cabin sensors; SDV-aligned
  bus topology; <1 ms fusion.
- **Safety** — micro-distraction mitigation + child-presence alert.
- **Business** — premium/EV cabin refinement, ADAS expansion to internal
  awareness, upgradeable SDV feature.

## Conventions for future work
- Sensor nodes publish dicts to the bus; never block — keep each node's loop at
  its `HZ`. To go to real hardware, swap only the node body; keep the schema.
- Fusion stays transparent/rule-based for the demo; an ML scorer can replace
  `_cognitive_load()` behind the same interface.
- HMI is intentionally build-free (CDN React) for hackathon speed; migrate to
  Vite only if the UI grows.
