# CabinSense — Technical Reference

> In-Cabin Disturbance Intelligence for Driver Cognitive-Load Reduction  
> Hackathon build · June 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Component Deep-Dive](#3-component-deep-dive)
   - 3.1 [Async Message Bus](#31-async-message-bus-buspy)
   - 3.2 [Shared World State](#32-shared-world-state-worldpy)
   - 3.3 [Sensor Nodes](#33-sensor-nodes)
   - 3.4 [Fusion & Decision Engine](#34-fusion--decision-engine-fusionpy)
   - 3.5 [FastAPI Server & WebSocket Layer](#35-fastapi-server--websocket-layer-mainpy)
   - 3.6 [React HMI Frontend](#36-react-hmi-frontend)
4. [Q1 — How Acoustic Anomaly Works](#4-q1--how-acoustic-anomaly-works)
5. [Q2 — All Factors in Cognitive Load Calculation](#5-q2--all-factors-in-cognitive-load-calculation)
6. [Q3 — Cognitive Scoring Criteria per Disturbance Attribute](#6-q3--cognitive-scoring-criteria-per-disturbance-attribute)
7. [Data Flow: End-to-End Walk-Through](#7-data-flow-end-to-end-walk-through)
8. [Demo Scenarios](#8-demo-scenarios)
9. [Adaptive Mitigations Reference](#9-adaptive-mitigations-reference)
10. [Repository Layout](#10-repository-layout)

---

## 1. System Overview

CabinSense is a **multi-modal in-cabin intelligence layer** that sits alongside ADAS/DMS systems. It continuously fuses data from five sensing modalities into a single **Cognitive-Load / Attention-Risk score (0–100)** and surfaces **non-intrusive, confirm-first mitigations** to the driver.

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Non-intrusive** | Every mitigation is *proposed*, not auto-applied. Driver must tap Apply. |
| **Privacy-preserving** | Radar handles sensitive detections (vitals, child-left-behind) without a camera. |
| **Real-time** | Fusion loop runs at 10 Hz; target end-to-end latency < 200 ms (actual < 1 ms on CPU). |
| **Graceful degradation** | Each sensor node fails independently; fusion works with whatever is live. |
| **SDV-aligned** | Pub/sub bus topology mirrors an automotive signal bus; swapping any node to real hardware requires changing only that node file. |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SENSING NODES  (each an async coroutine, independent Hz)               │
│                                                                         │
│  radar.py         → "radar"           topic  (10 Hz)  mmWave vitals    │
│  audio_yamnet.py  → "audio"           topic  (on-demand) AST/mic       │
│  vehicle.py       → "vehicle"         topic  (5 Hz)   speed/visibility │
│  vibration.py     → "vibration"       topic  (50 Hz)  IMU RMS          │
│  vision.py        → "vision_driver", "vision_objects"  (5 fps webcam)  │
│  qwen_vision.py   → "vision_all_seats","vision_driver" (per-frame GPU) │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │  async pub/sub  (bus.py — bounded queue)
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FUSION / DECISION ENGINE  (fusion.py, 10 Hz)                           │
│                                                                         │
│  _latest[topic]  ←  subscribe to all sensor topics                     │
│  _build_seat_configs()   →  merge world.seats + radar + vision         │
│  _cognitive_load()       →  score 0–100 + contributing factors list    │
│  _proposed()             →  active mitigation cards                    │
│  _reconcile()            →  merge confirm/dismiss state across ticks   │
│                                                                         │
│  Publishes fused state snapshot → "fused" topic                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FASTAPI SERVER  (main.py)                                              │
│                                                                         │
│  GET  /               →  serves frontend/index.html                    │
│  GET  /deck           →  serves pitch deck (frontend/deck.html)        │
│  GET  /api/scenarios  →  list of scenario names                        │
│  GET  /api/caps       →  live_audio / live_vision flags                │
│  POST /api/demo/play  →  start 8-step auto-play demo                   │
│  POST /api/demo/stop  →  cancel running demo (< 300 ms)                │
│  WS   /ws             →  bidirectional: streams fused state,           │
│                           accepts control commands                      │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │  WebSocket (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  REACT HMI  (frontend/index.html — CDN-free, zero build step)          │
│                                                                         │
│  Sensor View (3-column grid):                                           │
│   Col 1: Video | Vehicle + Road Quality                                 │
│   Col 2: Audio | Seat Status (compact) + IMU                           │
│   Col 3: Radar panel (full height, kind selector + HR sliders)         │
│  Timeline bar: mode indicator + scenario selector + auto-play          │
│  Infotainment View: cognitive-load gauge, live issues, mitigations     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Threading / Concurrency Model

The entire backend is single-process, single-thread, built on **Python asyncio**. All sensor nodes, the fusion engine, and the WebSocket pump are independent `asyncio.Task` coroutines running on the same event loop. Model inference (AST audio, Qwen vision) runs in a thread-pool executor via `loop.run_in_executor`.

---

## 3. Component Deep-Dive

### 3.1 Async Message Bus (`bus.py`)

A minimal in-process pub/sub that mirrors an automotive CAN/SOME-IP signal bus.

```
Bus
 ├─ subscribe(topic) → asyncio.Queue(maxsize=32)
 ├─ publish(topic, payload) → fans out to all subscriber queues
 │    └─ if queue full: drop oldest message to keep latency bounded
 └─ stream(topic) → async generator (wraps subscribe)
```

**Key design choice:** when a queue is full, the *oldest* message is dropped rather than blocking the publisher. This keeps end-to-end fusion latency bounded regardless of how fast any individual sensor publishes.

---

### 3.2 Shared World State (`world.py`)

A single Python dataclass instance (`world`) that acts as the **ground truth** for the synthetic sensor nodes. Scenarios and the HMI seat configurator mutate it; sensor nodes read from it to produce realistic, coherent streams.

```python
@dataclass
class Occupant:
    occupied: bool          # seat is occupied
    kind: str               # adult | child | pet | unknown
    buckled: bool           # seatbelt engaged
    distress: float         # 0.0–1.0 intensity of distress
    audio_event: str        # none | crying | happy | shouting | barking | talking
    heart_rate_bpm: float   # override (None = auto-derived from kind)
    respiration_rpm: float  # override (None = auto-derived from kind)
    emotion: str            # calm | stressed | tired | happy (driver only used in fusion)

@dataclass
class World:
    seats: dict[str, Occupant]  # driver, front_passenger, rear_left, rear_right
    driver_hr: float            # bpm baseline (used by radar node)
    driver_resp: float          # breaths/min baseline
    driver_emotion: str         # calm | stressed | tired | happy
    audio_label: str            # synthesised from per-seat audio_events
    audio_conf: float           # confidence 0.0–1.0
    speed_kmh: float
    pothole_ahead_m: float | None
    visibility: str             # good | low | rain | fog
    unsecured_object: bool
    object_motion: float        # 0.0–1.0
    vib_override: bool          # True = use manual vib settings instead of CSV
    vib_road_quality: str       # smooth | rough | pothole
    vib_rms: float              # override RMS value
    scenario: str               # current scenario name
    demo_running: bool          # auto-play demo active flag
```

When the HMI configures a seat, `_sync_audio_from_seats()` (in `main.py`) picks the highest-priority audio event across all occupied seats and writes it to `world.audio_label`. This is how per-seat audio config drives the global audio state.

**Audio event priority order (highest → lowest):**

| Event | Priority |
|-------|----------|
| `crying` | 5 |
| `barking` | 4 |
| `shouting` | 3 |
| `talking` | 2 |
| `happy` | 1 |
| `none` | 0 |

---

### 3.3 Sensor Nodes

#### Radar Node (`sensors/radar.py`) — 10 Hz

Simulates a **TI IWR6843/AWR-class mmWave radar** doing in-cabin occupancy + contactless vital signs via micro-Doppler.

For each occupied seat, it derives heart rate and respiration from world state, applies ±5% Gaussian jitter to simulate sensor noise, and computes a `motion` value that represents micro-displacement energy:

```
motion = distress × 0.6 + object_motion (driver only) + random(0, 0.08)
```

**Per-seat vital sign defaults (when no override set):**

| Seat / Kind | HR baseline | Resp baseline | Notes |
|-------------|-------------|---------------|-------|
| Driver | `world.driver_hr` (72) | `world.driver_resp` (14) | Emotion/state driven |
| Adult passenger | 75 bpm | 15 rpm | Calm baseline |
| Child | 100 + distress×25 bpm | 24 + distress×12 rpm | Higher at rest; distress raises further |
| Pet | 110 + distress×30 bpm | 30 + distress×20 rpm | Higher metabolic rate |

If `occ.heart_rate_bpm` or `occ.respiration_rpm` are set (via HMI seat config or the Radar panel HR slider), those values override the defaults before jitter is applied.

Published schema: `RadarFrame { seats: [SeatState], point_count: int }`  
`SeatState { seat, occupied, occupant, buckled, respiration_rpm, heart_rate_bpm, motion }`

---

#### Audio Node — Two Modes

##### Mode A: Live — AST Model (`sensors/audio_yamnet.py`)

Active when Python packages `transformers` + `torch` are available (`live_audio: true` from `/api/caps`).

**Pipeline:**
```
Browser mic (16 kHz)
  → AudioWorklet (0.96 s Float32 chunks at 15 360 samples)
  → WebSocket → chunk_queue (maxsize 8)
  → thread-pool executor
      → ASTFeatureExtractor (mel spectrogram, 128 bins)
      → ASTForAudioClassification (MIT/ast-finetuned-audioset-10-10-0.4593)
          80 M parameter model, ~40 ms GPU inference
      → softmax over 527 AudioSet classes
      → filter to CabinSense vocab (6 classes):
```

| AudioSet class index | Label published |
|----------------------|-----------------|
| 22 (Baby cry / infant cry) | `crying` |
| 21 (Crying / sobbing) | `crying` |
| 80 (Cat) | `animal` |
| 74 (Dog) | `animal` |
| 73 (Animal generic) | `animal` |
| 468 (Rattle) | `rattle` |
| 0 (Speech) | `speech` |

Confidence threshold: **0.25** — below this, label is `none`.

Model runs in `loop.run_in_executor(None, _infer, samples)` — off the asyncio event loop to avoid blocking sensor ticks. The `chunk_queue` has a maxsize of 8; if the model can't keep up, older chunks are dropped rather than backing up indefinitely.

##### Mode B: Synthetic (`sensors/audio.py`)

Used when live audio is unavailable. Simply reads `world.audio_label` and `world.audio_conf`, adds ±0.05 uniform noise to confidence, and publishes at 5 Hz.

---

#### Vehicle Node (`sensors/vehicle.py`) — 5 Hz

Reads `world.speed_kmh`, `world.visibility`, `world.pothole_ahead_m`. On each tick it counts down `pothole_ahead_m` by the distance the vehicle travels in one tick (`speed_kmh / 3.6 / 5 metres`), simulating the vehicle approaching the rough patch in real time.

Published schema: `VehicleContext { speed_kmh, pothole_ahead_m, visibility }`

---

#### Vibration Node (`sensors/vibration.py`) — 50 Hz

Two operating modes:

**CSV Replay mode (default):** Replays `backend/data/route_imu.csv` — a 50 Hz accelerometer recording with columns `(timestamp_ms, ax, ay, az, road_label)`. It loops continuously. A 0.5-second RMS window of the z-axis (vertical shock) classifies the current road quality:

| RMS z-axis (g) | Classification |
|----------------|----------------|
| < 0.18 | `smooth` |
| 0.18 – 0.55 | `rough` |
| ≥ 0.55 | `pothole` |

A 4-second look-ahead window (200 rows) scans upcoming rows; if that window has rough RMS ≥ 0.18, `world.pothole_ahead_m` is set to the estimated distance so the vehicle node can count it down and the fusion engine can issue the pre-emptive advisory.

**Manual Override mode:** When `world.vib_override = True` (toggled from the HMI IMU panel), the CSV is bypassed entirely and the node publishes `world.vib_road_quality` and `world.vib_rms` directly at 50 Hz.

---

#### Vision Node — Two Modes

##### Mode A: Qwen2-VL Live Vision (`sensors/qwen_vision.py`) — per frame, GPU

Active when `qwen_vision: true` is set in `main.py`. Runs **Qwen2-VL-2B-Instruct** (4-bit quantized via BitsAndBytes, ~2.5 GB VRAM, ~100–150 ms on RTX 3080).

Browser captures webcam frames (JPEG), sends via WebSocket to `frame_queue`. The model receives the frame and a structured prompt asking it to analyse all four seats, returning a JSON object:

```json
{
  "driver":            {"occupied": bool, "kind": "adult|child|infant|pet|unknown", "emotion": "calm|happy|stressed|tired|distressed", "buckled": bool},
  "front_passenger":   {...},
  "rear_left":         {...},
  "rear_right":        {...}
}
```

Publishes two bus topics:
- `"vision_all_seats"` — full per-seat dict for use in `_build_seat_configs()`
- `"vision_driver"` — legacy compat (drowsy/emotion from driver seat)

**Vision override guard:** Qwen inferences only update a seat's `kind` when the current world state has `kind in ("unknown", "")`. If the user has manually configured a seat as "child", "pet", or "adult", the vision model will not override that choice.

##### Mode B: MediaPipe + YOLOv8n (`sensors/vision.py`) — ~5 fps

Active when `mediapipe` + `ultralytics` are installed (but Qwen is not available).

1. **MediaPipe FaceLandmarker:** Detects 478 facial landmarks. Computes **Eye Aspect Ratio (EAR)** from 6 points per eye: `EAR = (v1 + v2) / (2 × horizontal)`. EAR < 0.20 → `drowsy: true`. Emotion approximated from mouth-open ratio.

2. **YOLOv8n:** Detects loose objects (backpack, suitcase, bottle, cup, book, laptop). Informs the pothole-aware advisory mitigation.

Publishes to:
- `"vision_driver"` → `{face_detected, ear, drowsy, emotion, mouth_ratio}`
- `"vision_objects"` → `{detections: [{label, confidence, box}]}`

---

### 3.4 Fusion & Decision Engine (`fusion.py`)

The heart of the system. Runs at **10 Hz** as an asyncio coroutine. Maintains `_latest[topic]` — one cached dict per sensor topic, updated whenever a new frame arrives on the bus.

Each tick:
1. Calls `_build_seat_configs()` → merges `world.seats` + radar + vision into per-seat config dict
2. Calls `_cognitive_load()` → `(score: int, factors: list[str])`
3. Calls `_proposed()` → list of mitigation dicts
4. Calls `_reconcile(proposed)` → merges with stored confirm/dismiss state
5. Publishes the full fused state snapshot to the `"fused"` bus topic
6. The WebSocket pump reads from `"fused"` and pushes to all connected browser clients

#### `_build_seat_configs()` — Seat State Merging

Merges three data sources per seat, with explicit priority:

```
1. world.seats (Occupant dataclass) — always current, mutated by configure_seat
2. radar SeatState — motion, radar-derived HR/resp (stale by up to 100ms)
3. vision_all_seats — kind/emotion from Qwen (only if kind is unknown)
```

Priority: `world.seats` wins for `kind`, `occupied`, `heart_rate_bpm`, `respiration_rpm`. Vision only fills in kind when `occ.kind in ("unknown", "")`.

#### `_effective_audio()` — Hybrid Audio Resolution

This function resolves which audio label to use for cognitive load and mitigation calculations:

```
IF sensor audio (from AST model) has label ≠ "none" AND confidence > 0.40:
    USE sensor audio   ← real mic detection takes priority

ELSE IF world.audio_label ≠ "none":
    USE world.audio_label / world.audio_conf   ← seat config fallback

ELSE:
    label = "none", conf = 0.0
```

#### `_cognitive_load()` — Scoring Data Source Priority

**Critical design:** `world.seats` is read directly rather than relying on radar's `_latest["radar"]`, which can be stale by up to 100 ms after a `configure_seat` command. This ensures the score updates on the *first* fusion tick after any manual configuration change.

A `_scored_hr_seats` set tracks which seats have had HR scored from `world.seats`, preventing double-counting when the radar loop also processes those seats.

#### `_proposed()` — Mitigation Data Source

Same principle: `child_pet_seat_ids` is built from the **union** of `world.seats` and radar data, so mitigations fire immediately on the first tick after any seat configuration, even before the radar node republishes.

Seatbelt check reads from `world.seats.buckled` directly (not from radar), for the same reason.

---

### 3.5 FastAPI Server & WebSocket Layer (`main.py`)

#### WebSocket Command Protocol

The `/ws` endpoint is bidirectional. The browser sends JSON commands; the server streams fused state back at 10 Hz.

**Inbound commands (browser → server):**

| `cmd` | Required fields | Effect |
|-------|----------------|--------|
| `scenario` | `name` | Cancel any running demo; apply named scenario |
| `configure_seat` | `seat` + any subset of: `occupied`, `kind`, `buckled`, `distress`, `audio_event`, `heart_rate_bpm`, `respiration_rpm`, `emotion` | Mutate that seat's Occupant; re-sync world audio |
| `configure_vehicle` | any of: `speed_kmh`, `visibility`, `pothole_ahead_m` | Update world vehicle state |
| `configure_vibration` | any of: `override`, `road_quality`, `rms` | Toggle CSV/manual; set override values |
| `confirm` | `id` | Mark mitigation as "active" (user accepted) |
| `dismiss` | `id` | Mark mitigation as "dismissed" |
| `audio_chunk` | `data: [float]` | Feed 16 kHz PCM samples to AST inference queue |
| `vision_frame` | `data: base64-jpeg` | Feed webcam frame to vision pipeline |

**Outbound state fields (server → browser, every 100 ms):**

```json
{
  "ts": 1719000000.0,
  "scenario": "idle",
  "demo_running": false,
  "cognitive_load": 42,
  "factors": ["child in cabin (rear right)", "child elevated HR (150 bpm)"],
  "radar": { "seats": [...], "point_count": 87 },
  "audio": { "label": "crying", "confidence": 0.87 },
  "vehicle": { "speed_kmh": 60, "pothole_ahead_m": null, "visibility": "good" },
  "vibration": { "rms_z": 0.12, "road_quality": "smooth", "pothole_ahead_m": null },
  "vision_driver": { "face_detected": true, "ear": 0.31, "drowsy": false, "emotion": "calm" },
  "vision_objects": { "detections": [] },
  "mitigations": [...],
  "latency_ms": 0.43,
  "seat_configs": {
    "driver": { "occupied": true, "kind": "adult", "buckled": true,
                "distress": 0.0, "audio_event": "none",
                "heart_rate_bpm": null, "respiration_rpm": null, "emotion": "calm" },
    "front_passenger": { ... },
    "rear_left": { ... },
    "rear_right": { ... }
  },
  "driver_emotion": "calm",
  "vib_override": false,
  "vib_road_quality": "smooth",
  "vib_rms": 0.05,
  "world_speed_kmh": 60.0,
  "world_visibility": "good",
  "world_pothole_ahead_m": null
}
```

#### Demo Auto-Play

`POST /api/demo/play` triggers `_run_demo()`, an 8-step coroutine. Each step applies a scenario, waits for sensors to settle (2 s), auto-confirms certain mitigations, then holds for a display period. All sleeps use `_interruptible_sleep()`, which polls `_demo_stop` every 0.3 seconds — so any scenario selection or Stop button cancels the demo within 300 ms.

```
DEMO_SCRIPT = [
  ("idle",              hold=3s,  auto-confirm=[]),
  ("child_crying_rear", hold=5s,  auto-confirm=["comfort_audio"]),
  ("driver_stress",     hold=5s,  auto-confirm=["persona_calm"]),
  ("driver_tired",      hold=5s,  auto-confirm=["persona_alert"]),
  ("seatbelt_misuse",   hold=4s,  auto-confirm=[]),
  ("pothole_object",    hold=6s,  auto-confirm=["secure_object"]),
  ("child_left_behind", hold=5s,  auto-confirm=[]),
  ("idle",              hold=2s,  auto-confirm=[]),
]
Total demo duration ≈ 45 seconds
```

---

### 3.6 React HMI Frontend

Built with **React 18 (CDN)** + **Babel 7.12.17** (pinned — later versions break with CDN React). Zero build step. Key constraints:
- No backtick template literals inside `<script type="text/babel">` blocks (AudioWorklet is defined in a plain `<script>` tag above Babel, using a Blob URL)
- All components are plain functions with `React.useState`, `React.useRef`, etc. (no JSX imports)

#### Two Views

The HMI has two top-level tabs:

**Sensor View** — raw sensor data, direct manual configuration  
**Infotainment View** — driver-facing: cognitive load gauge, active issues, mitigation cards

#### Sensor View Layout (3-column CSS Grid)

```
grid-template-columns: 1fr 1fr 1.3fr
grid-template-rows:    1fr 1fr 72px
```

| | Col 1 | Col 2 | Col 3 |
|---|---|---|---|
| **Row 1** | VideoWidget (live webcam) | AudioWidget (waveform + label) | RadarWidget (spans rows 1–2) |
| **Row 2** | VehicleWidget + RoadQualityWidget | SeatWidget (compact) + IMUWidget | ← RadarWidget cont. |
| **Row 3** | TimelineBar (spans all 3 columns) | | |

Every child has an explicit `gridColumn` / `gridRow` wrapper — CSS auto-placement is not used because conditional panel toggling would shift other panels and block clicks.

`.panel` CSS does **not** use `resize: both` — that property was removed because it created invisible overlay regions blocking pointer events on panels below.

Hidden panels keep an empty `<div>` placeholder in their grid slot to prevent layout reflow.

#### Input Mode System (TimelineBar)

The TimelineBar shows the current input mode as a coloured badge:

| Mode | Colour | Condition |
|------|--------|-----------|
| **Manual** | Blue `#82b4ff` | `scenario === "idle"`, no live sensors active |
| **Live A/V** | Teal `#00d4b0` | `live_audio` or `live_vision` caps are true |
| **Scenario: \<name\>** | Amber `#f6ad55` + "overrides manual" badge | Any scenario other than idle is active |

**Auto-clear:** When the user clicks a kind button in RadarWidget or changes occupant type in SeatConfigOverlay, the HMI automatically sends `{cmd:'scenario', name:'idle'}` if a scenario was active. This switches mode to Manual and clears scenario-driven overrides.

#### RadarWidget (col 3, spans both rows)

The radar panel contains:

1. **Canvas visualisation (160×115 px):** Animated mmWave point cloud with per-seat colour rings:
   - `—` empty: dim white
   - `👤` adult: teal `#00d4b0`
   - `🧒` child: amber `#f6ad55`
   - `🐾` pet: purple `#a78bfa`

2. **Inline seat configurator** for F.P (front passenger), R.L (rear left), R.R (rear right) — driver excluded:

   Each seat row shows:
   ```
   [Seat Label]  [—]  [👤 Adult]  [🧒 Kid]  [🐾 Pet]
   ```
   Clicking a kind button sends `{cmd:'configure_seat', seat, occupied:true/false, kind}` directly. If a scenario is active, it is cleared to idle first.

3. **HR Slider** — shown per seat when that seat is occupied:
   - Range: 40–180 bpm
   - Colour: green → amber → red based on kind threshold (child: 115, pet: 125, adult: 95)
   - Sends `{cmd:'configure_seat', seat, heart_rate_bpm: val}` on change

4. **Vitals bars** — live HR and respiration bars from radar synthetic data, per occupied seat

#### SeatConfigOverlay

Opened by clicking any seat in the CabinView or Seat Status compact panel. Props: `{seatId, config, send, onClose, scenario}`.

Key implementation details:
- **Local optimistic state** (`localKind`, `localOcc`) — buttons respond instantly in the UI without waiting for the server round-trip and WebSocket echo
- Occupant kind change sends a single combined message `{occupied:true, kind:k}` (not two separate messages, which caused a race condition)
- Clears active scenario on any manual occupant change
- Controls: occupant type, heart rate (40–180 bpm), respiration, distress, audio event, emotion, buckled

#### Infotainment View

Driver-facing summary panel shown on the second tab:

- **Vehicle Cabin header** with `● LIVE` badge (pulsing green) and mode tag (`Manual / Live` or `Scenario: <name>`)
- **CircularGauge** — cognitive load 0–100, colour bands: green (0–32) / amber (33–65) / red (66–100)
- **IssuePanel** — maps `factors[]` strings to human-readable labels via `ISSUE_MAP`:

| Factor substring | Display label |
|-----------------|---------------|
| `child elevated HR` | Child Heartbeat Alert |
| `pet elevated HR` | Pet Heartbeat Alert |
| `child in cabin` | Child Detected |
| `pet in cabin` | Pet Detected |
| `child distress` | Child Distress Detected |
| `pet distress` | Pet Distress Detected |
| `child rapid breathing` | Child Respiratory Alert |
| `pet rapid breathing` | Pet Respiratory Alert |
| `elevated heart rate` | Driver HR Elevated |
| `crying in cabin` | Crying Detected |
| *(etc.)* | *(standard labels)* |

- **RecommendationPanel** — active mitigation cards with Apply / Dismiss. `friendlyTitle` map translates mitigation titles (e.g., `"Child heartbeat elevated"` → `"Child Heartbeat Alert"`, `"Cabin monitoring active"` → `"Radar Monitoring Active"`).

#### Component Tree Summary

```
App
 ├── Sensor View (tab 1)
 │    ├── [col1,row1] VideoWidget
 │    ├── [col2,row1] AudioWidget
 │    ├── [col3,rows1-2] RadarWidget
 │    │       ├── Canvas (mmWave visualisation)
 │    │       ├── SeatRow × 3 (kind buttons + HR slider per seat)
 │    │       └── VitalsBars × occupied seats
 │    ├── [col1,row2] VehicleWidget + RoadQualityWidget
 │    ├── [col2,row2] SeatWidget (compact) + IMUWidget
 │    └── [col1-3,row3] TimelineBar (mode badge + scenario buttons + demo)
 │         └── SeatConfigOverlay (modal, on seat select)
 └── Infotainment View (tab 2)
      ├── Header (● LIVE + mode tag)
      ├── CircularGauge (cognitive load)
      ├── IssuePanel (factor chips)
      └── RecommendationPanel (mitigation cards: Apply / Dismiss)
```

---

## 4. Q1 — How Acoustic Anomaly Works

The **Acoustic Anomaly** feature has two operating paths that coexist through the `_effective_audio()` resolver.

### Path A: Live Mic → AST Model (active when `live_audio: true`)

```
User's browser microphone (16 kHz mono)
  │
  ▼  (AudioWorklet "Chunker" processor)
Accumulates PCM Float32 samples in a ring buffer
  │  emits when buffer ≥ 15 360 samples (≈ 0.96 seconds)
  ▼
WebSocket  {cmd: "audio_chunk", data: [float × 15360]}
  │
  ▼  backend/main.py  WS handler
chunk_queue (maxsize=8, drops on overflow)
  │
  ▼  sensors/audio_yamnet.py  (asyncio.run_in_executor → thread pool)
ASTFeatureExtractor
  → mel spectrogram (128 mel bins, 16 kHz)
  │
  ▼
ASTForAudioClassification
  MIT/ast-finetuned-audioset-10-10-0.4593
  80 M parameters · ~40 ms GPU (RTX 3080) or ~400 ms CPU
  → logits for 527 AudioSet classes
  → softmax → probabilities
  │
  ▼  CabinSense vocabulary filter
  Check 6 class indices {22,21,80,74,73,468,0}
  Pick highest-confidence match above threshold 0.25
  │
  ▼  bus.publish("audio", AudioEvent)
  {label: "crying"|"animal"|"rattle"|"speech"|"none", confidence: 0.0–1.0}
```

When the mic is active and the model detects something with confidence > 0.40, this result is used directly in cognitive load and mitigation calculations.

### Path B: Per-Seat Config → Synthetic World State (fallback / override)

When the user configures a seat in the HMI:

```
User selects seat → sets audio_event = "crying" (or "barking", "shouting", etc.)
  │
  ▼  WS: {cmd: "configure_seat", seat: "rear_left", audio_event: "crying"}
  │
  ▼  backend/main.py  _sync_audio_from_seats()
Scans all occupied seats, picks highest-priority audio_event
  → world.audio_label = "crying"
  → world.audio_conf  = 0.80 + distress × 0.15
  │
  ▼  _effective_audio() in fusion.py (called every 100 ms)
IF live sensor is quiet (label="none" or conf ≤ 0.40):
    returns (world.audio_label, world.audio_conf)
  │
  ▼  Used in _cognitive_load() + _proposed()
```

### Why Both Paths Are Needed

When the live AST model is running but the physical microphone is in a quiet environment (no real crying/barking/etc.), the model publishes `label="none"`. Without the world-state fallback, demo scenarios and manual seat configurations would be invisible to the cognitive load formula. The `_effective_audio()` resolver ensures seat configs always influence the score unless real audio overrides them.

### Audio Events and Their Sources

| Label | Source path | Example trigger |
|-------|-------------|-----------------|
| `crying` | AST (AudioSet class 21/22) or seat config | Child seat set to "Crying" |
| `animal` | AST (AudioSet class 73/74/80) | Cat or dog detected by model |
| `barking` | Seat config only | Pet seat set to "Barking" |
| `shouting` | Seat config only | Passenger seat set to "Shouting" |
| `rattle` | AST (AudioSet class 468) | Loose object vibrating |
| `speech` | AST (AudioSet class 0) | General conversation detected |
| `talking` | Seat config only | Seat set to "Talking" |
| `happy` | Seat config only | Positive state (child/pet) |
| `none` | Default / silence | No anomaly detected |

---

## 5. Q2 — All Factors in Cognitive Load Calculation

The cognitive load score is an **additive rule-based model** computed in `fusion._cognitive_load()` at 10 Hz. It returns an integer 0–100 (capped via `min(100, score)`). Each contributing condition appends a human-readable string to the `factors` list shown on the HMI.

Below is every check, in execution order:

### Group 1: Child / Pet Presence and Vitals (from `world.seats` — instant)

Source: `world.seats` — read directly, no radar lag. Scores immediately after any `configure_seat` command.

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| Child or pet in any non-driver seat | +5 per seat | `"child in cabin (rear right)"` / `"pet in cabin (rear left)"` |
| Occupant distress > 0.05 | `min(30, distress × 40)` per seat | `"child distress 85%"` |
| Child HR > 115 bpm (manual override) | `min(40, (HR−115) × 1.5)` | `"child elevated HR (150 bpm)"` |
| Pet HR > 125 bpm (manual override) | `min(40, (HR−125) × 1.5)` | `"pet elevated HR (130 bpm)"` |

The `_scored_hr_seats` dedup set prevents a seat's HR from being counted twice if it also appears in the radar data.

### Group 2: Driver Physiology (from Radar micro-Doppler)

Source: `_latest["radar"]` → driver seat SeatState

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| Driver HR > 95 bpm | `min(30, (HR − 95) × 1.5)` | `"elevated heart rate"` |
| Driver respiration > 20 rpm | `min(15, (resp − 20) × 2)` | `"rapid breathing"` |
| Driver HR < 60 bpm | +12 | `"low arousal / fatigue"` |

HR and respiration are read from the latest radar frame (which adds ±5% jitter to world.driver_hr / world.driver_resp or the seat-level override values).

### Group 3: Driver Emotional State (from World / Seat Config)

Source: `world.driver_emotion` (set via scenario or HMI driver-seat emotion selector)

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| `driver_emotion == "stressed"` | +10 | `"driver stressed"` |
| `driver_emotion == "tired"` | +8 | `"driver fatigued"` |
| `driver_emotion == "calm"` or `"happy"` | 0 | — |

### Group 4: Acoustic Anomaly (via `_effective_audio()`)

Source: AST model output OR world.audio_label (see Q1 above). Confidence threshold: **> 0.40**.

| Label | Score added | Factor label |
|-------|-------------|--------------|
| `crying` | +25 | `"crying in cabin"` |
| `barking` | +25 | `"barking in cabin"` |
| `animal` | +25 | `"animal in cabin"` |
| `shouting` | +20 | `"shouting in cabin"` |
| `rattle` | +10 | `"rattling object"` |
| `talking` | +5 | `"speech activity"` |
| `happy` / `none` / `speech` | 0 | — |

### Group 5: Vehicle Context

Source: `_latest["vehicle"]` (published by `sensors/vehicle.py` at 5 Hz from `world.speed_kmh` and `world.visibility`)

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| visibility ∈ {`"low"`, `"rain"`, `"fog"`} | +12 | `"reduced visibility"` |
| speed > 100 km/h | +8 | `"high speed"` |

### Group 6: Agitated Rear Occupants (from Radar)

Source: `_latest["radar"]` → all non-driver seats. HR from radar is skipped for seats already scored in Group 1 (via `_scored_hr_seats`).

| Condition | Score added (per seat) | Factor label |
|-----------|------------------------|--------------|
| Child/pet HR from radar > threshold (if not already scored) | `min(20, over × 0.8)` | `"child elevated HR"` |
| Child resp > 28 rpm | `min(10, (resp−28) × 1.5)` | `"child rapid breathing"` |
| Pet resp > 35 rpm | same formula | `"pet rapid breathing"` |
| motion > 0.40 | +10 | `"agitated child"` / `"agitated pet"` |

Motion is computed by the radar node as: `distress × 0.6 + random(0, 0.08)`. A distress setting of 0.67+ reliably pushes motion above the 0.40 threshold.

### Group 7: Road Quality (from IMU / Vibration)

Source: `_latest["vibration"]` (published by `sensors/vibration.py` at 50 Hz)

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| road_quality == `"pothole"` | +18 | `"pothole / road shock (IMU)"` |
| road_quality == `"rough"` | +8 | `"rough road (IMU)"` |
| pothole_ahead_m ≠ null AND < 80 m | +5 | `"rough road ahead (IMU)"` |

### Group 8: Vision — Driver Face (from Camera / MediaPipe or Qwen)

Source: `_latest["vision_driver"]` — only contributes when `face_detected: true`

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| `drowsy: true` (EAR < 0.20) | +15 | `"drowsy eyes (camera)"` |
| `emotion == "stressed"` | +10 | `"stress expression (camera)"` |

### Group 9: Vision — Loose Objects (from Camera / YOLOv8n)

Source: `_latest["vision_objects"]` — detections list

| Condition | Score added | Factor label |
|-----------|-------------|--------------|
| Any of {backpack, suitcase, bottle, cup, book, laptop} detected | +6 | `"loose <label> (camera)"` |

Only the first matching detection contributes (no double-counting multiple loose items).

---

## 6. Q3 — Cognitive Scoring Criteria per Disturbance Attribute

This section gives the complete scoring table with example configurations and their expected score contribution.

### Scoring Summary Table

| Disturbance Attribute | Sensor | Max contribution | Score formula |
|-----------------------|--------|-----------------|---------------|
| Child/pet presence (per seat) | world.seats | 5 per seat | flat +5 |
| Child/pet distress | world.seats | 30 per seat | `min(30, distress×40)` |
| Child HR > 115 bpm (manual) | world.seats HR override | 40 | `min(40, (HR−115)×1.5)` |
| Pet HR > 125 bpm (manual) | world.seats HR override | 40 | `min(40, (HR−125)×1.5)` |
| Child HR from radar (if not manually set) | Radar vitals | 20 | `min(20, over×0.8)` |
| Driver high heart rate | Radar (μ-Doppler) | 30 | `min(30, (HR−95)×1.5)` |
| Driver rapid breathing | Radar (μ-Doppler) | 15 | `min(15, (resp−20)×2)` |
| Driver low HR / fatigue | Radar (μ-Doppler) | 12 | flat +12 |
| Driver emotion: stressed | World state / seat config | 10 | flat +10 |
| Driver emotion: tired | World state / seat config | 8 | flat +8 |
| Child/pet crying or barking | Audio (AST or seat config) | 25 | flat +25 |
| Shouting in cabin | Audio (seat config) | 20 | flat +20 |
| Rattling object | Audio (AST) | 10 | flat +10 |
| Speech / talking | Audio (AST or seat config) | 5 | flat +5 |
| Reduced visibility | Vehicle context | 12 | flat +12 |
| High speed (>100 km/h) | Vehicle context | 8 | flat +8 |
| Agitated child (rear seat) | Radar motion | 10 per seat | flat +10 each |
| Agitated pet (rear seat) | Radar motion | 10 per seat | flat +10 each |
| Pothole / road shock | IMU (vibration) | 18 | flat +18 |
| Rough road | IMU (vibration) | 8 | flat +8 |
| Rough road ahead (<80 m) | IMU (vibration) | 5 | flat +5 |
| Driver drowsiness (EAR) | Camera (MediaPipe / Qwen) | 15 | flat +15 |
| Driver stress expression | Camera (MediaPipe / Qwen) | 10 | flat +10 |
| Loose object detected | Camera (YOLOv8n) | 6 | flat +6 |
| **Theoretical maximum** | — | **≈ 200+** | capped to **100** |

### Score Bands and HMI Colours

| Band | Range | Colour | Interpretation |
|------|-------|--------|----------------|
| Safe | 0 – 32 | Green (`#3ddc84`) | Cabin nominal; no action needed |
| Elevated | 33 – 65 | Amber (`#f5a623`) | Disturbances present; mitigation proposed |
| Critical | 66 – 100 | Red (`#ff4d4f`) | High cognitive load; immediate advisory |

### Example Scenarios with Expected Scores

**Scenario A: Child with elevated HR in rear seat (manual config via Radar panel)**

| Factor | Contribution |
|--------|-------------|
| Seat config: rear_right = child, HR = 150 bpm (set via radar HR slider) | — |
| → child in cabin | +5 |
| → child HR 150 > 115: (150−115) × 1.5 = 52.5 → capped | +40 |
| **Total** | **45** (Elevated → triggers critical hr_alert mitigation) |

**Scenario B: Child crying in rear seat (typical demo)**

| Factor | Contribution |
|--------|-------------|
| Seat config: rear_right = child, audio_event=crying, distress=0.85 | — |
| → child in cabin | +5 |
| → world.audio_label = "crying", conf = 0.93 | +25 |
| → child distress 0.85 × 40 = 34 → capped | +30 |
| → radar: child motion = 0.85×0.6 + noise ≈ 0.55 > 0.40 | +10 |
| **Total** | **≈ 70** (Critical) |

**Scenario C: Stressed driver, low visibility, rough road**

| Factor | Contribution |
|--------|-------------|
| Driver HR = 108 bpm → (108−95)×1.5 = 19.5 | +19 |
| Driver emotion = stressed | +10 |
| Visibility = rain | +12 |
| Road quality = rough | +8 |
| **Total** | **≈ 49** (Elevated) |

**Scenario D: Multi-disturbance overload**

| Factor | Contribution |
|--------|-------------|
| Driver HR = 115 bpm → (115−95)×1.5 = 30 (cap) | +30 |
| Driver resp = 26 rpm → (26−20)×2 = 12 | +12 |
| Driver emotion = stressed | +10 |
| crying in cabin (conf 0.92) | +25 |
| Agitated child (motion 0.55) | +10 |
| Pothole (IMU) | +18 |
| Rough road ahead (< 80 m) | +5 |
| **Raw total** | **110 → capped to 100** (Critical) |

**Scenario E: Child left behind (safety critical)**

| Factor | Contribution |
|--------|-------------|
| Driver seat: unoccupied (speed = 0) | 0 |
| Rear child present, speed = 0, no driver | → triggers `CHILD PRESENCE ALERT` critical mitigation (auto-active) |
| child distress = 0.4 → motion ≈ 0.26 (below 0.40 threshold) | 0 |
| **Cognitive load** | **≈ 5** (child in cabin) but critical mitigation always surfaced |

This highlights that **mitigations are independent of the cognitive load score** — safety-critical alerts (child left behind, unbuckled occupant, elevated child HR) are always surfaced regardless of the overall score.

### Distress Level → Motion → Score Relationship (child/pet)

The `distress` slider (0–100%) on the HMI seat configurator directly controls how the radar node renders micro-motion, which then feeds into the agitated-occupant check in the cognitive load formula:

| Distress | Radar motion (deterministic) | Triggers agitated check? | Score contribution |
|----------|------------------------------|--------------------------|-------------------|
| 0% | 0.00 + noise ≈ 0.04 | No (< 0.40) | 0 |
| 50% | 0.30 + noise ≈ 0.34 | Borderline | 0 (just below) |
| 67% | 0.40 + noise ≈ 0.44 | Yes | +10 |
| 85% | 0.51 + noise ≈ 0.55 | Yes | +10 |
| 100% | 0.60 + noise ≈ 0.64 | Yes | +10 |

> Note: ±5% jitter on HR/resp, and `random(0, 0.08)` additive noise on motion, means scores vary slightly between ticks. The ranges above are representative of expected values.

### HR Slider → Score Relationship (child/pet via Radar panel)

The HR slider in RadarWidget sets `occ.heart_rate_bpm` directly in `world.seats`, bypassing radar synthetic defaults:

| HR set (child, threshold=115) | Over-threshold | Score added |
|-------------------------------|----------------|-------------|
| 100 bpm | 0 | 0 |
| 120 bpm | 5 | +7.5 |
| 140 bpm | 25 | +37.5 |
| 150 bpm | 35 | +40 (capped) |
| 180 bpm | 65 | +40 (capped) |

Score becomes visible in Infotainment View after the next fusion tick (< 100 ms).

---

## 7. Data Flow: End-to-End Walk-Through

Example: **User selects child + sets HR = 150 bpm via Radar panel, then switches to Infotainment View.**

```
1. HMI: User clicks "🧒 Kid" button on rear_right row in RadarWidget
   WS send: {cmd:'scenario', name:'idle'}           ← clears active scenario
   WS send: {cmd:'configure_seat', seat:'rear_right', occupied:true, kind:'child'}
   Server: world.seats["rear_right"].kind = "child"
           world.seats["rear_right"].occupied = True

2. HMI: User drags HR slider to 150 bpm
   WS send: {cmd:'configure_seat', seat:'rear_right', heart_rate_bpm:150}
   Server: world.seats["rear_right"].heart_rate_bpm = 150

── fusion tick (100ms) ────────────────────────────────────────────────────

3. fusion._cognitive_load():
   Group 1 — world.seats scan:
     rear_right: kind=child, occupied=True → +5 ("child in cabin (rear right)")
     heart_rate_bpm=150 > 115 → (150−115)×1.5 = 52.5, capped → +40
                                 ("child elevated HR (150 bpm)")
     _scored_hr_seats.add("rear_right")
   All other groups: no contribution (driver calm, audio quiet, smooth road)
   Score = 5 + 40 = 45

4. fusion._proposed():
   world_child_pet = {"rear_right"}
   → cabin_monitoring mitigation: id="cabin_monitoring", title="Cabin monitoring active",
     severity="advisory"
   → hr_alert mitigation: id="hr_alert_rear_right",
     title="Child heartbeat elevated",
     detail="Rear Right — child heart rate 150 bpm (normal <115). Monitor and adjust cabin comfort.",
     severity="warning" (150 < 115+20=135? No, 150 > 135 → severity="critical")
     severity="critical"
   → no belt mitigation (rear_right buckled=True from world)
   → no comfort_audio (no audio event set)

5. fused state published:
   {cognitive_load: 45, factors: ["child in cabin (rear right)", "child elevated HR (150 bpm)"],
    mitigations: [{id:"cabin_monitoring",...}, {id:"hr_alert_rear_right", severity:"critical",...}]}

6. WebSocket pump → browser

7. HMI: Infotainment View renders:
   Gauge = 45 (Amber zone)
   IssuePanel: "Child Heartbeat Alert" (from ISSUE_MAP)
   RecommendationPanel: "Child heartbeat elevated" card (critical red, Apply/Dismiss)
```

---

## 8. Demo Scenarios

| Scenario name | What it sets in the world | Primary mitigation triggered |
|---------------|--------------------------|------------------------------|
| `idle` | All seats at baseline; calm driver; smooth road | None |
| `child_crying_rear` | rear_right: child, distress=0.85; audio=crying(0.92) | Soothe cabin (advisory) |
| `pet_agitated` | rear_left: pet, distress=0.70, unbuckled; audio=animal(0.80) | Soothe cabin (advisory) |
| `driver_stress` | driver HR=104, resp=22, emotion=stressed; visibility=low | Calming persona (advisory) |
| `driver_tired` | driver HR=58, resp=10, emotion=tired | Alertness boost (advisory) |
| `seatbelt_misuse` | driver: occupied but buckled=False | Seatbelt warning (auto-active) |
| `pothole_object` | unsecured_object=True; pothole_ahead=80m; speed=75 | Secure loose item (warning) |
| `child_left_behind` | driver seat empty; rear_right: child, distress=0.40; speed=0 | CHILD PRESENCE ALERT (critical, auto-active) |

Each scenario calls `_reset(world)` first (returning all seats to baseline) then applies its specific mutations. Applying a scenario from the HMI also cancels any running auto-play demo within 300 ms.

---

## 9. Adaptive Mitigations Reference

Mitigations are **propose-first**: they appear as cards requiring the driver to tap Apply. Only the `CHILD PRESENCE ALERT`, seatbelt warnings, and elevated child/pet HR alerts are auto-active or critical severity.

| ID | Title | Trigger condition | Severity | Confirm? |
|----|-------|------------------|----------|---------|
| `comfort_audio` | Soothe cabin | audio ∈ {crying, animal, barking} AND child/pet in rear (world or radar) | advisory | Yes |
| `baby_engagement` | Baby engagement | audio=crying AND child/infant present | advisory | Yes |
| `shouting_alert` | Passenger distress | audio = shouting AND conf > 0.40 | warning | Yes |
| `persona_calm` | Calming persona | driver HR > 95 OR emotion = stressed | advisory | Yes |
| `persona_alert` | Alertness boost | driver HR < 60 OR emotion = tired | advisory | Yes |
| `belt_<seat>` | Seatbelt not engaged | Any occupied seat in `world.seats` with `buckled=False` | warning | No (auto-active) |
| `secure_object` | Secure loose item | world.unsecured_object AND pothole_ahead ≤ 120 m | warning | Yes |
| `secure_object_cam` | Secure loose item (camera) | YOLO loose object AND pothole_ahead ≤ 120 m | warning | Yes |
| `child_left` | CHILD PRESENCE ALERT | Rear child detected AND driver seat empty | critical | No (auto-active) |
| `hr_alert_<seat>` | Child/pet heartbeat elevated | `world.seats[seat].heart_rate_bpm` > threshold (child:115, pet:125) | warning / critical | Yes |
| `cabin_monitoring` | Cabin monitoring active | Child/pet in `world.seats`, no audio-specific mitigation active | advisory | No |

**hr_alert severity escalation:** `warning` if HR ≤ threshold+20; `critical` if HR > threshold+20.  
Example: child HR=150 > 115+20=135 → critical.

**cabin_monitoring suppression:** Not shown if `comfort_audio` or `baby_engagement` is already active (those are more specific).

---

## 10. Repository Layout

```
technothon/
├── backend/
│   ├── main.py              FastAPI app: routes, WS handler, demo runner
│   ├── fusion.py            Cognitive-load + mitigation engine (10 Hz)
│   ├── world.py             Shared ground-truth dataclass (World + Occupant)
│   ├── bus.py               Async pub/sub message bus
│   ├── schemas.py           RadarFrame / AudioEvent / VehicleContext dataclasses
│   ├── scenarios.py         8 scripted demo scenarios that mutate world
│   ├── sensors/
│   │   ├── radar.py         Synthetic mmWave node (10 Hz)
│   │   ├── audio.py         Synthetic audio node (5 Hz, reads world.audio_label)
│   │   ├── audio_yamnet.py  Live AST audio classifier (PyTorch, on-demand)
│   │   ├── vehicle.py       Speed / visibility / pothole-distance node (5 Hz)
│   │   ├── vibration.py     IMU CSV replay or manual override (50 Hz)
│   │   ├── vision.py        MediaPipe face + YOLOv8n objects (~5 fps)
│   │   └── qwen_vision.py   Qwen2-VL-2B-Instruct all-seat vision (4-bit, RTX 3080)
│   ├── models/
│   │   └── face_landmarker.task   MediaPipe model file
│   ├── data/
│   │   └── route_imu.csv    50 Hz accelerometer recording for vibration replay
│   └── requirements.txt
├── frontend/
│   ├── index.html           React HMI (CDN React 18 + Babel 7.12.17, zero-build)
│   ├── deck.html            10-slide pitch deck
│   └── lib/
│       ├── react.min.js
│       ├── react-dom.min.js
│       └── babel.min.js     Pinned to 7.12.17 (newer breaks CDN React)
├── CLAUDE.md                Architecture guide + build plan
├── USAGE.md                 Setup + run + API reference
├── doc.md                   ← this document
├── environment.yml          conda env (cabinsense, CUDA-capable)
└── setup_frontend.sh        Downloads React/Babel libs locally
```

---

*Document updated from source code inspection — June 2026*
