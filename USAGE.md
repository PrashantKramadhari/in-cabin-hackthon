# CabinSense — Usage Guide

## What it is

CabinSense is a real-time in-cabin disturbance intelligence system. It fuses
**radar + audio + vision + vibration** into a single Cognitive-Load score and
surfaces non-intrusive, confirm-first mitigations on a live dashboard.

**Architecture & pipelines:** [ARCHITECTURE.md](ARCHITECTURE.md)  
**Cognitive load scoring:** [ARCHITECTURE.md §6](ARCHITECTURE.md#6-cognitive-load-scoring-flow)

```
Radar (mmWave)  ─┐
Mic  → AST model ─┤
Cam  → MediaPipe ─┼─ bus ─▶ Fusion engine ─▶ WebSocket ─▶ Cabin HMI
IMU  → CSV replay ─┤         (CL score + mitigations)
Vehicle/GPS      ─┘
```

---

## 1. Prerequisites

| Tool | Version |
|---|---|
| Conda | Miniconda or Anaconda (any recent version) |
| Python | managed by conda (3.11 pinned in `environment.yml`) |
| Node / npm | not required — HMI is CDN React, zero-build |
| Browser | Chrome / Edge recommended (mic + camera WebRTC) |

---

## 2. First-time setup (conda)

```bash
# from the repo root — one command creates and populates the environment
conda env create -f environment.yml

# activate it (do this every time you open a new terminal)
conda activate cabinsense
```

Download the local JS libs (React, Babel — avoids CDN blank-screen issues):

```bash
bash setup_frontend.sh
```

Then download the MediaPipe face landmarker model (one-time, ~3.6 MB):

```bash
mkdir -p backend/models
curl -L -o backend/models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

YOLOv8n (`yolov8n.pt`, ~6 MB) and the AST audio model (~85 MB) download
automatically on first server start and are cached locally.

> **To update the environment later** (e.g. after a `git pull`):
> ```bash
> conda env update -f environment.yml --prune
> ```

---

## 3. Start the server

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --port 8000
```

Open **http://localhost:8000** in your browser.

Expected startup output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

The first start takes ~15–20 s while the AST audio model downloads from
HuggingFace. Subsequent starts are instant (models are cached).

---

## 4. The HMI at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  CabinSense   In-Cabin Disturbance Intelligence  ● live 0ms │
├──────────────────────────────────────────────────────────────┤
│  [▶ Run Demo]  ████████░░░░  Step 3/8: Driver under stress   │
├────────────────┬────────────────────────┬────────────────────┤
│ Cognitive Load │  In-Cabin Radar        │ Adaptive           │
│      47        │  ┌driver┐ ┌front_pass┐│ Mitigations        │
│  ████████░░    │  │  HR  │ │  empty   ││                    │
│                │  │  92  │ └──────────┘│ [Calming persona]  │
│ Acoustic Anomaly│  └──────┘ ┌rear_left┐│  proposed  [Apply] │
│   crying  92%  │  ┌rear_r┐  │  empty  ││                    │
│ [🎤 use mic]   │  │child │  └─────────┘│ [Soothe cabin]     │
│                │  └──────┘             │  proposed  [Apply] │
│ IMU / Vibration│  speed  visibility    │                    │
│  rough  ██░░   │  75km/h   good        │                    │
│  ⚠ 23m ahead  │                       │                    │
│                │  Vision · Face+Objects│                    │
│ Demo Scenarios │  emotion: stressed    │                    │
│ [idle][crying] │  EAR: 0.24  drowsy: no│                   │
│ [stress][tired]│  [📷 use cam]         │                    │
└────────────────┴────────────────────────┴────────────────────┘
```

### Panel descriptions

| Panel | What it shows |
|---|---|
| **Cognitive Load** | 0–100 attention-risk score. Green < 33, amber 33–66, red > 66. Chips below explain contributing factors. |
| **Acoustic Anomaly** | Current audio event (`crying / animal / rattle / speech / quiet`) and confidence. Click **🎤 use mic** for live inference. |
| **IMU / Vibration** | Real-time road quality from CSV replay (`smooth / rough / pothole`), RMS bar, metres to next rough patch. |
| **Demo Scenarios** | Buttons to inject any cabin situation instantly. |
| **In-Cabin Radar** | Per-seat occupancy, occupant type, HR, breathing rate, buckle state. |
| **Vehicle context** | Speed, visibility, predicted distance to rough road. |
| **Vision** | Driver face emotion + EAR drowsiness flag, YOLO object detections. Click **📷 use cam** for live webcam. |
| **Adaptive Mitigations** | Cards for each active anomaly. Confirm-first: click **Apply** to activate, **Dismiss** to suppress. |

---

## 5. Running the auto-play demo

Click **▶ Run Demo** in the top bar.

It plays 8 scenarios hands-free (~45 seconds) with automatic mitigation
confirmation, so judges see the complete story without you touching anything:

| Step | Scenario | What triggers |
|---|---|---|
| 1 | Idle | Cabin nominal, no mitigations |
| 2 | Child crying in rear seat | Audio + radar → Soothe cabin advisory → auto-confirmed |
| 3 | Driver under stress | Radar HR ↑ + breathing ↑ → Calming persona → auto-confirmed |
| 4 | Driver fatigue | Radar HR ↓ + low EAR → Alertness boost → auto-confirmed |
| 5 | Seatbelt misuse | Radar: driver occupied, not buckled → Warning (no confirm needed) |
| 6 | Unsecured object + pothole | IMU predicts rough road, YOLO sees object → Secure item advisory |
| 7 | Child left behind | Rear child detected, no driver → Critical alert (horn/lights) |
| 8 | Idle | Back to nominal |

---

## 6. Manual scenario injection

Click any button in the **Demo Scenarios** panel to jump to a specific
situation without waiting for the auto-play sequence. Useful for answering
judge questions ("can you show me the seatbelt case?").

Available scenarios:

| Button | Situation simulated |
|---|---|
| `idle` | All quiet — baseline cabin state |
| `child crying rear` | Child in rear-right seat, distress 85%, audio confidence 92% |
| `pet agitated` | Pet in rear-left seat, audio: animal sound 80% |
| `driver stress` | Driver HR 104 bpm, breathing 22 rpm, low visibility |
| `driver tired` | Driver HR 58 bpm, breathing 10 rpm |
| `seatbelt misuse` | Driver seat occupied, belt not properly worn |
| `pothole object` | Unsecured object + rough road 80 m ahead at 75 km/h |
| `child left behind` | Rear child present, driver absent — safety-critical alert |

---

## 7. Using live sensors

### Live microphone (audio)

1. Start the server. Confirm `live_audio: true` at `GET /api/caps`.
2. Open the HMI. Click **🎤 use mic** in the Acoustic panel.
3. Allow browser mic access when prompted.
4. Speak, play a crying sound, or hold a phone near the mic — the label
   and confidence update in real time.

The mic captures at 16 kHz, chunks into 0.96 s windows, streams to the
backend, and the **Audio Spectrogram Transformer (AST)** model classifies
across 527 AudioSet classes. Results appear within ~1 second.

> **Note:** Live mic and scenario injection work simultaneously. The scenario
> sets the radar/vehicle world state; the mic replaces the synthetic audio.

### Live webcam (vision)

1. Click **📷 use cam** in the Vision panel.
2. Allow browser camera access.
3. The camera sends 320×240 JPEG frames at 5 fps to the backend.
4. **MediaPipe FaceLandmarker** extracts eye-aspect-ratio (EAR) and mouth
   geometry → drowsiness flag + emotion estimate.
5. **YOLOv8n** detects objects in the frame → loose-item advisories.

> Try: look down / half-close your eyes → EAR drops → `drowsy: yes` →
> cognitive load increases → Alertness-boost mitigation appears.

---

## 8. API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Cabin HMI (React dashboard) |
| `GET` | `/api/caps` | `{live_audio, live_vision}` capability flags |
| `GET` | `/api/scenarios` | List of scenario names + current active |
| `POST` | `/api/demo/play` | Start the auto-play demo sequence |
| `WS` | `/ws` | Bidirectional stream — receive fused state, send commands |

### WebSocket message format

**Receive** (server → browser, 10 Hz):
```json
{
  "ts": 1719000000.0,
  "scenario": "child_crying_rear",
  "cognitive_load": 47,
  "factors": ["crying in cabin", "agitated child"],
  "radar":   { "seats": [...], "point_count": 84 },
  "audio":   { "label": "crying", "confidence": 0.92 },
  "vehicle": { "speed_kmh": 60, "pothole_ahead_m": null, "visibility": "good" },
  "vibration":{ "rms_z": 0.25, "road_quality": "rough", "pothole_ahead_m": 23.4 },
  "vision_driver":  { "face_detected": true, "ear": 0.27, "drowsy": false, "emotion": "calm" },
  "vision_objects": { "detections": [] },
  "mitigations": [
    { "id": "comfort_audio", "title": "Soothe cabin", "status": "proposed",
      "severity": "advisory", "confirm": true, "detail": "..." }
  ],
  "latency_ms": 0.04
}
```

**Send** (browser → server):
```json
{ "cmd": "scenario",     "name": "driver_stress" }   // inject scenario
{ "cmd": "confirm",      "id": "comfort_audio"   }   // apply mitigation
{ "cmd": "dismiss",      "id": "comfort_audio"   }   // dismiss mitigation
{ "cmd": "audio_chunk",  "data": [0.01, -0.02, ...] }// 16kHz Float32 mic chunk
{ "cmd": "vision_frame", "data": "<base64-jpeg>"     }// 320x240 webcam frame
```

---

## 9. Project structure

```
technothon/
├── backend/
│   ├── main.py          FastAPI server — websocket + scenario + demo endpoints
│   ├── bus.py           Async pub/sub message bus (latency-bounded, drop-oldest)
│   ├── fusion.py        Cognitive-load score + mitigation rules + confirm state
│   ├── world.py         Shared ground-truth cabin state (mutated by scenarios)
│   ├── scenarios.py     8 scripted demo situations
│   ├── schemas.py       RadarFrame / AudioEvent / VehicleContext dataclasses
│   ├── sensors/
│   │   ├── radar.py         Synthetic mmWave node (occupancy + vitals)
│   │   ├── audio.py         Synthetic audio node (fallback)
│   │   ├── audio_yamnet.py  Live AST inference node (PyTorch)
│   │   ├── vehicle.py       Speed / visibility / pothole context
│   │   ├── vibration.py     IMU CSV replay at 50 Hz (pre-emptive warnings)
│   │   └── vision.py        MediaPipe FaceLandmarker + YOLOv8n
│   ├── data/
│   │   └── route_imu.csv    90-second synthetic route (smooth→rough→pothole)
│   └── models/
│       └── face_landmarker.task  MediaPipe model (downloaded at setup)
├── frontend/
│   └── index.html       React Cabin HMI (CDN React, no build step)
├── CLAUDE.md            Architecture + day-by-day build plan
├── USAGE.md             ← this file
└── README.md            Quick-start summary
```

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `live_audio: false` at `/api/caps` | `pip install torch transformers` and restart |
| `live_vision: false` at `/api/caps` | `pip install mediapipe ultralytics opencv-python-headless` and restart |
| AST model download slow on first start | Wait ~30 s; it caches in `~/.cache/huggingface/` |
| Mic button missing / greyed out | Server returned `live_audio: false` — check install |
| Camera shows `no face` | Ensure good lighting; camera must face you directly |
| HMI blank / black screen | Run `bash setup_frontend.sh` (CDN blocked). Then hard-refresh Ctrl+Shift+R |
| `torchvision` import error | Run `conda env create -f environment.yml` from scratch — it pins the correct versions |
| MediaPipe `no attribute solutions` | You have mp 0.10+; the node uses the Tasks API — should work |
| Port 8000 in use | `uvicorn main:app --port 8001` and open `http://localhost:8001` |
