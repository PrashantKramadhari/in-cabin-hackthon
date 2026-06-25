# Cabin Video Feed Panel — How It Works

A study of the **Cabin Video Feed** panel: what you control, what runs on the backend, which models are involved, and how results flow back into the dashboard.

---

## 1. What the panel actually does

The panel is the React `VideoWidget` in `frontend/index.html`. It has two jobs:

1. **Show video** in a `<video>` element (preview only).
2. **Sample frames** and send them to the backend over WebSocket for vision inference.

It does **not** run ML in the browser. All detection happens server-side.

### User-controllable parameter (only one in this panel)

| Dropdown option | What happens |
|-----------------|--------------|
| **Camera Off** | Stops camera/file, no frames sent |
| **Default Video (`video_1.mp4`)** | Loops `/static/video_1.mp4`, samples frames |
| **Live Camera** | `getUserMedia({video:true})`, samples webcam frames |

**Live Camera** only appears if `/api/caps` reports `live_vision: true` (a vision backend node loaded successfully).

There is no slider or tuning UI in this panel — mode is the only direct control.

---

## 2. End-to-end data flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (VideoWidget)                                          │
│  <video> → hidden <canvas> → JPEG base64 → WebSocket            │
└────────────────────────────┬────────────────────────────────────┘
                             │  {cmd: "vision_frame", data: "..."}
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  backend/main.py  →  frame_queue (max 4)  →  vision node         │
│       qwen_vision.py  OR  vision.py (MediaPipe + YOLO)          │
└────────────────────────────┬────────────────────────────────────┘
                             │  bus topics
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  fusion.py  →  cognitive_load + mitigations + seat_configs      │
└────────────────────────────┬────────────────────────────────────┘
                             │  fused state (~10 Hz)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  WebSocket  →  HMI (gauge, mitigations, emotion overlay)       │
└─────────────────────────────────────────────────────────────────┘
```

### Step by step

1. **Capture** — Video plays in `<video>`. A hidden canvas draws the current frame, scaled so the longest side ≤ **480px**.
2. **Encode** — `canvas.toDataURL('image/jpeg', 0.85)` → strip the `data:image/jpeg;base64,` prefix.
3. **Send** — WebSocket message:
   ```json
   { "cmd": "vision_frame", "data": "<base64-jpeg>" }
   ```
4. **Queue** — `main.py` pushes into `frame_queue` (max **4** frames; drops if full).
5. **Infer** — Vision node decodes JPEG, runs model(s), publishes bus topics.
6. **Fuse** — `fusion.py` merges vision with radar, audio, vehicle, scenarios.
7. **Return** — Fused JSON streams back on the same WebSocket (~10 Hz).
8. **Display** — Panel shows `vision_driver` (driver emotion/drowsy). Infotainment tab shows `vision_all_seats` when Qwen is active.

---

## 3. Frame sampling rates

| Mode | Interval | Effective rate | Notes |
|------|----------|----------------|-------|
| Live camera | 2000 ms | 0.5 fps | Matches Qwen latency budget |
| Default video | 2000 ms | 0.5 fps | Same pipeline as live |
| Camera off | — | 0 | Preview stopped |

There is also `startSynth()` (canvas + baby audio) in the code, but it is **not** in the current dropdown — only off / default video / live are exposed.

---

## 4. Which vision backend runs?

At startup, `backend/main.py` picks **one** vision node (priority order):

```
Qwen2-VL  →  MediaPipe + YOLO  →  none
```

| Priority | Module | `vision_node` in `/api/caps` | Requirements |
|----------|--------|-------------------------------|--------------|
| 1 | `backend/sensors/qwen_vision.py` | `qwen2vl` | GPU, CUDA, `bitsandbytes`, ~2.5 GB VRAM |
| 2 | `backend/sensors/vision.py` | `mediapipe` | CPU OK, OpenCV, MediaPipe, Ultralytics |
| 3 | — | `none` | No live vision; dropdown hides Live Camera |

On a CPU-only VM, Qwen typically fails and MediaPipe/YOLO is attempted next.

---

## 5. Models and what they output

### Path A — Qwen2-VL (`qwen_vision.py`)

| Item | Detail |
|------|--------|
| **Model** | `Qwen/Qwen2-VL-2B-Instruct` (4-bit NF4) |
| **Local cache** | `backend/models/qwen2-vl-2b/` if present |
| **Input** | Single cabin image + fixed JSON prompt |
| **Output** | Per-seat JSON for all 4 seats |

Per seat:

- `occupied`, `kind` (adult / child / infant / pet / unknown)
- `emotion` (calm / happy / stressed / tired / distressed)
- `buckled`
- `objects` (loose items: box, bag, laptop, etc.)

**Bus topics published:**

- `vision_all_seats` — full 4-seat analysis
- `vision_driver` — legacy driver-only fields for fusion rules

Qwen does **not** publish `vision_objects` (no bounding boxes).

### Path B — MediaPipe + YOLO (`vision.py`)

| Model | File | Role |
|-------|------|------|
| **MediaPipe Face Landmarker** | `backend/models/face_landmarker.task` | Driver face, 478 landmarks |
| **YOLOv8n** | `backend/models/yolov8n.pt` | Object detection (COCO classes) |

**Face pipeline:**

- **EAR** (eye aspect ratio) from landmarks → drowsiness if EAR < **0.20**
- **Mouth geometry** → emotion heuristic: `tired` / `stressed` / `happy` / `calm`

**YOLO pipeline:**

- `conf ≥ 0.35`
- Detections: `{label, confidence, box}`

**Bus topics:**

- `vision_driver` — `face_detected`, `ear`, `drowsy`, `emotion`, `mouth_ratio`
- `vision_objects` — detection list

---

## 6. How results change system behavior (fusion)

Vision does not update the HMI directly. `backend/fusion.py` consumes bus topics and affects:

### Cognitive load score (`_cognitive_load`)

| Signal | Score impact |
|--------|----------------|
| Drowsy eyes (MediaPipe EAR) | +15 |
| Stressed expression (camera) | +10 |
| Loose YOLO object (backpack, bottle, laptop, …) | +6 |
| Qwen unsecured objects on a seat | +4 per object (max +10) |

### Mitigations (`_proposed`)

Examples tied to vision:

- **Persona tuning** — camera sees stressed/tired driver → Calming persona / Alertness boost
- **Secure loose item** — YOLO or Qwen objects + pothole ahead → pre-emptive advisory
- **Unsecured item on seat** — Qwen `objects[]` on any seat

### Seat configs shown in HMI (`_build_seat_configs`)

Fusion merges:

- **World state** (scenarios, Seat Status panel, Radar sliders) — manual config
- **Qwen `vision_all_seats`** — can set `occupied`, `kind`, `emotion` when world has no manual kind

Manual seat config from other panels **wins** over vision for `kind` when already set.

---

## 7. What changes when you use the video panel vs other panels

| Action | Where | Effect on vision pipeline |
|--------|-------|---------------------------|
| Change dropdown to Live / Default Video | Video panel | Starts/stops frame stream |
| Change dropdown to Off | Video panel | Stops frames; last vision results age out in fusion |
| Configure seat (occupant, HR, audio) | Seat / Radar panels | `configure_seat` over WebSocket → `scenarios.world` — **not** from video |
| Click demo scenario | Timeline | Mutates world; vision still runs in parallel |
| Drag panel / theme | UI only | No effect on vision |

The video panel’s `send` prop is passed in but **unused** — it only uses `ws` for `vision_frame`.

---

## 8. What you see in the UI

### Sensor tab — Video panel

- Video preview
- If `vision_driver.face_detected`: **Driver emotion** + drowsy warning

### Infotainment tab

- **Vision strip** under cabin map from `vision_all_seats` (Qwen path)
- Occupant list uses fused `seat_configs` (world + vision overlay)

### Elsewhere (not in video panel)

- Cognitive load gauge, mitigation cards, factor chips — all from fused state

---

## 9. Architecture role in CabinSense

Video fits the **Sense** layer of the single pipeline:

```
Sense → Detect → Fuse → Feedback
```

```
Browser camera/file  →  vision_frame  →  Vision node (Qwen or MediaPipe+YOLO)
                                              ↓
                                        vision_driver / vision_all_seats / vision_objects
                                              ↓
                                        fusion.py  →  cognitive_load + mitigations
```

Radar and audio provide corroboration; vision adds **driver face/state** and **cabin objects/occupants** without replacing radar for vitals or mic for crying detection.

---

## 10. Key source files

| File | Role |
|------|------|
| `frontend/index.html` — `VideoWidget` | Capture, encode, send frames; show preview + driver emotion |
| `backend/main.py` | Vision node selection; WebSocket `vision_frame` handler |
| `backend/sensors/qwen_vision.py` | Qwen2-VL all-seat inference |
| `backend/sensors/vision.py` | MediaPipe face + YOLOv8n objects |
| `backend/fusion.py` | Merge vision into score, mitigations, seat configs |
| `backend/bus.py` | Async pub/sub between nodes |

---

## 11. Practical notes

- **Server restart** does not reset vision logic; models reload on startup.
- **No GPU** → Qwen fails → MediaPipe/YOLO fallback (if deps installed).
- **No camera permission** → Live mode alerts and reverts to Off.
- **Default video** works without camera — good for demos with `video_1.mp4` under `frontend/`.
- Vision is **slow by design** (~0.5 fps) because Qwen inference is ~100–150 ms per frame on GPU and the queue is bounded.

---

## 12. Related docs

- [USAGE.md](USAGE.md) — run instructions and HMI overview
- [CLAUDE.md](CLAUDE.md) — project architecture and MVP scope
