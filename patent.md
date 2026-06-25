# CabinSense — Patent Analysis & Filing Guide

> **Disclaimer:** This document is a technical prior-art and novelty analysis, not legal advice.
> Consult a registered patent attorney (preferably automotive/AI IP) before filing.

---

## Public Disclosure Warning

The hackathon presentation is a **public disclosure event**. Once the work is publicly
presented, a 12-month grace period applies in the US (35 U.S.C. § 102(b)(1)) and
**no grace period** applies in most other jurisdictions (EU, China, India).

**File a provisional patent application BEFORE the presentation** to preserve the
priority date in all jurisdictions. Cost: ~$320 USD (USPTO micro-entity fee).

---

## Invention Title (Proposed)

> **"Multi-Modal In-Cabin Disturbance Intelligence System with Confirm-First
> Adaptive Mitigation and Pre-Emptive Hazard Advisory"**

---

## Claims Summary

### Claim 1 — Core System (Independent, Broadest)

A real-time in-cabin monitoring system comprising:

- A plurality of sensing nodes operating independently at their own frequencies,
  each publishing sensor frames to a bounded asynchronous pub/sub message bus;
- A world-state store maintaining a ground-truth representation of cabin occupant
  configuration, decoupled from sensor latency;
- A fusion engine operating at a fixed cadence that accumulates a scalar
  Cognitive Load score from multi-modal sensor inputs via a transparent,
  rule-based weighted accumulator;
- A mitigation rule engine that derives a set of confirm-first advisory actions
  from the fused score and sensor state;
- A grace-period state machine that retains user-confirmed or user-dismissed
  mitigation states for a fixed interval after the triggering condition clears,
  preventing flicker from transient sensor noise;
- A human-machine interface that presents mitigations non-intrusively and
  requires explicit user confirmation before any cabin actuator is engaged.

---

### Claim 2 — Pre-Emptive Object-Securing Advisory (STRONGEST)

*Dependent on Claim 1. Most novel — likely no direct prior art.*

A method for pre-emptive loose-object hazard advisory comprising:

1. Detecting one or more unsecured objects on vehicle seats via a computer vision
   pipeline (vision-language model and/or object detection network);
2. Concurrently receiving a road-quality prediction signal including a
   distance-to-hazard metric (pothole or rough road ahead in metres) from an
   inertial measurement unit or GPS/map fusion source;
3. Generating a warning mitigation when the distance-to-hazard falls below a
   configurable threshold (default 120 m), naming the detected object and
   estimated time to impact;
4. Presenting the advisory to the driver with sufficient lead time for corrective
   action before the road event occurs.

**Dependent sub-claims:**
- The vision pipeline is a vision-language model (VLM) that returns a structured
  per-seat JSON including an `objects` array describing unsecured items by name.
- The advisory severity escalates from advisory to warning as distance decreases.
- The system suppresses duplicate advisories during the grace period if the driver
  has already dismissed the mitigation.

---

### Claim 3 — Two-Stage Cabin Audio Classification Cascade

*Dependent on Claim 1. Novel architecture in automotive cabin context.*

A cascaded audio classification system for in-cabin sound event detection comprising:

1. A first-stage lightweight convolutional neural network (fast-path classifier)
   operating on mel-spectrogram representations of short audio frames (~0.96 s),
   trained on domain-specific cabin audio classes (infant crying, infant talking,
   infant happy, none);
2. A confidence threshold gate: when the fast-path classifier confidence exceeds
   a defined threshold (e.g., 0.60), the classification result is accepted and
   the second stage is bypassed;
3. A second-stage large pre-trained audio classification model (Audio Spectrogram
   Transformer or equivalent) activated only when fast-path confidence falls below
   the threshold, providing broad-class fallback across 527+ AudioSet categories;
4. A unified bus topic publication carrying label and confidence regardless of
   which stage produced the result.

**Dependent sub-claims:**
- The fast-path model is trained on-device using augmented audio samples (noise,
  gain shift, time-shift, polarity flip, zero-masking).
- The fast-path model weights are persisted to a local path and retrained on
  first run if absent.

---

### Claim 4 — VLM Per-Seat Structured Occupant Analysis for Automotive HMI

*Dependent on Claim 1. Novel application of VLMs in automotive context.*

A vision-based per-seat cabin analysis method comprising:

1. Receiving a camera frame from an in-cabin camera and encoding it as a base64
   image;
2. Submitting the image to an on-device vision-language model with a structured
   prompt requesting per-seat occupant analysis;
3. Parsing the model response as a JSON object keyed by seat identifier
   (driver, front_passenger, rear_left, rear_right), each seat entry containing:
   - `occupied` (boolean)
   - `kind` (adult | child | infant | pet | unknown)
   - `emotion` (calm | happy | stressed | tired | distressed)
   - `buckled` (boolean)
   - `objects` (array of strings — loose/unsecured items on or near the seat)
4. Applying a vision override guard such that VLM-inferred occupant kind never
   overwrites a manually configured kind, preserving driver intent;
5. Publishing the structured per-seat data to the cabin fusion engine for
   cognitive load scoring and mitigation generation.

**Dependent sub-claims:**
- The VLM is quantized to 4-bit precision (NF4) to fit within vehicle-grade
  GPU VRAM constraints (~2.5 GB).
- The trust rule: if the VLM infers a child/infant/pet kind, the seat is marked
  occupied regardless of the VLM's `occupied` field, prioritising kind over
  occupancy flag.

---

### Claim 5 — Confirm-First Mitigation UX with Grace-Period State Retention

*Dependent on Claim 1. Novel UX safety contract for automotive cabin systems.*

A mitigation management method for in-cabin advisory systems comprising:

1. Classifying each generated mitigation as one of: advisory (requires user
   confirmation before actuator engagement), warning (requires confirmation),
   or critical (immediate activation, no confirmation required);
2. Maintaining per-mitigation state: proposed → active (after Apply) or
   dismissed (after Dismiss);
3. Upon condition clearance, retaining the mitigation in its current state
   (active or dismissed) for a configurable grace period (default 6 seconds)
   before removal from the active mitigation set;
4. Ensuring that no cabin actuator (volume, temperature, lighting) is engaged
   without explicit user confirmation for non-critical mitigations.

---

### Claim 6 — Cognitive Load Score Accumulator with Direct World-State Read

*Dependent on Claim 1.*

A cognitive load computation method comprising:

- A scalar accumulator initialised to zero at each fusion tick;
- Additive contribution rules sourced independently from: radar-derived driver
  vitals (heart rate, respiration rate), vision-derived driver emotion, audio
  classification labels, per-seat child/pet welfare state, inertial road quality,
  and vehicle speed/visibility;
- A deduplication mechanism preventing double-scoring of heart-rate anomalies
  when the same seat is represented in both the radar sensor frame and the world
  state store;
- Reading child/pet presence and heart-rate overrides directly from the world
  state store (bypassing potentially stale sensor frames) to achieve zero-lag
  response to manual occupant configuration;
- Capping the accumulated score at 100 and mapping to labelled bands (Low /
  Medium / High / Critical) for HMI presentation.

---

## Prior Art Landscape

| Feature | Prior Art | Notes |
|---|---|---|
| Radar contactless HR/vitals | TI IWR6843, Vayyar, Bosch, Infineon | Vitals alone not novel |
| Child-left-behind detection | GM, Hyundai, HEAT Act (US law) | Detection alone not novel |
| Cognitive load scoring | Seeing Machines, Smart Eye, hundreds of papers | Broad concept crowded |
| Driver drowsiness (EAR) | Patented since ~2005, Bosch, Mobileye | Not novel |
| Seatbelt + occupancy mismatch | Standard OEM feature | Not novel |
| Multi-modal driver monitoring | Broad category — very crowded | Combination angle needed |
| Audio event detection in car | Some OEM work, but narrow | BabyNet cascade likely novel |
| VLM for structured cabin analysis | No known automotive prior art (as of 2026-06) | Strong novelty |
| Pre-emptive pothole + object advisory | No known prior art | Strongest claim |
| Confirm-first HMI contract | No known automotive patent | UX pattern angle |

---

## Recommended IPC / CPC Classification Codes

For prior art search and filing classification:

- `B60R 21/015` — Electrical circuits for triggering passive safety arrangements
- `B60W 40/08` — Determining cognitive aspects of driver state
- `B60W 50/14` — Means for informing the driver
- `G06V 20/59` — Recognition of objects in vehicle interior
- `G06V 40/10` — Image recognition of persons
- `G10L 25/51` — Speech analysis for detecting emotional state
- `G08B 21/04` — Alarms for child/person presence
- `G06N 3/04` — Neural networks (BabyNet, AST)
- `G06F 3/16` — Multimodal HMI

---

## Search Queries (Google Patents / USPTO)

```
"in-cabin" AND "cognitive load" AND "radar" AND "fusion"
"loose object" AND "pothole" AND "vehicle" AND "advisory"
"infant cry" AND "vehicle" AND "classification" AND "neural network"
"vision language model" AND "vehicle cabin" AND "occupant"
"confirm" AND "mitigation" AND "cabin" AND "non-intrusive"
"child left" AND "radar" AND "contactless" AND "vehicle seat"
```

---

## Filing Strategy

### Step 1 — File Provisional Now (before presentation)

- **Cost:** ~$320 USD (USPTO micro-entity)
- **Buys:** 12-month window to file full application, establishes priority date
- **Covers:** All 6 claims above
- **Form:** USPTO Form SB/16 + written description (this document + CLAUDE.md +
  architecture diagrams from the presentation)

### Step 2 — PCT Application (within 12 months of provisional)

- Covers 150+ countries with one filing
- **Cost:** ~$3,000–5,000 USD
- Target jurisdictions: US, EU (EPO), India, Japan, South Korea, China

### Step 3 — National Phase (within 30 months of priority date)

- Enter national phase in key automotive markets
- Priority: Germany (DE), USA, Japan, South Korea, China

### Estimated Total Cost (if pursuing seriously)

| Stage | Cost (USD) |
|---|---|
| Provisional (self-filed) | $320 |
| Attorney-drafted provisional | $2,000–4,000 |
| PCT application | $3,000–5,000 |
| National phase (5 countries) | $15,000–30,000 |
| **Total to granted patent** | **~$25,000–45,000** |

---

## Strongest Differentiator for Claims

The pre-emptive object-securing advisory (Claim 2) is the most defensible:

1. **Novel:** No known patent combines vision-detected cabin objects + IMU
   road-quality prediction + distance-threshold trigger into a timed advisory.
2. **Non-obvious:** The insight that a loose laptop becomes a projectile hazard
   specifically *before* a pothole (not after) requires combining two unrelated
   sensor streams with a forward-looking time window.
3. **Useful:** Directly prevents injury — clear utility.
4. **Enabled:** Fully implemented and demonstrated in this codebase.
5. **Specific:** Not a broad "detect objects + warn driver" claim — the specific
   distance threshold, escalation logic, and VLM object naming are all concrete.

---

## Contact for Next Steps

- **USPTO Patent Center:** patentcenter.uspto.gov
- **Provisional application guide:** uspto.gov/patents/basics/provisional-application
- **Patent search:** patents.google.com
- **Find an attorney:** patentbar.com (search automotive + AI specialty)
