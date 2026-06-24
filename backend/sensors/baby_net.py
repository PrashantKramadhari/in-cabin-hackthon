"""BabyNet: lightweight mel-spec CNN fine-tuned on project baby audio samples.

Classes:  0=none  1=crying  2=talking  3=happy
Cascade:  BabyNet fast-path (≥0.60 conf) → AST fallback for all other sounds.
Latency:  ~10 ms GPU / ~30 ms CPU for BabyNet alone.

Bus contract: same as audio_yamnet.py — publishes AudioEvent to 'audio' topic.
chunk_queue is drained by this node (replaces audio_yamnet when loaded).
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

from bus import bus
from schemas import AudioEvent

# ── Paths ───────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent.parent
_AUDIO_DIR  = _ROOT / "audio"
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baby_net.pt"

# ── Constants ────────────────────────────────────────────────────────────────
SR      = 16_000
CHUNK   = 15_360          # 0.96 s @ 16 kHz  (matches browser AudioWorklet)
N_MELS  = 64
CLASSES = ["none", "crying", "talking", "happy"]
_THRESH = 0.60            # min BabyNet confidence to skip AST fallback

chunk_queue: asyncio.Queue[list[float]] = asyncio.Queue(maxsize=8)


# ── Model ────────────────────────────────────────────────────────────────────
class BabyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = AT.MelSpectrogram(
            sample_rate=SR, n_fft=400, hop_length=160,
            n_mels=N_MELS, f_min=50, f_max=8000,
        )
        self.db = AT.AmplitudeToDB(top_db=80)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(128 * 4 * 4, len(CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, CHUNK] float32 waveform at SR
        s = self.db(self.mel(x)).unsqueeze(1)   # [B, 1, N_MELS, T]
        return self.head(self.conv(s).flatten(1))


# ── Data helpers ─────────────────────────────────────────────────────────────
def _load_mono(path: Path) -> torch.Tensor:
    w, sr = torchaudio.load(str(path))
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    if sr != SR:
        w = AF.resample(w, sr, SR)
    return w.squeeze(0)


def _chunks(w: torch.Tensor, stride: int) -> list[torch.Tensor]:
    out, n = [], w.shape[0]
    i = 0
    while i + CHUNK <= n:
        out.append(w[i: i + CHUNK])
        i += stride
    return out or [nn.functional.pad(w, (0, CHUNK - w.shape[0]))]


def _augment(w: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Fast augmentation: noise, gain, time-shift, polarity — no resampling."""
    out = []
    L = w.shape[0]
    for _ in range(n):
        x = w.clone()
        # Random time-shift crop
        if L >= CHUNK:
            s = random.randint(0, L - CHUNK)
            x = x[s: s + CHUNK]
        else:
            x = nn.functional.pad(x, (0, CHUNK - L))
        # Gaussian noise
        x = x + torch.randn_like(x) * random.uniform(0.001, 0.02)
        # Random gain
        x = x * random.uniform(0.3, 1.8)
        # Polarity flip
        if random.random() < 0.5:
            x = -x
        # Zero-out random short segment (SpecAugment-style on time domain)
        mask_start = random.randint(0, CHUNK - CHUNK // 6)
        mask_len   = random.randint(0, CHUNK // 6)
        x[mask_start: mask_start + mask_len] = 0.0
        out.append(x.clamp(-1.0, 1.0))
    return out


def _none_samples(n: int = 150) -> list[torch.Tensor]:
    out = []
    for _ in range(n):
        amp = random.uniform(0.0005, 0.04)
        out.append(torch.randn(CHUNK) * amp)
    return out


def _build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    label_map = {
        "baby_crying.mp3": 1,
        "baby_talking.mp3": 2,
        "baby_happy.mp3": 3,
    }
    xs: list[torch.Tensor] = []
    ys: list[int] = []

    # None class
    for s in _none_samples(150):
        xs.append(s); ys.append(0)

    # Baby classes
    for fname, label in label_map.items():
        path = _AUDIO_DIR / fname
        if not path.exists():
            print(f"[BabyNet] Warning: {path} not found — skipping")
            continue
        w = _load_mono(path)
        base = _chunks(w, stride=SR)         # 1 s stride — fast enough
        n_aug = max(8, 80 // max(1,len(base)))  # target ~80 samples per class
        for chunk in base:
            for aug in _augment(chunk, n=n_aug):
                xs.append(aug); ys.append(label)
        print(f"[BabyNet] {fname}: {len(base)} chunks × {n_aug} aug = "
              f"{len(base)*n_aug} samples")

    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


# ── Training ─────────────────────────────────────────────────────────────────
def _train() -> BabyCNN:
    print("[BabyNet] Training from audio samples …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = _build_dataset()

    # Shuffle
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]

    model = BabyCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150)
    loss_fn = nn.CrossEntropyLoss()

    BATCH, N_EPOCH = 32, 150
    for epoch in range(N_EPOCH):
        model.train()
        total = 0.0
        for i in range(0, len(X), BATCH):
            xb = X[i: i + BATCH].to(device)
            yb = Y[i: i + BATCH].to(device)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward(); opt.step()
            total += l.item()
        sched.step()
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}/{N_EPOCH}  loss={total:.3f}")

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _MODEL_PATH)
    print(f"[BabyNet] Saved → {_MODEL_PATH}")
    return model.eval()


# ── AST fallback (lazy-loaded) ────────────────────────────────────────────────
_ast_processor: Any = None
_ast_model: Any     = None
_AST_MAP  = {22: "crying", 21: "crying", 80: "animal", 74: "animal",
             73: "animal", 468: "rattle", 0: "speech"}
_AST_ID   = "MIT/ast-finetuned-audioset-10-10-0.4593"
_AST_THRESH = 0.15


def _load_ast() -> None:
    global _ast_processor, _ast_model
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    _ast_processor = ASTFeatureExtractor.from_pretrained(_AST_ID)
    _ast_model     = ASTForAudioClassification.from_pretrained(_AST_ID)
    _ast_model.eval()
    print("[BabyNet] AST fallback loaded")


def _ast_infer(samples: list[float]) -> tuple[str, float]:
    arr = np.array(samples, dtype=np.float32)
    inp = _ast_processor(arr, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(_ast_model(**inp).logits, dim=-1).squeeze()
    best_l, best_c = "none", 0.0
    for idx, lbl in _AST_MAP.items():
        c = float(probs[idx])
        if c > best_c:
            best_c, best_l = c, lbl
    if best_c < _AST_THRESH:
        best_l = "none"
    return best_l, round(best_c, 3)


# ── Module-level state ────────────────────────────────────────────────────────
_model:  BabyCNN | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _load_or_train() -> None:
    global _model
    m = BabyCNN().to(_device)
    if _MODEL_PATH.exists():
        m.load_state_dict(torch.load(_MODEL_PATH, map_location=_device, weights_only=True))
        print(f"[BabyNet] Loaded weights from {_MODEL_PATH}")
    else:
        m = _train().to(_device)
    m.eval()
    _model = m


def classify(samples: list[float]) -> tuple[str, float]:
    """Cascade: BabyNet fast → AST fallback if low-confidence or 'none'."""
    assert _model is not None, "BabyNet not loaded"
    x = torch.tensor(samples, dtype=torch.float32).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=-1).squeeze()
    top_val, top_idx = probs.max(0)
    label, conf = CLASSES[int(top_idx)], float(top_val)

    if conf >= _THRESH and label != "none":
        return label, round(conf, 3)

    # AST fallback
    if _ast_model is not None:
        return _ast_infer(samples)
    return "none", 0.0


async def run() -> None:
    loop = asyncio.get_event_loop()
    print("[BabyNet] Initialising …")
    await loop.run_in_executor(None, _load_or_train)
    await loop.run_in_executor(None, _load_ast)
    print("[BabyNet] Ready")
    while True:
        samples = await chunk_queue.get()
        try:
            label, conf = await loop.run_in_executor(None, classify, samples)
        except Exception as exc:
            print(f"[BabyNet] inference error: {exc}")
            label, conf = "none", 0.0
        await bus.publish(
            "audio",
            AudioEvent(ts=time.time(), label=label, confidence=conf).to_dict(),
        )
