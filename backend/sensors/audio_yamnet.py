"""Live audio classification node — PyTorch / HuggingFace AST.

Uses Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593),
a pure-PyTorch model trained on the same 527-class AudioSet as YAMNet.
Identical bus contract to the synthetic audio node.

Browser captures mic at 16 kHz, sends 0.96 s Float32 chunks over websocket as
{cmd: "audio_chunk", data: [...]}.  This node drains chunk_queue, runs AST
inference on a thread pool, and publishes AudioEvent to the "audio" bus topic.

AudioSet class indices used (same numbering as YAMNet):
  22  — Baby cry / infant cry  → "crying"
  21  — Crying / sobbing       → "crying"
  80  — Cat                    → "animal"
  74  — Dog                    → "animal"
  73  — Animal (generic)       → "animal"
  468 — Rattle                 → "rattle"
  0   — Speech                 → "speech"
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import torch

from bus import bus
from schemas import AudioEvent

_processor: Any = None
_model: Any = None

_LABEL_MAP = {
    22: "crying",
    21: "crying",
    80: "animal",
    74: "animal",
    73: "animal",
    468: "rattle",
    0:  "speech",
}
_THRESHOLD = 0.25
_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

chunk_queue: asyncio.Queue[list[float]] = asyncio.Queue(maxsize=8)


def _load_model() -> None:
    global _processor, _model
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    _processor = ASTFeatureExtractor.from_pretrained(_MODEL_ID)
    _model = ASTForAudioClassification.from_pretrained(_MODEL_ID)
    _model.eval()


def _infer(samples: list[float]) -> tuple[str, float]:
    arr = np.array(samples, dtype=np.float32)
    inputs = _processor(arr, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze()

    # check our vocab first
    best_label, best_conf = "none", 0.0
    for idx, label in _LABEL_MAP.items():
        conf = float(probs[idx])
        if conf > best_conf:
            best_conf, best_label = conf, label

    if best_conf < _THRESHOLD:
        best_label = "none"
    return best_label, round(best_conf, 3)


async def run() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)

    while True:
        samples = await chunk_queue.get()
        label, conf = await loop.run_in_executor(None, _infer, samples)
        await bus.publish("audio", AudioEvent(ts=time.time(), label=label, confidence=conf).to_dict())
