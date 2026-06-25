"""Zero-shot audio classification via CLAP with temporal stability.

- Larger chunks (config audio.chunk_samples) for richer context
- t+1 delay: classify the previous chunk when the next one arrives
- Majority vote over the last N classifications before publishing
"""
from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from bus import bus
from config import audio as acfg
from schemas import AudioEvent

SR = 16_000
CLAP_SR = 48_000

chunk_queue: asyncio.Queue[list[float]] = asyncio.Queue(maxsize=8)

_HUB_ID = acfg.clap_model
_LOCAL = Path(__file__).resolve().parent.parent / "models" / "clap-htsat-fused"
_MODEL_ID = str(_LOCAL) if _LOCAL.exists() else _HUB_ID

_model: Any = None
_processor: Any = None
_device: str = "cpu"

_PROMPTS: list[str] = []
_PROMPT_CAT: list[dict] = []

_pending: list[float] | None = None
_smooth_hist: deque[tuple[str, float, str, str]] = deque(
    maxlen=max(1, acfg.clap_smooth_chunks),
)


def _build_prompt_index() -> None:
    global _PROMPTS, _PROMPT_CAT
    _PROMPTS, _PROMPT_CAT = [], []
    for cat in acfg.clap_categories:
        for prompt in cat["prompts"]:
            _PROMPTS.append(prompt)
            _PROMPT_CAT.append({
                "id": cat["id"],
                "label": cat["label"],
                "distress_class": cat["distress_class"],
                "prompt": prompt,
            })


_build_prompt_index()


def _load_model() -> None:
    global _model, _processor, _device
    from transformers import ClapModel, ClapProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLAP] Loading {_MODEL_ID} on {_device} …")
    _processor = ClapProcessor.from_pretrained(_MODEL_ID)
    _model = ClapModel.from_pretrained(_MODEL_ID).to(_device)
    _model.eval()
    print(
        f"[CLAP] Ready — {len(_PROMPTS)} prompts, "
        f"chunk={acfg.chunk_samples / SR:.1f}s, "
        f"delay={acfg.clap_delay_chunks} smooth={acfg.clap_smooth_chunks}"
    )


def _classify(samples: list[float]) -> tuple[str, float, str, str]:
    """Return (fusion_label, confidence, distress_class, winning_prompt)."""
    assert _model is not None and _processor is not None

    chunk = acfg.chunk_samples
    arr = np.array(samples, dtype=np.float32)
    if arr.shape[0] < chunk:
        arr = np.pad(arr, (0, chunk - arr.shape[0]))
    elif arr.shape[0] > chunk:
        arr = arr[:chunk]

    wav = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    wav48 = AF.resample(wav, SR, CLAP_SR).squeeze(0).numpy()

    inputs = _processor(
        text=_PROMPTS,
        audio=wav48,
        sampling_rate=CLAP_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits_per_audio[0]
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    best_per_cat: dict[str, tuple[int, float]] = {}
    for i, cat in enumerate(_PROMPT_CAT):
        cid = cat["id"]
        if cid not in best_per_cat or probs[i] > best_per_cat[cid][1]:
            best_per_cat[cid] = (i, float(probs[i]))

    ranked = sorted(best_per_cat.items(), key=lambda x: x[1][1], reverse=True)
    win_id, (win_idx, win_prob) = ranked[0]
    win_cat = next(c for c in acfg.clap_categories if c["id"] == win_id)

    none_prob = best_per_cat.get("none", (0, 0.0))[1]
    if win_id != "none" and win_prob < acfg.clap_conf_threshold:
        return "none", round(win_prob, 3), "none", ""
    if win_id != "none" and "none" in best_per_cat:
        if win_prob - none_prob < acfg.clap_margin_over_none:
            return "none", round(none_prob, 3), "none", ""

    return (
        win_cat["label"],
        round(win_prob, 3),
        win_cat["distress_class"],
        _PROMPT_CAT[win_idx]["prompt"],
    )


def _smooth_vote() -> tuple[str, float, str, str]:
    """Majority label over recent classifications; mean conf for winner."""
    if not _smooth_hist:
        return "none", 0.0, "none", ""
    labels = [h[0] for h in _smooth_hist]
    winner = Counter(labels).most_common(1)[0][0]
    matches = [h for h in _smooth_hist if h[0] == winner]
    conf = round(sum(h[1] for h in matches) / len(matches), 3)
    distress = matches[-1][2]
    prompt = matches[-1][3]
    return winner, conf, distress, prompt


def reset_smoothing() -> None:
    global _pending
    _pending = None
    _smooth_hist.clear()


async def run() -> None:
    global _pending
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)

    while True:
        samples = await chunk_queue.get()

        # t+1: classify the previous chunk when this one arrives
        if _pending is not None:
            try:
                result = await loop.run_in_executor(None, _classify, _pending)
                _smooth_hist.append(result)
                label, conf, distress, prompt = _smooth_vote()
            except Exception as exc:
                print(f"[CLAP] inference error: {exc}")
                label, conf, distress, prompt = "none", 0.0, "none", ""

            await bus.publish(
                "audio",
                AudioEvent(
                    ts=time.time(),
                    label=label,
                    confidence=conf,
                    source="clap_zs",
                    distress_class=distress,
                    prompt=prompt,
                ).to_dict(),
            )

        _pending = samples
