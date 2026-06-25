"""Zero-shot audio classification via CLAP (Contrastive Language-Audio Pretraining).

Scores each 0.96 s browser chunk against natural-language prompts — no fine-tuning
required.  Add or edit prompts in config.yaml → audio.clap_categories.

Publishes the same AudioEvent bus contract as baby_net / AST, plus:
  distress_class  kid | human | pet | vehicle | none
  prompt          winning text prompt (for Pipeline Debug)

Model: laion/clap-htsat-fused (~400 MB, CPU-friendly at ~0.5–2 s/chunk).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from bus import bus
from config import audio as acfg
from schemas import AudioEvent

SR = 16_000          # browser capture rate
CLAP_SR = 48_000     # laion/clap-htsat-fused expects 48 kHz
CHUNK = 15_360       # 0.96 s @ 16 kHz — matches browser AudioWorklet

chunk_queue: asyncio.Queue[list[float]] = asyncio.Queue(maxsize=8)

_HUB_ID = acfg.clap_model
_LOCAL = Path(__file__).resolve().parent.parent / "models" / "clap-htsat-fused"
_MODEL_ID = str(_LOCAL) if _LOCAL.exists() else _HUB_ID

_model: Any = None
_processor: Any = None
_device: str = "cpu"

# Flatten config categories → parallel prompt / metadata lists (built at import).
_PROMPTS: list[str] = []
_PROMPT_CAT: list[dict] = []  # {label, distress_class, id}


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
    print(f"[CLAP] Ready — {len(_PROMPTS)} prompts across "
          f"{len(acfg.clap_categories)} categories")


def _classify(samples: list[float]) -> tuple[str, float, str, str]:
    """Return (fusion_label, confidence, distress_class, winning_prompt)."""
    assert _model is not None and _processor is not None

    arr = np.array(samples, dtype=np.float32)
    if arr.shape[0] < CHUNK:
        arr = np.pad(arr, (0, CHUNK - arr.shape[0]))
    elif arr.shape[0] > CHUNK:
        arr = arr[:CHUNK]

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
        logits = _model(**inputs).logits_per_audio[0]  # [num_prompts]
    probs = F.softmax(logits, dim=-1).cpu().numpy()

    # Best prompt per category (max over prompts in that category).
    best_per_cat: dict[str, tuple[int, float]] = {}
    for i, cat in enumerate(_PROMPT_CAT):
        cid = cat["id"]
        if cid not in best_per_cat or probs[i] > best_per_cat[cid][1]:
            best_per_cat[cid] = (i, float(probs[i]))

    # Pick winning category (highest prob); require margin over 'none' if present.
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


async def run() -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load_model)

    while True:
        samples = await chunk_queue.get()
        try:
            label, conf, distress, prompt = await loop.run_in_executor(
                None, _classify, samples
            )
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
