"""Multi-second feed analysis — corroborate audio, vision objects, and emotions."""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

from config import fusion as cfg

_LOOSE_LABELS = frozenset({
    "backpack", "suitcase", "bottle", "cup", "book", "laptop",
    "handbag", "cell phone", "remote", "mouse", "teddy bear", "wine glass",
})
_DISTRESS_EMOTIONS = frozenset({"stressed", "distressed", "tired"})
_CHILD_PET = frozenset({"child", "infant", "pet"})

_WINDOW_S = cfg.recommend_analysis_s
_MIN_RATIO = cfg.recommend_min_ratio
_AUDIO_THRESH = cfg.score.audio_conf_threshold


class FeedAnalyzer:
  """Ring buffer of fusion ticks; sustained-signal queries for recommendations."""

  def __init__(self) -> None:
    self._ticks: list[dict[str, Any]] = []
    self._started_at: float | None = None

  def reset(self) -> None:
    self._ticks.clear()
    self._started_at = None

  def observe(self, tick: dict[str, Any]) -> None:
    now = time.time()
    if self._started_at is None:
      self._started_at = now
    self._ticks.append({"ts": now, **tick})
    cutoff = now - _WINDOW_S
    self._ticks = [t for t in self._ticks if t["ts"] >= cutoff]

  def ready(self) -> bool:
    if len(self._ticks) < 4:
      return False
    span = self._ticks[-1]["ts"] - self._ticks[0]["ts"]
    return span >= _WINDOW_S * 0.75

  def _ratio(self, pred) -> float:
    if not self._ticks:
      return 0.0
    return sum(1 for t in self._ticks if pred(t)) / len(self._ticks)

  def sustained_audio(self, *labels: str) -> tuple[bool, str, float]:
    labels_set = set(labels)

    def match(t: dict) -> bool:
      lbl = t.get("audio_label", "none")
      conf = float(t.get("audio_conf", 0) or 0)
      return lbl in labels_set and conf > _AUDIO_THRESH

    ratio = self._ratio(match)
    if ratio < _MIN_RATIO:
      return False, "none", 0.0
    # dominant label among matching ticks
    counts: Counter[str] = Counter()
    conf_sum: dict[str, float] = defaultdict(float)
    conf_n: dict[str, int] = defaultdict(int)
    for t in self._ticks:
      lbl = t.get("audio_label", "none")
      conf = float(t.get("audio_conf", 0) or 0)
      if lbl in labels_set and conf > _AUDIO_THRESH:
        counts[lbl] += 1
        conf_sum[lbl] += conf
        conf_n[lbl] += 1
    if not counts:
      return False, "none", 0.0
    best = counts.most_common(1)[0][0]
    avg_conf = conf_sum[best] / conf_n[best]
    return True, best, round(avg_conf, 2)

  def sustained_emotion(self, seat_id: str, *emotions: str) -> bool:
    emo_set = set(emotions)

    def match(t: dict) -> bool:
      seats = t.get("seats") or {}
      sv = seats.get(seat_id) or {}
      return bool(sv.get("occupied")) and sv.get("emotion") in emo_set

    return self._ratio(match) >= _MIN_RATIO

  def seats_with_sustained_emotion(self, *emotions: str) -> list[str]:
    emo_set = set(emotions)
    seat_ids = set()
    for t in self._ticks:
      for sid, sv in (t.get("seats") or {}).items():
        if sv.get("occupied") and sv.get("emotion") in emo_set:
          seat_ids.add(sid)
    out = []
    for sid in seat_ids:
      if self.sustained_emotion(sid, *emotions):
        out.append(sid)
    return out

  def sustained_objects(self) -> dict[str, list[str]]:
    """Seat → object labels seen in ≥min_ratio of ticks."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    n = len(self._ticks) or 1
    for t in self._ticks:
      for sid, objs in (t.get("objects_by_seat") or {}).items():
        for obj in objs:
          counts[sid][obj] += 1
    out: dict[str, list[str]] = {}
    need = max(1, int(n * _MIN_RATIO))
    for sid, ctr in counts.items():
      stable = [obj for obj, c in ctr.items() if c >= need]
      if stable:
        out[sid] = stable
    return out

  def child_pet_seats(self) -> set[str]:
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    for t in self._ticks:
      for sid, sv in (t.get("seats") or {}).items():
        if sv.get("occupied"):
          kinds[sid][sv.get("kind", "unknown")] += 1
    out: set[str] = set()
    need = max(1, int(len(self._ticks) * _MIN_RATIO))
    for sid, ctr in kinds.items():
      top = ctr.most_common(1)[0][0] if ctr else "unknown"
      if top in _CHILD_PET and ctr[top] >= need:
        out.add(sid)
    return out

  def sustained_drowsy(self) -> bool:
    return self._ratio(lambda t: bool(t.get("driver_drowsy"))) >= _MIN_RATIO

  def snapshot(self) -> dict[str, Any]:
    n = len(self._ticks)
    span = round(self._ticks[-1]["ts"] - self._ticks[0]["ts"], 1) if n > 1 else 0.0
    ok, aud, conf = self.sustained_audio(
      "crying", "barking", "animal", "shouting", "talking", "happy", "rattle",
    )
    if not ok:
      aud, conf = "none", 0.0
    objs = self.sustained_objects()
    emo_seats = {
      sid: sv.get("emotion", "calm")
      for t in [self._ticks[-1]] if self._ticks
      for sid, sv in (t.get("seats") or {}).items()
      if sv.get("occupied")
    } if self._ticks else {}
    return {
      "ready": self.ready(),
      "window_s": _WINDOW_S,
      "ticks": n,
      "span_s": span,
      "dominant_audio": aud,
      "audio_conf": conf,
      "distressed_seats": self.seats_with_sustained_emotion(
        "stressed", "distressed", "tired",
      ),
      "objects_by_seat": objs,
      "emotions": emo_seats,
      "child_pet_seats": sorted(self.child_pet_seats()),
    }


_feed = FeedAnalyzer()


def get_feed() -> FeedAnalyzer:
  return _feed


def reset_feed() -> None:
  _feed.reset()
