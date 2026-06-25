"""Download all remote models into backend/models/ for offline packaging.

Usage:
    python download_models.py

After running, the models/ directory will contain:
    models/
    ├── baby_net.pt              (auto-trained on first run if missing)
    ├── face_landmarker.task     (MediaPipe — download separately, see below)
    ├── yolov8n.pt               (YOLOv8n — 6 MB)
    ├── qwen2-vl-2b/             (Qwen2-VL-2B-Instruct 4-bit — ~4 GB)
    └── ast-audioset/            (MIT AST AudioSet — ~330 MB)

face_landmarker.task must be placed manually if not present:
    wget -q -O models/face_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
from __future__ import annotations

import sys
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"
MODELS.mkdir(exist_ok=True)


def _download_yolo() -> None:
    dest = MODELS / "yolov8n.pt"
    if dest.exists():
        print(f"[yolov8n] already present ({dest.stat().st_size // 1024} KB)")
        return
    print("[yolov8n] Downloading …")
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")  # downloads to CWD or cache
    import shutil
    src = Path("yolov8n.pt")
    if src.exists():
        shutil.move(str(src), str(dest))
    print(f"[yolov8n] Saved → {dest}")


def _download_hf(hub_id: str, local_name: str) -> None:
    dest = MODELS / local_name
    if dest.exists():
        print(f"[{local_name}] already present")
        return
    print(f"[{local_name}] Downloading {hub_id} …")
    from transformers import AutoProcessor, AutoModelForAudioClassification, Qwen2VLForConditionalGeneration
    dest.mkdir(parents=True, exist_ok=True)
    if "Qwen" in hub_id:
        import torch
        proc  = __import__("transformers").AutoProcessor.from_pretrained(hub_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(hub_id, torch_dtype=torch.float16)
        proc.save_pretrained(str(dest))
        model.save_pretrained(str(dest))
    else:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        proc  = ASTFeatureExtractor.from_pretrained(hub_id)
        model = ASTForAudioClassification.from_pretrained(hub_id)
        proc.save_pretrained(str(dest))
        model.save_pretrained(str(dest))
    print(f"[{local_name}] Saved → {dest}")


def main() -> None:
    _download_yolo()
    _download_hf("MIT/ast-finetuned-audioset-10-10-0.4593", "ast-audioset")
    _download_hf("Qwen/Qwen2-VL-2B-Instruct", "qwen2-vl-2b")

    mp = MODELS / "face_landmarker.task"
    if not mp.exists():
        print("\n[face_landmarker] NOT FOUND — download manually:")
        print("  wget -q -O models/face_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
    else:
        print(f"[face_landmarker] present ({mp.stat().st_size // 1024} KB)")

    print("\nAll models ready in backend/models/")


if __name__ == "__main__":
    main()
