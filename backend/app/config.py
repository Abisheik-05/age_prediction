from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Settings:
    app_title: str = "Chrono AI"
    version: str = "0.1.0"
    base_dir: Path = Path(__file__).resolve().parent.parent
    model_dir: Path = base_dir / "models"
    model_path: Path = model_dir / "age_resnet101_finetuned_best.pth"
    uploads_dir: Path = base_dir / "uploads"
    mongodb_uri: str | None = os.getenv("MONGODB_URI")
    allowed_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


settings = Settings()


def detect_device() -> str:
    """Return the active PyTorch device name."""
    return "cuda" if torch.cuda.is_available() else "cpu"
