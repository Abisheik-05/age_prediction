from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
from torchvision.models import resnet101

_MODEL_CACHE: Final[dict[str, nn.Module]] = {}


def detect_device() -> torch.device:
    """Return the active PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(device: torch.device) -> nn.Module:
    """Build the exact ResNet101 regression model used for age prediction."""
    model = resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return model.to(device)


def load_model(model_path: str | Path, device: torch.device | None = None) -> tuple[nn.Module, torch.device]:
    """Load the trained age model once and return both the model and active device."""
    file_path = Path(model_path).expanduser().resolve()
    active_device = device or detect_device()
    cache_key = f"{file_path}:{active_device.type}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], active_device

    if not file_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {file_path}")

    model = build_model(active_device)

    try:
        checkpoint = torch.load(str(file_path), map_location=active_device)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint from {file_path}: {exc}") from exc

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {file_path}. Expected a state_dict dictionary."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Model architecture mismatch while loading {file_path}. "
            "Ensure the checkpoint was trained with the exact ResNet101 head: "
            "Linear(in_features, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 1)."
        ) from exc

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model, active_device
