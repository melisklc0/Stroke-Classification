"""Checkpoint normalization before uploading to Hugging Face (strip wrapper prefixes)."""

from __future__ import annotations

import torch


def unwrap_checkpoint(raw: object) -> dict:
    """Torch save payload -> flat state dict."""
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            return dict(raw["state_dict"])
        if "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
            return dict(raw["model_state_dict"])
        return raw
    raise TypeError(f"Expected a state dict or dict wrapper, got {type(raw)}")


def normalize_state_dict(state: dict) -> dict:
    """Strip ``model.`` / ``module.`` prefixes from wrapped checkpoints."""
    if any(k.startswith("model.") for k in state):
        state = {k.replace("model.", ""): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def prepare_hub_checkpoint(local_path: str, device: torch.device | None = None) -> dict:
    """Load local .pth and return canonical state dict for Hub upload."""
    device = device or torch.device("cpu")
    raw = torch.load(local_path, map_location=device, weights_only=False)
    state = unwrap_checkpoint(raw)
    return normalize_state_dict(state)
