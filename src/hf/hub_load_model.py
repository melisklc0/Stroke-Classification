"""
Standalone inference module for the Hugging Face model repo.

This file is uploaded as ``load_model.py`` next to ``model.pth``. Copy-paste friendly:
only needs ``torch``, ``torchvision``, ``pillow``, ``huggingface_hub``.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models, transforms
from PIL import Image

IMAGE_SIZE = 299
CLASS_NAMES = ("No-Stroke", "Stroke")


class StrokeHead(nn.Sequential):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )


def build_model(num_classes: int = 2) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = StrokeHead(in_features, num_classes)
    return m


def _normalize_state_dict(state: dict) -> dict:
    if any(k.startswith("model.") for k in state):
        state = {k.replace("model.", ""): v for k, v in state.items()}
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def _unwrap(raw: object) -> dict:
    if isinstance(raw, dict) and "state_dict" in raw:
        return dict(raw["state_dict"])
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return dict(raw["model_state_dict"])
    if not isinstance(raw, dict):
        raise TypeError(f"Expected state dict, got {type(raw)}")
    return raw


def eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_model(
    repo_id: str,
    *,
    weights_name: str = "model.pth",
    image_size: int = IMAGE_SIZE,
    token: str | None = None,
):
    """
    Download weights from the Hub, build EfficientNet-B0 + head, return ``(model, transform)``.
    """
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_name,
        repo_type="model",
        token=tok,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(path, map_location=device, weights_only=False)
    state = _normalize_state_dict(_unwrap(raw))

    model = build_model()
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, eval_transform(image_size)


def predict_proba(model: nn.Module, tfm: transforms.Compose, img: Image.Image):
    device = next(model.parameters()).device
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    return {CLASS_NAMES[i]: p[i].item() for i in range(len(CLASS_NAMES))}
