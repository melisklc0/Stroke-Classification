"""Upload to HF Model Hub: full bundle (``model.pth``, ``load_model.py``, ``model_config.json``, README) and/or ``model.onnx`` only."""

from __future__ import annotations

import json
import os
import tempfile

import torch
from huggingface_hub import HfApi

from src.hf.checkpoint import prepare_hub_checkpoint


def push_model(
    config: dict,
    weight_path: str | None = None,
    onnx_path: str | None = None,
) -> None:
    if not weight_path and not onnx_path:
        print("ERROR: provide weight_path and/or onnx_path.")
        return
    if weight_path and not os.path.isfile(weight_path):
        print(f"ERROR: checkpoint not found: {weight_path}")
        return
    if onnx_path and not os.path.isfile(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return

    hf_user = config["data"]["hf_repo_id"].split("/")[0]
    slug = config["model"].get("hf_model_id")
    if not slug:
        print("ERROR: set model.hf_model_id in config.yaml (Hub repo name without user).")
        return
    repo_id = f"{hf_user}/{slug}"

    img_size = config["data"].get("image_size", 299)
    classes = config["data"].get("classes", {})

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: create_repo: {e}")

    if not weight_path:
        print(f"Uploading ONNX only to https://huggingface.co/{repo_id}")
        try:
            api.upload_file(
                path_or_fileobj=onnx_path,
                path_in_repo="model.onnx",
                repo_id=repo_id,
                repo_type="model",
            )
            print("Uploaded model.onnx")
            print(f"Done: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"ERROR: {e}")
        return

    state = prepare_hub_checkpoint(weight_path)
    fd, tmp_pth = tempfile.mkstemp(suffix=".pth")
    os.close(fd)
    torch.save(state, tmp_pth)

    cfg = {
        "framework": "pytorch",
        "architecture": "efficientnet_b0_custom_head",
        "num_classes": config["model"].get("num_classes", 2),
        "image_size": img_size,
        "class_indices": classes,
        "weights_file": "model.pth",
        "onnx_file": "model.onnx",
    }
    fd2, tmp_cfg = tempfile.mkstemp(suffix=".json")
    os.close(fd2)
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    readme = f"""---
language: en
license: apache-2.0
tags:
- pytorch
- medical
- image-classification
- stroke-detection
pipeline_tag: image-classification
---

# Stroke classifier (EfficientNet-B0)

Distilled EfficientNet-B0 for binary **No-Stroke / Stroke** logits on head CT–style RGB images.

**Not for clinical use** — research / demo only.

## Files

| File | Role |
|------|------|
| `model.pth` | PyTorch state dict (canonical keys) |
| `model.onnx` | ONNX opset 18 (optional; for ONNX Runtime / Spaces) |
| `load_model.py` | Architecture + `load_model(repo_id)` |
| `model_config.json` | Input size and class indices |

Input: RGB, **{img_size}×{img_size}**, ImageNet normalization. ONNX input name: `input`, shape `(N,3,{img_size},{img_size})`.

## Install

```bash
pip install torch torchvision pillow huggingface_hub
```

## Load (minimal)

```python
from huggingface_hub import hf_hub_download
import torch

from load_model import build_model  # save ``load_model.py`` from this repo next to your script

path = hf_hub_download(repo_id="{repo_id}", filename="model.pth", repo_type="model")
model = build_model()
model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()
```

Or download weights automatically:

```python
from load_model import load_model, predict_proba
from PIL import Image

m, tfm = load_model("{repo_id}")
print(predict_proba(m, tfm, Image.open("image.png")))
```
"""

    readme_path = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    readme_path.write(readme)
    readme_path.close()

    loader_src = os.path.join(os.path.dirname(__file__), "hub_load_model.py")

    print(f"Pushing to https://huggingface.co/{repo_id}")
    try:
        api.upload_file(path_or_fileobj=tmp_pth, path_in_repo="model.pth", repo_id=repo_id, repo_type="model")
        api.upload_file(path_or_fileobj=loader_src, path_in_repo="load_model.py", repo_id=repo_id, repo_type="model")
        api.upload_file(path_or_fileobj=tmp_cfg, path_in_repo="model_config.json", repo_id=repo_id, repo_type="model")
        api.upload_file(path_or_fileobj=readme_path.name, path_in_repo="README.md", repo_id=repo_id, repo_type="model")
        if onnx_path:
            api.upload_file(
                path_or_fileobj=onnx_path,
                path_in_repo="model.onnx",
                repo_id=repo_id,
                repo_type="model",
            )
            print("Uploaded model.onnx")
        print(f"Done: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        for p in (tmp_pth, tmp_cfg, readme_path.name):
            if os.path.isfile(p):
                os.remove(p)
