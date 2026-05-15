"""Export EfficientNet-B0 + stroke head to ONNX (opset 18, dynamic batch) for Hub / ONNX Runtime."""

from __future__ import annotations

import os

import torch

from src.hf.checkpoint import prepare_hub_checkpoint
from src.hf.hub_load_model import build_model


def export_to_onnx(pth_path: str, onnx_path: str) -> None:
    print(f"Loading model from {pth_path}...")
    state = prepare_hub_checkpoint(pth_path)
    model = build_model()
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 299, 299)
    parent = os.path.dirname(os.path.abspath(onnx_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    print("Export complete!")
