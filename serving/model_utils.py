import os
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

IMAGE_SIZE = 299
CLASS_NAMES = ("No-Stroke", "Stroke")
REPO_ID = os.environ.get("STROKE_MODEL_REPO", "melisklc0/efficientnet-b0-stroke-distilled")
ONNX_FILENAME = "model.onnx"
# Set when the ONNX file is baked into the image at build time; empty means pull from the Hub.
MODEL_PATH = os.environ.get("STROKE_MODEL_PATH", "")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def preprocess_image(img: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB, resize, ImageNet normalize -> NCHW float32."""
    rgb = img.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0)


def resolve_onnx_path() -> str:
    """Local ONNX file when baked into the image, otherwise downloaded from the Hub."""
    if MODEL_PATH:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"STROKE_MODEL_PATH is set but missing: {MODEL_PATH}")
        return MODEL_PATH
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=ONNX_FILENAME,
        repo_type="model",
    )


def load_stroke_model():
    """Build an inference session from the baked-in or Hub-downloaded ONNX file."""
    onnx_path = resolve_onnx_path()
    providers: list[str] = ["CPUExecutionProvider"]
    if ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, preprocess_image


def predict(session: ort.InferenceSession, preprocess: Any, img: Image.Image):
    x = preprocess(img)
    inp = session.get_inputs()[0].name
    logits = session.run(None, {inp: x})[0]
    probs = _softmax(logits[0])

    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    pred_idx = int(np.argmax(probs))
    prediction = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return prediction, confidence, results
