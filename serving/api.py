"""FastAPI entrypoint — same inference path as the Streamlit app, over HTTP."""

import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES, load_stroke_model, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# One JSON object per line, nothing around it: that is what Cloud Logging parses into fields.
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("stroke.predict")


@lru_cache(maxsize=1)
def get_model():
    """Cached, so the ONNX session is built once rather than per request."""
    return load_stroke_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the session before the first request: the cold start absorbs it, not a caller.
    get_model()
    yield


# Callers are authenticated by Cloud Run IAM before a request reaches this app.
app = FastAPI(
    title="Stroke Classification API",
    version="0.1.0",
    description="Research demo — not for clinical use.",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "classes": list(CLASS_NAMES)}


@app.post("/predict")
async def predict_image(
    file: Annotated[UploadFile, File()],
    model: Annotated[tuple, Depends(get_model)],
) -> dict:
    # Client-declared size first so an oversized body is refused before it is read into memory;
    # it is absent on chunked uploads, hence the second check.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image") from None

    session, preprocess = model
    started = time.perf_counter()
    prediction, confidence, probabilities = predict(session, preprocess, image)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    # Scan itself is never logged — only what monitoring needs.
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "latency_ms": latency_ms,
                "image_bytes": len(raw),
            }
        )
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
