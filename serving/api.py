"""FastAPI entrypoint — same inference path as the Streamlit app, over HTTP."""

import io
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES, load_stroke_model, predict

API_KEY = os.environ.get("STROKE_API_KEY", "")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@lru_cache(maxsize=1)
def get_model():
    """Cached, so the ONNX session is built once rather than per request."""
    return load_stroke_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the session before the first request: the cold start absorbs it, not a caller.
    get_model()
    yield


app = FastAPI(
    title="Stroke Classification API",
    version="0.1.0",
    description="Research demo — not for clinical use.",
    lifespan=lifespan,
)


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "classes": list(CLASS_NAMES)}


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict_image(
    file: Annotated[UploadFile, File()],
    model: Annotated[tuple, Depends(get_model)],
) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image") from None

    session, preprocess = model
    prediction, confidence, probabilities = predict(session, preprocess, image)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
