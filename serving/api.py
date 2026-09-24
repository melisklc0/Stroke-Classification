"""FastAPI entrypoint — same inference path as the Streamlit app, over HTTP."""

import io
import os
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from model_utils import CLASS_NAMES, load_stroke_model, predict

API_KEY = os.environ.get("STROKE_API_KEY", "")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Stroke Classification API",
    version="0.1.0",
    description="Research demo — not for clinical use.",
)

# Loaded once at import: the cold start pays for this, not every request.
_session, _preprocess = load_stroke_model()


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "classes": list(CLASS_NAMES)}


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict_image(file: Annotated[UploadFile, File()]) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a readable image")

    prediction, confidence, probabilities = predict(_session, _preprocess, image)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }
