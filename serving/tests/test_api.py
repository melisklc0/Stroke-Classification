import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api
from model_utils import preprocess_image

# TestClient is built without a `with` block on purpose: lifespan never runs, so no test
# loads the real ONNX file.


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def stroke_model(fake_session):
    api.app.dependency_overrides[api.get_model] = lambda: (
        fake_session([0.0, 5.0]),
        preprocess_image,
    )
    yield
    api.app.dependency_overrides.clear()


def png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 120, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_healthz_lists_the_classes(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "classes": ["No-Stroke", "Stroke"]}


def test_predict_returns_class_and_probabilities(client, stroke_model):
    response = client.post("/predict", files={"file": ("scan.png", png_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "Stroke"
    assert body["confidence"] == pytest.approx(body["probabilities"]["Stroke"])


def test_predict_rejects_a_non_image(client, stroke_model):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})

    assert response.status_code == 400
    assert response.json()["detail"] == "File is not a readable image"


def test_predict_rejects_an_oversized_upload(client, stroke_model):
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * api.MAX_UPLOAD_BYTES
    response = client.post("/predict", files={"file": ("big.png", oversized, "image/png")})

    assert response.status_code == 413


def test_api_key_is_enforced_when_configured(client, stroke_model, monkeypatch):
    monkeypatch.setattr(api, "API_KEY", "secret")
    files = {"file": ("scan.png", png_bytes(), "image/png")}

    assert client.post("/predict", files=files).status_code == 401
    assert client.post("/predict", files=files, headers={"X-API-Key": "secret"}).status_code == 200
