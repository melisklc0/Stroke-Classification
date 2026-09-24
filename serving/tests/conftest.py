import numpy as np
import pytest
from PIL import Image


class FakeSession:
    """Stands in for an onnxruntime session, returning fixed logits."""

    def __init__(self, logits: list[float]):
        self._logits = logits
        self.received: dict | None = None

    def get_inputs(self):
        class _Input:
            name = "input"

        return [_Input()]

    def run(self, _outputs, feeds):
        self.received = feeds
        return [np.array([self._logits], dtype=np.float32)]


@pytest.fixture
def fake_session():
    return FakeSession


@pytest.fixture
def white_image() -> Image.Image:
    return Image.new("RGB", (64, 48), color=(255, 255, 255))
