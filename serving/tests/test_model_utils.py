import numpy as np
import pytest

from model_utils import CLASS_NAMES, _softmax, predict, preprocess_image


def test_preprocess_returns_nchw_float32(white_image):
    arr = preprocess_image(white_image)
    assert arr.shape == (1, 3, 299, 299)
    assert arr.dtype == np.float32


def test_preprocess_applies_imagenet_normalisation(white_image):
    arr = preprocess_image(white_image)
    # A pure white pixel becomes (1 - mean) / std per channel.
    expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(arr[0, :, 0, 0], expected, atol=1e-5)


def test_softmax_sums_to_one():
    probs = _softmax(np.array([2.0, -1.0], dtype=np.float32))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] > probs[1]


def test_predict_reports_the_argmax_class(white_image, fake_session):
    prediction, confidence, probabilities = predict(
        fake_session([0.1, 5.0]), preprocess_image, white_image
    )

    assert prediction == "Stroke"
    assert confidence > 0.9
    assert set(probabilities) == set(CLASS_NAMES)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_predict_feeds_the_preprocessed_tensor(white_image, fake_session):
    session = fake_session([1.0, 0.0])
    predict(session, preprocess_image, white_image)

    assert session.received is not None
    assert session.received["input"].shape == (1, 3, 299, 299)
