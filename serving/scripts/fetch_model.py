"""Download the ONNX model at build time so the container starts without calling the Hub."""

import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ID = os.environ.get("STROKE_MODEL_REPO", "melisklc0/efficientnet-b0-stroke-distilled")
# Hub commit of the model this image serves. Bumping it is how a new model is promoted:
# the change goes through a PR, and merging it deploys.
REVISION = "913f4b756868f508e1fc5724e68977d1ac4e1678"
ONNX_FILENAME = "model.onnx"
DEST = Path(os.environ.get("STROKE_MODEL_PATH", "model/model.onnx"))


def main() -> None:
    cached = hf_hub_download(
        repo_id=REPO_ID, filename=ONNX_FILENAME, repo_type="model", revision=REVISION
    )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, DEST)
    logger.info(
        "Baked %s@%s (%.1f MB) into %s", REPO_ID, REVISION[:8], DEST.stat().st_size / 1e6, DEST
    )


if __name__ == "__main__":
    main()
