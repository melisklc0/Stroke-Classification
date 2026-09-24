# Stroke serving image — FastAPI + ONNX Runtime inference API.
# Needs BuildKit for the uv cache mount: `make build` locally, `make release` for Cloud Build.
#
# ==========================================
# Builder Stage
# ==========================================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Bake the ONNX file in: a cold start must not depend on the Hub being up.
COPY scripts ./scripts
ENV STROKE_MODEL_PATH=/app/model/model.onnx
RUN /app/.venv/bin/python scripts/fetch_model.py

# ==========================================
# Runtime Stage
# ==========================================
FROM python:3.13-slim AS runtime

WORKDIR /app

RUN useradd --create-home --shell /usr/sbin/nologin app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/model /app/model
COPY --chown=app:app api.py model_utils.py ./
COPY --chown=app:app assets ./assets

# Ensure runtime entrypoints stay executable
RUN find /app/.venv/bin -type f -exec chmod 755 {} +

ENV PYTHONUNBUFFERED=1 \
    STROKE_MODEL_PATH=/app/model/model.onnx

USER app
EXPOSE 8080

# Cloud Run ignores this; it keeps local runs honest.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD /app/.venv/bin/python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz',timeout=5).read()"

# Shell form so $PORT expands; exec so uvicorn is PID 1 and receives SIGTERM.
CMD exec /app/.venv/bin/uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}
