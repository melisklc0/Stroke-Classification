#!/usr/bin/env bash
# Local container smoke test: start the image, hit /healthz and /predict with the sample scans.
set -euo pipefail

IMAGE="${IMAGE:-stroke-api:local}"
PORT="${PORT:-9090}"
NAME="stroke-smoke"
BASE="http://127.0.0.1:${PORT}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker run -d --name "$NAME" -p "${PORT}:${PORT}" -e "PORT=${PORT}" "$IMAGE" >/dev/null

for i in $(seq 1 40); do
  if curl -sf "${BASE}/healthz" >/dev/null 2>&1; then
    echo "ready after ${i}s"
    break
  fi
  sleep 1
done

echo "healthz   : $(curl -s "${BASE}/healthz")"
echo "stroke    : $(curl -s -F "file=@${HERE}/assets/sample_stroke.png" "${BASE}/predict")"
echo "no-stroke : $(curl -s -F "file=@${HERE}/assets/sample_no_stroke.png" "${BASE}/predict")"
echo "bad input : $(curl -s -o /dev/null -w '%{http_code}' -F "file=@${HERE}/pyproject.toml" "${BASE}/predict")"
echo "runs as   : $(docker exec "$NAME" whoami)"
