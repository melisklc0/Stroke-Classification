.PHONY: install lint test build run smoke release clean

# Run from WSL: the Docker daemon lives there, not on the Windows host.

TAG ?= $(shell git rev-parse --short HEAD)
PROJECT ?= stoke-serving

install:  ## sync deps + dev group
	uv sync

lint:  ## ruff check + format
	uv run ruff check .
	uv run ruff format --check .

test:  ## unit tests — no docker, no network, no model weights
	uv run pytest -q

build:  ## build the serving image locally
	DOCKER_BUILDKIT=1 docker build -t stroke-api:local .

run:  ## run the local image on :8080 (Swagger UI at /docs)
	docker run --rm -p 8080:8080 stroke-api:local

smoke:  ## start the built image and assert /healthz + both sample scans
	bash scripts/smoke_test.sh

release:  ## build and push through Cloud Build, tagged with the current commit
	gcloud builds submit . --config cloudbuild.yaml --project $(PROJECT) --substitutions=_TAG=$(TAG)

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
