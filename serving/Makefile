.PHONY: install lint test build run smoke release url invoke logs clean

# Run from WSL: the Docker daemon lives there, not on the Windows host.

TAG ?= $(shell git rev-parse --short HEAD)
PROJECT ?= stoke-serving
REGION ?= europe-west1
SERVICE ?= stroke-api

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

release:  ## build, push and deploy through Cloud Build, tagged with the current commit
	gcloud builds submit . --config cloudbuild.yaml --project $(PROJECT) \
	  --substitutions=_TAG=$(TAG),_REGION=$(REGION),_SERVICE=$(SERVICE)

url:  ## print the deployed service URL
	@gcloud run services describe $(SERVICE) --project $(PROJECT) --region $(REGION) --format='value(status.url)'

invoke:  ## classify a sample scan on the deployed service, with an IAM identity token
	@curl -s -H "Authorization: Bearer $$(gcloud auth print-identity-token)" \
	  -F "file=@assets/sample_stroke.png" "$$($(MAKE) -s url)/predict"

logs:  ## tail the prediction log lines
	gcloud beta run services logs tail $(SERVICE) --project $(PROJECT) --region $(REGION)

clean:  ## remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
