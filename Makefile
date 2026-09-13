# Grounded Answer Engine
#
# Everything runs from here. `make` on its own lists every target.
#
# Layout:
#   services/llm-gateway   containerised - talks to Gemini, knows no tax law
#   services/tax-agent     containerised - retrieval, grounding, citations
#   corpus-builder         LOCAL ONLY - turns PDFs into a Qdrant index
#   data/                  raw PDFs, corpus.db, embedded index

# --- Paths and tools --------------------------------------------------------
# Absolute interpreter path, so every target works whether or not the venv is
# activated, and survives the `cd` inside the run/test targets.
VENV      ?= .venv
PY        := $(CURDIR)/$(VENV)/bin/python
GATEWAY   := services/llm-gateway
AGENT     := services/tax-agent
BUILDER   := corpus-builder

# --- Corpus settings --------------------------------------------------------
DB         ?= data/corpus.db
QDRANT     ?= http://localhost:6333
COLLECTION ?= ita_sections
PDF_2025   ?= data/raw/ita-2025.pdf
PDF_1961   ?= data/raw/ita-1961.PDF

# --- AWS --------------------------------------------------------------------
AWS_REGION  ?= ap-south-1
AWS_ACCOUNT ?= 268666185034
ECR_REPO    ?= backend
ECS_CLUSTER ?= firstCluster
ECS_SERVICE ?= api-task
ECR_URI     := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)

.DEFAULT_GOAL := help
.PHONY: help install install-agent corpus-deps install-all run run-agent test lint \
        format format-check check parse parse-2025 parse-1961 index index-1961 \
        index-naive index-embedded search corpus-stats build up down restart logs \
        ps shell-agent shell-gateway clean qdrant-up qdrant-ui collections \
        aws-audit aws-spend ecr-login ecr-push ecs-start ecs-stop ecs-status

help: ## Show this help
	@echo "Grounded Answer Engine"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} \
		/^##@/ { printf "\n%s\n", substr($$0, 5) } \
		/^[a-zA-Z_-]+:.*?##/ { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables you can override, e.g. make search Q=\"section 80C\""
	@echo "  Q  QDRANT  COLLECTION  DB  ACT  YEAR  TOP_K"
	@echo ""

##@ Setup
install: ## Install llm-gateway dependencies
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r $(GATEWAY)/requirements.txt

install-agent: ## Install tax-agent dependencies
	$(PY) -m pip install -r $(AGENT)/requirements.txt

corpus-deps: ## Install corpus-builder dependencies (pypdf, qdrant, openai)
	$(PY) -m pip install -r $(BUILDER)/requirements.txt

install-all: install install-agent corpus-deps ## Install everything

##@ Develop
run: ## Run llm-gateway locally on :8000 with reload
	cd $(GATEWAY) && $(PY) -m uvicorn app.main:create_app --factory --reload --port 8000

run-agent: ## Run tax-agent locally on :8080 with reload (needs qdrant + gateway)
	cd $(AGENT) && $(PY) -m uvicorn app.main:create_app --factory --reload --port 8080

test: ## Run llm-gateway tests
	cd $(GATEWAY) && $(PY) -m pytest

lint: ## ruff check across the repo
	$(PY) -m ruff check .

format: ## ruff format across the repo
	$(PY) -m ruff format .

format-check: ## Fail if anything is unformatted
	$(PY) -m ruff format --check .

check: lint format-check test ## Lint, format check, and tests

##@ Corpus (local only - never runs in a container)
parse: parse-2025 parse-1961 ## Parse both Acts into data/corpus.db

parse-2025: ## Parse the 2025 Act (536 sections + 16 schedules)
	$(PY) $(BUILDER)/scripts/parse.py --pdf $(PDF_2025) --act ITA-2025 --db $(DB)

parse-1961: ## Parse the 1961 Act (861 sections + 10 schedules)
	$(PY) $(BUILDER)/scripts/parse.py --pdf $(PDF_1961) --act ITA-1961 --db $(DB)

index: ## Embed both Acts into Qdrant (~3 cents, needs EMBEDDING_API_KEY)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-2025 \
		--qdrant $(QDRANT) --collection $(COLLECTION)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-1961 \
		--qdrant $(QDRANT) --collection $(COLLECTION)

index-naive: ## Build the naive baseline collection, for A/B comparison
	$(PY) $(BUILDER)/scripts/index.py --strategy naive --pdf $(PDF_2025) --act ITA-2025 \
		--qdrant $(QDRANT) --collection ita_naive

index-embedded: ## Build the index on disk instead of a server (production shape)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-2025 \
		--qdrant data/index --collection $(COLLECTION)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-1961 \
		--qdrant data/index --collection $(COLLECTION)

# Usage: make search Q="deduction for life insurance premium" YEAR=2027
Q     ?= deduction for life insurance premium
TOP_K ?= 5
search: ## Query the index. make search Q="..." [YEAR=2027] [ACT=ITA-1961]
	$(PY) $(BUILDER)/scripts/search.py "$(Q)" --qdrant $(QDRANT) \
		--collection $(COLLECTION) --top-k $(TOP_K) \
		$(if $(YEAR),--tax-year $(YEAR),) $(if $(ACT),--act $(ACT),)

corpus-stats: ## Section and character counts per Act in corpus.db
	@$(PY) -c "import sqlite3; \
		[print(f'  {a:<10} {n:>4} sections  {c:>10,} chars') \
		 for a, n, c in sqlite3.connect('$(DB)').execute( \
		 'SELECT act, COUNT(*), SUM(length(text)) FROM sections GROUP BY act')]"

##@ Docker
build: ## Build all images
	docker compose build

up: ## Start qdrant + llm-gateway + tax-agent in the background
	docker compose up -d

down: ## Stop all services (keeps the qdrant volume)
	docker compose down

restart: ## Rebuild and restart everything
	docker compose up -d --build

logs: ## Follow logs from all services
	docker compose logs -f

ps: ## Show service status and ports
	docker compose ps

shell-agent: ## Shell inside the running tax-agent container
	docker compose exec tax-agent /bin/sh

shell-gateway: ## Shell inside the running llm-gateway container
	docker compose exec llm-gateway /bin/sh

clean: ## Stop services AND delete the qdrant volume (destroys the index)
	docker compose down -v

##@ Qdrant
qdrant-up: ## Start only Qdrant
	docker compose up -d qdrant

qdrant-ui: ## Open the Qdrant dashboard
	open http://localhost:6333/dashboard

collections: ## List collections with point counts
	@curl -s $(QDRANT)/collections | $(PY) -c "import json,sys; \
		[print('  ' + c['name']) for c in json.load(sys.stdin)['result']['collections']]"
	@for c in $$(curl -s $(QDRANT)/collections | $(PY) -c \
		"import json,sys; print(' '.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))"); do \
		curl -s $(QDRANT)/collections/$$c | $(PY) -c \
		"import json,sys; d=json.load(sys.stdin)['result']; \
		 print(f\"  $$c: {d['points_count']} points, {d['config']['params']['vectors']['size']} dims\")"; \
	done

##@ API smoke tests
health: ## Check both services are answering
	@curl -s -m 3 http://localhost:8080/health && echo "" || echo "  tax-agent down"

ask: ## Ask a question through the full stack. make ask Q="..."
	@curl -s -X POST http://localhost:8080/ask -H 'Content-Type: application/json' \
		-d '{"question":"$(Q)","max_tokens":500}' | $(PY) -m json.tool

api-search: ## Hit the /search endpoint. make api-search Q="..." [YEAR=2027]
	@curl -s -X POST http://localhost:8080/search -H 'Content-Type: application/json' \
		-d '{"question":"$(Q)"$(if $(YEAR),\,"tax_year":$(YEAR),)}' | $(PY) -m json.tool

##@ AWS
aws-audit: ## List anything that costs money right now
	@echo "EC2 instances:"; aws ec2 describe-instances --region $(AWS_REGION) \
		--query 'Reservations[].Instances[?State.Name!=`terminated`].InstanceId' --output text
	@echo "Load balancers:"; aws elbv2 describe-load-balancers --region $(AWS_REGION) \
		--query 'LoadBalancers[].LoadBalancerName' --output text
	@echo "Elastic IPs:"; aws ec2 describe-addresses --region $(AWS_REGION) \
		--query 'Addresses[].PublicIp' --output text
	@echo "NAT gateways:"; aws ec2 describe-nat-gateways --region $(AWS_REGION) \
		--query 'NatGateways[?State==`available`].NatGatewayId' --output text
	@echo "Running ECS tasks:"; aws ecs list-tasks --cluster $(ECS_CLUSTER) \
		--region $(AWS_REGION) --query 'taskArns' --output text

aws-spend: ## Month-to-date spend (free; Cost Explorer's API is not)
	@aws budgets describe-budgets --account-id $(AWS_ACCOUNT) \
		--query 'Budgets[].{limit:BudgetLimit.Amount,spend:CalculatedSpend.ActualSpend.Amount}' \
		--output table

ecr-login: ## Authenticate docker to ECR (token lasts 12 hours)
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com

# Fargate is x86; your Mac is arm64. Without --platform the task fails to pull.
ecr-push: ## Build for amd64 and push. make ecr-push TAG=v3
	docker build --platform linux/amd64 --provenance=false \
		-t $(ECR_URI):$(or $(TAG),latest) $(AGENT)
	docker push $(ECR_URI):$(or $(TAG),latest)

ecs-start: ## Scale the ECS service to 1 task (~$0.46/day)
	aws ecs update-service --cluster $(ECS_CLUSTER) --service $(ECS_SERVICE) \
		--desired-count 1 --region $(AWS_REGION) --query 'service.desiredCount' --output text

ecs-stop: ## Scale the ECS service to 0 (cost drops to zero)
	aws ecs update-service --cluster $(ECS_CLUSTER) --service $(ECS_SERVICE) \
		--desired-count 0 --region $(AWS_REGION) --query 'service.desiredCount' --output text

ecs-status: ## Desired vs running task count
	@aws ecs describe-services --cluster $(ECS_CLUSTER) --services $(ECS_SERVICE) \
		--region $(AWS_REGION) \
		--query 'services[0].{desired:desiredCount,running:runningCount,taskDef:taskDefinition}' \
		--output table
