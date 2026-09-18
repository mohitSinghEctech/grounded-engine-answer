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
ECR_REPO       ?= backend
AGENT_ECR_REPO ?= tax-agent
ECS_CLUSTER ?= firstCluster
ECS_SERVICE ?= api-task
ECR_URI        := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPO)
AGENT_ECR_URI  := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com/$(AGENT_ECR_REPO)

.DEFAULT_GOAL := help
.PHONY: help install install-agent corpus-deps install-all run run-agent test lint \
        format format-check check parse parse-2025 parse-1961 index index-1961 \
        index-naive index-embedded index-standalone image-standalone search corpus-stats build up down restart logs \
        ps shell-agent shell-gateway clean qdrant-up qdrant-ui collections \
        aws-audit aws-spend ecr-login ecr-push ecr-push-agent ecr-images ecs-start ecs-stop ecs-status \
        manifest manifest-check health ask ask-stream api-search index-guidance reindex corpus-export \
        eval eval-smoke eval-baseline grade grade-summary \
        graph-on graph-off graph-which graph-draw graph-mermaid map-sections

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
manifest: ## Record source URLs and hashes for everything in data/raw
	$(PY) $(BUILDER)/scripts/manifest.py --update

manifest-check: ## Re-download sources and detect if the law was republished
	$(PY) $(BUILDER)/scripts/manifest.py --check

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

index-guidance: ## Embed the department's guidance pages (separate act label)
	$(PY) $(BUILDER)/scripts/index.py --strategy guidance \
		--qdrant $(QDRANT) --collection $(COLLECTION)

reindex: ## Delete both collections and rebuild everything from scratch
	@$(PY) -c "from qdrant_client import QdrantClient; \
		c = QdrantClient(url='$(QDRANT)'); \
		[ (c.delete_collection(x.name), print('  dropped ' + x.name)) \
		  for x in c.get_collections().collections ]"
	$(MAKE) index
	$(MAKE) index-guidance

index-naive: ## Build the naive baseline collection, for A/B comparison
	$(PY) $(BUILDER)/scripts/index.py --strategy naive --pdf $(PDF_2025) --act ITA-2025 \
		--qdrant $(QDRANT) --collection ita_naive

INDEX_PATH ?= data/index

map-sections: ## Populate maps_to_1961 by matching titles across the Acts
	$(PY) $(BUILDER)/scripts/map_sections.py --db $(DB) --write

index-standalone: ## Build a COMPLETE on-disk index for the self-contained image
	@rm -rf $(INDEX_PATH)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-2025 \
		--qdrant $(INDEX_PATH) --collection $(COLLECTION)
	$(PY) $(BUILDER)/scripts/index.py --strategy sections --db $(DB) --act ITA-1961 \
		--qdrant $(INDEX_PATH) --collection $(COLLECTION)
	$(PY) $(BUILDER)/scripts/index.py --strategy guidance \
		--qdrant $(INDEX_PATH) --collection $(COLLECTION)
	@echo ""
	@$(PY) -c "from qdrant_client import QdrantClient; \
		c = QdrantClient(path='$(INDEX_PATH)'); \
		i = c.get_collection('$(COLLECTION)'); \
		print(f'  $(INDEX_PATH): {i.points_count} points, {i.config.params.vectors.size} dims'); \
		c.close()"

# Kept as an alias; index-standalone is the one to use, because this name
# used to omit the guidance pages and so produced an index that could not
# answer any procedural question.
index-embedded: index-standalone ## Deprecated alias for index-standalone

# Usage: make search Q="deduction for life insurance premium" YEAR=2027
Q     ?= deduction for life insurance premium
TOP_K ?= 5
search: ## Query the index. make search Q="..." [YEAR=2027] [ACT=ITA-1961]
	$(PY) $(BUILDER)/scripts/search.py "$(Q)" --qdrant $(QDRANT) \
		--collection $(COLLECTION) --top-k $(TOP_K) \
		$(if $(YEAR),--tax-year $(YEAR),) $(if $(ACT),--act $(ACT),)

corpus-export: ## Dump corpus.db to readable files in data/export/
	$(PY) $(BUILDER)/scripts/export.py --format index
	$(PY) $(BUILDER)/scripts/export.py --format jsonl

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

##@ Pipeline
graph-on: ## Switch /ask to the LangGraph pipeline and restart the agent
	PIPELINE=graph docker compose up -d --build tax-agent

graph-off: ## Switch /ask back to the linear pipeline
	PIPELINE=linear docker compose up -d --build tax-agent

graph-which: ## Which orchestrator the running agent is using
	@docker compose exec tax-agent printenv PIPELINE || echo "linear (unset)"

graph-mermaid: ## Regenerate docs/graph.mmd from the compiled graph
	@cd $(AGENT) && $(PY) -c "from app.graph.build import build_ask_graph; \
		print(build_ask_graph().get_graph().draw_mermaid())" > $(CURDIR)/docs/graph.mmd
	@echo "wrote docs/graph.mmd - generated from the graph, so it cannot go stale"

graph-draw: ## Print the graph's nodes and edges. Needs no running service
	@cd $(AGENT) && $(PY) -c "from app.graph.build import build_ask_graph; \
		g = build_ask_graph().get_graph(); \
		print('nodes:', ', '.join(n for n in g.nodes if not n.startswith('__'))); \
		[print(f'  {e.source:14} -> {e.target:14} {e.data or \"\"}') for e in g.edges]"

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

ask-stream: ## Watch the status stream live. make ask-stream Q="..."
	@curl -sN -X POST http://localhost:8080/ask/stream \
		-H 'Content-Type: application/json' \
		-d '{"question":"$(Q)","max_tokens":1200}' \
		| $(PY) eval/watch.py

ask: ## Ask a question through the full stack. make ask Q="..."
	@curl -s -X POST http://localhost:8080/ask -H 'Content-Type: application/json' \
		-d '{"question":"$(Q)","max_tokens":2000}' | $(PY) -m json.tool

##@ Evaluation
eval: ## Score the question set. make eval [OUT=...] [DELAY=3] [CATEGORY=...]
	$(PY) eval/run.py --delay $(or $(DELAY),3) \
		--out $(or $(OUT),eval/runs/latest.csv) \
		$(if $(CATEGORY),--category $(CATEGORY),)

eval-smoke: ## One question per run, to check the harness is wired up
	$(PY) eval/run.py --limit 1 --out eval/runs/smoke.csv

eval-baseline: ## The run everything later is compared against
	$(PY) eval/run.py --delay 3 --out eval/runs/baseline.csv

grade: ## Grade answer_correct by hand. make grade [CSV=...] [CATEGORY=...]
	@$(PY) eval/grade.py --csv $(or $(CSV),eval/runs/baseline.csv) \
		$(if $(CATEGORY),--category $(CATEGORY),) $(if $(ID),--id $(ID),)

grade-summary: ## Grading progress and rates, changing nothing
	@$(PY) eval/grade.py --csv $(or $(CSV),eval/runs/baseline.csv) --summary

api-search: ## Hit the /search endpoint. make api-search Q="..." [YEAR=2027]
	@$(PY) -c "import json,os,urllib.request as u; \
		body={'question':os.environ['Q']}; \
		body.update({'tax_year':int(os.environ['YEAR'])} if os.environ.get('YEAR') else {}); \
		r=u.urlopen(u.Request('http://localhost:8080/search', \
		  json.dumps(body).encode(), {'Content-Type':'application/json'})); \
		d=json.load(r); \
		print(f\"tax_year={d['tax_year']} source={d['tax_year_source']}\"); \
		[print(f\"  {p['score']:.4f}  {p['act']} s.{p['section_number']} - {(p['section_title'] or '')[:56]}\") \
		 for p in d['passages']]" Q="$(Q)" YEAR="$(YEAR)"

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
# Which service goes to ECR. The deployed task (api-task) runs the
# llm-gateway on port 8000, so that is the default - this target used to
# hardcode the tax-agent, which would have pushed an image listening on
# 8080 and demanding QDRANT_URL into the repo the gateway task pulls from.
# The task would then fail its health check and never start.
SERVICE ?= $(GATEWAY)

ecr-push: ## Build for amd64 and push. make ecr-push TAG=v3 [SERVICE=services/tax-agent]
	@echo "  service   $(SERVICE)"
	@echo "  image     $(ECR_URI):$(or $(TAG),latest)"
	docker build --platform linux/amd64 --provenance=false \
		-t $(ECR_URI):$(or $(TAG),latest) $(SERVICE)
	docker push $(ECR_URI):$(or $(TAG),latest)

image-standalone: ## Build the self-contained tax-agent (index baked in)
	@test -d $(INDEX_PATH) || { echo "  no index at $(INDEX_PATH) - run: make index-standalone"; exit 1; }
	docker build --platform linux/amd64 --provenance=false \
		-t tax-agent:dev $(AGENT)
	docker build --platform linux/amd64 --provenance=false \
		-f $(AGENT)/Dockerfile.standalone \
		-t tax-agent:standalone .
	@echo ""
	@docker images tax-agent --format "  {{.Repository}}:{{.Tag}}  {{.Size}}"

ecr-push-agent: ## Push the standalone tax-agent. make ecr-push-agent TAG=v1
	@test -n "$(TAG)" || { echo "  give a TAG, e.g. make ecr-push-agent TAG=v1"; exit 1; }
	@test -d $(INDEX_PATH) || { echo "  no index - run: make index-standalone"; exit 1; }
	$(MAKE) image-standalone
	docker tag tax-agent:standalone $(AGENT_ECR_URI):$(TAG)
	docker push $(AGENT_ECR_URI):$(TAG)
	@echo ""
	@echo "  pushed $(AGENT_ECR_URI):$(TAG)"

ecr-images: ## What is already in the ECR repo, oldest first
	@aws ecr describe-images --repository-name $(ECR_REPO) --region $(AWS_REGION) \
		--query 'sort_by(imageDetails,&imagePushedAt)[].{tag:imageTags,pushed:imagePushedAt}' \
		--output table

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
