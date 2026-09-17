# Grounded Answer Engine

A production-oriented FastAPI service for generating answers through an LLM, with a clean application boundary around the model provider.

The project focuses on:

- Clean API architecture
- Provider abstraction
- Explicit error handling
- Retry policies with exponential backoff and jitter
- Request IDs and structured logging
- Input and output validation
- Automated tests without requiring an API key
- Ruff linting and formatting
- Continuous integration with GitHub Actions

> **Current status:** System 1 — LLM API shell complete. Retrieval and grounding are planned for the next phase.

---

## Architecture

The current request flow is:

```text
Client
  │
  ▼
Request ID Middleware
  │
  ▼
FastAPI Router
  │
  ▼
LLM Service
  │
  ▼
OpenAI-Compatible Provider
  │
  ▼
LLM API
  │
  ▼
LLMResult
  │
  ▼
API Response
```

The application separates provider-specific SDK behavior from the rest of the application through the `LLMClient` protocol.

This allows the provider implementation to be replaced without changing the API layer and allows tests to use a fake LLM client instead of making real API calls.

### Planned Grounded Architecture

The next phase will extend the request path to:

```text
Client
  │
  ▼
Request ID Middleware
  │
  ▼
FastAPI Router
  │
  ▼
Grounded Answer Service
  │
  ├──────────────► Retriever
  │                    │
  │                    ▼
  │              Relevant Chunks
  │                    │
  │                    ▼
  │              Context Builder
  │
  ▼
LLM Client
  │
  ▼
Grounded Answer + Sources
```

---

# Project Structure

Two containerised services, one local-only corpus builder, and an evaluation
harness that scores the whole thing. Every file's job in one line.

```text
grounded_answer_engine/
│
├── docker-compose.yml          qdrant + llm-gateway + tax-agent, and the timeout ladder
├── Makefile                    every command lives here; `make` lists them
│
├── services/tax-agent/         RETRIEVAL AND GROUNDING - the interesting service
│   └── app/
│       ├── main.py             startup: picks an embedder and a store by name
│       ├── config.py           every setting, with its default and its reason
│       ├── prompt.py           the six rules, and the citation parser
│       ├── scope.py            the refusal the pipeline cannot decide alone
│       ├── tax_year.py         "FY 2026-27" -> 2026, which picks the Act
│       ├── schemas.py          request and response shapes, incl. refusal_reason
│       │
│       ├── retrieval/          PLUGGABLE: swap either half by config
│       │   ├── base.py           the two sockets - Embedder, VectorStore
│       │   ├── embedders.py      OpenAI-compatible + the registry
│       │   ├── stores.py         Qdrant, InMemory + the registry
│       │   ├── sections.py       spotting "section 139" in a question
│       │   └── retriever.py      the policy: exact-first, cap, truncate
│       │
│       ├── routers/
│       │   ├── ask.py            retrieve -> prompt -> generate -> verify cites
│       │   ├── search.py         retrieval only, no model, no cost
│       │   └── health.py
│       │
│       └── services/
│           ├── base.py           LLMGateway and Retriever protocols
│           └── gateway_client.py talks to llm-gateway over HTTP
│
├── services/llm-gateway/       THE ONLY THING THAT TALKS TO A MODEL
│   └── app/
│       ├── vendors.py            PLUGGABLE: openai, gemini - and what each accepts
│       ├── retry.py              exponential backoff with jitter
│       └── services/
│           └── openai_compatible.py  one client for every OpenAI-shaped API
│
├── corpus-builder/             LOCAL ONLY - never runs in a container
│   ├── corpus/
│   │   ├── pdf.py                text out of the Act PDFs
│   │   ├── dialects/             each Act numbers its sections differently
│   │   ├── chunking.py           section-aware and naive strategies
│   │   ├── guidance.py           the department's web pages, not statute
│   │   ├── storage.py            SQLite, with UNIQUE(act, section_number)
│   │   └── vectorstore.py        embedding and upload
│   └── scripts/                  parse, index, search, export, manifest
│
├── eval/                       TURNING "IT WORKS" INTO A NUMBER
│   ├── questions.yaml           40 questions and their ground truth
│   ├── run.py                   asks each one, scores six signals
│   ├── grade.py                 the human pass over answer_correct
│   ├── compare.py               tells a real change from run-to-run noise
│   └── PLAN.md                  what is measured, and why
│
└── data/                       gitignored except manifest.json
    ├── raw/                     the source PDFs and scraped guidance
    ├── corpus.db                parsed sections
    └── manifest.json            source URLs and hashes, so a run is reproducible
```

## Where to start reading

1. `eval/PLAN.md` - what "good" means here, and why it is measured this way
2. `services/tax-agent/app/routers/ask.py` - the whole request, top to bottom
3. `services/tax-agent/app/retrieval/base.py` - the two sockets, and where the
   boundary sits
4. `services/tax-agent/app/prompt.py` - the rules the model is held to

---

# Getting Started

## Requirements

- Python 3.12+
- An OpenAI-compatible LLM API endpoint
- An API key for the configured provider

Python 3.12 is the version used by the CI environment.

## 1. Clone the Repository

```bash
git clone git@github.com:mohitSinghEctech/grounded-engine-answer.git
cd grounded-engine-answer
```

Alternatively, using HTTPS:

```bash
git clone https://github.com/mohitSinghEctech/grounded-engine-answer.git
cd grounded-engine-answer
```

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If `python3.12` is not available:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.12.x
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Verify:

```bash
python --version
```

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env`.

At minimum:

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model
```

The project also supports:

```env
APP_NAME=Grounded Answer Engine
APP_VERSION=0.1.0
ENVIRONMENT=development
LOG_LEVEL=INFO

REASONING_EFFORT=low

REQUEST_TIMEOUT_SECONDS=30

MAX_RETRIES=2
RETRY_BASE_DELAY_SECONDS=0.5
RETRY_MAX_DELAY_SECONDS=5.0
MAX_RETRY_TIME_SECONDS=60
```

Never commit `.env` or real API keys.

---

# Running the Application

Start the development server:

```bash
uvicorn app.main:create_app --factory --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

---

# API

## GET `/health`

Returns the current application health status.

Example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "app_name": "Grounded Answer Engine",
  "app_version": "0.1.0",
  "environment": "development"
}
```

The health endpoint also supports request IDs.

```bash
curl \
  -H "X-Request-ID: example-request-123" \
  http://127.0.0.1:8000/health
```

The response contains the same request ID in the response headers:

```text
X-Request-ID: example-request-123
```

---

# POST `/ask`

Generates an answer using the configured LLM provider.

## Request

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is FastAPI?",
    "max_tokens": 200
  }'
```

## Request Body

```json
{
  "question": "What is FastAPI?",
  "max_tokens": 200
}
```

### Fields

| Field | Type | Required | Constraints |
|---|---|---:|---|
| `question` | string | Yes | 1–2000 characters |
| `max_tokens` | integer | No | 1–8000, default 2000 |

Whitespace surrounding `question` is stripped during validation.

---

# POST `/ask/stream`

Same question, same answer, but it tells you what it is doing while it works.
Built for a UI: three seconds of silence feels much longer than three
narrated ones.

The transport is **Server-Sent Events** - a long-lived HTTP response with one
`event:`/`data:` pair per update and a blank line between them. Simpler than
a WebSocket because it only goes one way, which is all this needs.

## Request

Identical to `/ask`.

```bash
make ask-stream Q="which section charges interest for a late filed return?"
```

## What comes back

One `status` frame per step, then exactly one `result` frame carrying the
same body `/ask` returns - or one `error` frame.

```text
  0.00s  received       question_length=71
  0.00s  year_resolved  tax_year=None source=none governing_act=None
  0.00s  embedding      model=text-embedding-3-small
  0.63s  embedded       dimensions=1536 took_ms=646
  0.63s  searching      store=qdrant:ita_sections limit=24
  0.64s  searched       candidates=24
  0.64s  filtering      min_score=0.3 kept=24 dropped=0
  0.64s  capping        max_per_section=2 before=24 after=6
  0.64s  retrieved      count=6 acts=['ITA-1961', 'ITA-2025']
            ITA-1961  s.234A  Interest for defaults in furnishing...  0.666
            ITA-2025  s.423   Interest for defaults in furnishing...  0.604
  0.64s  prompting      provisions=6 rules=6
  0.64s  generating     prompt_characters=7676 max_tokens=1200
  1.84s  generated      model=gpt-4.1-mini completion_tokens=42 took_ms=1204
  1.84s  verified       cited=['ITA-1961 s.234A'] invented=[]
  1.84s  ANSWER         refused=False reason=none
```

The timings are the useful part: embedding is a network call and costs about
0.6s, the vector search is local and nearly free at 30ms, and the model is
the rest. Optimising retrieval would buy almost nothing.

## The steps

| step | means |
|---|---|
| `received` | the question passed validation |
| `resolving` / `year_resolved` | which tax year applies, where it came from, and which Act that puts in charge |
| `embedding` / `embedded` | turning the question into a vector; a network call |
| `searching` / `searched` | querying the vector store; how many candidates |
| `filtering` | dropping candidates below the score floor |
| `looking_up` / `looked_up` | the question named a section, so it is fetched exactly rather than by similarity |
| `capping` | limiting how much of the window one section may occupy |
| `retrieved` | the final set, with acts, sections, titles and scores |
| `prompting` / `generating` / `generated` | the provisions go to the model; model, tokens and timing come back |
| `verifying` / `verified` | every citation checked against what was supplied, split into `cited` and `invented` |
| `refused` | stopping without an answer, with the reason and whether the model was called at all |

Steps carry **facts, not sentences**. `retrieved` reports which sections were
found, never "Found 6 provisions!" - wording belongs to the UI, and changing
it should not mean changing this service. A client should ignore any step it
does not recognise rather than display it raw.

## What the stream deliberately does not do

**It does not stream the answer text.** Citations can only be checked once
the answer is complete, because a fabricated reference may be in the last
sentence. Around two answers in forty try to cite a provision that was never
supplied, so streaming the prose would mean showing a reader an invented
section and retracting it afterwards.

That trade also costs little here. The model spends most of its time before
the first token, then writes ~104 words quickly, so word-by-word delivery
would save about a second of a three-second request. The steps cover the
whole wait.

If token streaming is added later, the safe shape is to release a paragraph
at a time once its citations check out - never to stream and correct.

## Both endpoints are one implementation

`/ask` and `/ask/stream` call the same `run_pipeline`. The plain endpoint
passes a `Progress.discarded()` emitter, which accepts every step and keeps
none, so there is no `if streaming:` branch anywhere and the two cannot
drift. A test asserts the answer bodies match.

---

# Example `/ask` Response

The following is an actual response returned by the running application.

```json
{
  "answer": "**FastAPI** is a modern, high-performance web framework for building APIs with Python (version 3.8+) based on standard Python-type hints. \n\nIt was created by Sebastián Ramírez and released in 2018. Today, it is one of the most popular Python frameworks, used by tech giants like Microsoft, Uber, Netflix, and Cisco.\n\nHere is a breakdown of what makes FastAPI special:\n\n---\n\n### Key Features and Benefits\n\n1. **High Performance (Very Fast):**\n   * It is on par with **Node.js** and **Go** in terms of speed, making it one of the fastest Python frameworks available. \n   * It achieves this by running on top of **Starlette** (for web routing) and **Uvicorn** (an ASGI web server).\n\n2. **Automatic Interactive Documentation:**\n   * FastAPI automatically generates interactive API documentation for your endpoints based on OpenAPI standards.",
  "model": "gemini-3.7-flash",
  "latency_ms": 2269,
  "prompt_tokens": 5,
  "completion_tokens": 196,
  "reasoning_tokens": 0,
  "total_tokens": 201,
  "finish_reason": "length"
}
```

The response exposes both the generated answer and execution metadata.

The `finish_reason` of `length` indicates that this particular request reached the configured token limit.

---

# Response Fields

| Field | Description |
|---|---|
| `answer` | Generated text from the LLM |
| `model` | Model used for generation |
| `latency_ms` | End-to-end LLM request latency |
| `prompt_tokens` | Number of prompt tokens |
| `completion_tokens` | Number of completion tokens |
| `reasoning_tokens` | Tokens consumed by model reasoning, when reported |
| `total_tokens` | Total token usage |
| `finish_reason` | Reason the model stopped generating |

These fields provide useful information for later performance, cost, and reliability analysis.

---

# Error Handling

The API uses a consistent error response format.

Example:

```json
{
  "error_code": "UPSTREAM_TIMEOUT",
  "message": "The upstream LLM service timed out.",
  "request_id": "example-request-123",
  "details": null
}
```

## Error Codes

| Error Code | HTTP Status | Meaning |
|---|---:|---|
| `UPSTREAM_TIMEOUT` | 504 | The LLM provider timed out |
| `UPSTREAM_UNAVAILABLE` | 503 | The LLM provider is unavailable |
| `INVALID_UPSTREAM_RESPONSE` | 502 | The provider returned an invalid or unexpected response |
| `UPSTREAM_RATE_LIMITED` | 429 | The provider rate-limited the request |
| `INTERNAL_SERVER_ERROR` | 500 | Unexpected application error |
| Validation error | 422 | Request validation failed |

For validation errors, the response includes details describing the invalid request fields.

Internal exception messages and tracebacks are not exposed to API clients.

Every error response contains a `request_id` to make the corresponding server-side log entry easier to locate.

---

# Retry Behavior

The application owns its retry policy rather than delegating retry behavior to the provider SDK.

Retryable failures include:

- Upstream rate limiting
- Upstream unavailability

Non-retryable failures include:

- Invalid upstream responses
- Other client-side/API errors
- Application validation errors

The retry policy uses:

```text
Exponential Backoff
        +
Jitter
        +
Retry Time Budget
        +
Retry-After when provided
```

Conceptually:

```text
Attempt 1
   │
   ▼
Failure
   │
   ▼
Retry-After / Exponential Backoff
   │
   ▼
Attempt 2
   │
   ▼
Failure
   │
   ▼
Retry-After / Exponential Backoff
   │
   ▼
Attempt 3
```

The retry loop stops when either:

- The request succeeds
- The maximum retry count is reached
- The retry time budget cannot accommodate another retry

---

# Design Decisions

## LLMClient Protocol

The application defines an `LLMClient` protocol rather than coupling the API directly to a concrete provider implementation.

```python
class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
        max_tokens: int,
    ) -> LLMResult: ...
```

The API layer therefore does not need to know whether the underlying provider is OpenAI, Gemini, or another OpenAI-compatible service.

It also makes the service easy to test using a fake implementation.

## Application-Owned Retries

The OpenAI SDK is configured with:

```text
max_retries=0
```

The application owns retry behavior instead.

This avoids having two independent retry systems operating at the same time.

Having a single retry policy means retry count, delay, jitter, rate-limit handling, and the overall retry time budget are observable and controllable by the application.

## Why Retries Exclude Most 4xx Errors

Most 4xx errors represent a problem with the request rather than a temporarily unavailable upstream service.

Automatically retrying those requests can:

- Waste provider capacity
- Increase latency
- Repeat requests that cannot succeed without changing input
- Hide the actual failure

The retry policy therefore focuses on failures that are plausibly transient.

## Why Jitter

Multiple application instances can fail at approximately the same time.

Without jitter:

```text
Instance A ── failure ── wait 1s ── retry
Instance B ── failure ── wait 1s ── retry
Instance C ── failure ── wait 1s ── retry
Instance D ── failure ── wait 1s ── retry
```

With jitter:

```text
Instance A ── wait 1.08s
Instance B ── wait 1.21s
Instance C ── wait 1.04s
Instance D ── wait 1.18s
```

This reduces synchronized retry spikes against the upstream provider.

## Reasoning Tokens and `max_tokens`

Reasoning-capable models can consume part of their token budget while reasoning before producing visible output.

Therefore, `max_tokens` cannot necessarily be interpreted simply as the number of visible answer tokens.

The application preserves reasoning-token usage separately:

```text
prompt tokens
completion tokens
reasoning tokens
total tokens
```

The application also warns when the provider reports a length-based finish reason so truncated responses can be diagnosed.

## Python Version

The project standardizes on Python 3.12 across local development and CI.

Python 3.12 was selected so the development environment and CI environment use the same Python version.

This also provides a stable compatibility baseline for the upcoming retrieval stack, which may include packages such as FAISS and sentence-transformers.

---

# Dependency Injection

The FastAPI application uses dependency injection to provide the LLM client to API routes.

Conceptually:

```text
FastAPI Request
      │
      ▼
/ask Router
      │
      ▼
Depends(get_llm_client)
      │
      ▼
Application LLM Client
```

This avoids creating a new LLM SDK client for every request.

The LLM client is created during application startup and stored in application state.

Tests can override the dependency and provide a fake client.

This gives the application:

- Reusable provider clients
- Cleaner route handlers
- Easier testing
- Lower coupling between the API and infrastructure

---

# Application Lifespan

The application creates infrastructure resources during startup and manages their lifetime through FastAPI's lifespan mechanism.

The LLM SDK client is initialized when the application starts.

Conceptually:

```text
Application Startup
        │
        ▼
Create LLM Client
        │
        ▼
Application Running
        │
        ▼
Serve Requests
        │
        ▼
Application Shutdown
```

This keeps infrastructure initialization separate from individual request handling.

---

# Request IDs

Every request receives a request ID.

If the client sends:

```text
X-Request-ID: example-request-123
```

the application preserves that value.

If the client does not provide one, the middleware generates a request ID.

The request ID is available in logs and API error responses.

Example:

```text
Request
   │
   ├── Request ID: abc123
   │
   ▼
Router
   │
   ▼
LLM Service
   │
   ▼
Error
   │
   ▼
Response
   │
   └── Request ID: abc123
```

---

# Logging

The application uses Python's logging infrastructure rather than printing application events directly.

Important events such as upstream failures and request completion are logged with useful execution information.

Examples include:

- Request ID
- Error code
- Upstream status
- Latency
- Token usage
- Finish reason

This provides a foundation for future centralized logging and observability.

---

# Validation

Request validation is handled using Pydantic models.

The `/ask` request validates:

- Required question
- Question length
- Maximum token range

Whitespace surrounding the question is removed before processing.

Invalid requests are rejected at the API boundary instead of being passed to the LLM provider.

Example:

```json
{
  "question": "",
  "max_tokens": 200
}
```

This results in a `422 Unprocessable Entity` response.

---

# Upstream Response Validation

The provider response is not assumed to always be valid.

The LLM client explicitly validates:

```text
choices exists
      │
      ▼
choices is not empty
      │
      ▼
first choice exists
      │
      ▼
message content exists
      │
      ▼
valid LLMResult
```

Invalid responses are translated into:

```text
INVALID_UPSTREAM_RESPONSE
```

rather than allowing low-level exceptions such as `IndexError` or `AttributeError` to escape as unexpected server errors.

---

# Tests Do Not Require an API Key

The test suite does not make real LLM requests.

Instead:

```text
Test
  │
  ▼
Fake / Mock LLM Client
  │
  ▼
Deterministic Result
```

This provides:

- Fast tests
- No API costs
- No dependency on provider availability
- No secrets in CI
- Deterministic failure testing

The SDK-to-application error translation is separately tested using mocked SDK responses and exceptions.

---

# Testing

Run the complete test suite:

```bash
pytest
```

Or:

```bash
python -m pytest
```

Run a specific test file:

```bash
pytest tests/test_ask.py -v
```

Run retry tests:

```bash
pytest tests/test_retry.py -v
```

Run LLM provider boundary tests:

```bash
pytest tests/test_llm_client.py -v
```

The test suite covers:

- Health endpoint
- Request ID behavior
- Request validation
- Successful `/ask` requests
- Application error handling
- Unexpected exception handling
- SDK exception translation
- Invalid provider responses
- Retry behavior
- Retry time budgets
- `Retry-After`
- Exponential backoff
- Jitter

The tests are designed to run without making network requests to an actual LLM provider.

---

# Code Quality

Ruff is used for linting and formatting.

Run linting:

```bash
ruff check .
```

Check formatting:

```bash
ruff format --check .
```

Automatically format the project:

```bash
ruff format .
```

---

# Continuous Integration

GitHub Actions runs the same quality checks automatically for pushes and pull requests.

The CI pipeline performs:

```text
Checkout
   │
   ▼
Python 3.12
   │
   ▼
Install Dependencies
   │
   ▼
Ruff Lint
   │
   ▼
Ruff Format Check
   │
   ▼
Pytest
```

The CI configuration is located at:

```text
.github/workflows/ci.yml
```

The goal is to prevent broken code, formatting issues, or failing tests from being merged.

---

# Troubleshooting

## `ModuleNotFoundError`

If imports such as:

```text
ModuleNotFoundError: No module named 'app'
```

appear while running tests, make sure the command is being run from the project root.

The project also configures:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

so the `app` package can be imported during tests.

## Unexpected Response Truncation

Reasoning-capable models can consume tokens from the model's output budget.

If a response unexpectedly ends early, inspect:

```text
finish_reason
reasoning_tokens
completion_tokens
total_tokens
```

A length-based finish reason can indicate that the configured token budget was exhausted.

For example:

```json
{
  "completion_tokens": 196,
  "reasoning_tokens": 0,
  "total_tokens": 201,
  "finish_reason": "length"
}
```

Increasing `max_tokens` may allow the model to produce a longer response.

## `choices[0]` Errors

Provider responses should not be assumed to always contain a valid first choice.

The LLM client explicitly validates:

```text
choices exists
      ↓
choices is not empty
      ↓
message exists
      ↓
content is not None
```

Invalid responses are translated into:

```text
INVALID_UPSTREAM_RESPONSE
```

rather than allowing an `IndexError` or `AttributeError` to escape as an internal server error.

## Provider Rate Limiting

When the provider returns a `Retry-After` header, the retry policy uses that value when possible.

Otherwise, the application falls back to its exponential-backoff strategy with jitter.

## Port 8000 Already in Use

If the application fails to start because port `8000` is already in use:

```bash
lsof -i :8000
```

Alternatively, run the application on another port:

```bash
uvicorn app.main:create_app --factory --reload --port 8001
```

---

# Roadmap

## System 1 — LLM API Shell

- [x] FastAPI application structure
- [x] Application configuration
- [x] Lifespan-managed LLM client
- [x] LLM provider abstraction
- [x] OpenAI-compatible provider implementation
- [x] Request validation
- [x] Error hierarchy
- [x] Request IDs
- [x] Logging
- [x] Retry policy
- [x] Retry time budget
- [x] Unit and API tests
- [x] Ruff linting and formatting
- [x] GitHub Actions CI

## System 2 — Grounding / RAG

Planned:

- [ ] Document ingestion
- [ ] Document chunking
- [ ] Embedding generation
- [ ] Vector database integration
- [ ] Retriever abstraction
- [ ] Top-k retrieval
- [ ] Context construction
- [ ] Grounded prompt construction
- [ ] Source attribution
- [ ] Retrieval evaluation
- [ ] Grounding failure handling
- [ ] End-to-end RAG tests

## Production Hardening

Planned:

- [ ] Metrics
- [ ] Distributed tracing
- [ ] Authentication
- [ ] Rate limiting
- [ ] Deployment
- [ ] Production configuration
- [ ] Cost monitoring
- [ ] Retrieval and answer quality evaluation

---

# Security

Never commit secrets.

The following files should remain local:

```text
.env
```

Use `.env.example` as the template for required configuration.

Before releasing a version, inspect Git history for accidentally committed API keys:

```bash
git log -p | grep -iE "AIza|sk-|gsk_"
```

Also consider scanning the repository using a dedicated secret-scanning tool before publishing releases.

GitHub secret scanning and push protection should be enabled for the repository where available.

If a secret is accidentally committed, do not simply delete it from the latest commit. Treat the credential as compromised and rotate or revoke it with the provider.

---

# Release

The first release represents the completion of the LLM API shell:

```text
v0.1.0
```

Before tagging a release, run:

```bash
ruff check .
ruff format --check .
pytest
```

Then verify that the application can be installed and started from a fresh clone using only the instructions in this README.

The release should not contain:

```text
.env
API keys
provider secrets
local virtual environments
Python cache files
```

---

# Future Architecture

The current application intentionally focuses on the LLM API boundary first.

The next major step is introducing retrieval and grounding.

The intended architecture is:

```text
                         ┌──────────────────┐
                         │    Client        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    FastAPI       │
                         │      API         │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Grounded Answer Service │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
             ┌──────────────┐          ┌──────────────┐
             │   Retriever  │          │ LLM Client   │
             └──────┬───────┘          └──────┬───────┘
                    │                         │
                    ▼                         │
             ┌──────────────┐                 │
             │ Relevant     │                 │
             │ Chunks       │                 │
             └──────┬───────┘                 │
                    │                         │
                    ▼                         │
             ┌──────────────┐                 │
             │   Context    │─────────────────┘
             │   Builder    │
             └──────────────┘
                    │
                    ▼
             ┌─────────────────┐
             │ Grounded Answer │
             │ + Sources       │
             └─────────────────┘
```

This separation allows retrieval, context construction, and generation to evolve independently.

---

# Project Goals

The long-term goal is to build a production-oriented grounded-answer system rather than a simple "send prompt to model" application.

The project is being developed incrementally around the following principles:

1. Keep infrastructure separate from application logic.
2. Depend on interfaces rather than concrete providers.
3. Make failures explicit and predictable.
4. Make retry behavior application-controlled.
5. Make requests traceable using request IDs.
6. Keep tests deterministic and independent of external APIs.
7. Validate both incoming requests and upstream responses.
8. Keep local development and CI environments aligned.
9. Introduce retrieval and grounding as separate components.
10. Build observability and production hardening on top of a clean foundation.

---

# License

This project is currently intended as a portfolio and learning project.
