# Custom Agents

Self-hosted research and writing agents with **Deep Agents + LangGraph**, PostgreSQL,
LiteLLM and local image generation. MIT licensed. The first agent is a small news studio.

Research a topic → read sources → draft → review against sources → create a cover → save.
The service produces **drafts**, with citations and an explicitly labelled AI illustration.
It does not publish articles or execute code from the web.

## Repository layout

| Path | Purpose |
| --- | --- |
| `apps/server` | FastAPI UI, API, A2A adapter and durable worker |
| `agents/news` | News agent manifest and reusable websearch skill |
| `packages/core` | Shared settings, PostgreSQL queue and bounded public-web reader |
| `deploy/helm/custom-agents` | Small reusable chart, PostgreSQL and internal SearXNG |
| `tests` | Tool safety, citation, authentication, queue and protocol contracts |
| `.github/workflows` | Test and publish a SHA-tagged container to GHCR |

Actual environment values, deployment credentials and model files belong in a separate
private infrastructure repo. This public repo contains no production keys or user articles.

## Design

A **single durable worker** uses a PostgreSQL queue and session advisory lock. LangGraph's
PostgreSQL checkpointer saves stage results. On restart, interrupted jobs are reclaimed and
resume from the last completed stage. Failed jobs remain visible and can be retried.
Research uses a Deep Agent with search/read tools and the websearch skill. It has no shell,
host filesystem or delegated agents. The other stages use the same local Qwen route through
LiteLLM. Source IDs are validated before the final article is saved. Final body length
must be within 20% of the requested target (Chinese characters for Chinese, words for
other languages). Up to two bounded revisions correct length before cover generation.

SearXNG is free, self-hosted and internal. Its upstream search engines can rate-limit or fail;
a run with insufficient readable sources fails visibly instead of inventing a story. Search
and page text are untrusted evidence. The reader pins verified public IPs, checks redirects,
limits response sizes, and rejects private networks. Model review helps editing; it does not
guarantee factual accuracy. An editor should review all drafts before publication.

The queue is at least once. A crash during a running model/API call may repeat that stage;
completed graph stages are reused. Cover files are written atomically and reused on retry.
Do not scale this first version above one worker without revisiting GPU concurrency.

Article metadata and checkpoints live in PostgreSQL. Markdown, JSON and PNG artifacts live
on a retained PVC. Reboot/pod replacement preserves them on intact persistent storage.
There is no automatic backup component in this chart.

## Run locally

Use Python 3.12+ and PostgreSQL 17+; provide a SearXNG instance with JSON search enabled.
Copy `.env.example` to `.env`, fill in a **scoped** LiteLLM key, and export the variables.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .
uvicorn custom_agents.app:create_app --factory --port 8000
```

The UI is `/agents/news/`. All UI/API paths require `Authorization: Bearer <AGENT_API_KEY>`
by default. In Kubernetes the browser path uses Authentik ForwardAuth. Enable `TRUST_SSO`
only when ingress overwrites `X-authentik-username` and NetworkPolicy blocks direct access.
Browser writes require a matching Origin; machine API calls use the bearer key.

```sh
curl http://localhost:8000/agents/news/jobs \
  -H "Authorization: Bearer $AGENT_API_KEY" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: example-news-001' \
  -d '{"topic":"Recent open-weight local AI model releases","language":"繁體中文（香港用語）","length":750,"cover":true}'
```

Poll `GET /agents/news/jobs/{id}`. Download `/jobs/{id}/files/article.md`, `result.json`,
and `cover.png` under the same prefix. Explicitly retry with `POST /jobs/{id}/retry`.
`/health/live` checks the process; `/health/ready` checks DB connectivity and worker lifetime.

## LiteLLM agent registration

The service is an **agent**, while Qwen and MOSS remain **models**. Register this service in
LiteLLM's A2A agent registry rather than pretending it is a model. Agent discovery is at
`/.well-known/agent-card.json` (also `/a2a/.well-known/agent-card.json` for clients
that resolve discovery relative to the RPC URL). The JSON-RPC endpoint `/a2a` implements **A2A 0.3**
`message/send` and `tasks/get`. Streaming, push notifications and cancellation are not advertised.
A send returns a durable submitted/working task immediately; poll its ID for artifacts.
The completed cover is embedded as A2A file bytes so API clients need no browser SSO session.
Message IDs are idempotent. Supply an optional `params.metadata.news_request` object to set
language/length/cover; otherwise the message text is the topic and defaults apply.

Register `agent_name: news-writer`, the fetched card in `agent_card_params`, and
`static_headers.Authorization: Bearer <AGENT_API_KEY>` with an operator's LiteLLM admin key.
The agent itself receives only a virtual key limited to the Qwen and MOSS routes.
Keep registration headers out of Git. Give client virtual keys explicit access to the agent.

## Deploy with GitOps

Build/publish the container, make its GHCR package public, and pin the resulting digest in
your private Argo CD values. The reusable chart uses existing Kubernetes Secrets for app,
PostgreSQL and SearXNG configuration; see `values.yaml`. Argo CD deploys the chart from an
immutable Git commit plus private environment values. CI builds images; Argo CD reconciles
Kubernetes. Credentials do not enter GitHub Actions.

The chart defaults to one app pod, one small PostgreSQL pod and one internal SearXNG pod.
It requests no GPU: inference stays in existing model services. A retained storage class is
required for reboot persistence. Configure an authenticated ingress separately. Plain local
Helm users can port-forward the service and use bearer API authentication.

## Tests

```sh
pip install --no-deps -e '.[test]'
pytest -q
```

Optional PostgreSQL integration tests use `TEST_DATABASE_URL` and a disposable database.
The initial live acceptance run also exercises real search, Qwen writing, MOSS cover, A2A
through LiteLLM, SSO and restart persistence; those require a configured private environment.
