# Travel Agent Evaluation Pipeline (`travel-agent-eval-pipeline`)

> Offline evaluation pipeline for the Travel Booking AI Agent system. Includes Ragas/DeepEval quality scoring, A/B model testing with statistical significance, human-in-the-loop review queue, golden dataset CI/CD gates, deterministic rule-based checkers, and Grafana quality dashboards — powered by Langfuse as the data bridge.

---

## Architecture & Separation Boundary

This repository is completely **decoupled from the hot path** of the production Travel Booking Agent (`agent-repo`).
- **Agent Repo** retains inline runtime routing (`node_fact_checker`, `node_evaluator_agent`, `route_from_evaluator`), inline Prometheus latency/token metrics (`port 9090`), Grafana (`port 3001`), and the non-blocking Langfuse callback (`port 3000`).
- **Eval Pipeline Repo (this repo)** pulls traces asynchronously from **Langfuse (`port 3000`)**, executes 3-layer offline quality scoring, runs statistical significance tests for A/B model comparisons, manages the Human-in-the-Loop (HITL) review queue, enforces CI/CD Golden Dataset regression gates, and pushes quality metrics to **Prometheus Pushgateway (`port 9091`)**.

---

## Safe Port Allocation Map (Zero Collisions)

| Port | Service | Repository | Purpose |
|---:|---|---|---|
| `3000` | `langfuse_server` | Agent Repo | Event-driven Data Bridge (Traces, Scores & Annotations) |
| `3001` | `booking_grafana` | Agent Repo | Main Production Grafana Dashboard |
| `5432` | `langfuse_postgres` | Agent Repo | Langfuse PostgreSQL |
| `5672` / `15672` | `booking_rabbitmq` | Agent Repo | Async Memory Broker |
| `6333` / `6334` | `booking_qdrant_server` | Agent Repo | Vector Database |
| `6379` | `booking_redis` | Agent Repo | Checkpoint & Cache Store |
| `8002` / `50051` | `booking_grpc_server` | Agent Repo | gRPC Streaming & Metrics |
| `8080` | `booking_api_server` | Agent Repo | FastAPI REST & SSE Server |
| `8123` | `booking_langgraph_studio` | Agent Repo | LangGraph Studio |
| `9090` | `booking_prometheus` | Agent Repo | Main Prometheus Time-Series DB |
| `20128` | `booking_9router` | Agent Repo | 9Router LLM Proxy |
| **`8088`** | **`eval_api_server`** | **Eval Repo (Safe)** | FastAPI Evaluation, A/B Report & HITL Webhook Server |
| **`8089`** | **`eval_hourly_worker`** | **Eval Repo (Safe)** | Hourly Async Scorer & HITL Queue Worker Health Check |
| **`9091`** | **`eval_pushgateway`** | **Eval Repo (Safe)** | Prometheus Pushgateway for Offline Eval Metrics |
| **`3002`** | **`eval_standalone_grafana`** | **Eval Repo (Safe)** | Optional Standalone Grafana (`--profile standalone`) |

---

## Repository Structure

```text
travel-agent-eval-pipeline/
├── README.md
├── pyproject.toml
├── .env.example
├── dockerfile
├── docker-compose.yml              # Joins external agent_network (Ports: 9091, 8088, 8089, 3002)
├── .github/workflows/
│   └── eval_ci_gate.yml            # CI/CD Quality Gate workflow
├── config/
│   ├── settings.py                 # Env + YAML configuration loader
│   ├── thresholds.yaml             # Faithfulness >= 0.90, CSR >= 0.95, HITL triggers
│   └── ab_test_config.yaml         # Min sample size = 500, alpha = 0.05, metric definitions
├── evaluators/
│   ├── deterministic_checker.py    # Layer 1: Zero-LLM rules + empty-tool False Negative fix
│   ├── ragas_scorer.py             # Layer 2: Faithfulness, AnswerRelevance, Context Precision/Recall
│   ├── deepeval_runner.py          # Layer 2: DeepEval unit assertions & G-Eval itinerary rubric
│   └── confusion_matrix.py         # TP/FP/FN/TN, Precision, Recall, F1 per decision node
├── ab_testing/
│   ├── trace_fetcher.py            # Deterministic Hash(SessionId)%100 + Langfuse stratified sampler
│   ├── significance.py             # Mann-Whitney U, Welch t-test, Chi-Square, Power/Sample guard
│   ├── analyzer.py                 # Variant grouping & guardrail verification
│   └── report.py                   # Markdown & JSON Decision Matrix generator
├── human_review/
│   ├── queue_manager.py            # HITL trigger engine (forced_accept, contradictions, low score)
│   ├── annotation_client.py        # Langfuse Score API wrapper & Confusion Matrix ground-truth sync
│   └── notifier.py                 # Slack webhook alerting for high-risk traces
├── golden_dataset/
│   ├── dataset.json                # Curated Ground Truth benchmark cases
│   ├── runner.py                   # Offline & live HTTP benchmark executor
│   └── regression_report.py        # CI/CD blocking gate evaluator
├── dashboard/
│   └── metrics_exporter.py         # Prometheus CollectorRegistry & Pushgateway (port 9091) exporter
├── grafana/
│   ├── dashboards/
│   │   ├── quality_overview.json   # 4-row Quality Control, HITL, A/B & Confusion Matrix dashboard
│   │   └── ab_comparison.json      # Statistical significance & variant comparison dashboard
│   └── provisioning/
├── workers/
│   ├── hourly_scorer.py            # Cron/daemon worker (10% hourly sample -> score -> push)
│   └── ci_regression.py            # CLI entrypoint for CI/CD regression gate
├── api/
│   └── main.py                     # FastAPI server on port 8088
└── tests/
    ├── test_deterministic.py
    ├── test_significance.py
    └── test_confusion_matrix_and_hitl.py
```

---

## Quick Start

### 1. Local Installation & Unit Tests

```bash
pip install -e ".[dev]"
pytest -v
```

### 2. Run CI/CD Golden Dataset Quality Gate

```bash
python -m workers.ci_regression --mode offline --output-json regression_report.json
```

### 3. Run Hourly Offline Scorer (One-Shot or Daemon)

```bash
# Single batch run
python -m workers.hourly_scorer

# Continuous daemon (exposes health check on port 8089)
python -m workers.hourly_scorer --daemon
```

### 4. Start Docker Services on `agent_network`

```bash
cp .env.example .env
docker compose up -d --build
```

To allow `booking_prometheus` (`port 9090` in Agent Repo) to scrape the offline evaluation metrics from `eval_pushgateway` (`port 9091`), add this job to `agent/prometheus.yml`:

```yaml
  - job_name: "eval-pipeline-pushgateway"
    honor_labels: true
    scrape_interval: 15s
    static_configs:
      - targets: ["eval_pushgateway:9091"]
        labels:
          service: "travel-agent-eval-pipeline"
```
