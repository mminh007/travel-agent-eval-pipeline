# 📦 Eval Pipeline Repo — Tổng Quan Kiến Trúc & Đề Xuất

---

## ✅ Xác Nhận Phân Chia Trách Nhiệm

### Agent Repo — Giữ nguyên 100%

| Thành phần | Giữ lý do |
|---|---|
| Prometheus metrics (`metrics.py`) | Đo inline trong hot path: TTFT, token, latency, node duration |
| Grafana dashboards (hiện tại) | Đã provisioned, gắn với Prometheus của agent |
| Langfuse `CallbackHandler` (`chat_stream_service.py`) | Non-blocking callback — không ảnh hưởng latency |
| `node_fact_checker`, `node_evaluator_agent` | Điều khiển rework loop runtime — không thể tách |
| `route_from_evaluator()` | Routing logic phụ thuộc trực tiếp vào eval output |

### Eval Pipeline Repo — Nhiệm vụ mới hoàn toàn

| Module | Mô tả |
|---|---|
| **Ragas / DeepEval Scorer** | Đo Faithfulness, AnswerRelevance, ContextRecall offline |
| **A/B Testing Analyzer** | So sánh model variants, statistical significance |
| **Human Review Queue** | Pull flagged traces, Slack notify, Langfuse annotation |
| **Golden Dataset** | Bộ test chuẩn + CI/CD regression runner |
| **Deterministic Checker** | Rule-based: price, date, entity validation (zero LLM) |
| **Grafana Panels (Quality)** | Dashboard riêng cho quality metrics, A/B comparison |
| **Prometheus Pushgateway** | Push offline eval scores vào Prometheus ecosystem |
| **Statistical Significance** | Mann-Whitney U, chi-square, sample size calculator |

---

## 🔌 Port Map Toàn Bộ Agent Repo

> Tổng hợp từ: `docker-compose.yml`, `docker-compose.tracing.yml`, `docker-compose.9router.yml`, `docker-compose.studio.yml`

| Port (Host) | Container Port | Service | File |
|---|---|---|---|
| **5432** | 5432 | `langfuse_postgres` (PostgreSQL) | `docker-compose.tracing.yml` |
| **5672** | 5672 | `booking_rabbitmq` (AMQP) | `docker-compose.yml` |
| **6333** | 6333 | `booking_qdrant_server` (HTTP API) | `docker-compose.yml` |
| **6334** | 6334 | `booking_qdrant_server` (gRPC) | `docker-compose.yml` |
| **6379** | 6379 | `booking_redis` | `docker-compose.yml` |
| **8002** | 8001 | `booking_grpc_server` (metrics) | `docker-compose.yml` |
| **8080** | 8080 | `booking_api_server` (FastAPI HTTP) | `docker-compose.yml` |
| **8123** | 8123 | `booking_langgraph_studio` | `docker-compose.studio.yml` |
| **9090** | 9090 | `booking_prometheus` | `docker-compose.yml` |
| **15672** | 15672 | `booking_rabbitmq` (Management UI) | `docker-compose.yml` |
| **20128** | 20128 | `booking_9router` (LLM Proxy) | `docker-compose.9router.yml` |
| **3000** | 3000 | `langfuse_server` | `docker-compose.tracing.yml` |
| **3001** | 3000 | `booking_grafana` | `docker-compose.yml` |
| **50051** | 50051 | `booking_grpc_server` (gRPC) | `docker-compose.yml` |

> [!WARNING]
> **Port 3000** đã bị chiếm bởi `langfuse_server`.
> **Port 3001** đã bị chiếm bởi `booking_grafana`.
> Nếu Eval Repo muốn chạy Grafana riêng → dùng **port 3002** hoặc cao hơn.

---

## 🟢 Ports An Toàn Cho Eval Pipeline Repo

Các port chưa được sử dụng, khuyến nghị dùng cho Eval Repo:

| Port đề xuất | Dùng cho service gì |
|---|---|
| **3002** | Grafana riêng của Eval Repo (nếu chạy standalone) |
| **9091** | Prometheus Pushgateway (nhận metrics từ eval workers) |
| **9092** | Prometheus riêng của Eval Repo (nếu tách hoàn toàn) |
| **8088** | Eval API server (FastAPI — expose endpoints cho CI/CD) |
| **8089** | Eval worker health check endpoint |

> [!TIP]
> **Khuyến nghị:** Eval Repo **không cần chạy Prometheus/Grafana riêng** nếu cùng network `agent_network`.
> Chỉ cần chạy **Prometheus Pushgateway** ở port **9091** để push offline metrics vào
> Prometheus của agent-repo (`booking_prometheus:9090`). Grafana agent-repo sẽ tự hiển thị.

---

## 🔗 Kiến Trúc Network Kết Hợp

```
agent_network (Docker external network - đã có sẵn)
│
├── booking_prometheus:9090        ← Prometheus gốc của agent
├── booking_grafana:3001           ← Grafana gốc
├── langfuse_server:3000           ← Data bridge
│
│  [Eval Repo join vào agent_network]
│
├── eval_pushgateway:9091          ← Eval Repo push offline metrics vào đây
│   └── booking_prometheus scrape 9091  ← Tự động có trong Grafana agent
│
└── eval_api:8088                  ← Eval Repo expose API cho CI/CD webhook
```

```yaml
# eval-pipeline/docker-compose.yml
networks:
  agent_network:
    external: true  # Join vào network có sẵn của agent-repo
```

---

## 📝 Đề Xuất Tên GitHub Repo

### Option 1 — Tên kỹ thuật rõ ràng ⭐ Recommended

```
travel-agent-eval-pipeline
```
**Description:**
> Offline evaluation pipeline for the Travel Booking AI Agent system. Includes Ragas/DeepEval quality scoring, A/B model testing with statistical significance, human-in-the-loop review queue, golden dataset CI/CD gates, deterministic rule-based checkers, and Grafana quality dashboards — powered by Langfuse as the data bridge.

---

### Option 2 — Tên theo chuẩn LLMOps

```
llmops-eval-pipeline
```
**Description:**
> LLMOps evaluation and quality control pipeline for the Travel Booking Agent. Implements automated scoring (Ragas, DeepEval), A/B testing, human review workflows, and CI/CD regression gates to ensure LLM output quality across model upgrades and prompt changes.

---

### Option 3 — Tên ngắn gọn

```
agent-quality-control
```
**Description:**
> Quality control system for the Travel Booking AI Agent. Offline LLM evaluation, A/B testing framework, human review queue, and CI/CD quality gates — built to complement the production agent without affecting runtime latency.

---

### Option 4 — Tên domain-specific

```
booking-agent-observatory
```
**Description:**
> Observability and evaluation layer for the Travel Booking AI Agent. Measures response faithfulness, detects hallucinations offline, runs model A/B tests, and enforces quality gates before production deployments — decoupled from the agent runtime via Langfuse tracing.

---

## 📁 Cấu Trúc Repo Gợi Ý

```
travel-agent-eval-pipeline/
│
├── README.md
├── pyproject.toml
├── .env.example                    # LANGFUSE_*, AGENT_API_URL, SLACK_WEBHOOK
├── docker-compose.yml              # pushgateway:9091, eval_api:8088
│
├── evaluators/                     # Core scoring logic
│   ├── ragas_scorer.py
│   ├── deepeval_runner.py
│   └── deterministic_checker.py    # Zero-LLM rule checks
│
├── ab_testing/                     # A/B experiment analysis
│   ├── trace_fetcher.py            # Pull từ Langfuse
│   ├── analyzer.py
│   ├── significance.py             # Mann-Whitney U, chi-square
│   └── report.py
│
├── human_review/                   # HITL workflow
│   ├── queue_manager.py
│   ├── annotation_client.py        # Langfuse Score API
│   └── notifier.py                 # Slack / email
│
├── golden_dataset/                 # Ground truth & CI gate
│   ├── dataset.json
│   ├── runner.py                   # Gọi agent HTTP API
│   └── regression_report.py
│
├── workers/                        # Scheduled jobs
│   ├── hourly_scorer.py            # Cron: pull → score → push
│   └── ci_regression.py            # CI/CD entrypoint
│
├── grafana/                        # Quality dashboard panels
│   └── dashboards/
│       ├── quality_overview.json
│       └── ab_comparison.json
│
├── api/                            # Eval API (webhook cho CI/CD)
│   └── main.py                     # FastAPI — port 8088
│
└── tests/
    └── test_deterministic.py
```

---

## 🎯 Tóm Tắt Quyết Định

```
AGENT REPO                           EVAL PIPELINE REPO
──────────────────────────────────   ────────────────────────────────────
✅ Giữ toàn bộ Prometheus metrics    ✅ Pushgateway (port 9091) → feed vào
✅ Giữ toàn bộ Grafana (port 3001)      Prometheus của agent
✅ Giữ Langfuse callback             ✅ Grafana quality panels (import vào
✅ Giữ fact_checker, evaluator          Grafana agent nếu cùng network)
✅ Giữ rework routing logic          ✅ Ragas / DeepEval offline
                                     ✅ A/B testing + significance
                                     ✅ Human review queue
                                     ✅ Golden dataset + CI/CD gate
                                     ✅ Deterministic checker
```

**Langfuse (port 3000) = API trung gian duy nhất giữa 2 repo.**
```
