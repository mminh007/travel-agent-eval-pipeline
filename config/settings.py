"""Centralized settings and YAML configuration loader for the Eval Pipeline."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from config.compat import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

_DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "quality_gates": {
        "faithfulness_min": 0.90,
        "answer_relevance_min": 0.85,
        "context_precision_min": 0.80,
        "context_recall_min": 0.85,
        "constraint_satisfaction_rate_min": 0.95,
        "deterministic_pass_rate_min": 1.00,
    },
    "hitl_triggers": {
        "consistency_score_min": 0.75,
        "max_contradictions": 0,
        "max_ungrounded_claims": 2,
        "min_eval_dimension_score": 5,
        "max_rework_cycles": 2,
        "flag_on_deterministic_failure": True,
    },
    "confusion_matrix_targets": {
        "fact_checker_recall_min": 0.90,
        "input_guardrail_precision_min": 0.95,
        "evaluator_f1_min": 0.85,
    },
}

_DEFAULT_AB_CONFIG: Dict[str, Any] = {
    "experiment": {
        "name": "model_comparison_v1",
        "control_variant": "model_a",
        "treatment_variant": "model_b",
        "traffic_split_treatment": 0.10,
        "min_sample_size_per_variant": 500,
        "significance_level_alpha": 0.05,
        "minimum_statistical_power": 0.80,
    },
    "decision_rules": {
        "consistency_score_min": 0.90,
        "constraint_satisfaction_rate_min": 0.95,
        "ttft_p95_max_multiplier": 1.20,
        "allow_forced_accept_increase": False,
        "allow_rework_rate_increase": False,
    },
}


class EvalSettings(BaseModel):
    """Environment-backed settings adhering to the safe port allocation map."""

    # Langfuse Data Bridge (Port 3000 on agent_network)
    langfuse_host: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )
    langfuse_public_key: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-replace-me")
    )
    langfuse_secret_key: str = Field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-replace-me")
    )

    # Agent Production / Staging Endpoints (Port 8080 HTTP / Port 50051 gRPC)
    agent_api_url: str = Field(
        default_factory=lambda: os.getenv("AGENT_API_URL", "http://localhost:8080")
    )
    agent_grpc_host: str = Field(
        default_factory=lambda: os.getenv("AGENT_GRPC_HOST", "localhost:50051")
    )

    # Judge LLM Configuration (Distinct from Agent Tier 2 to prevent Sycophancy Bias)
    judge_llm_provider: str = Field(
        default_factory=lambda: os.getenv("JUDGE_LLM_PROVIDER", "openai")
    )
    judge_llm_model: str = Field(
        default_factory=lambda: os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini")
    )
    judge_llm_api_key: str = Field(
        default_factory=lambda: os.getenv("JUDGE_LLM_API_KEY", "dummy-key")
    )
    judge_llm_base_url: str = Field(
        default_factory=lambda: os.getenv("JUDGE_LLM_BASE_URL", "http://localhost:20128/v1")
    )

    # Safe Eval Ports (Avoids Agent Repo ports: 3000, 3001, 5432, 5672, 6333, 6334, 6379, 8002, 8080, 8123, 9090, 15672, 20128, 50051)
    pushgateway_url: str = Field(
        default_factory=lambda: os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
    )
    eval_api_host: str = Field(
        default_factory=lambda: os.getenv("EVAL_API_HOST", "0.0.0.0")
    )
    eval_api_port: int = Field(
        default_factory=lambda: int(os.getenv("EVAL_API_PORT", "8088"))
    )
    eval_worker_health_port: int = Field(
        default_factory=lambda: int(os.getenv("EVAL_WORKER_HEALTH_PORT", "8089"))
    )

    # Human-in-the-Loop Notifications
    slack_webhook_url: str = Field(
        default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", "")
    )
    hitl_review_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "HITL_REVIEW_BASE_URL", "http://localhost:3000/project/default/traces"
        )
    )

    # Experiment & Sampling Defaults
    default_experiment_name: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_EXPERIMENT_NAME", "model_comparison_v1")
    )
    sample_rate: float = Field(
        default_factory=lambda: float(os.getenv("SAMPLE_RATE", "0.10"))
    )


def load_thresholds_config(path: Path | None = None) -> Dict[str, Any]:
    """Loads quality gates and HITL thresholds from config/thresholds.yaml."""
    target = path or (BASE_DIR / "config" / "thresholds.yaml")
    try:
        import yaml  # type: ignore

        with open(target, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return dict(_DEFAULT_THRESHOLDS)


def load_ab_test_config(path: Path | None = None) -> Dict[str, Any]:
    """Loads A/B testing experiment rules from config/ab_test_config.yaml."""
    target = path or (BASE_DIR / "config" / "ab_test_config.yaml")
    try:
        import yaml  # type: ignore

        with open(target, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return dict(_DEFAULT_AB_CONFIG)


settings = EvalSettings()
