"""Layer 1: Zero-LLM Deterministic Checker for Travel Booking Agent outputs.

Eliminates LLM-as-a-Judge sycophancy bias for objective business rules:
1. Price validation (0 < price <= user_budget)
2. Date validation (reference_date <= check_in < check_out)
3. Entity grounding (claimed hotel_id / flight_id / hotel names exist in tool_observations)
4. Constraint Satisfaction Rate (CSR = satisfied_constraints / total_user_constraints)
5. Multilingual Language Drift detection (e.g., CJK leakage in Vietnamese/English responses)
6. Structural False Negative prevention when tool_observations is empty (returns None instead of 1.0)
"""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Dict, List, Literal, Optional
from config.compat import BaseModel, Field

# Regex for CJK Unified Ideographs to detect Qwen/multilingual language drift
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


class DeterministicCheckResult(BaseModel):
    """Structured output of zero-LLM deterministic rule checks."""

    price_in_range: bool = Field(
        default=True, description="True if all claimed prices > 0 and <= user_budget."
    )
    date_valid: bool = Field(
        default=True,
        description="True if check_in < check_out and not in the past relative to reference_date.",
    )
    entity_exists: bool = Field(
        default=True,
        description="True if all claimed hotel/flight entities exist in ground-truth tool observations.",
    )
    constraint_satisfaction_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ratio of user constraints satisfied in the response (0.0 to 1.0).",
    )
    language_consistent: bool = Field(
        default=True,
        description="True if response matches expected language without CJK/language drift.",
    )
    verification_status: Literal["verified", "no_tool_data", "failed"] = Field(
        default="verified",
        description="Distinguishes verified consistency from unverifiable empty tool observations.",
    )
    is_consistent: Optional[bool] = Field(
        default=True,
        description="None when tool_observations is empty (prevents False Negative inflation).",
    )
    consistency_score: Optional[float] = Field(
        default=1.0,
        description="None when tool_observations is empty; otherwise 0.0-1.0.",
    )
    failed_checks: List[str] = Field(
        default_factory=list,
        description="Failure tags for Prometheus agent_deterministic_check_failures_total.",
    )
    details: List[str] = Field(
        default_factory=list,
        description="Human-readable explanation of any rule violations.",
    )

    @property
    def passed_all(self) -> bool:
        return (
            self.price_in_range
            and self.date_valid
            and self.entity_exists
            and self.language_consistent
            and self.constraint_satisfaction_rate >= 0.95
            and len(self.failed_checks) == 0
        )


class DeterministicChecker:
    """Zero-LLM rule-based verifier for Travel Booking responses and state payloads."""

    def __init__(self, csr_threshold: float = 0.95) -> None:
        self.csr_threshold = csr_threshold

    def check(
        self,
        response_text: str,
        tool_observations: Optional[List[Dict[str, Any]]] = None,
        claimed_entities: Optional[List[str]] = None,
        claimed_prices: Optional[List[float]] = None,
        user_budget: Optional[float] = None,
        check_in: Optional[str | date] = None,
        check_out: Optional[str | date] = None,
        reference_date: Optional[str | date] = None,
        user_constraints: Optional[List[str]] = None,
        expected_language: str = "Vietnamese",
        requires_tool_grounding: bool = True,
    ) -> DeterministicCheckResult:
        """Executes all deterministic assertions without invoking any LLM."""
        failed_checks: List[str] = []
        details: List[str] = []

        # 1. Price Validation
        price_in_range = True
        prices_to_check = list(claimed_prices or [])
        if prices_to_check:
            for p in prices_to_check:
                if p <= 0:
                    price_in_range = False
                    details.append(f"Invalid non-positive price detected: {p}")
                elif user_budget is not None and p > user_budget:
                    price_in_range = False
                    details.append(f"Claimed price {p} exceeds user budget {user_budget}")
            if not price_in_range:
                failed_checks.append("price_out_of_range")

        # 2. Date Validation
        date_valid = True
        if check_in is not None and check_out is not None:
            try:
                ci_date = self._parse_date(check_in)
                co_date = self._parse_date(check_out)
                if ci_date >= co_date:
                    date_valid = False
                    details.append(f"check_in ({ci_date}) must be strictly before check_out ({co_date})")
                if reference_date is not None:
                    ref_date = self._parse_date(reference_date)
                    if ci_date < ref_date:
                        date_valid = False
                        details.append(f"check_in ({ci_date}) is in the past relative to {ref_date}")
            except ValueError as exc:
                date_valid = False
                details.append(f"Date parse failure: {exc}")

            if not date_valid:
                failed_checks.append("date_invalid")

        # 3. Entity Grounding & Empty Tool Observation Guard (False Negative Fix)
        entity_exists = True
        has_tool_obs = bool(tool_observations and len(tool_observations) > 0)

        if not has_tool_obs and requires_tool_grounding:
            # Fix for structural False Negative in evaluator.py:
            # "không thể verify" != "đã verify và ổn"
            verification_status: Literal["verified", "no_tool_data", "failed"] = "no_tool_data"
            is_consistent: Optional[bool] = None
            consistency_score: Optional[float] = None
            if claimed_entities:
                entity_exists = False
                failed_checks.append("entity_not_found")
                details.append(
                    f"Unverified entities claimed without tool observations: {claimed_entities}"
                )
        else:
            verification_status = "verified"
            if claimed_entities and has_tool_obs:
                obs_corpus = self._serialize_observations(tool_observations or []).lower()
                missing_entities = [
                    ent for ent in claimed_entities if str(ent).lower() not in obs_corpus
                ]
                if missing_entities:
                    entity_exists = False
                    failed_checks.append("entity_not_found")
                    details.append(
                        f"Claimed entities missing from tool observations: {missing_entities}"
                    )

            is_consistent = price_in_range and date_valid and entity_exists
            consistency_score = 1.0 if is_consistent else 0.0

        # 4. Constraint Satisfaction Rate (CSR)
        csr = self._compute_csr(response_text, tool_observations, user_constraints)
        if csr < self.csr_threshold:
            failed_checks.append("constraint_violated")
            details.append(
                f"Constraint Satisfaction Rate {csr:.2f} below threshold {self.csr_threshold:.2f}"
            )

        # 5. Language Consistency (Language Drift Check)
        language_consistent = self._check_language_consistency(response_text, expected_language)
        if not language_consistent:
            failed_checks.append("language_drift")
            details.append(
                f"Language drift detected in response (expected {expected_language})."
            )

        if failed_checks and verification_status == "verified":
            verification_status = "failed"
            is_consistent = False
            consistency_score = max(
                0.0, 1.0 - (len(failed_checks) * 0.25)
            )

        return DeterministicCheckResult(
            price_in_range=price_in_range,
            date_valid=date_valid,
            entity_exists=entity_exists,
            constraint_satisfaction_rate=round(csr, 4),
            language_consistent=language_consistent,
            verification_status=verification_status,
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            failed_checks=failed_checks,
            details=details,
        )

    @staticmethod
    def _parse_date(val: str | date) -> date:
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, date):
            return val
        return date.fromisoformat(str(val).strip()[:10])

    @staticmethod
    def _serialize_observations(observations: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for obs in observations:
            parts.append(str(obs))
        return " ".join(parts)

    @classmethod
    def _compute_csr(
        cls,
        response_text: str,
        tool_observations: Optional[List[Dict[str, Any]]],
        user_constraints: Optional[List[str]],
    ) -> float:
        if not user_constraints:
            return 1.0
        combined_corpus = (
            (response_text or "") + " " + cls._serialize_observations(tool_observations or [])
        ).lower()
        satisfied = sum(
            1 for constraint in user_constraints if str(constraint).lower() in combined_corpus
        )
        return satisfied / len(user_constraints)

    @staticmethod
    def _check_language_consistency(response_text: str, expected_language: str) -> bool:
        if not response_text or not response_text.strip():
            return False
        exp_lower = expected_language.lower()
        if exp_lower in ("vietnamese", "vi", "english", "en", "spanish", "es"):
            # Detect CJK character leakage (Language Drift in Qwen/multilingual models)
            if _CJK_PATTERN.search(response_text):
                return False
        return True
