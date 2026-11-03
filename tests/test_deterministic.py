"""Unit tests for Layer 1 DeterministicChecker & Structural False Negative Fix."""
from __future__ import annotations

from evaluators.deterministic_checker import DeterministicChecker


def test_valid_hotel_response_passes_all_deterministic_checks():
    checker = DeterministicChecker(csr_threshold=0.95)
    result = checker.check(
        response_text="Khách sạn Sala Danang Beach Hotel (HTL-DN-101) tại Đà Nẵng có hồ bơi và ban công với giá 1500000 VND.",
        tool_observations=[
            {
                "hotel_id": "HTL-DN-101",
                "name": "Sala Danang Beach Hotel",
                "city": "Đà Nẵng",
                "price_per_night": 1500000,
                "amenities": ["hồ bơi", "ban công"],
            }
        ],
        claimed_entities=["HTL-DN-101", "Sala Danang Beach Hotel"],
        claimed_prices=[1500000],
        user_budget=2000000,
        check_in="2026-12-10",
        check_out="2026-12-12",
        reference_date="2026-11-03",
        user_constraints=["Đà Nẵng", "hồ bơi", "ban công"],
        expected_language="Vietnamese",
    )
    assert result.passed_all is True
    assert result.price_in_range is True
    assert result.date_valid is True
    assert result.entity_exists is True
    assert result.constraint_satisfaction_rate == 1.0
    assert result.language_consistent is True
    assert result.verification_status == "verified"
    assert result.is_consistent is True
    assert result.consistency_score == 1.0


def test_empty_tool_observations_prevents_false_negative():
    """Verifies the fix for evaluator.py line 33-37 where empty tool_observations returned 1.0."""
    checker = DeterministicChecker()
    result = checker.check(
        response_text="Khách sạn Grand Hotel có hồ bơi Olympic tiêu chuẩn quốc tế.",
        tool_observations=[],
        claimed_entities=["Grand Hotel"],
        requires_tool_grounding=True,
    )
    assert result.verification_status == "no_tool_data"
    assert result.is_consistent is None
    assert result.consistency_score is None
    assert result.entity_exists is False
    assert "entity_not_found" in result.failed_checks


def test_price_and_date_and_language_drift_violations():
    checker = DeterministicChecker(csr_threshold=0.95)
    result = checker.check(
        response_text="Khách sạn giá 2500000 VND 酒店推荐",  # Contains CJK drift
        tool_observations=[{"hotel_id": "HTL-1", "price": 2500000}],
        claimed_entities=["HTL-999-MISSING"],
        claimed_prices=[2500000],
        user_budget=2000000,
        check_in="2026-12-15",
        check_out="2026-12-10",  # Invalid: check_in > check_out
        reference_date="2026-11-03",
        user_constraints=["hồ bơi", "ban công", "spa"],
        expected_language="Vietnamese",
    )
    assert result.passed_all is False
    assert result.price_in_range is False
    assert result.date_valid is False
    assert result.entity_exists is False
    assert result.language_consistent is False
    assert set(result.failed_checks) == {
        "price_out_of_range",
        "date_invalid",
        "entity_not_found",
        "constraint_violated",
        "language_drift",
    }
