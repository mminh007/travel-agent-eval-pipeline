"""Confusion Matrix Evaluator for Agent Decision Nodes.

Applies the 4-cell Confusion Matrix framework discussed in Confusion_Matrix_for_LLM_Systems.md
to audit the accuracy of:
1. node_fact_checker    -> Priority: High Recall (>= 0.90) to prevent False Negatives (escaped hallucinations)
2. node_input_guardrail -> Priority: High Precision (>= 0.95) to block OOD requests cleanly
3. node_evaluator_agent -> Priority: Balanced F1 (>= 0.85) to balance rework token cost vs quality
"""
from __future__ import annotations

from typing import Iterable, Literal, Tuple
from config.compat import BaseModel, Field

NodeName = Literal["fact_checker", "input_guardrail", "evaluator_agent"]


class ConfusionMatrixMetrics(BaseModel):
    """Complete Confusion Matrix breakdown for an agent binary decision node."""

    node_name: NodeName
    true_positive: int = Field(ge=0, description="Actual issue present AND system flagged issue.")
    false_positive: int = Field(ge=0, description="No actual issue, BUT system flagged issue (false alarm).")
    false_negative: int = Field(
        ge=0, description="Actual issue present, BUT system missed it (HIGH RISK)."
    )
    true_negative: int = Field(ge=0, description="No actual issue AND system confirmed clean.")

    precision: float = Field(ge=0.0, le=1.0, description="TP / (TP + FP)")
    recall: float = Field(ge=0.0, le=1.0, description="TP / (TP + FN)")
    f1_score: float = Field(ge=0.0, le=1.0, description="2 * (P * R) / (P + R)")
    false_negative_rate: float = Field(ge=0.0, le=1.0, description="FN / (TP + FN)")
    false_positive_rate: float = Field(ge=0.0, le=1.0, description="FP / (FP + TN)")
    meets_target: bool = Field(description="Whether the node meets its strategic metric target.")
    recommendation: str = Field(description="Actionable threshold/prompt tuning advice.")


class NodeConfusionMatrixEvaluator:
    """Computes TP/FP/FN/TN and strategic recommendations against Human Ground Truth."""

    def __init__(
        self,
        fact_checker_recall_min: float = 0.90,
        guardrail_precision_min: float = 0.95,
        evaluator_f1_min: float = 0.85,
    ) -> None:
        self.fact_checker_recall_min = fact_checker_recall_min
        self.guardrail_precision_min = guardrail_precision_min
        self.evaluator_f1_min = evaluator_f1_min

    def evaluate(
        self,
        node_name: NodeName,
        predictions_and_truths: Iterable[Tuple[bool, bool]],
    ) -> ConfusionMatrixMetrics:
        """Computes confusion matrix from pairs of (system_flagged_issue, ground_truth_has_issue)."""
        tp = fp = fn = tn = 0
        for predicted_issue, actual_issue in predictions_and_truths:
            if predicted_issue and actual_issue:
                tp += 1
            elif predicted_issue and not actual_issue:
                fp += 1
            elif not predicted_issue and actual_issue:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        meets_target, recommendation = self._assess_node_strategy(
            node_name=node_name,
            precision=precision,
            recall=recall,
            f1=f1,
            fn=fn,
        )

        return ConfusionMatrixMetrics(
            node_name=node_name,
            true_positive=tp,
            false_positive=fp,
            false_negative=fn,
            true_negative=tn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_negative_rate=round(fnr, 4),
            false_positive_rate=round(fpr, 4),
            meets_target=meets_target,
            recommendation=recommendation,
        )

    def _assess_node_strategy(
        self,
        node_name: NodeName,
        precision: float,
        recall: float,
        f1: float,
        fn: int,
    ) -> Tuple[bool, str]:
        if node_name == "fact_checker":
            meets = recall >= self.fact_checker_recall_min
            if not meets:
                return (
                    False,
                    f"CRITICAL: fact_checker Recall ({recall:.2f}) < {self.fact_checker_recall_min:.2f} "
                    f"({fn} hallucinations escaped as False Negatives). Lower consistency threshold or enable DeterministicChecker pre-filter.",
                )
            return True, "fact_checker Recall meets safety target."

        if node_name == "input_guardrail":
            meets = precision >= self.guardrail_precision_min
            if not meets:
                return (
                    False,
                    f"WARNING: input_guardrail Precision ({precision:.2f}) < {self.guardrail_precision_min:.2f}. "
                    "Tune intent_confidence_threshold to reduce false classifications.",
                )
            return True, "input_guardrail Precision meets target."

        # evaluator_agent
        meets = f1 >= self.evaluator_f1_min
        if not meets:
            return (
                False,
                f"WARNING: evaluator_agent F1 ({f1:.2f}) < {self.evaluator_f1_min:.2f}. "
                "Decouple judge model from executor model to reduce self-enhancement bias.",
            )
        return True, "evaluator_agent F1 balance meets target."
