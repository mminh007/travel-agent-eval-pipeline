"""Slack & Webhook Notifier for Human-in-the-Loop Review Alerts."""
from __future__ import annotations

from typing import Optional

from config.settings import settings
from human_review.queue_manager import HumanReviewTask


class HITLNotifier:
    """Dispatches real-time alerts when traces enter the Human Review Queue."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        review_base_url: Optional[str] = None,
    ) -> None:
        self.webhook_url = webhook_url if webhook_url is not None else settings.slack_webhook_url
        self.review_base_url = review_base_url or settings.hitl_review_base_url

    def build_payload(self, task: HumanReviewTask) -> dict:
        """Constructs a structured Slack message payload for a flagged trace."""
        badge = "CRITICAL" if task.severity == "critical" else task.severity.upper()
        trace_url = f"{self.review_base_url}/{task.trace_id}"
        reasons_str = ", ".join(task.trigger_reasons)
        return {
            "text": (
                f"[{badge}] Human Review Required | Variant: `{task.model_variant}` | "
                f"Session: `{task.session_id}` | Triggers: `{reasons_str}`\n"
                f"Review Trace: {trace_url}"
            ),
            "trace_id": task.trace_id,
            "severity": task.severity,
            "reasons": task.trigger_reasons,
        }

    async def notify(self, task: HumanReviewTask) -> bool:
        """Sends alert to configured Slack webhook URL (no-op if URL is empty/placeholder)."""
        payload = self.build_payload(task)
        if not self.webhook_url or "T000/B000/X000" in self.webhook_url:
            return False
        try:
            import httpx  # type: ignore

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self.webhook_url, json={"text": payload["text"]})
                return resp.status_code < 300
        except Exception:
            return False
