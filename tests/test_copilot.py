from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.db import init_db
from app.main import app, settings
from app.services.classifier import RulesClassifier
from app.services.copilot import build_templated_draft_reply


class StubDraftService:
    def __init__(self, reply: str = "I can help with that.", should_fail: bool = False) -> None:
        self.reply = reply
        self.should_fail = should_fail

    async def draft_reply(self, **_: object) -> str:
        if self.should_fail:
            raise ConnectionError("LM Studio unavailable")
        return self.reply

    def is_unreachable_error(self, exc: Exception) -> bool:
        return isinstance(exc, ConnectionError)


@pytest.fixture(autouse=True)
def setup_app_state(tmp_path) -> None:
    app.state.classifier = RulesClassifier()
    app.state.draft_service = StubDraftService()
    app.state.metrics = InMemoryMetrics()
    app.state.rate_limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)
    settings.review_threshold = 0.70
    db_path = str(tmp_path / "test.sqlite3")
    init_db(db_path)
    app.state.db_path = db_path


async def post_copilot(text: str, channel: str = "webchat"):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/copilot", json={"text": text, "channel": channel})


async def get_review(limit: int = 20):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/review?limit={limit}")


async def post_review(request_id: str, final_category: str, final_reply: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            f"/review/{request_id}",
            json={"final_category": final_category, "final_reply": final_reply},
        )


async def get_recent(limit: int = 20):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/recent?limit={limit}")


@pytest.mark.asyncio
async def test_copilot_complaint_returns_high_priority_and_expected_actions():
    response = await post_copilot("I want a refund. The item is broken.")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "complaint"
    assert body["priority"] == "high"
    assert "Apologize and acknowledge the issue" in body["next_actions"]
    assert "Ask for order number / reference" in body["next_actions"]
    assert "Confirm refund/replacement preference" in body["next_actions"]
    assert body["classifier_used"] == "rules"
    assert body["request_id"]
    assert body["needs_review"] is False
    assert body["reply_is_draft"] is False


@pytest.mark.asyncio
async def test_copilot_sales_returns_medium_priority_and_non_empty_draft_reply():
    app.state.draft_service = StubDraftService(reply="Great question. I can prepare a quote today.")
    response = await post_copilot("How much does this cost for bulk orders?", channel="email")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "sales"
    assert body["priority"] == "medium"
    assert body["draft_reply"].strip()
    assert body["latency_ms"] >= 0
    assert body["needs_review"] is False


@pytest.mark.asyncio
async def test_copilot_uses_template_reply_when_lmstudio_draft_fails():
    app.state.draft_service = StubDraftService(should_fail=True)
    response = await post_copilot("How do I reset my password?", channel="whatsapp")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "question"
    assert body["draft_reply"] == build_templated_draft_reply("question", "whatsapp")


@pytest.mark.asyncio
async def test_low_confidence_result_sets_needs_review():
    settings.review_threshold = 0.99
    response = await post_copilot("How do I reset my password?")
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["confidence"] < settings.review_threshold
    assert body["needs_review"] is True
    assert body["reply_is_draft"] is True
    assert body["priority"] == "medium"


@pytest.mark.asyncio
async def test_severe_complaint_sets_needs_review_even_with_high_confidence():
    settings.review_threshold = 0.70
    response = await post_copilot(
        "I want a refund immediately or I will file a chargeback with my bank."
    )
    body = response.json()

    assert response.status_code == 200
    assert body["intent"]["category"] == "complaint"
    assert body["intent"]["confidence"] >= settings.review_threshold
    assert body["needs_review"] is True
    assert body["reply_is_draft"] is True
    assert body["priority"] == "high"


@pytest.mark.asyncio
async def test_review_endpoint_returns_pending_items():
    settings.review_threshold = 0.99
    created = await post_copilot("How do I reset my password?")
    created_id = created.json()["request_id"]

    response = await get_review(limit=20)
    assert response.status_code == 200
    items = response.json()
    assert any(item["request_id"] == created_id for item in items)


@pytest.mark.asyncio
async def test_posting_review_updates_record():
    settings.review_threshold = 0.99
    created = await post_copilot("How do I reset my password?")
    request_id = created.json()["request_id"]

    submit = await post_review(
        request_id=request_id,
        final_category="question",
        final_reply="Thanks for your question. Please try resetting from settings.",
    )
    assert submit.status_code == 200
    body = submit.json()
    assert body["request_id"] == request_id
    assert body["status"] == "reviewed"
    assert body["reviewed_at"]

    queue = await get_review(limit=20)
    assert queue.status_code == 200
    assert all(item["request_id"] != request_id for item in queue.json())

    recent = await get_recent(limit=20)
    assert recent.status_code == 200
    row = next(item for item in recent.json() if item["request_id"] == request_id)
    assert row["final_category"] == "question"
    assert row["final_reply"] == "Thanks for your question. Please try resetting from settings."
    assert row["reviewed_at"]
