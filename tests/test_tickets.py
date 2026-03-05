from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import InMemoryMetrics
from app.core.rate_limit import InMemoryRateLimiter
from app.db import init_db
from app.main import app, settings
from app.services.classifier import RulesClassifier
from app.services.ticket_provider import MockTicketProvider


class StubDraftService:
    async def draft_reply(self, **_: object) -> str:
        return "Thanks for raising this ticket. We are looking into it."

    def is_unreachable_error(self, exc: Exception) -> bool:  # noqa: ARG002
        return False


@pytest.fixture(autouse=True)
def setup_app_state(tmp_path) -> None:
    app.state.classifier = RulesClassifier()
    app.state.draft_service = StubDraftService()
    app.state.ticket_provider = MockTicketProvider()
    app.state.metrics = InMemoryMetrics()
    app.state.rate_limiter = InMemoryRateLimiter(max_requests=1000, window_seconds=60)
    settings.review_threshold = 0.70
    db_path = str(tmp_path / "test.sqlite3")
    init_db(db_path)
    app.state.db_path = db_path


async def post_ticket_triage(payload: dict):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/tickets/triage", json=payload)


@pytest.mark.asyncio
async def test_ticket_triage_returns_intent_priority_and_team():
    response = await post_ticket_triage(
        {
            "ticket_id": "INC-901",
            "subject": "Need pricing",
            "description": "Can I get a quote for 200 seats?",
            "channel": "email",
            "requester": "ops@company.com",
        }
    )
    body = response.json()
    assert response.status_code == 200
    assert body["ticket_id"] == "INC-901"
    assert body["intent"]["category"] == "sales"
    assert body["priority"] == "medium"
    assert body["assigned_team"] == "sales"
    assert body["draft_reply"]


@pytest.mark.asyncio
async def test_ticket_triage_severe_complaint_sets_review():
    response = await post_ticket_triage(
        {
            "ticket_id": "INC-902",
            "subject": "Refund dispute",
            "description": "I will file a chargeback with my bank.",
            "channel": "email",
        }
    )
    body = response.json()
    assert response.status_code == 200
    assert body["intent"]["category"] == "complaint"
    assert body["needs_review"] is True
    assert body["reply_is_draft"] is True
    assert body["priority"] == "high"
    assert body["assigned_team"] == "escalations"


@pytest.mark.asyncio
async def test_list_mock_tickets_returns_seed_data():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/tickets/mock?limit=2")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 2
    assert "ticket_id" in items[0]
    assert "subject" in items[0]
