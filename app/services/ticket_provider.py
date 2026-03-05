from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.models.schemas import Channel


@dataclass(frozen=True)
class TicketRecord:
    ticket_id: str
    subject: str
    description: str
    channel: Channel
    requester: str


class TicketProvider(Protocol):
    def list_tickets(self, limit: int = 20) -> list[TicketRecord]: ...

    def get_ticket(self, ticket_id: str) -> TicketRecord | None: ...


class MockTicketProvider:
    def __init__(self) -> None:
        self._items = [
            TicketRecord(
                ticket_id="INC-1001",
                subject="VPN login failing",
                description="I cannot connect to VPN after password reset.",
                channel="webchat",
                requester="alex@company.com",
            ),
            TicketRecord(
                ticket_id="INC-1002",
                subject="Refund request for duplicate charge",
                description="I was charged twice and want a refund immediately.",
                channel="email",
                requester="jamie@customer.com",
            ),
            TicketRecord(
                ticket_id="INC-1003",
                subject="Bulk license quote",
                description="Can you quote annual pricing for 250 seats?",
                channel="email",
                requester="buyer@customer.com",
            ),
        ]

    def list_tickets(self, limit: int = 20) -> list[TicketRecord]:
        safe_limit = max(1, min(100, int(limit)))
        return self._items[:safe_limit]

    def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        for item in self._items:
            if item.ticket_id == ticket_id:
                return item
        return None
