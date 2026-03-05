from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["question", "complaint", "sales", "spam", "other"]


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def text_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ClassifyResponse(BaseModel):
    category: Category
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_reply: str = Field(..., min_length=1, max_length=500)


class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)

    @field_validator("texts")
    @classmethod
    def items_cannot_be_blank(cls, values: list[str]) -> list[str]:
        for index, value in enumerate(values):
            if not value.strip():
                raise ValueError(f"texts[{index}] must not be blank")
            if len(value) > 4000:
                raise ValueError(f"texts[{index}] must be at most 4000 characters")
        return values


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class MetricsResponse(BaseModel):
    counters: dict[str, int]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
