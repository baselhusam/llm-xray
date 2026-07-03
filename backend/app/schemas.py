"""Pydantic request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)


class NextToken(BaseModel):
    token: str
    token_id: int
    prob: float


class GenerateResponse(BaseModel):
    prompt: str
    tokens: list[str]
    token_ids: list[int]
    next_token: NextToken
