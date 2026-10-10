"""JEV decision-signal adapter."""

from __future__ import annotations

import math
import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_dream_job_engine.models import (
    DecisionQuestion,
    DecisionSignal,
    SignalStatus,
)
from ai_dream_job_engine.providers.base import ProviderError


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"


class _JevAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question_version: str = Field(min_length=1)
    noul: Any = None


class _JevResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[_JevAnswer]


class JevSignalProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not resolved_key or not resolved_key.strip():
            raise ProviderError("JEV API key is required")
        self._api_key = resolved_key
        if client is not None:
            self._client = client
        else:
            try:
                self._client = httpx.Client()
            except Exception:
                raise ProviderError("JEV client initialization failed") from None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={JEV_MODEL!r})"

    def judge(
        self, job_description: str, questions: list[DecisionQuestion]
    ) -> list[DecisionSignal]:
        """Reject malformed records; localize missing or invalid probabilities."""
        question_ids = [question.id for question in questions]
        if len(question_ids) != len(set(question_ids)):
            raise ProviderError("duplicate question ID is not allowed")

        payload = {
            "model": JEV_MODEL,
            "input": {
                "job_description": job_description,
                "questions": [question.model_dump() for question in questions],
            },
        }
        try:
            response = self._client.post(
                JEV_ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
            body = response.json()
        except Exception:
            raise ProviderError("JEV provider request failed") from None

        try:
            parsed = _JevResponse.model_validate(body)
        except (ValidationError, ValueError, TypeError):
            raise ProviderError("JEV provider returned an invalid response") from None

        requested_keys = {
            (question.id, question.version) for question in questions
        }
        answers_by_key: dict[tuple[str, str], _JevAnswer] = {}
        for answer in parsed.answers:
            key = (answer.question_id, answer.question_version)
            if key not in requested_keys or key in answers_by_key:
                raise ProviderError("JEV provider returned an invalid response")
            answers_by_key[key] = answer

        signals = []
        for question in questions:
            answer = answers_by_key.get((question.id, question.version))
            probability = self._valid_probability(answer)
            signals.append(
                DecisionSignal(
                    question_id=question.id,
                    question_version=question.version,
                    probability=probability,
                    status=(
                        SignalStatus.AVAILABLE
                        if probability is not None
                        else SignalStatus.INVALID
                    ),
                    provider="jev",
                    model=JEV_MODEL,
                )
            )
        return signals

    @staticmethod
    def _valid_probability(answer: _JevAnswer | None) -> float | None:
        """Missing or invalid values affect only their requested output."""
        if answer is None:
            return None
        value = answer.noul
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return None
        return probability
