"""JEV decision-signal adapter."""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from ai_dream_job_engine.models import (
    DecisionQuestion,
    DecisionSignal,
    SignalStatus,
)
from ai_dream_job_engine.providers.base import ProviderError


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"


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
        self._client = client or httpx.Client()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={JEV_MODEL!r})"

    def judge(
        self, job_description: str, questions: list[DecisionQuestion]
    ) -> list[DecisionSignal]:
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
        except (httpx.HTTPError, ValueError):
            raise ProviderError("JEV provider request failed") from None

        if not isinstance(body, dict):
            raise ProviderError("JEV provider returned an invalid response")
        answers = body.get("answers", [])
        if not isinstance(answers, list):
            raise ProviderError("JEV provider returned an invalid response")

        by_question: dict[str, list[Any]] = {}
        for answer in answers:
            if isinstance(answer, dict) and isinstance(
                answer.get("question_id"), str
            ):
                by_question.setdefault(answer["question_id"], []).append(answer)

        signals = []
        for question in questions:
            matching = by_question.get(question.id, [])
            probability = self._valid_probability(matching)
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
    def _valid_probability(answers: list[Any]) -> float | None:
        if len(answers) != 1:
            return None
        value = answers[0].get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            return None
        return probability
