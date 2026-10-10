"""Structural interfaces implemented by external model providers."""

from typing import Protocol

from ai_dream_job_engine.models import (
    CandidateClaim,
    DecisionQuestion,
    DecisionSignal,
    ExtractedFact,
    Rubric,
)


class ProviderError(RuntimeError):
    """A provider failed without exposing request data or credentials."""


class JobFactExtractor(Protocol):
    def extract(
        self, job_description: str, rubric: Rubric
    ) -> list[ExtractedFact]: ...


class CandidateClaimExtractor(Protocol):
    def extract(self, resume_text: str) -> list[CandidateClaim]: ...


class DecisionSignalProvider(Protocol):
    def judge(
        self, job_description: str, questions: list[DecisionQuestion]
    ) -> list[DecisionSignal]: ...
