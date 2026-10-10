"""Provider contracts for extraction and model-generated signals."""

from ai_dream_job_engine.providers.base import (
    CandidateClaimExtractor,
    DecisionSignalProvider,
    JobFactExtractor,
)

__all__ = [
    "CandidateClaimExtractor",
    "DecisionSignalProvider",
    "JobFactExtractor",
]
