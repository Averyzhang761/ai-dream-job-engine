"""Provider contracts for extraction and model-generated signals."""

from ai_dream_job_engine.providers.base import (
    CandidateClaimExtractor,
    DecisionSignalProvider,
    JobFactExtractor,
    ProviderError,
)
from ai_dream_job_engine.providers.jev import JevSignalProvider
from ai_dream_job_engine.providers.openai_compatible import OpenAICompatibleProvider

__all__ = [
    "CandidateClaimExtractor",
    "DecisionSignalProvider",
    "JevSignalProvider",
    "JobFactExtractor",
    "OpenAICompatibleProvider",
    "ProviderError",
]
