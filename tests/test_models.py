import pytest
from pydantic import ValidationError

from ai_dream_job_engine.models import (
    CandidateClaim,
    ClaimSource,
    DecisionResult,
    DecisionSignal,
    FactDefinition,
    Gap,
    RecommendationLevel,
    ValueDimension,
)


def test_agent_suggested_claim_requires_confirmation() -> None:
    with pytest.raises(ValidationError, match="cannot be confirmed directly"):
        CandidateClaim(
            id="claim-1",
            claim_type="skill",
            value="Python",
            source=ClaimSource.AGENT_SUGGESTED,
            confirmed=True,
        )


def test_fact_definition_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        FactDefinition(
            id="fact-1",
            name="Role family",
            definition="The primary role family.",
            data_type="object",
            extraction_instructions="Extract the named role family.",
        )


def test_dimension_rejects_negative_weight() -> None:
    with pytest.raises(ValidationError):
        ValueDimension(
            id="dimension-1",
            name="Applied AI depth",
            definition="How directly the role builds applied AI products.",
            weight=-0.1,
        )


def test_signal_probability_must_be_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        DecisionSignal(
            question_id="applied_ai",
            question_version="1",
            probability=1.01,
        )


def test_decision_result_requires_evidence_for_every_gap() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        DecisionResult(
            preference_score=0.8,
            qualification_score=0.5,
            recommendation=RecommendationLevel.APPLY_AFTER_REVIEW,
            gaps=[
                Gap(
                    kind="evidence_gap",
                    subject_id="python",
                    message="Python experience is not established.",
                )
            ],
            rubric_version="ai-job-v1",
        )
