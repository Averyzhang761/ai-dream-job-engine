import pytest
from pydantic import ValidationError

from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    ClaimDecision,
    ClaimSource,
    DecisionResult,
    DecisionSignal,
    EvaluationCase,
    EvaluationReport,
    FactDefinition,
    Gap,
    GapKind,
    JobReviewRecord,
    JobReviewStatus,
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


@pytest.mark.parametrize(
    ("action", "edited_value"),
    [
        ("accept", None),
        ("accept", "ignored"),
        ("reject", None),
        ("reject", "ignored"),
    ],
)
def test_claim_decision_forbids_edited_value_without_edit(
    action: str, edited_value: object
) -> None:
    with pytest.raises(ValidationError, match="edited_value"):
        ClaimDecision(
            claim_id="claim-1", action=action, edited_value=edited_value
        )


def test_claim_decision_edit_requires_explicit_value() -> None:
    with pytest.raises(ValidationError, match="edited_value"):
        ClaimDecision(claim_id="claim-1", action="edit")

    decision = ClaimDecision(claim_id="claim-1", action="edit", edited_value=None)

    assert "edited_value" in decision.model_fields_set


def test_fact_definition_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        FactDefinition(
            id="fact-1",
            name="Role family",
            definition="The primary role family.",
            data_type="object",
            extraction_instructions="Extract the named role family.",
        )


def test_fact_definition_requires_explicit_qualification_policy_and_weight() -> None:
    preference = FactDefinition(
        id="role",
        name="Role",
        definition="Role family.",
        data_type="text",
        extraction_instructions="Extract role family.",
    )
    qualification = FactDefinition(
        id="authorization",
        name="Authorization",
        definition="Allowed authorization paths.",
        data_type="list",
        extraction_instructions="Extract allowed paths.",
        candidate_comparison="allowed_alternatives",
        qualification_weight=1,
    )

    assert preference.candidate_comparison == "none"
    assert preference.qualification_weight == 0
    assert qualification.candidate_comparison == "allowed_alternatives"

    with pytest.raises(ValidationError, match="qualification_weight"):
        FactDefinition(
            id="invalid",
            name="Invalid",
            definition="Invalid policy.",
            data_type="text",
            extraction_instructions="Extract value.",
            candidate_comparison="exact",
        )

    with pytest.raises(ValidationError, match="list data_type"):
        FactDefinition(
            id="invalid-list-policy",
            name="Invalid list policy",
            definition="Invalid list policy.",
            data_type="text",
            extraction_instructions="Extract value.",
            candidate_comparison="allowed_alternatives",
            qualification_weight=1,
        )


def test_dimension_rejects_negative_weight() -> None:
    with pytest.raises(ValidationError):
        ValueDimension(
            id="dimension-1",
            name="Applied AI depth",
            definition="How directly the role builds applied AI products.",
            weight=-0.1,
            scoring_direction="higher_is_better",
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


def test_evidence_gap_can_explicitly_record_absent_source_evidence() -> None:
    result = DecisionResult(
        recommendation=RecommendationLevel.APPLY_AFTER_REVIEW,
        gaps=[
            Gap(
                kind="evidence_gap",
                subject_id="python",
                message="Python evidence is absent.",
                input_ids=["python"],
                missing_evidence=True,
            )
        ],
        rubric_version="ai-job-v1",
        needs_review=True,
    )

    assert result.gaps[0].job_evidence == []
    assert result.gaps[0].missing_evidence is True


@pytest.mark.parametrize("direction", ["higher_is_better", "lower_is_better"])
def test_dimension_accepts_supported_scoring_directions(direction: str) -> None:
    dimension = ValueDimension(
        id="dimension-1",
        name="Applied AI depth",
        definition="How directly the role builds applied AI products.",
        weight=1,
        scoring_direction=direction,
    )

    assert dimension.scoring_direction == direction


def test_dimension_rejects_unknown_scoring_direction() -> None:
    with pytest.raises(ValidationError):
        ValueDimension(
            id="dimension-1",
            name="Applied AI depth",
            definition="How directly the role builds applied AI products.",
            weight=1,
            scoring_direction="sideways",
        )


def test_analysis_record_rejects_negative_latency_ms() -> None:
    record = AnalysisRecord(
        id="analysis-1",
        input_hash="sha256:example",
        status="success",
        latency_ms=0,
    )

    assert record.latency_ms == 0

    with pytest.raises(ValidationError):
        AnalysisRecord(
            id="analysis-1",
            input_hash="sha256:example",
            status="success",
            latency_ms=-1,
        )


def test_job_review_status_has_only_supported_values() -> None:
    assert {status.value for status in JobReviewStatus} == {
        "unreviewed",
        "kept",
        "ignored",
        "ready_to_apply",
    }


def test_job_review_record_requires_analysis_id() -> None:
    with pytest.raises(ValidationError):
        JobReviewRecord(status=JobReviewStatus.UNREVIEWED)


def test_evaluation_case_requires_human_labels_before_system_result() -> None:
    assert EvaluationCase.model_fields["human_recommendation"].is_required()
    assert EvaluationCase.model_fields["human_gap_labels"].is_required()

    with pytest.raises(ValidationError):
        EvaluationCase(
            id="case-1",
            job_description="Build an applied AI product.",
            system_recommendation=RecommendationLevel.PRIORITY_APPLY,
            system_gap_labels=[],
        )


def test_evaluation_report_carries_all_named_metrics() -> None:
    report = EvaluationReport(
        cases=[
            EvaluationCase(
                id="case-1",
                job_description="Build an applied AI product.",
                human_recommendation=RecommendationLevel.PRIORITY_APPLY,
                human_gap_labels=[GapKind.EVIDENCE_GAP],
                system_recommendation=RecommendationLevel.APPLY_AFTER_REVIEW,
                system_gap_labels=[GapKind.EVIDENCE_GAP],
                system_preference_score=0.7,
                system_qualification_score=0.8,
                human_top_five_usefulness=0.8,
                human_review_time_seconds=120,
            )
        ],
        attempted_count=1,
        completed_count=1,
        failed_count=0,
        rerun_attempted_count=1,
        rerun_completed_count=1,
        rerun_failed_count=0,
        run_complete=True,
        recommendation_agreement=0.8,
        blocker_precision=1.0,
        gap_type_confusion_counts={
            GapKind.EVIDENCE_GAP: {GapKind.EVIDENCE_GAP: 1}
        },
        unsupported_fact_count=0,
        unsupported_signal_count=0,
        unsupported_gap_count=0,
        unsupported_contribution_count=0,
        unsupported_claim_count=0,
        rerun_stability=1.0,
        gap_label_agreement=1.0,
        top_five_case_ids=["case-1"],
        evaluation_runtime_seconds=2.5,
        top_five_usefulness=0.8,
        review_time_seconds=120,
    )

    assert report.recommendation_agreement == 0.8
    assert report.unsupported_claim_count == 0
    assert report.review_time_seconds == 120
    assert report.run_complete is True
    assert report.evaluation_runtime_seconds == 2.5
