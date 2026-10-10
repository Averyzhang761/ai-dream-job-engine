import pytest
from pydantic import ValidationError

from ai_dream_job_engine.models import RecommendationLevel, RubricPatch
from ai_dream_job_engine.rubric import apply_rubric_patch, load_default_rubric


EXPECTED_DIMENSION_IDS = {
    "role_family_fit",
    "applied_ai_depth",
    "product_customer_proximity",
    "ownership_impact",
    "infrastructure_balance",
    "work_arrangement",
    "compensation",
    "work_authorization",
}


def test_default_rubric_contains_ai_job_dimensions() -> None:
    rubric = load_default_rubric()

    assert rubric.version == "ai-job-v1"
    assert {dimension.id for dimension in rubric.dimensions} == EXPECTED_DIMENSION_IDS


def test_default_questions_include_product_applied_ai_and_infrastructure() -> None:
    rubric = load_default_rubric()

    assert {question.id for question in rubric.questions} == {
        "product_facing",
        "applied_ai",
        "pure_infrastructure",
    }


def test_patch_can_change_weight_and_threshold() -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(
        dimension_updates={"applied_ai_depth": {"weight": 2.5, "threshold": 0.7}},
        threshold_updates={RecommendationLevel.PRIORITY_APPLY: 0.85},
    )

    updated = apply_rubric_patch(rubric, patch)

    applied_ai = next(
        dimension
        for dimension in updated.dimensions
        if dimension.id == "applied_ai_depth"
    )
    assert applied_ai.weight == 2.5
    assert applied_ai.threshold == 0.7
    assert updated.recommendation_thresholds[
        RecommendationLevel.PRIORITY_APPLY
    ] == 0.85
    assert next(
        dimension
        for dimension in rubric.dimensions
        if dimension.id == "applied_ai_depth"
    ).weight != 2.5


@pytest.mark.parametrize(
    "updates",
    [
        {"fact_ids": ["missing_fact"]},
        {"signal_ids": ["missing_signal"]},
    ],
)
def test_patch_cannot_reference_missing_fact_or_signal(
    updates: dict[str, list[str]],
) -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(dimension_updates={"applied_ai_depth": updates})

    with pytest.raises(ValueError, match="unknown (fact|signal) reference"):
        apply_rubric_patch(rubric, patch)


def test_missing_fact_policy_remains_unknown() -> None:
    rubric = load_default_rubric()

    assert rubric.facts
    assert {fact.missing_policy for fact in rubric.facts} == {"unknown"}
    assert all(fact.default_value is None for fact in rubric.facts)


def test_patch_rejects_unknown_declarative_fields() -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(
        dimension_updates={"applied_ai_depth": {"callback": "run_arbitrary_code"}}
    )

    with pytest.raises(ValidationError):
        apply_rubric_patch(rubric, patch)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_patch_rejects_invalid_thresholds(threshold: float) -> None:
    rubric = load_default_rubric()

    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_rubric_patch(
            rubric,
            RubricPatch(
                threshold_updates={RecommendationLevel.PRIORITY_APPLY: threshold}
            ),
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_rubric_patch(
            rubric,
            RubricPatch(
                dimension_updates={"applied_ai_depth": {"threshold": threshold}}
            ),
        )


def test_patch_rejects_negative_weights() -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(dimension_updates={"applied_ai_depth": {"weight": -0.1}})

    with pytest.raises(ValidationError):
        apply_rubric_patch(rubric, patch)


def test_patch_rejects_duplicate_ids_in_input_rubric() -> None:
    rubric = load_default_rubric()
    rubric.dimensions.append(rubric.dimensions[0].model_copy(deep=True))

    with pytest.raises(ValueError, match="duplicate dimension id"):
        apply_rubric_patch(rubric, RubricPatch())
