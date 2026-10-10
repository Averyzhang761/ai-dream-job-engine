import os
from pathlib import Path
import subprocess
import sys
import zipfile

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


class CustomDefault:
    pass


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


def test_built_wheel_contains_template_and_loads_outside_checkout(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        assert "ai_dream_job_engine/templates/ai_job_v1.json" in archive.namelist()
        archive.extractall(installed)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from ai_dream_job_engine.rubric import load_default_rubric; "
                "print(load_default_rubric().version)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "ai-job-v1"


@pytest.mark.parametrize("default_value", [lambda: None, CustomDefault()])
def test_patch_rejects_executable_or_custom_defaults(default_value: object) -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(
        fact_updates={
            "role_family": {
                "missing_policy": "default",
                "default_value": default_value,
            }
        }
    )

    with pytest.raises(ValueError, match="JSON-compatible"):
        apply_rubric_patch(rubric, patch)


@pytest.mark.parametrize(
    ("data_type", "default_value", "enum_values"),
    [
        ("boolean", "yes", []),
        ("number", True, []),
        ("number", float("inf"), []),
        ("text", 42, []),
        ("enum", "platform", ["applied_ai"]),
        ("list", "applied_ai", []),
    ],
)
def test_patch_validates_default_against_fact_data_type(
    data_type: str,
    default_value: object,
    enum_values: list[str],
) -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(
        fact_updates={
            "role_family": {
                "data_type": data_type,
                "missing_policy": "default",
                "default_value": default_value,
                "enum_values": enum_values,
            }
        }
    )

    with pytest.raises(ValueError, match="default_value"):
        apply_rubric_patch(rubric, patch)


@pytest.mark.parametrize("weight", [float("inf"), float("-inf"), float("nan")])
def test_patch_rejects_non_finite_weights(weight: float) -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(dimension_updates={"applied_ai_depth": {"weight": weight}})

    with pytest.raises(ValueError):
        apply_rubric_patch(rubric, patch)


def test_patch_rejects_inverted_recommendation_thresholds() -> None:
    rubric = load_default_rubric()
    patch = RubricPatch(
        threshold_updates={
            RecommendationLevel.PRIORITY_APPLY: 0.1,
            RecommendationLevel.SKIP: 0.9,
        }
    )

    with pytest.raises(ValueError, match="priority_apply > apply_after_review"):
        apply_rubric_patch(rubric, patch)


def test_rubric_requires_all_recommendation_thresholds() -> None:
    rubric = load_default_rubric()
    del rubric.recommendation_thresholds[RecommendationLevel.HOLD]

    with pytest.raises(ValueError, match="all recommendation levels"):
        apply_rubric_patch(rubric, RubricPatch())


def test_default_work_arrangement_has_no_preferred_mode() -> None:
    rubric = load_default_rubric()
    work_arrangement = next(
        dimension for dimension in rubric.dimensions if dimension.id == "work_arrangement"
    )

    assert work_arrangement.positive_keywords == []
    assert work_arrangement.negative_keywords == []


def test_infrastructure_balance_models_pure_infrastructure_consistently() -> None:
    rubric = load_default_rubric()
    infrastructure = next(
        dimension
        for dimension in rubric.dimensions
        if dimension.id == "infrastructure_balance"
    )

    assert infrastructure.signal_ids == ["pure_infrastructure"]
    assert infrastructure.scoring_direction == "lower_is_better"
    assert "pure infrastructure" in infrastructure.positive_keywords
    assert "product integration" in infrastructure.negative_keywords
