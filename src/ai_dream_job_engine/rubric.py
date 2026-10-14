"""Load and declaratively update the default AI-job rubric."""

from __future__ import annotations

from importlib.resources import files
import hashlib
import json
import math
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_dream_job_engine.models import (
    DecisionQuestion,
    FactDataType,
    FactDefinition,
    RecommendationLevel,
    RequirementLevel,
    Rubric,
    RubricPatch,
    ValueDimension,
)


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def requirement_weight(level: RequirementLevel | str) -> float:
    return {
        RequirementLevel.IGNORED: 0.0,
        RequirementLevel.NORMAL: 1.0,
        RequirementLevel.STRONG: 1.5,
        RequirementLevel.REQUIRED: 1.0,
    }[RequirementLevel(level)]


def requirement_is_hard(level: RequirementLevel | str) -> bool:
    return RequirementLevel(level) is RequirementLevel.REQUIRED


def load_default_rubric() -> Rubric:
    """Load and validate the bundled AI-job rubric."""
    template = files("ai_dream_job_engine").joinpath("templates", "ai_job_v1.json")
    rubric = Rubric.model_validate_json(template.read_text(encoding="utf-8"))
    rubric = rubric.model_copy(update={"revision": rubric_revision(rubric)})
    _validate_rubric(rubric)
    return rubric


def apply_rubric_patch(rubric: Rubric, patch: RubricPatch) -> Rubric:
    """Return a validated rubric with declarative updates applied."""
    _validate_rubric(rubric)

    facts = [
        _normalize_fact_default(fact)
        for fact in _apply_updates(rubric.facts, patch.fact_updates, FactDefinition)
    ]
    dimension_updates = {
        item_id: (
            {**changes, "legacy_weight_override": None}
            if "requirement_level" in changes
            else changes
        )
        for item_id, changes in patch.dimension_updates.items()
    }
    dimensions = _apply_updates(rubric.dimensions, dimension_updates, ValueDimension)
    questions = _apply_updates(
        rubric.questions, patch.question_updates, DecisionQuestion
    )
    thresholds = dict(rubric.recommendation_thresholds)
    thresholds.update(patch.threshold_updates)

    updated = Rubric(
        id=rubric.id,
        version=rubric.version,
        storage_revision=rubric.storage_revision,
        facts=facts,
        dimensions=dimensions,
        questions=questions,
        pending_preferences=rubric.pending_preferences,
        recommendation_thresholds=thresholds,
    )
    updated = updated.model_copy(update={"revision": rubric_revision(updated)})
    _validate_rubric(updated)
    return updated


def rubric_revision(rubric: Rubric) -> str:
    """Return the immutable digest of the rubric's semantic content."""
    payload = rubric.model_dump(
        mode="json", exclude={"revision", "storage_revision"}
    )
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _apply_updates(
    items: list[_ModelT],
    updates: dict[str, dict[str, Any]],
    model_type: type[_ModelT],
) -> list[_ModelT]:
    by_id = {item.id: item for item in items}

    for item_id, changes in updates.items():
        if "id" in changes and changes["id"] != item_id:
            raise ValueError(f"patch id {changes['id']!r} does not match {item_id!r}")

        payload = (
            by_id[item_id].model_dump(mode="python") if item_id in by_id else {"id": item_id}
        )
        payload.update(changes)
        by_id[item_id] = model_type.model_validate(payload)

    return list(by_id.values())


def _validate_rubric(rubric: Rubric) -> None:
    _require_unique_ids("fact", rubric.facts)
    _require_unique_ids("dimension", rubric.dimensions)
    topics = [
        " ".join(item.topic_id.casefold().split())
        for item in rubric.dimensions
    ]
    if len(topics) != len(set(topics)):
        raise ValueError("dimension topic_id values must be unique")
    _require_unique_ids("question", rubric.questions)

    facts_by_id = {fact.id: fact for fact in rubric.facts}
    fact_ids = set(facts_by_id)
    signal_ids = {question.id for question in rubric.questions}
    for fact in rubric.facts:
        _validate_fact_default(fact)
        if not math.isfinite(fact.qualification_weight):
            raise ValueError(f"fact {fact.id} qualification_weight must be finite")

    for dimension in rubric.dimensions:
        if not math.isfinite(dimension.weight):
            raise ValueError(f"dimension {dimension.id} weight must be finite")
        if (
            dimension.legacy_weight_override is not None
            and not math.isfinite(dimension.legacy_weight_override)
        ):
            raise ValueError(
                f"dimension {dimension.id} legacy_weight_override must be finite"
            )
        _require_threshold(dimension.threshold, f"dimension {dimension.id}")
        for fact_id in dimension.fact_ids:
            if fact_id not in fact_ids:
                raise ValueError(
                    f"unknown fact reference {fact_id!r} in dimension {dimension.id!r}"
                )
        for signal_id in dimension.signal_ids:
            if signal_id not in signal_ids:
                raise ValueError(
                    f"unknown signal reference {signal_id!r} in dimension {dimension.id!r}"
                )
        if dimension.hard_constraint and not (
            dimension.fact_ids or dimension.signal_ids
        ):
            raise ValueError(
                f"hard_constraint dimension {dimension.id!r} must reference an input"
            )

    expected_levels = set(RecommendationLevel)
    if set(rubric.recommendation_thresholds) != expected_levels:
        raise ValueError("thresholds are required for all recommendation levels")

    for level, threshold in rubric.recommendation_thresholds.items():
        _require_threshold(threshold, f"recommendation {level.value}")

    thresholds = rubric.recommendation_thresholds
    if not (
        thresholds[RecommendationLevel.PRIORITY_APPLY]
        > thresholds[RecommendationLevel.APPLY_AFTER_REVIEW]
        > thresholds[RecommendationLevel.HOLD]
        > thresholds[RecommendationLevel.SKIP]
    ):
        raise ValueError(
            "recommendation thresholds must satisfy priority_apply > "
            "apply_after_review > hold > skip"
        )
    if thresholds[RecommendationLevel.SKIP] != 0:
        raise ValueError("skip threshold must be zero")


def _validate_fact_default(fact: FactDefinition) -> None:
    _normalize_fact_default(fact)


def _normalize_fact_default(fact: FactDefinition) -> FactDefinition:
    value = fact.default_value
    if value is None:
        return fact

    try:
        value = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"fact {fact.id} default_value must be JSON-compatible"
        ) from error

    valid_type = {
        FactDataType.BOOLEAN: type(value) is bool,
        FactDataType.NUMBER: (
            type(value) in (int, float) and math.isfinite(value)
        ),
        FactDataType.TEXT: isinstance(value, str),
        FactDataType.ENUM: isinstance(value, str),
        FactDataType.LIST: isinstance(value, list),
    }[fact.data_type]
    if not valid_type:
        raise ValueError(
            f"fact {fact.id} default_value must match {fact.data_type.value}"
        )
    if fact.data_type is FactDataType.ENUM and value not in fact.enum_values:
        raise ValueError(f"fact {fact.id} default_value must be in enum_values")
    return fact.model_copy(update={"default_value": value})


def _require_unique_ids(label: str, items: list[BaseModel]) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = str(item.id)
        if item_id in seen:
            raise ValueError(f"duplicate {label} id {item_id!r}")
        seen.add(item_id)


def _require_threshold(value: float | None, label: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{label} threshold must be between 0 and 1")
