"""Load and declaratively update the default AI-job rubric."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ai_dream_job_engine.models import (
    DecisionQuestion,
    FactDefinition,
    Rubric,
    RubricPatch,
    ValueDimension,
)


_DEFAULT_RUBRIC_PATH = Path(__file__).with_name("templates") / "ai_job_v1.json"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def load_default_rubric() -> Rubric:
    """Load and validate the bundled AI-job rubric."""
    rubric = Rubric.model_validate_json(_DEFAULT_RUBRIC_PATH.read_text(encoding="utf-8"))
    _validate_rubric(rubric)
    return rubric


def apply_rubric_patch(rubric: Rubric, patch: RubricPatch) -> Rubric:
    """Return a validated rubric with declarative updates applied."""
    _validate_rubric(rubric)

    facts = _apply_updates(rubric.facts, patch.fact_updates, FactDefinition)
    dimensions = _apply_updates(
        rubric.dimensions, patch.dimension_updates, ValueDimension
    )
    questions = _apply_updates(
        rubric.questions, patch.question_updates, DecisionQuestion
    )
    thresholds = dict(rubric.recommendation_thresholds)
    thresholds.update(patch.threshold_updates)

    updated = Rubric(
        id=rubric.id,
        version=rubric.version,
        facts=facts,
        dimensions=dimensions,
        questions=questions,
        recommendation_thresholds=thresholds,
    )
    _validate_rubric(updated)
    return updated


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
    _require_unique_ids("question", rubric.questions)

    fact_ids = {fact.id for fact in rubric.facts}
    signal_ids = {question.id for question in rubric.questions}
    for dimension in rubric.dimensions:
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

    for level, threshold in rubric.recommendation_thresholds.items():
        _require_threshold(threshold, f"recommendation {level.value}")


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
