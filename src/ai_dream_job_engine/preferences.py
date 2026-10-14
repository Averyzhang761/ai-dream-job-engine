"""Stage and explicitly resolve custom job preferences."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from ai_dream_job_engine.models import (
    CriterionOrigin,
    CustomPreferenceProposal,
    DecisionQuestion,
    PendingPreference,
    PreferenceConflictKind,
    PreferenceResolution,
    Rubric,
    ValueDimension,
)
from ai_dream_job_engine.rubric import (
    _validate_rubric,
    requirement_is_hard,
    requirement_weight,
    rubric_revision,
)


def stage_preference(
    rubric: Rubric, proposal: CustomPreferenceProposal
) -> Rubric:
    conflict, conflicting_id = _classify(rubric, proposal)
    pending = PendingPreference(
        id=f"preference-{uuid4()}",
        **proposal.model_dump(mode="python"),
        conflict_kind=conflict,
        conflicting_dimension_id=conflicting_id,
    )
    return _replace(
        rubric,
        pending_preferences=[*rubric.pending_preferences, pending],
    )


def resolve_preference(
    rubric: Rubric,
    draft_id: str,
    resolution: PreferenceResolution,
) -> Rubric:
    draft = next(
        (item for item in rubric.pending_preferences if item.id == draft_id),
        None,
    )
    if draft is None:
        raise ValueError("pending preference was not found")

    proposal = resolution.edited_proposal or CustomPreferenceProposal.model_validate(
        draft.model_dump(
            mode="python",
            exclude={"id", "conflict_kind", "conflicting_dimension_id"},
        )
    )
    conflict, conflicting_id = _classify(rubric, proposal)
    action = resolution.action
    allowed = {
        PreferenceConflictKind.NONE: {"accept_new"},
        PreferenceConflictKind.DUPLICATE: {"keep_existing"},
        PreferenceConflictKind.DIRECT_CONFLICT: {
            "replace_existing",
            "keep_existing",
        },
        PreferenceConflictKind.SEMANTIC_TENSION: {
            "accept_tension",
            "keep_existing",
        },
        PreferenceConflictKind.MANUAL_REVIEW: set(),
    }[conflict]
    if action not in allowed:
        raise ValueError(f"{action} cannot resolve {conflict.value}")

    pending = [item for item in rubric.pending_preferences if item.id != draft_id]
    if action == "keep_existing":
        return _replace(rubric, pending_preferences=pending)

    existing = next(
        (item for item in rubric.dimensions if item.id == conflicting_id),
        None,
    )
    dimension = _dimension_from_proposal(proposal, existing)
    if action == "replace_existing":
        dimensions = [
            dimension if item.id == conflicting_id else item
            for item in rubric.dimensions
        ]
    else:
        dimensions = [*rubric.dimensions, dimension]
    question = _question_from_proposal(proposal)
    referenced_signals = {
        signal_id for item in dimensions for signal_id in item.signal_ids
    }
    questions = [
        item
        for item in rubric.questions
        if item.id != question.id and item.id in referenced_signals
    ]
    questions.append(question)
    return _replace(
        rubric,
        dimensions=dimensions,
        questions=questions,
        pending_preferences=pending,
    )


def _classify(
    rubric: Rubric, proposal: CustomPreferenceProposal
) -> tuple[PreferenceConflictKind, str | None]:
    required = (
        proposal.topic_id,
        proposal.category,
        proposal.name,
        proposal.desired_value,
        proposal.requirement_level,
        proposal.evidence_guidance,
    )
    if any(item is None or (isinstance(item, str) and not item.strip()) for item in required):
        return PreferenceConflictKind.MANUAL_REVIEW, None
    if not proposal.positive_keywords and not proposal.negative_keywords:
        return PreferenceConflictKind.MANUAL_REVIEW, None

    topic = _normalized(proposal.topic_id)
    existing = next(
        (
            item
            for item in rubric.dimensions
            if _normalized(item.topic_id) == topic
        ),
        None,
    )
    if existing is not None:
        if (
            _normalized(existing.desired_value)
            == _normalized(proposal.desired_value)
            and existing.requirement_level == proposal.requirement_level
        ):
            return PreferenceConflictKind.DUPLICATE, existing.id
        return PreferenceConflictKind.DIRECT_CONFLICT, existing.id

    dimensions_by_topic = {
        _normalized(item.topic_id): item for item in rubric.dimensions
    }
    for related_topic in proposal.related_topic_ids:
        related = dimensions_by_topic.get(_normalized(related_topic))
        if related is not None:
            return PreferenceConflictKind.SEMANTIC_TENSION, related.id

    return PreferenceConflictKind.NONE, None


def _dimension_from_proposal(
    proposal: CustomPreferenceProposal,
    existing: ValueDimension | None,
) -> ValueDimension:
    if (
        proposal.topic_id is None
        or proposal.category is None
        or proposal.name is None
        or proposal.desired_value is None
        or proposal.requirement_level is None
        or proposal.evidence_guidance is None
    ):
        raise ValueError("manual_review proposal is incomplete")

    signal_id = _custom_signal_id(proposal.topic_id)
    payload = existing.model_dump(mode="python") if existing is not None else {
        "id": f"custom:{proposal.topic_id}",
        "examples": [],
        "scoring_direction": "higher_is_better",
        "positive_keywords": [],
        "negative_keywords": [],
        "threshold": None,
    }
    payload.update(
        name=proposal.name,
        definition=proposal.evidence_guidance,
        topic_id=proposal.topic_id,
        category=proposal.category,
        desired_value=proposal.desired_value,
        requirement_level=proposal.requirement_level,
        origin=CriterionOrigin.CUSTOM,
        fact_ids=[],
        signal_ids=[signal_id],
        weight=requirement_weight(proposal.requirement_level),
        legacy_weight_override=None,
        hard_constraint=requirement_is_hard(proposal.requirement_level),
        explanation_template=proposal.evidence_guidance,
    )
    return ValueDimension.model_validate(payload)


def _question_from_proposal(proposal: CustomPreferenceProposal) -> DecisionQuestion:
    if proposal.topic_id is None or proposal.desired_value is None:
        raise ValueError("manual_review proposal is incomplete")
    question = (
        f"Does this role satisfy the preference '{proposal.desired_value}'? "
        f"Use this evidence guidance: {proposal.evidence_guidance}."
    )
    version = hashlib.sha256(question.encode()).hexdigest()[:16]
    return DecisionQuestion(
        id=_custom_signal_id(proposal.topic_id),
        version=f"sha256:{version}",
        question=question,
        evaluation="local_keywords",
        positive_keywords=proposal.positive_keywords,
        negative_keywords=proposal.negative_keywords,
    )


def _custom_signal_id(topic_id: str) -> str:
    topic_hash = hashlib.sha256(_normalized(topic_id).encode()).hexdigest()[:16]
    return f"custom:{topic_hash}:fit"


def _replace(rubric: Rubric, **updates: object) -> Rubric:
    payload = rubric.model_dump(mode="python", exclude={"revision"})
    payload.update(updates)
    updated = Rubric.model_validate(payload)
    _validate_rubric(updated)
    return updated.model_copy(update={"revision": rubric_revision(updated)})


def _normalized(value: str | None) -> str:
    return " ".join((value or "").casefold().split())
