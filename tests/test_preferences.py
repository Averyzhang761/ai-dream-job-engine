from copy import deepcopy

import pytest

from ai_dream_job_engine.decision import decide_job
from ai_dream_job_engine.models import (
    CandidateProfile,
    CustomPreferenceProposal,
    PreferenceResolution,
)
from ai_dream_job_engine.preferences import resolve_preference, stage_preference
from ai_dream_job_engine.rubric import load_default_rubric


def proposal(**updates: object) -> CustomPreferenceProposal:
    payload: dict[str, object] = {
        "statement": "I prefer roles with direct customer discovery.",
        "topic_id": "customer_discovery",
        "category": "role_content",
        "name": "Customer discovery",
        "desired_value": "Direct customer discovery",
        "requirement_level": "strong",
        "evidence_guidance": "Look for explicit customer interviews or discovery.",
        "fact_ids": ["product_customer_scope"],
        "signal_ids": [],
        "related_topic_ids": [],
    }
    payload.update(updates)
    return CustomPreferenceProposal.model_validate(payload)


def test_new_topic_stages_without_affecting_effective_dimensions() -> None:
    rubric = load_default_rubric()

    staged = stage_preference(rubric, proposal())

    assert staged.dimensions == rubric.dimensions
    assert len(staged.pending_preferences) == 1
    assert staged.pending_preferences[0].conflict_kind == "none"
    assert staged.revision != rubric.revision


def test_same_topic_and_value_is_duplicate() -> None:
    rubric = load_default_rubric()
    existing = rubric.dimensions[1]

    staged = stage_preference(
        rubric,
        proposal(
            topic_id=existing.topic_id,
            name=existing.name,
            desired_value=existing.desired_value,
            requirement_level=existing.requirement_level,
            fact_ids=existing.fact_ids,
            signal_ids=existing.signal_ids,
        ),
    )

    draft = staged.pending_preferences[0]
    assert draft.conflict_kind == "duplicate"
    assert draft.conflicting_dimension_id == existing.id


def test_same_topic_and_different_value_is_direct_conflict() -> None:
    rubric = load_default_rubric()
    existing = rubric.dimensions[1]

    staged = stage_preference(
        rubric,
        proposal(topic_id=existing.topic_id, desired_value="Avoid applied AI"),
    )

    draft = staged.pending_preferences[0]
    assert draft.conflict_kind == "direct_conflict"
    assert draft.conflicting_dimension_id == existing.id


def test_declared_related_topic_is_semantic_tension() -> None:
    rubric = load_default_rubric()

    staged = stage_preference(
        rubric,
        proposal(related_topic_ids=["product_customer_proximity"]),
    )

    draft = staged.pending_preferences[0]
    assert draft.conflict_kind == "semantic_tension"
    assert draft.conflicting_dimension_id == "product_customer_proximity"


def test_incomplete_proposal_requires_manual_review() -> None:
    rubric = load_default_rubric()

    staged = stage_preference(
        rubric,
        CustomPreferenceProposal(statement="Something about team quality"),
    )

    assert staged.pending_preferences[0].conflict_kind == "manual_review"
    with pytest.raises(ValueError, match="manual_review"):
        resolve_preference(
            staged,
            staged.pending_preferences[0].id,
            PreferenceResolution(action="accept_new"),
        )


def test_replace_resolution_leaves_one_effective_topic_rule() -> None:
    rubric = load_default_rubric()
    existing = rubric.dimensions[1]
    staged = stage_preference(
        rubric,
        proposal(
            topic_id=existing.topic_id,
            desired_value="Applied AI tied to measurable product outcomes",
            fact_ids=existing.fact_ids,
            signal_ids=existing.signal_ids,
        ),
    )

    resolved = resolve_preference(
        staged,
        staged.pending_preferences[0].id,
        PreferenceResolution(action="replace_existing"),
    )

    matching = [
        item for item in resolved.dimensions if item.topic_id == existing.topic_id
    ]
    assert len(matching) == 1
    assert matching[0].origin == "custom"
    assert matching[0].desired_value == "Applied AI tied to measurable product outcomes"
    assert resolved.pending_preferences == []


def test_unresolved_draft_does_not_change_decision_result() -> None:
    rubric = load_default_rubric()
    staged = stage_preference(rubric, proposal())

    original = decide_job([], [], CandidateProfile(), rubric)
    pending = decide_job([], [], CandidateProfile(), staged)

    assert pending.preference_score == original.preference_score
    assert pending.preference_contributions == original.preference_contributions
    assert pending.gaps == original.gaps


def test_resolution_changes_effective_revision() -> None:
    rubric = load_default_rubric()
    staged = stage_preference(rubric, proposal())

    resolved = resolve_preference(
        staged,
        staged.pending_preferences[0].id,
        PreferenceResolution(action="accept_new"),
    )

    assert resolved.revision != staged.revision
    assert resolved.pending_preferences == []
    assert any(item.topic_id == "customer_discovery" for item in resolved.dimensions)


def test_staging_does_not_mutate_input_rubric() -> None:
    rubric = load_default_rubric()
    before = deepcopy(rubric)

    stage_preference(rubric, proposal())

    assert rubric == before
