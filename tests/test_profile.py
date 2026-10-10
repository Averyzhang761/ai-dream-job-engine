from datetime import datetime, timezone

import pytest

from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    DataClass,
    OutboundDisclosure,
)
from ai_dream_job_engine.profile import (
    confirm_claims,
    disconnect_memory_source,
    preview_memory_import,
    propose_resume_claims,
)


OBSERVED_AT = datetime(2026, 10, 10, tzinfo=timezone.utc)


class FakeResumeExtractor:
    def __init__(self, claims: list[CandidateClaim]) -> None:
        self.claims = claims
        self.calls: list[str] = []

    def extract(self, resume_text: str) -> list[CandidateClaim]:
        self.calls.append(resume_text)
        return self.claims


class RemoteResumeExtractor(FakeResumeExtractor):
    requires_disclosure = True
    provider = "remote-resume-api"


def memory_claim(
    claim_id: str,
    claim_type: str,
    value: object,
    source_agent: str = "career-agent",
) -> CandidateClaim:
    return CandidateClaim(
        id=claim_id,
        claim_type=claim_type,
        value=value,
        source=ClaimSource.AGENT_SUGGESTED,
        source_quote="User described this preference.",
        provenance=f"agent-memory://{source_agent}/{claim_id}",
        source_agent=source_agent,
        observed_at=OBSERVED_AT,
        confidence=0.8,
    )


def test_resume_claims_keep_exact_source_quote() -> None:
    source_quote = "Built 3 production retrieval systems.\nLed evaluation design."
    extracted = CandidateClaim(
        id="resume-1",
        claim_type="experience",
        value="Production RAG",
        source=ClaimSource.RESUME_EVIDENCE,
        source_quote=source_quote,
    )

    claims = propose_resume_claims(
        "Resume text is not retained.", FakeResumeExtractor([extracted])
    )

    assert claims[0].source_quote == source_quote


def test_memory_import_is_preview_only() -> None:
    profile = CandidateProfile()
    candidate = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    profile_before = profile.model_dump()
    candidate_before = candidate.model_dump()

    preview = preview_memory_import(
        profile,
        source_agent="career-agent",
        scope=["preferences"],
        candidates=[candidate],
    )

    assert preview.added_claim_ids == ["memory-1"]
    assert profile.model_dump() == profile_before
    assert candidate.model_dump() == candidate_before


def test_previewed_candidate_can_be_confirmed_and_unknown_ids_are_rejected() -> None:
    profile = CandidateProfile()
    candidate = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    preview = preview_memory_import(
        profile, "career-agent", ["preferences"], [candidate]
    )

    result = confirm_claims(
        profile,
        [ClaimDecision(claim_id="memory-1", action="accept")],
        preview=preview,
    )

    assert result.claims[0].id == "memory-1"
    assert result.claims[1].id.startswith("agent-import:")
    exact_replay = confirm_claims(
        result,
        [ClaimDecision(claim_id="memory-1", action="accept")],
        preview=preview,
    )
    assert exact_replay == result

    with pytest.raises(ValueError, match="unknown claim IDs"):
        confirm_claims(
            profile,
            [ClaimDecision(claim_id="not-previewed", action="accept")],
            preview=preview,
        )


def test_unconfirmed_memory_claim_is_not_active() -> None:
    suggestion = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    profile = CandidateProfile(claims=[suggestion])

    result = confirm_claims(profile, [])

    assert result == profile
    assert not any(claim.confirmed for claim in result.claims)


def test_confirmed_memory_claim_becomes_user_confirmed_and_keeps_provenance() -> None:
    suggestion = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    profile = CandidateProfile(claims=[suggestion])

    result = confirm_claims(
        profile, [ClaimDecision(claim_id="memory-1", action="accept")]
    )

    confirmed = next(claim for claim in result.claims if claim.confirmed)
    retained = next(claim for claim in result.claims if not claim.confirmed)
    assert retained == suggestion
    assert confirmed.id.startswith("agent-import:")
    assert confirmed is not suggestion
    assert confirmed.source is ClaimSource.USER_CONFIRMED
    assert confirmed.value == "Applied AI PM"
    assert confirmed.source_quote == suggestion.source_quote
    assert confirmed.source_agent == suggestion.source_agent
    assert confirmed.provenance == suggestion.provenance
    assert confirmed.observed_at == suggestion.observed_at
    assert confirmed.confidence == suggestion.confidence
    assert confirmed.import_origin is not None
    assert confirmed.import_origin.candidate_id == suggestion.id
    assert confirmed.import_origin.confirmed_value == confirmed.value
    assert suggestion.source is ClaimSource.AGENT_SUGGESTED
    assert suggestion.confirmed is False

    persisted = CandidateProfile.model_validate_json(result.model_dump_json())
    assert persisted == result

    repeated = confirm_claims(
        result, [ClaimDecision(claim_id="memory-1", action="accept")]
    )
    assert repeated == result


def test_remote_resume_extraction_requires_explicit_disclosure_confirmation() -> None:
    extractor = RemoteResumeExtractor([])

    with pytest.raises(ValueError, match="disclosure"):
        propose_resume_claims("private resume", extractor)

    assert extractor.calls == []

    disclosure = OutboundDisclosure(
        provider="remote-resume-api",
        fields=["resume_text"],
        purpose="Extract candidate claims from the supplied resume.",
        data_class=DataClass.RESTRICTED_PERSONAL_DATA,
        confirmed=True,
    )
    propose_resume_claims("private resume", extractor, disclosure=disclosure)

    assert extractor.calls == ["private resume"]


def test_refresh_reports_conflict_without_overwrite() -> None:
    active = CandidateClaim(
        id="confirmed-1",
        claim_type="preferred_role",
        value="Applied AI PM",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        provenance="agent-memory://career-agent/memory-1",
        source_agent="career-agent",
    )
    profile = CandidateProfile(claims=[active])
    refreshed = memory_claim(
        "memory-2", "preferred_role", "Machine Learning Engineer"
    )
    profile_before = profile.model_dump()

    preview = preview_memory_import(
        profile,
        source_agent="career-agent",
        scope=["preferences"],
        candidates=[refreshed],
    )

    assert preview.conflicting_claim_ids == ["memory-2"]
    assert preview.changed_claim_ids == []
    assert profile.model_dump() == profile_before
    assert profile.claims[0].value == "Applied AI PM"


def test_same_source_same_id_refresh_is_an_explicit_candidate_subject() -> None:
    original = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    initial_preview = preview_memory_import(
        CandidateProfile(), "career-agent", ["preferences"], [original]
    )
    initial = confirm_claims(
        CandidateProfile(),
        [ClaimDecision(claim_id="memory-1", action="accept")],
        preview=initial_preview,
    )
    changed = memory_claim("memory-1", "preferred_role", "ML Engineer")
    refresh = preview_memory_import(
        initial, "career-agent", ["preferences"], [changed]
    )

    assert refresh.changed_claim_ids == ["memory-1"]
    assert refresh.conflicting_claim_ids == ["memory-1"]
    with pytest.raises(ValueError, match="conflict"):
        confirm_claims(
            initial,
            [ClaimDecision(claim_id="memory-1", action="accept")],
            preview=refresh,
        )

    resolved = confirm_claims(
        initial,
        [
            ClaimDecision(
                claim_id="memory-1", action="edit", edited_value="Applied AI PM"
            )
        ],
        preview=refresh,
    )

    retained = next(claim for claim in resolved.claims if claim.id == "memory-1")
    confirmed = [claim for claim in resolved.claims if claim.confirmed]
    assert retained.value == "ML Engineer"
    assert len(confirmed) == 2
    assert all(claim.value == "Applied AI PM" for claim in confirmed)

    disconnected = disconnect_memory_source(resolved, "career-agent", True)
    assert [claim.id for claim in disconnected.claims] == ["memory-1"]


def test_cross_pool_id_collision_is_rejected_when_identity_is_ambiguous() -> None:
    manual = CandidateClaim(
        id="memory-1",
        claim_type="location",
        value="Remote",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
    )
    candidate = memory_claim("memory-1", "location", "Remote")
    profile = CandidateProfile(claims=[manual])
    preview = preview_memory_import(
        profile, "career-agent", ["preferences"], [candidate]
    )

    with pytest.raises(ValueError, match="ambiguous cross-pool"):
        confirm_claims(
            profile,
            [ClaimDecision(claim_id="memory-1", action="accept")],
            preview=preview,
        )


def test_conflicting_confirmations_are_atomic_and_order_invariant() -> None:
    profile = CandidateProfile()
    candidates = [
        memory_claim("memory-1", "preferred_role", "Applied AI PM"),
        memory_claim("memory-2", "preferred_role", "ML Engineer"),
    ]
    preview = preview_memory_import(
        profile, "career-agent", ["preferences"], candidates
    )
    decisions = [
        ClaimDecision(claim_id="memory-1", action="accept"),
        ClaimDecision(claim_id="memory-2", action="accept"),
    ]

    for ordered in (decisions, list(reversed(decisions))):
        with pytest.raises(ValueError, match="conflict"):
            confirm_claims(profile, ordered, preview=preview)

    assert profile.claims == []


def test_non_conflicting_confirmation_is_decision_order_invariant() -> None:
    profile = CandidateProfile(
        claims=[
            memory_claim("memory-1", "preferred_role", "Applied AI PM"),
            memory_claim("memory-2", "location", "Remote"),
        ]
    )
    decisions = [
        ClaimDecision(claim_id="memory-1", action="accept"),
        ClaimDecision(claim_id="memory-2", action="accept"),
    ]

    forward = confirm_claims(profile, decisions)
    reverse = confirm_claims(profile, list(reversed(decisions)))

    assert forward == reverse


def test_conflict_requires_reject_or_edit_to_existing_value() -> None:
    active = CandidateClaim(
        id="manual-1",
        claim_type="preferred_role",
        value="Applied AI PM",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
    )
    candidate = memory_claim("memory-2", "preferred_role", "ML Engineer")
    profile = CandidateProfile(claims=[active])
    preview = preview_memory_import(
        profile, "career-agent", ["preferences"], [candidate]
    )

    with pytest.raises(ValueError, match="conflict"):
        confirm_claims(
            profile,
            [ClaimDecision(claim_id="memory-2", action="accept")],
            preview=preview,
        )

    resolved = confirm_claims(
        profile,
        [
            ClaimDecision(
                claim_id="memory-2", action="edit", edited_value="Applied AI PM"
            )
        ],
        preview=preview,
    )
    rejected = confirm_claims(
        profile,
        [ClaimDecision(claim_id="memory-2", action="reject")],
        preview=preview,
    )

    assert [claim.id for claim in resolved.claims[:2]] == ["manual-1", "memory-2"]
    assert len([claim for claim in resolved.claims if claim.confirmed]) == 2
    imported = resolved.claims[-1]
    assert imported.source_agent == "career-agent"
    assert imported.provenance == candidate.provenance
    assert rejected == profile


def test_same_value_accepts_preserve_each_candidate_provenance_record() -> None:
    career = memory_claim("career-1", "location", "Remote", "career-agent")
    learning = memory_claim("learning-1", "location", "Remote", "learning-agent")
    profile = CandidateProfile(claims=[career, learning])

    result = confirm_claims(
        profile,
        [
            ClaimDecision(claim_id="career-1", action="accept"),
            ClaimDecision(claim_id="learning-1", action="accept"),
        ],
    )

    confirmed = [claim for claim in result.claims if claim.confirmed]
    assert len(confirmed) == 2
    assert {claim.source_agent for claim in confirmed} == {
        "career-agent",
        "learning-agent",
    }
    assert {claim.provenance for claim in confirmed} == {
        career.provenance,
        learning.provenance,
    }


def test_disconnect_can_remove_only_selected_source() -> None:
    career_suggestion = memory_claim(
        "memory-1", "preferred_role", "Applied AI PM"
    )
    learning_suggestion = memory_claim(
        "memory-2", "learning_style", "Examples first", "learning-agent"
    )
    profile = CandidateProfile(claims=[career_suggestion, learning_suggestion])
    imported = confirm_claims(
        profile,
        [
            ClaimDecision(claim_id="memory-1", action="accept"),
            ClaimDecision(claim_id="memory-2", action="accept"),
        ],
    )
    spoofed_manual = CandidateClaim(
        id=(
            "agent-import:"
            "849e2edaf8693c8c98ef2a66c24442d337a8f9c2fa21c55882ed11c7d2edd298"
        ),
        claim_type="location",
        value="Remote",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        provenance="agent-memory://career-agent/spoof-source",
        source_agent="career-agent",
    )
    resume = CandidateClaim(
        id="resume-1",
        claim_type="experience",
        value="Production RAG",
        source=ClaimSource.RESUME_EVIDENCE,
        confirmed=True,
        source_quote="Built a production retrieval system.",
    )
    profile = imported.model_copy(
        update={"claims": imported.claims + [spoofed_manual, resume]}, deep=True
    )

    unchanged = disconnect_memory_source(profile, "career-agent", False)
    disconnected = disconnect_memory_source(profile, "career-agent", True)

    assert unchanged == profile
    remaining_ids = [claim.id for claim in disconnected.claims]
    career_confirmed_id = next(
        claim.id
        for claim in profile.claims
        if claim.confirmed and claim.source_agent == "career-agent"
    )
    assert career_confirmed_id not in remaining_ids
    assert "memory-1" in remaining_ids
    assert "memory-2" in remaining_ids
    assert spoofed_manual.id in remaining_ids
    assert "resume-1" in remaining_ids
    assert any(
        claim.confirmed and claim.source_agent == "learning-agent"
        for claim in disconnected.claims
    )
    assert len(profile.claims) == 6


def test_disconnect_preserves_forged_pair_with_different_manual_value() -> None:
    suggestion = memory_claim("forged-source", "location", "Remote")
    imported = confirm_claims(
        CandidateProfile(claims=[suggestion]),
        [ClaimDecision(claim_id="forged-source", action="accept")],
    )
    template = next(claim for claim in imported.claims if claim.confirmed)
    forged = template.model_copy(update={"value": "On-site"}, deep=True)
    profile = CandidateProfile(claims=[suggestion, forged])

    disconnected = disconnect_memory_source(profile, "career-agent", True)

    assert disconnected == profile


def test_memory_candidate_provenance_must_match_selected_agent() -> None:
    candidate = memory_claim("memory-1", "preferred_role", "Applied AI PM")
    spoofed = candidate.model_copy(
        update={"provenance": "agent-memory://other-agent/memory-1"}
    )

    with pytest.raises(ValueError, match="provenance"):
        preview_memory_import(
            CandidateProfile(), "career-agent", ["preferences"], [spoofed]
        )


@pytest.mark.parametrize("entrypoint", ["preview", "confirm"])
def test_duplicate_profile_ids_are_rejected(entrypoint: str) -> None:
    first = memory_claim("duplicate", "preferred_role", "Applied AI PM")
    second = memory_claim("duplicate", "location", "Remote")
    profile = CandidateProfile(claims=[first, second])

    with pytest.raises(ValueError, match="duplicate profile claim IDs"):
        if entrypoint == "preview":
            preview_memory_import(profile, "career-agent", [], [])
        else:
            confirm_claims(profile, [])


def test_duplicate_candidate_and_decision_ids_are_rejected() -> None:
    first = memory_claim("duplicate", "preferred_role", "Applied AI PM")
    second = memory_claim("duplicate", "location", "Remote")

    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        preview_memory_import(
            CandidateProfile(), "career-agent", [], [first, second]
        )

    profile = CandidateProfile(claims=[first])
    decisions = [
        ClaimDecision(claim_id="duplicate", action="accept"),
        ClaimDecision(claim_id="duplicate", action="reject"),
    ]
    with pytest.raises(ValueError, match="duplicate decision"):
        confirm_claims(profile, decisions)


def test_claim_values_are_json_safe_finite_and_type_preserving() -> None:
    candidates = [
        memory_claim("bool", "preference", True),
        memory_claim("int", "preference", 1),
        memory_claim("float", "preference", 1.0),
        memory_claim("null", "preference", None),
        memory_claim("list", "evidence", ["Python", {"years": 3}]),
        memory_claim("dict-a", "details", {"b": 2, "a": 1}),
        memory_claim("dict-b", "details", {"a": 1, "b": 2}),
    ]

    preview = preview_memory_import(
        CandidateProfile(), "career-agent", [], candidates
    )

    assert preview.added_claim_ids == [
        "bool",
        "int",
        "float",
        "null",
        "list",
        "dict-a",
    ]


@pytest.mark.parametrize(
    "unsupported",
    [float("nan"), float("inf"), ("tuple",), {1: "non-string key"}, object()],
)
def test_claim_values_reject_unsupported_or_non_finite_values(
    unsupported: object,
) -> None:
    candidate = memory_claim("memory-1", "preference", unsupported)

    with pytest.raises(ValueError, match="JSON-safe"):
        preview_memory_import(
            CandidateProfile(), "career-agent", [], [candidate]
        )
