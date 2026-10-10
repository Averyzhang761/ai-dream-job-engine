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
    assert confirmed is not suggestion
    assert confirmed.source is ClaimSource.USER_CONFIRMED
    assert confirmed.value == "Applied AI PM"
    assert confirmed.source_quote == suggestion.source_quote
    assert confirmed.source_agent == suggestion.source_agent
    assert confirmed.provenance == suggestion.provenance
    assert confirmed.observed_at == suggestion.observed_at
    assert confirmed.confidence == suggestion.confidence
    assert suggestion.source is ClaimSource.AGENT_SUGGESTED
    assert suggestion.confirmed is False


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


def test_disconnect_can_remove_only_selected_source() -> None:
    selected = CandidateClaim(
        id="confirmed-career",
        claim_type="preferred_role",
        value="Applied AI PM",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        provenance="agent-memory://career-agent/memory-1",
        source_agent="career-agent",
    )
    other_agent = CandidateClaim(
        id="confirmed-learning",
        claim_type="learning_style",
        value="Examples first",
        source=ClaimSource.USER_CONFIRMED,
        confirmed=True,
        provenance="agent-memory://learning-agent/memory-2",
        source_agent="learning-agent",
    )
    resume = CandidateClaim(
        id="resume-1",
        claim_type="experience",
        value="Production RAG",
        source=ClaimSource.RESUME_EVIDENCE,
        confirmed=True,
        source_quote="Built a production retrieval system.",
    )
    profile = CandidateProfile(claims=[selected, other_agent, resume])

    unchanged = disconnect_memory_source(profile, "career-agent", False)
    disconnected = disconnect_memory_source(profile, "career-agent", True)

    assert unchanged == profile
    assert [claim.id for claim in disconnected.claims] == [
        "confirmed-learning",
        "resume-1",
    ]
    assert [claim.id for claim in profile.claims] == [
        "confirmed-career",
        "confirmed-learning",
        "resume-1",
    ]
