"""Consent-gated candidate profile updates."""

from __future__ import annotations

import re
from collections.abc import Hashable
from typing import Any

from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    DataClass,
    MemoryImportPreview,
    OutboundDisclosure,
)
from ai_dream_job_engine.providers import CandidateClaimExtractor


def _normalized_value(value: Any) -> Hashable:
    if isinstance(value, str):
        return ("str", re.sub(r"\s+", " ", value).strip().casefold())
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (str(key), _normalized_value(item))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_normalized_value(item) for item in value))
    return (type(value).__name__, repr(value))


def _normalized_claim_key(claim: CandidateClaim) -> tuple[str, Hashable]:
    """Deduplicate by normalized claim type and value, retaining type boundaries."""

    claim_type = re.sub(r"\s+", " ", claim.claim_type).strip().casefold()
    return claim_type, _normalized_value(claim.value)


def _validate_memory_candidate(
    candidate: CandidateClaim, source_agent: str
) -> None:
    if candidate.source is not ClaimSource.AGENT_SUGGESTED:
        raise ValueError("memory candidates must be agent_suggested")
    if candidate.confirmed:
        raise ValueError("memory candidates must be unconfirmed")
    if candidate.source_agent != source_agent:
        raise ValueError("memory candidate source_agent does not match import source")
    if not candidate.provenance or not candidate.provenance.strip():
        raise ValueError("memory candidates must carry provenance")


def _validate_disclosure(
    extractor: CandidateClaimExtractor, disclosure: OutboundDisclosure | None
) -> None:
    if not getattr(extractor, "requires_disclosure", False):
        return

    provider = getattr(extractor, "provider", None)
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("remote extractor must declare a non-empty provider")
    if (
        disclosure is None
        or not disclosure.confirmed
        or disclosure.provider != provider
        or disclosure.data_class is not DataClass.RESTRICTED_PERSONAL_DATA
        or "resume_text" not in disclosure.fields
        or not disclosure.purpose.strip()
    ):
        raise ValueError("confirmed disclosure for resume_text is required")


def propose_resume_claims(
    resume_text: str,
    extractor: CandidateClaimExtractor,
    *,
    disclosure: OutboundDisclosure | None = None,
) -> list[CandidateClaim]:
    """Return extractor proposals without retaining or rewriting resume content."""

    _validate_disclosure(extractor, disclosure)
    return extractor.extract(resume_text)


def preview_memory_import(
    profile: CandidateProfile,
    source_agent: str,
    scope: list[str],
    candidates: list[CandidateClaim],
) -> MemoryImportPreview:
    """Classify host-supplied memory claims without changing profile state."""

    for candidate in candidates:
        _validate_memory_candidate(candidate, source_agent)

    existing_by_id = {claim.id: claim for claim in profile.claims}
    existing_keys = {_normalized_claim_key(claim) for claim in profile.claims}
    active_by_type: dict[str, list[CandidateClaim]] = {}
    for claim in profile.claims:
        if claim.confirmed and claim.source is ClaimSource.USER_CONFIRMED:
            claim_type = _normalized_claim_key(claim)[0]
            active_by_type.setdefault(claim_type, []).append(claim)

    added: list[str] = []
    changed: list[str] = []
    conflicting: list[str] = []
    seen_keys: set[tuple[str, Hashable]] = set()
    for candidate in candidates:
        key = _normalized_claim_key(candidate)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        active_claims = active_by_type.get(key[0], [])
        if active_claims and all(
            _normalized_claim_key(active) != key for active in active_claims
        ):
            conflicting.append(candidate.id)
        elif candidate.id in existing_by_id:
            if _normalized_claim_key(existing_by_id[candidate.id]) != key:
                changed.append(candidate.id)
        elif key not in existing_keys:
            added.append(candidate.id)

    return MemoryImportPreview(
        source_agent=source_agent,
        scope=list(scope),
        candidates=[candidate.model_copy(deep=True) for candidate in candidates],
        added_claim_ids=added,
        changed_claim_ids=changed,
        conflicting_claim_ids=conflicting,
    )


def confirm_claims(
    profile: CandidateProfile, decisions: list[ClaimDecision]
) -> CandidateProfile:
    """Apply explicit decisions while leaving the input profile untouched."""

    decisions_by_id: dict[str, ClaimDecision] = {}
    for decision in decisions:
        if decision.claim_id in decisions_by_id:
            raise ValueError(f"duplicate decision for claim {decision.claim_id}")
        decisions_by_id[decision.claim_id] = decision

    known_ids = {claim.id for claim in profile.claims}
    unknown_ids = decisions_by_id.keys() - known_ids
    if unknown_ids:
        raise ValueError(f"unknown claim IDs: {', '.join(sorted(unknown_ids))}")

    result_claims: list[CandidateClaim] = []
    for claim in profile.claims:
        decision = decisions_by_id.get(claim.id)
        if decision is None or decision.action == "reject":
            result_claims.append(claim.model_copy(deep=True))
            continue
        if claim.confirmed:
            raise ValueError(f"claim {claim.id} is already confirmed")

        value = decision.edited_value if decision.action == "edit" else claim.value
        confirmed = claim.model_copy(
            deep=True,
            update={
                "value": value,
                "source": ClaimSource.USER_CONFIRMED,
                "confirmed": True,
            },
        )
        confirmed_key = _normalized_claim_key(confirmed)
        conflicts = [
            active
            for active in profile.claims
            if active.id != claim.id
            and active.confirmed
            and active.source is ClaimSource.USER_CONFIRMED
            and _normalized_claim_key(active)[0] == confirmed_key[0]
            and _normalized_claim_key(active) != confirmed_key
        ]
        if conflicts:
            raise ValueError(f"claim {claim.id} conflicts with confirmed profile data")
        already_confirmed = any(
            existing.id != claim.id
            and existing.confirmed
            and existing.source is ClaimSource.USER_CONFIRMED
            and _normalized_claim_key(existing) == confirmed_key
            for existing in profile.claims
        )
        if not already_confirmed and not any(
            _normalized_claim_key(existing) == confirmed_key
            for existing in result_claims
        ):
            result_claims.append(confirmed)

    return profile.model_copy(update={"claims": result_claims}, deep=True)


def disconnect_memory_source(
    profile: CandidateProfile, source_agent: str, delete_imported: bool
) -> CandidateProfile:
    """Optionally revoke confirmed claims imported from one Agent source."""

    if not delete_imported:
        return profile.model_copy(deep=True)

    claims = [
        claim.model_copy(deep=True)
        for claim in profile.claims
        if not (
            claim.source is ClaimSource.USER_CONFIRMED
            and claim.confirmed
            and claim.source_agent == source_agent
            and claim.provenance
        )
    ]
    return profile.model_copy(update={"claims": claims}, deep=True)
