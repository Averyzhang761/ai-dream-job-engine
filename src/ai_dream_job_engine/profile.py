"""Consent-gated candidate profile updates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Hashable, Sequence
from typing import Any
from urllib.parse import urlsplit

from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimImportOrigin,
    ClaimDecision,
    ClaimSource,
    DataClass,
    MemoryImportPreview,
    OutboundDisclosure,
)
from ai_dream_job_engine.providers import CandidateClaimExtractor


def _canonical_value(value: Any) -> Hashable:
    """Return a deterministic, type-preserving key for JSON-safe values."""

    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("claim values must be finite and JSON-safe")
        return ("float", value.hex())
    if type(value) is str:
        return ("str", re.sub(r"\s+", " ", value).strip().casefold())
    if type(value) is list:
        return ("list", tuple(_canonical_value(item) for item in value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("claim values must use JSON-safe string object keys")
        return (
            "dict",
            tuple((key, _canonical_value(value[key])) for key in sorted(value)),
        )
    raise ValueError("claim values must be JSON-safe primitives, lists, or objects")


def _normalized_claim_key(claim: CandidateClaim) -> tuple[str, Hashable]:
    """Deduplicate by normalized claim type and canonical value."""

    claim_type = re.sub(r"\s+", " ", claim.claim_type).strip().casefold()
    return claim_type, _canonical_value(claim.value)


def _duplicate_ids(claims: Sequence[CandidateClaim]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for claim in claims:
        if claim.id in seen:
            duplicates.add(claim.id)
        seen.add(claim.id)
    return sorted(duplicates)


def _validate_profile(profile: CandidateProfile) -> None:
    duplicates = _duplicate_ids(profile.claims)
    if duplicates:
        raise ValueError(f"duplicate profile claim IDs: {', '.join(duplicates)}")
    for claim in profile.claims:
        _canonical_value(claim.value)
        if claim.import_origin is not None:
            _canonical_value(claim.import_origin.candidate_value)
            _canonical_value(claim.import_origin.confirmed_value)


def _validate_memory_candidate(
    candidate: CandidateClaim, source_agent: str
) -> None:
    if candidate.source is not ClaimSource.AGENT_SUGGESTED:
        raise ValueError("memory candidates must be agent_suggested")
    if candidate.confirmed:
        raise ValueError("memory candidates must be unconfirmed")
    if candidate.import_origin is not None:
        raise ValueError("memory candidates cannot carry import_origin")
    if candidate.source_agent != source_agent:
        raise ValueError("memory candidate source_agent does not match import source")
    provenance = urlsplit(candidate.provenance or "")
    if (
        provenance.scheme != "agent-memory"
        or provenance.netloc != source_agent
        or provenance.path != f"/{candidate.id}"
    ):
        raise ValueError("memory candidate provenance does not match import source")
    _canonical_value(candidate.value)


def _validate_candidates(
    candidates: Sequence[CandidateClaim], source_agent: str
) -> None:
    duplicates = _duplicate_ids(candidates)
    if duplicates:
        raise ValueError(f"duplicate candidate IDs: {', '.join(duplicates)}")
    for candidate in candidates:
        _validate_memory_candidate(candidate, source_agent)


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
    claims = extractor.extract(resume_text)
    duplicates = _duplicate_ids(claims)
    if duplicates:
        raise ValueError(f"duplicate resume claim IDs: {', '.join(duplicates)}")
    for claim in claims:
        _canonical_value(claim.value)
    return claims


def preview_memory_import(
    profile: CandidateProfile,
    source_agent: str,
    scope: list[str],
    candidates: list[CandidateClaim],
) -> MemoryImportPreview:
    """Classify host-supplied memory claims without changing profile state."""

    _validate_profile(profile)
    _validate_candidates(candidates, source_agent)

    existing_by_id = {claim.id: claim for claim in profile.claims}
    existing_keys = {_normalized_claim_key(claim) for claim in profile.claims}
    active_by_type: dict[str, set[tuple[str, Hashable]]] = {}
    for claim in profile.claims:
        if claim.confirmed and claim.source is ClaimSource.USER_CONFIRMED:
            key = _normalized_claim_key(claim)
            active_by_type.setdefault(key[0], set()).add(key)

    added: list[str] = []
    changed: list[str] = []
    conflicting: list[str] = []
    seen_keys: set[tuple[str, Hashable]] = set()
    for candidate in candidates:
        key = _normalized_claim_key(candidate)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        existing = existing_by_id.get(candidate.id)
        if existing is not None and _normalized_claim_key(existing) != key:
            changed.append(candidate.id)

        active_keys = active_by_type.get(key[0], set())
        if active_keys and key not in active_keys:
            conflicting.append(candidate.id)
        elif existing is None and key not in existing_keys:
            added.append(candidate.id)

    return MemoryImportPreview(
        source_agent=source_agent,
        scope=list(scope),
        candidates=[candidate.model_copy(deep=True) for candidate in candidates],
        added_claim_ids=added,
        changed_claim_ids=changed,
        conflicting_claim_ids=conflicting,
    )


def _import_origin(candidate: CandidateClaim, value: Any) -> ClaimImportOrigin:
    return ClaimImportOrigin(
        source_agent=candidate.source_agent,
        candidate_id=candidate.id,
        provenance=candidate.provenance,
        claim_type=candidate.claim_type,
        candidate_value=candidate.value,
        confirmed_value=value,
    )


def _confirmed_claim_id(
    candidate: CandidateClaim, origin: ClaimImportOrigin
) -> str:
    source = json.dumps(
        {
            "origin": origin.model_dump(mode="json"),
            "source_quote": candidate.source_quote,
            "observed_at": (
                candidate.observed_at.isoformat() if candidate.observed_at else None
            ),
            "confidence": candidate.confidence,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"agent-import:{hashlib.sha256(source).hexdigest()}"


def _confirmed_claim(candidate: CandidateClaim, value: Any) -> CandidateClaim:
    if candidate.source is ClaimSource.AGENT_SUGGESTED:
        origin = _import_origin(candidate, value)
        claim_id = _confirmed_claim_id(candidate, origin)
    else:
        origin = None
        source = json.dumps(
            [candidate.source.value, candidate.id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        claim_id = f"user-confirmed:{hashlib.sha256(source).hexdigest()}"
    return candidate.model_copy(
        deep=True,
        update={
            "id": claim_id,
            "value": value,
            "source": ClaimSource.USER_CONFIRMED,
            "confirmed": True,
            "import_origin": origin,
        },
    )


def _validate_decisions(decisions: Sequence[ClaimDecision]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for decision in decisions:
        if decision.claim_id in seen:
            duplicates.add(decision.claim_id)
        seen.add(decision.claim_id)
        supplied = "edited_value" in decision.model_fields_set
        if decision.action == "edit" and not supplied:
            raise ValueError("edited_value must be supplied for edit")
        if decision.action != "edit" and supplied:
            raise ValueError("edited_value is only allowed for edit")
    if duplicates:
        raise ValueError(f"duplicate decision IDs: {', '.join(sorted(duplicates))}")


def confirm_claims(
    profile: CandidateProfile,
    decisions: list[ClaimDecision],
    *,
    preview: MemoryImportPreview | None = None,
) -> CandidateProfile:
    """Apply explicit decisions atomically while preserving source suggestions."""

    _validate_profile(profile)
    _validate_decisions(decisions)

    preview_candidates = preview.candidates if preview else []
    if preview:
        _validate_candidates(preview_candidates, preview.source_agent)
    profile_by_id = {claim.id: claim for claim in profile.claims}
    preview_by_id = {claim.id: claim for claim in preview_candidates}
    preview_ids = {claim.id for claim in preview_candidates}
    for claim_id in profile_by_id.keys() & preview_ids:
        existing = profile_by_id[claim_id]
        incoming = preview_by_id[claim_id]
        if (
            existing.source is not ClaimSource.AGENT_SUGGESTED
            or existing.confirmed
            or existing.source_agent != incoming.source_agent
            or existing.provenance != incoming.provenance
        ):
            raise ValueError(f"ambiguous cross-pool candidate ID: {claim_id}")
        _validate_memory_candidate(existing, preview.source_agent)

    candidates_by_id = {**profile_by_id, **preview_by_id}
    unknown_ids = {decision.claim_id for decision in decisions} - candidates_by_id.keys()
    if unknown_ids:
        raise ValueError(f"unknown claim IDs: {', '.join(sorted(unknown_ids))}")

    accepted: list[tuple[CandidateClaim, CandidateClaim]] = []
    accepted_preview_ids: set[str] = set()
    for decision in decisions:
        candidate = candidates_by_id[decision.claim_id]
        if decision.action == "reject":
            continue
        if candidate.confirmed:
            raise ValueError(f"claim {candidate.id} is already confirmed")
        if candidate.source is ClaimSource.AGENT_SUGGESTED:
            _validate_memory_candidate(candidate, candidate.source_agent or "")
        value = decision.edited_value if decision.action == "edit" else candidate.value
        _canonical_value(value)
        accepted.append((candidate, _confirmed_claim(candidate, value)))
        if candidate.id in preview_ids:
            accepted_preview_ids.add(candidate.id)

    if not accepted:
        return profile.model_copy(deep=True)

    active_claims = [
        claim
        for claim in profile.claims
        if claim.confirmed and claim.source is ClaimSource.USER_CONFIRMED
    ]
    active_keys = {_normalized_claim_key(claim) for claim in active_claims}
    active_by_type: dict[str, set[tuple[str, Hashable]]] = {}
    for key in active_keys:
        active_by_type.setdefault(key[0], set()).add(key)

    proposed_by_type: dict[str, set[tuple[str, Hashable]]] = {}
    for _, confirmed in accepted:
        key = _normalized_claim_key(confirmed)
        proposed_by_type.setdefault(key[0], set()).add(key)

    all_types = active_by_type.keys() | proposed_by_type.keys()
    if any(
        len(
            active_by_type.get(claim_type, set())
            | proposed_by_type.get(claim_type, set())
        )
        > 1
        for claim_type in all_types
    ):
        raise ValueError("confirmation conflict requires reject or edit resolution")

    existing_by_id = {claim.id: claim for claim in profile.claims}
    base_claims = [
        (
            preview_by_id[claim.id].model_copy(deep=True)
            if claim.id in accepted_preview_ids
            else claim.model_copy(deep=True)
        )
        for claim in profile.claims
    ]
    new_suggestions = sorted(
        (
            claim.model_copy(deep=True)
            for claim in preview_candidates
            if claim.id in accepted_preview_ids and claim.id not in profile_by_id
        ),
        key=lambda claim: claim.id,
    )
    new_confirmed: list[CandidateClaim] = []
    for _, confirmed in sorted(accepted, key=lambda pair: pair[1].id):
        existing = existing_by_id.get(confirmed.id)
        if existing is not None:
            if existing != confirmed:
                raise ValueError(f"confirmed claim ID collision: {confirmed.id}")
            continue
        new_confirmed.append(confirmed)

    claims = [
        *base_claims,
        *new_suggestions,
        *new_confirmed,
    ]
    result_data = profile.model_dump()
    result_data["claims"] = [claim.model_dump() for claim in claims]
    result = CandidateProfile.model_validate(result_data)
    _validate_profile(result)
    return result


def _is_linked_import(claim: CandidateClaim, source_agent: str) -> bool:
    origin = claim.import_origin
    if origin is None or origin.source_agent != source_agent:
        return False
    try:
        _canonical_value(origin.candidate_value)
        confirmed_value_matches = _canonical_value(
            origin.confirmed_value
        ) == _canonical_value(claim.value)
    except ValueError:
        return False
    provenance = urlsplit(origin.provenance)
    return (
        provenance.scheme == "agent-memory"
        and provenance.netloc == source_agent
        and provenance.path == f"/{origin.candidate_id}"
        and confirmed_value_matches
        and claim.id == _confirmed_claim_id(claim, origin)
        and claim.source is ClaimSource.USER_CONFIRMED
        and claim.confirmed
        and claim.claim_type == origin.claim_type
        and claim.provenance == origin.provenance
        and claim.source_agent == origin.source_agent
    )


def disconnect_memory_source(
    profile: CandidateProfile, source_agent: str, delete_imported: bool
) -> CandidateProfile:
    """Optionally revoke module-confirmed claims linked to one Agent source."""

    _validate_profile(profile)
    if not delete_imported:
        return profile.model_copy(deep=True)

    claims = [
        claim.model_copy(deep=True)
        for claim in profile.claims
        if not _is_linked_import(claim, source_agent)
    ]
    return profile.model_copy(update={"claims": claims}, deep=True)
