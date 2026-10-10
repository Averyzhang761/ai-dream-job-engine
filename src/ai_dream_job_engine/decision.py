"""Deterministic, provider-free job decision rules."""

from __future__ import annotations

import math
from typing import Any, Iterable

from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimSource,
    DecisionContribution,
    DecisionResult,
    DecisionSignal,
    ExtractedFact,
    FactDataType,
    FactDefinition,
    Gap,
    GapKind,
    RecommendationLevel,
    Rubric,
    ScoringDirection,
    SignalStatus,
    ValueDimension,
)
from ai_dream_job_engine.rubric import _validate_rubric


_ACTIVE_SOURCES = {ClaimSource.RESUME_EVIDENCE, ClaimSource.USER_CONFIRMED}
_FALSE_TEXT = {"false", "no", "none", "missing", "lack", "lacks", "not held"}


def decide_job(
    facts: list[ExtractedFact],
    signals: list[DecisionSignal],
    profile: CandidateProfile,
    rubric: Rubric,
) -> DecisionResult:
    """Return a deterministic recommendation from validated local inputs."""
    _validate_rubric(rubric)
    facts_by_id = _unique_by_id(facts, "fact_id", "fact")
    signals_by_id = _unique_by_id(signals, "question_id", "signal")
    fact_definitions = {fact.id: fact for fact in rubric.facts}
    active_claims = _active_claims(profile.claims)

    preference_contributions: list[DecisionContribution] = []
    review_gaps: list[Gap] = []
    needs_review = False

    for dimension in sorted(rubric.dimensions, key=lambda item: item.id):
        scored_inputs: list[tuple[float, str, list[str]]] = []
        dimension_needs_review = False

        for fact_id in sorted(dimension.fact_ids):
            definition = fact_definitions[fact_id]
            fact = facts_by_id.get(fact_id)
            if fact is None or fact.is_unknown:
                dimension_needs_review = True
                if definition.requires_evidence:
                    review_gaps.append(_unknown_fact_gap(definition, fact))
                continue
            if definition.requires_evidence and not fact.evidence:
                dimension_needs_review = True
                review_gaps.append(_unknown_fact_gap(definition, fact))
                continue
            score = _fact_score(fact.value, definition, dimension)
            if score is None:
                dimension_needs_review = True
                if definition.requires_evidence:
                    review_gaps.append(_unknown_fact_gap(definition, fact))
                continue
            scored_inputs.append((score, fact.fact_id, sorted(fact.evidence)))

        for signal_id in sorted(dimension.signal_ids):
            signal = signals_by_id.get(signal_id)
            if (
                signal is None
                or signal.status is not SignalStatus.AVAILABLE
                or not signal.evidence
            ):
                dimension_needs_review = True
                review_gaps.append(_unknown_signal_gap(signal_id, signal))
                continue
            scored_inputs.append(
                (
                    _directed_score(signal.probability, dimension.scoring_direction),
                    signal.question_id,
                    sorted(signal.evidence),
                )
            )

        needs_review = needs_review or dimension_needs_review
        if scored_inputs and dimension.weight > 0:
            preference_contributions.append(
                DecisionContribution(
                    subject_id=dimension.id,
                    score=sum(item[0] for item in scored_inputs) / len(scored_inputs),
                    weight=dimension.weight,
                    input_ids=sorted(item[1] for item in scored_inputs),
                    job_evidence=sorted(
                        evidence
                        for item in scored_inputs
                        for evidence in item[2]
                    ),
                )
            )

    qualification_contributions: list[DecisionContribution] = []
    qualification_gaps: list[Gap] = []
    hard_blockers: list[Gap] = []

    for definition in sorted(rubric.facts, key=lambda item: item.id):
        if not definition.requires_evidence:
            continue
        fact = facts_by_id.get(definition.id)
        if fact is None or fact.is_unknown or not fact.evidence:
            continue
        if not _is_required(fact.value, definition):
            continue

        matching_claims = active_claims.get(definition.id, [])
        if not matching_claims:
            qualification_gaps.append(
                Gap(
                    kind=GapKind.EVIDENCE_GAP,
                    subject_id=definition.id,
                    message=f"Candidate evidence is missing for {definition.name}.",
                    input_ids=[fact.fact_id],
                    job_evidence=sorted(fact.evidence),
                )
            )
            needs_review = True
            continue

        selected_claim = _select_claim(matching_claims)
        candidate_evidence = _claim_evidence(selected_claim)
        lacks_capability = _explicitly_false(selected_claim.value)
        dimensions = [
            dimension
            for dimension in rubric.dimensions
            if definition.id in dimension.fact_ids
        ]
        weight = sum(dimension.weight for dimension in dimensions) or 1.0
        contribution = DecisionContribution(
            subject_id=definition.id,
            score=0.0 if lacks_capability else 1.0,
            weight=weight,
            input_ids=sorted([fact.fact_id, selected_claim.id]),
            job_evidence=sorted(fact.evidence),
            candidate_evidence=candidate_evidence,
        )
        qualification_contributions.append(contribution)

        if not lacks_capability:
            continue
        is_hard_blocker = any(dimension.hard_constraint for dimension in dimensions)
        gap = Gap(
            kind=(GapKind.HARD_BLOCKER if is_hard_blocker else GapKind.CAPABILITY_GAP),
            subject_id=definition.id,
            message=f"Confirmed candidate evidence conflicts with {definition.name}.",
            input_ids=contribution.input_ids,
            job_evidence=contribution.job_evidence,
            candidate_evidence=contribution.candidate_evidence,
        )
        (hard_blockers if is_hard_blocker else qualification_gaps).append(gap)

    preference_score = _weighted_score(preference_contributions)
    qualification_score = _weighted_score(qualification_contributions)

    if hard_blockers:
        gaps = _sorted_gaps(hard_blockers)
        recommendation = RecommendationLevel.SKIP
    else:
        gaps = _sorted_gaps([*review_gaps, *qualification_gaps])
        if any(gap.kind is GapKind.EVIDENCE_GAP for gap in gaps):
            recommendation = RecommendationLevel.APPLY_AFTER_REVIEW
            needs_review = True
        else:
            recommendation = _score_recommendation(
                preference_score, qualification_score, rubric
            )
            needs_review = (
                needs_review
                or recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
            )
            if recommendation is RecommendationLevel.SKIP:
                evidence = sorted(
                    item
                    for contribution in preference_contributions
                    for item in contribution.job_evidence
                )
                gaps.append(
                    Gap(
                        kind=GapKind.PREFERENCE_MISMATCH,
                        subject_id="preference",
                        message=(
                            "The evidenced role preference score is below the hold "
                            "threshold."
                        ),
                        input_ids=sorted(
                            item
                            for contribution in preference_contributions
                            for item in contribution.input_ids
                        ),
                        job_evidence=evidence,
                    )
                )
                gaps = _sorted_gaps(gaps)

    return DecisionResult(
        preference_score=preference_score,
        qualification_score=qualification_score,
        recommendation=recommendation,
        gaps=gaps,
        preference_contributions=sorted(
            preference_contributions, key=lambda item: item.subject_id
        ),
        qualification_contributions=sorted(
            qualification_contributions, key=lambda item: item.subject_id
        ),
        rubric_version=rubric.version,
        needs_review=needs_review,
    )


def _unique_by_id(items: Iterable[Any], field: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, field)
        if item_id in result:
            raise ValueError(f"duplicate {label} id {item_id!r}")
        result[item_id] = item
    return result


def _active_claims(claims: list[CandidateClaim]) -> dict[str, list[CandidateClaim]]:
    result: dict[str, list[CandidateClaim]] = {}
    for claim in sorted(claims, key=lambda item: item.id):
        if claim.confirmed and claim.source in _ACTIVE_SOURCES:
            result.setdefault(claim.claim_type, []).append(claim)
    return result


def _select_claim(claims: list[CandidateClaim]) -> CandidateClaim:
    return sorted(
        claims,
        key=lambda claim: (
            not _explicitly_false(claim.value),
            claim.source.value,
            claim.id,
        ),
    )[0]


def _fact_score(
    value: Any, definition: FactDefinition, dimension: ValueDimension
) -> float | None:
    if definition.data_type is FactDataType.BOOLEAN:
        if type(value) is not bool:
            raise ValueError(f"fact {definition.id} must be boolean")
        raw_score = float(value)
    elif definition.data_type is FactDataType.NUMBER:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"fact {definition.id} must be a finite number")
        if not 0 <= value <= 1:
            raise ValueError(f"fact {definition.id} number must be between 0 and 1")
        raw_score = float(value)
    elif definition.data_type in {FactDataType.TEXT, FactDataType.ENUM}:
        if not isinstance(value, str):
            raise ValueError(f"fact {definition.id} must be text")
        raw_score = _keyword_score(value, dimension)
    elif definition.data_type is FactDataType.LIST:
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"fact {definition.id} must be a list of text values")
        raw_score = _keyword_score(" ".join(value), dimension)
    else:
        raise ValueError(f"unsupported fact data type {definition.data_type}")
    if raw_score is None:
        return None
    return _directed_score(raw_score, dimension.scoring_direction)


def _keyword_score(value: str, dimension: ValueDimension) -> float | None:
    normalized = value.casefold()
    positive = any(
        keyword.casefold() in normalized for keyword in dimension.positive_keywords
    )
    negative = any(
        keyword.casefold() in normalized for keyword in dimension.negative_keywords
    )
    if positive == negative:
        return None
    return 1.0 if positive else 0.0


def _directed_score(value: float | None, direction: ScoringDirection) -> float:
    if value is None or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score input must be a finite number between 0 and 1")
    if direction is ScoringDirection.HIGHER_IS_BETTER:
        return float(value)
    return 1 - float(value)


def _is_required(value: Any, definition: FactDefinition) -> bool:
    if definition.data_type is FactDataType.BOOLEAN:
        if type(value) is not bool:
            raise ValueError(f"fact {definition.id} must be boolean")
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in _FALSE_TEXT
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"fact {definition.id} must be finite")
        return value != 0
    if isinstance(value, list):
        return bool(value)
    raise ValueError(f"fact {definition.id} has unsupported requirement value")


def _explicitly_false(value: Any) -> bool:
    if type(value) is bool:
        return not value
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("candidate numeric claim must be finite")
        return value == 0
    if isinstance(value, str):
        return value.strip().casefold() in _FALSE_TEXT
    if isinstance(value, list):
        return not value
    if value is None:
        return True
    raise ValueError("candidate claim has unsupported semantic type")


def _claim_evidence(claim: CandidateClaim) -> list[str]:
    evidence = claim.source_quote or claim.provenance
    return [evidence or f"{claim.source.value}:{claim.id}"]


def _unknown_fact_gap(
    definition: FactDefinition, fact: ExtractedFact | None
) -> Gap:
    evidence = (
        sorted(fact.evidence)
        if fact and fact.evidence
        else [f"fact:{definition.id}:unknown"]
    )
    return Gap(
        kind=GapKind.EVIDENCE_GAP,
        subject_id=definition.id,
        message=f"Required job evidence is unresolved for {definition.name}.",
        input_ids=[definition.id],
        job_evidence=evidence,
    )


def _unknown_signal_gap(
    signal_id: str, signal: DecisionSignal | None
) -> Gap:
    status = signal.status.value if signal else "missing"
    evidence = (
        sorted(signal.evidence)
        if signal and signal.evidence
        else [f"signal:{signal_id}:{status}"]
    )
    return Gap(
        kind=GapKind.EVIDENCE_GAP,
        subject_id=signal_id,
        message=f"Decision signal {signal_id} requires review.",
        input_ids=[signal_id],
        job_evidence=evidence,
    )


def _weighted_score(contributions: list[DecisionContribution]) -> float | None:
    total_weight = sum(item.weight for item in contributions)
    if total_weight == 0:
        return None
    return sum(item.score * item.weight for item in contributions) / total_weight


def _score_recommendation(
    preference_score: float | None,
    qualification_score: float | None,
    rubric: Rubric,
) -> RecommendationLevel:
    if preference_score is None or qualification_score is None:
        return RecommendationLevel.APPLY_AFTER_REVIEW
    score = min(preference_score, qualification_score)
    for level in (
        RecommendationLevel.PRIORITY_APPLY,
        RecommendationLevel.APPLY_AFTER_REVIEW,
        RecommendationLevel.HOLD,
        RecommendationLevel.SKIP,
    ):
        if score >= rubric.recommendation_thresholds[level]:
            return level
    raise ValueError("recommendation thresholds do not cover the score range")


def _sorted_gaps(gaps: list[Gap]) -> list[Gap]:
    unique = {
        (gap.kind.value, gap.subject_id, tuple(gap.input_ids)): gap for gap in gaps
    }
    return sorted(
        unique.values(),
        key=lambda gap: (gap.kind.value, gap.subject_id, tuple(gap.input_ids)),
    )
