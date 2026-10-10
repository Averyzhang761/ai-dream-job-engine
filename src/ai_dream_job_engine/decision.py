"""Deterministic, provider-free job decision rules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from ai_dream_job_engine.models import (
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
_UNKNOWN_VALUES = {"unknown", "unavailable", "invalid"}
_UNRESOLVED = object()


@dataclass(frozen=True)
class _Value:
    id: str
    value: Any
    evidence: tuple[str, ...]
    missing_evidence: bool

    @property
    def ready(self) -> bool:
        return self.value is not _UNRESOLVED


@dataclass(frozen=True)
class _Inputs:
    facts: dict[str, _Value]
    signals: dict[str, _Value]
    claims: dict[str, tuple[_Value, ...]]


def decide_job(
    facts: list[ExtractedFact],
    signals: list[DecisionSignal],
    profile: CandidateProfile,
    rubric: Rubric,
) -> DecisionResult:
    """Return a deterministic recommendation from validated local inputs."""
    inputs = _normalize(facts, signals, profile, rubric)
    preference, preference_gaps = _preference(inputs, rubric)
    qualification, qualification_gaps, blockers = _qualification(inputs, rubric)
    preference_score = _weighted(preference)
    qualification_score = _weighted(qualification)

    if blockers:
        recommendation = RecommendationLevel.SKIP
        gaps = [*preference_gaps, *qualification_gaps, *blockers]
    else:
        gaps = [*preference_gaps, *qualification_gaps]
        if any(gap.kind is GapKind.EVIDENCE_GAP for gap in gaps):
            recommendation = RecommendationLevel.APPLY_AFTER_REVIEW
        else:
            recommendation = _recommend(preference_score, qualification_score, rubric)
            if (
                recommendation is RecommendationLevel.SKIP
                and preference_score is not None
                and qualification_score is not None
                and preference_score < qualification_score
            ):
                gaps.append(_preference_gap(preference))

    gaps = _sort_gaps(gaps)
    return DecisionResult(
        preference_score=preference_score,
        qualification_score=qualification_score,
        recommendation=recommendation,
        gaps=gaps,
        preference_contributions=preference,
        qualification_contributions=qualification,
        rubric_version=rubric.version,
        needs_review=(
            recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
            or any(gap.kind is GapKind.EVIDENCE_GAP for gap in gaps)
        ),
    )


def _normalize(
    facts: list[ExtractedFact],
    signals: list[DecisionSignal],
    profile: CandidateProfile,
    rubric: Rubric,
) -> _Inputs:
    _validate_rubric(rubric)
    fact_inputs = _unique(facts, "fact_id", "fact")
    signal_inputs = _unique(signals, "question_id", "signal")
    _unique(profile.claims, "id", "candidate claim")
    definitions = {item.id: item for item in rubric.facts}
    questions = {item.id: item for item in rubric.questions}

    normalized_facts = {
        fact_id: _normalize_fact(definition, fact_inputs.get(fact_id))
        for fact_id, definition in definitions.items()
    }
    normalized_signals = {
        question_id: _normalize_signal(
            question_id, question.version, signal_inputs.get(question_id)
        )
        for question_id, question in questions.items()
    }
    claims: dict[str, list[_Value]] = {}
    for claim in sorted(profile.claims, key=lambda item: item.id):
        if not claim.confirmed or claim.source not in _ACTIVE_SOURCES:
            continue
        if claim.claim_type not in definitions:
            raise ValueError(f"candidate claim {claim.id} has unknown claim_type")
        evidence = (claim.source_quote or claim.provenance,)
        if evidence == (None,):
            evidence = ()
        value = _typed(definitions[claim.claim_type], claim.value, "candidate claim")
        claims.setdefault(claim.claim_type, []).append(
            _normalized_value(claim.id, value, evidence)
        )
    return _Inputs(
        normalized_facts,
        normalized_signals,
        {key: tuple(value) for key, value in sorted(claims.items())},
    )


def _normalize_fact(definition: FactDefinition, fact: ExtractedFact | None) -> _Value:
    if fact is None:
        return _Value(definition.id, _UNRESOLVED, (), True)
    evidence = tuple(sorted(fact.evidence))
    if fact.is_unknown:
        return _Value(fact.fact_id, _UNRESOLVED, evidence, not evidence)
    value = _typed(definition, fact.value, "fact")
    return _normalized_value(fact.fact_id, value, evidence)


def _normalize_signal(question_id: str, version: str, signal: DecisionSignal | None) -> _Value:
    if signal is None:
        return _Value(question_id, _UNRESOLVED, (), True)
    evidence = tuple(sorted(signal.evidence))
    valid = (
        signal.status is SignalStatus.AVAILABLE
        and signal.question_version == version
        and bool(evidence)
    )
    value = signal.probability if valid else _UNRESOLVED
    return _Value(signal.question_id, value, evidence, not evidence)


def _normalized_value(input_id: str, value: Any, evidence: tuple[str, ...]) -> _Value:
    if _unknown(value) or not evidence:
        value = _UNRESOLVED
    return _Value(input_id, value, evidence, not evidence)


def _typed(definition: FactDefinition, value: Any, source: str) -> Any:
    label = f"{source} {definition.id}"
    if definition.data_type is FactDataType.BOOLEAN:
        if type(value) is not bool:
            raise ValueError(f"{label} must be boolean")
        return value
    if definition.data_type is FactDataType.NUMBER:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        if not 0 <= value <= 1:
            raise ValueError(f"{label} must be between 0 and 1")
        return float(value)
    if definition.data_type in {FactDataType.TEXT, FactDataType.ENUM}:
        if not isinstance(value, str):
            raise ValueError(f"{label} must be text")
        if definition.data_type is FactDataType.ENUM and value not in definition.enum_values:
            raise ValueError(f"{label} must be in enum_values")
        return value.strip()
    if definition.data_type is FactDataType.LIST:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label} must be a list of text values")
        return tuple(item.strip() for item in value)
    raise ValueError(f"{label} has unsupported data type")


def _unknown(value: Any) -> bool:
    if isinstance(value, str):
        return value.casefold() in _UNKNOWN_VALUES
    return (
        isinstance(value, tuple)
        and any(item.casefold() in _UNKNOWN_VALUES for item in value)
    )


def _preference(inputs: _Inputs, rubric: Rubric) -> tuple[list[DecisionContribution], list[Gap]]:
    contributions: list[DecisionContribution] = []
    gaps: list[Gap] = []
    for dimension in sorted(rubric.dimensions, key=lambda item: item.id):
        if dimension.weight == 0:
            continue
        values = [inputs.facts[item] for item in sorted(dimension.fact_ids)]
        values += [inputs.signals[item] for item in sorted(dimension.signal_ids)]
        unresolved = [value for value in values if not value.ready]
        if unresolved:
            gaps += [
                _missing_gap(value.id, [value.id], value.evidence, value.missing_evidence)
                for value in unresolved
            ]
            continue
        scores = [_dimension_score(value.value, dimension) for value in values]
        if not scores or any(score is None for score in scores):
            gaps.append(_missing_gap(dimension.id, [value.id for value in values], _evidence(values)))
            continue
        score = sum(score for score in scores if score is not None) / len(scores)
        if dimension.threshold is not None:
            score = float(score >= dimension.threshold)  # Equality passes.
        contributions.append(
            DecisionContribution(
                subject_id=dimension.id,
                score=score,
                weight=dimension.weight,
                input_ids=sorted(value.id for value in values),
                job_evidence=_evidence(values),
            )
        )
    return contributions, _sort_gaps(gaps)


def _dimension_score(value: Any, dimension: ValueDimension) -> float | None:
    if type(value) in (bool, int, float):
        score = float(value)
    elif isinstance(value, (str, tuple)):
        if not dimension.positive_keywords and not dimension.negative_keywords:
            score = 1.0 if isinstance(value, tuple) else None
        else:
            text = value if isinstance(value, str) else " ".join(value)
            positive = any(word.casefold() in text.casefold() for word in dimension.positive_keywords)
            negative = any(word.casefold() in text.casefold() for word in dimension.negative_keywords)
            score = None if positive == negative else float(positive)
    else:
        return None
    if dimension.scoring_direction is ScoringDirection.LOWER_IS_BETTER:
        return 1 - score
    return score


def _qualification(
    inputs: _Inputs, rubric: Rubric
) -> tuple[list[DecisionContribution], list[Gap], list[Gap]]:
    contributions: list[DecisionContribution] = []
    gaps: list[Gap] = []
    blockers: list[Gap] = []
    for definition in sorted(rubric.facts, key=lambda item: item.id):
        if definition.candidate_comparison == "none":
            continue
        fact = inputs.facts[definition.id]
        if not fact.ready:
            gaps.append(_missing_gap(definition.id, [definition.id], fact.evidence, fact.missing_evidence))
            continue
        if (
            definition.candidate_comparison == "allowed_alternatives"
            and not fact.value
        ):
            contributions.append(
                DecisionContribution(
                    subject_id=definition.id,
                    score=1,
                    weight=definition.qualification_weight,
                    input_ids=[fact.id],
                    job_evidence=list(fact.evidence),
                )
            )
            continue
        claims = inputs.claims.get(definition.id, ())
        conflicts = [
            item
            for item in claims
            if item.ready
            and not _compatible(
                fact.value, item.value, definition.candidate_comparison
            )
        ]
        matches = [
            item
            for item in claims
            if item.ready
            and _compatible(fact.value, item.value, definition.candidate_comparison)
        ]
        unresolved = [item for item in claims if not item.ready]
        weight = definition.qualification_weight
        if conflicts:
            claim = conflicts[0]
            contribution = _qualification_contribution(definition.id, fact, claim, 0, weight)
            contributions.append(contribution)
            hard = any(
                item.hard_constraint and definition.id in item.fact_ids
                for item in rubric.dimensions
            )
            gap = Gap(
                kind=GapKind.HARD_BLOCKER if hard else GapKind.CAPABILITY_GAP,
                subject_id=definition.id,
                message=f"Confirmed candidate value conflicts with {definition.name}.",
                input_ids=contribution.input_ids,
                job_evidence=contribution.job_evidence,
                candidate_evidence=contribution.candidate_evidence,
            )
            (blockers if hard else gaps).append(gap)
        elif matches:
            contributions.append(_qualification_contribution(definition.id, fact, matches[0], 1, weight))
        else:
            claim = unresolved[0] if unresolved else None
            gaps.append(
                _missing_gap(
                    definition.id,
                    [definition.id, *([claim.id] if claim else [])],
                    fact.evidence,
                    claim is None or claim.missing_evidence,
                    claim.evidence if claim else (),
                )
            )
    return contributions, _sort_gaps(gaps), _sort_gaps(blockers)


def _compatible(required: Any, candidate: Any, policy: str) -> bool:
    if policy == "allowed_alternatives":
        required_set = {item.casefold() for item in required}
        candidate_set = {item.casefold() for item in candidate}
        return not required_set or bool(required_set & candidate_set)
    if isinstance(required, tuple):
        return tuple(item.casefold() for item in required) == tuple(
            item.casefold() for item in candidate
        )
    if isinstance(required, str):
        return required.casefold() == candidate.casefold()
    return required == candidate


def _qualification_contribution(
    subject: str, fact: _Value, claim: _Value, score: float, weight: float
) -> DecisionContribution:
    return DecisionContribution(
        subject_id=subject,
        score=score,
        weight=weight,
        input_ids=sorted([fact.id, claim.id]),
        job_evidence=list(fact.evidence),
        candidate_evidence=list(claim.evidence),
    )


def _missing_gap(
    subject: str,
    input_ids: list[str],
    job_evidence: Iterable[str],
    missing: bool = False,
    candidate_evidence: Iterable[str] = (),
) -> Gap:
    return Gap(
        kind=GapKind.EVIDENCE_GAP,
        subject_id=subject,
        message=f"Evidence for {subject} requires review.",
        input_ids=sorted(input_ids),
        job_evidence=sorted(job_evidence),
        candidate_evidence=sorted(candidate_evidence),
        missing_evidence=missing,
    )


def _preference_gap(contributions: list[DecisionContribution]) -> Gap:
    return Gap(
        kind=GapKind.PREFERENCE_MISMATCH,
        subject_id="preference",
        message="The evidenced preference score is below the hold threshold.",
        input_ids=sorted({value for item in contributions for value in item.input_ids}),
        job_evidence=sorted({value for item in contributions for value in item.job_evidence}),
    )


def _evidence(values: Iterable[_Value]) -> list[str]:
    return sorted({item for value in values for item in value.evidence})


def _weighted(contributions: list[DecisionContribution]) -> float | None:
    weight = sum(item.weight for item in contributions)
    if not weight:
        return None
    return sum(item.score * item.weight for item in contributions) / weight


def _recommend(
    preference: float | None, qualification: float | None, rubric: Rubric
) -> RecommendationLevel:
    if preference is None or qualification is None:
        return RecommendationLevel.APPLY_AFTER_REVIEW
    score = min(preference, qualification)
    for level in RecommendationLevel:
        if score >= rubric.recommendation_thresholds[level]:
            return level
    return RecommendationLevel.SKIP


def _unique(items: Iterable[Any], field: str, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, field)
        if item_id in result:
            raise ValueError(f"duplicate {label} id {item_id!r}")
        result[item_id] = item
    return result


def _sort_gaps(gaps: list[Gap]) -> list[Gap]:
    unique = {
        (gap.kind.value, gap.subject_id, tuple(gap.input_ids)): gap for gap in gaps
    }
    return sorted(
        unique.values(),
        key=lambda gap: (gap.kind.value, gap.subject_id, gap.input_ids),
    )
