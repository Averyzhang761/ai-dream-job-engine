#!/usr/bin/env python3
"""Run a local, human-first evaluation without provider setup at import time."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from pydantic import ValidationError

from ai_dream_job_engine.models import (
    EvaluationCase,
    EvaluationReport,
    GapKind,
    RecommendationLevel,
)
from ai_dream_job_engine.service import JobAnalysisService


class EvaluationInputError(ValueError):
    """A safe input error that never includes case contents."""


class AnalysisService(Protocol):
    def analyze_job(self, job_description: str) -> object: ...


def run_evaluation(
    cases: list[EvaluationCase], service: JobAnalysisService
) -> EvaluationReport:
    """Compare system outputs with prior human judgments, never ground truth."""
    if not cases:
        raise ValueError("at least one evaluation case is required")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    for case in cases:
        if (
            case.system_recommendation is not None
            or case.system_gap_labels is not None
        ):
            raise ValueError("pre-populated system result is not allowed")

    source_snapshots = [case.model_dump(mode="json") for case in cases]
    started = time.monotonic()
    evaluated: list[EvaluationCase] = []
    first_results: dict[str, tuple[RecommendationLevel, list[GapKind]]] = {}
    failure_ids: list[str] = []
    unsupported_claim_count = 0

    for case in cases:
        try:
            record = service.analyze_job(case.job_description)
            recommendation, gaps = _result_values(record)
        except Exception:
            failure_ids.append(case.id)
            continue
        first_results[case.id] = (recommendation, gaps)
        unsupported_claim_count += _unsupported_claims(record)
        evaluated.append(
            case.model_copy(
                update={
                    "system_recommendation": recommendation,
                    "system_gap_labels": gaps,
                },
                deep=True,
            )
        )

    if [case.model_dump(mode="json") for case in cases] != source_snapshots:
        raise ValueError("source evaluation case changed during analysis")

    stable = 0
    rerun_count = 0
    for case in cases:
        first = first_results.get(case.id)
        if first is None:
            continue
        rerun_count += 1
        try:
            second = _result_values(service.analyze_job(case.job_description))
        except Exception:
            continue
        if second == first:
            stable += 1

    completed = len(evaluated)
    recommendation_agreement = (
        sum(
            case.human_recommendation is case.system_recommendation
            for case in evaluated
        )
        / completed
        if completed
        else 0.0
    )
    predicted_blockers = sum(
        GapKind.HARD_BLOCKER in (case.system_gap_labels or [])
        for case in evaluated
    )
    true_blockers = sum(
        GapKind.HARD_BLOCKER in case.human_gap_labels
        and GapKind.HARD_BLOCKER in (case.system_gap_labels or [])
        for case in evaluated
    )

    return EvaluationReport(
        cases=evaluated,
        recommendation_agreement=recommendation_agreement,
        blocker_precision=(
            true_blockers / predicted_blockers if predicted_blockers else None
        ),
        gap_type_confusion_counts=_gap_confusion(evaluated),
        unsupported_claim_count=unsupported_claim_count,
        rerun_stability=stable / rerun_count if rerun_count else 0.0,
        top_five_usefulness=_top_five_label_agreement(evaluated),
        review_time_seconds=time.monotonic() - started,
        failure_case_ids=failure_ids,
    )


def _result_values(record: object) -> tuple[RecommendationLevel, list[GapKind]]:
    result = getattr(record, "result", None)
    if result is None:
        raise ValueError("analysis did not return a result")
    recommendation = RecommendationLevel(result.recommendation)
    gaps = [GapKind(gap.kind) for gap in result.gaps]
    return recommendation, gaps


def _unsupported_claims(record: object) -> int:
    result = getattr(record, "result", None)
    if result is None:
        return 0
    return sum(
        not gap.job_evidence
        and not gap.candidate_evidence
        and not gap.missing_evidence
        for gap in result.gaps
    )


def _gap_confusion(
    cases: list[EvaluationCase],
) -> dict[GapKind, dict[GapKind, int]]:
    counts: dict[GapKind, dict[GapKind, int]] = {}
    for case in cases:
        human_remaining = list(dict.fromkeys(case.human_gap_labels))
        system_remaining = list(dict.fromkeys(case.system_gap_labels or []))
        for label in list(human_remaining):
            if label in system_remaining:
                counts.setdefault(label, {}).setdefault(label, 0)
                counts[label][label] += 1
                human_remaining.remove(label)
                system_remaining.remove(label)
        for human, system in zip(human_remaining, system_remaining, strict=False):
            counts.setdefault(human, {}).setdefault(system, 0)
            counts[human][system] += 1
    return counts


def _top_five_label_agreement(cases: list[EvaluationCase]) -> float:
    if not cases:
        return 0.0
    scores = []
    for case in cases:
        predicted = list(dict.fromkeys(case.system_gap_labels or []))[:5]
        expected = set(case.human_gap_labels)
        if not predicted:
            scores.append(1.0 if not expected else 0.0)
        else:
            scores.append(sum(label in expected for label in predicted) / len(predicted))
    return sum(scores) / len(scores)


def _load_cases(path: Path) -> list[EvaluationCase]:
    cases = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationInputError("could not read UTF-8 evaluation input") from None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationInputError(f"line {line_number}: blank line is invalid")
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise EvaluationInputError(
                f"line {line_number}: invalid evaluation case"
            ) from None
    return cases


class OfflineSyntheticService:
    """Deterministic fake for the committed invented fixture only."""

    def analyze_job(self, job_description: str) -> object:
        lowered = job_description.lower()
        if "relocation required" in lowered:
            recommendation = RecommendationLevel.SKIP
            gap_kinds = [GapKind.HARD_BLOCKER]
        elif "evidence must be confirmed" in lowered:
            recommendation = RecommendationLevel.APPLY_AFTER_REVIEW
            gap_kinds = [GapKind.EVIDENCE_GAP]
        else:
            recommendation = RecommendationLevel.PRIORITY_APPLY
            gap_kinds = []
        gaps = [
            SimpleNamespace(
                kind=kind,
                job_evidence=["offline-synthetic-evidence"],
                candidate_evidence=[],
                missing_evidence=False,
            )
            for kind in gap_kinds
        ]
        return SimpleNamespace(
            result=SimpleNamespace(recommendation=recommendation, gaps=gaps)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local evaluation JSONL file.")
    parser.add_argument("cases", type=Path, help="UTF-8 JSONL input path")
    parser.add_argument("report", type=Path, help="JSON report output path")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use the deterministic fake for the committed synthetic fixture",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    service: AnalysisService | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cases.resolve() == args.report.resolve():
            raise EvaluationInputError("report path must differ from input path")
        cases = _load_cases(args.cases)
        active_service = service or (OfflineSyntheticService() if args.offline else None)
        if active_service is None:
            raise EvaluationInputError(
                "no analysis service configured; use dependency injection or --offline"
            )
        report = run_evaluation(cases, active_service)  # type: ignore[arg-type]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except (EvaluationInputError, ValueError):
        print("Evaluation failed: " + _safe_error_message(sys.exc_info()[1]), file=sys.stderr)
        return 2
    print(f"Evaluation report written: {args.report}")
    return 0


def _safe_error_message(error: BaseException | None) -> str:
    if isinstance(error, EvaluationInputError):
        return str(error)
    return "evaluation could not be completed"


if __name__ == "__main__":
    raise SystemExit(main())
