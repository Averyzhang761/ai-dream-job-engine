#!/usr/bin/env python3
"""Run a local, human-first evaluation without provider setup at import time."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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


_RECOMMENDATION_RANK = {
    RecommendationLevel.PRIORITY_APPLY: 0,
    RecommendationLevel.APPLY_AFTER_REVIEW: 1,
    RecommendationLevel.HOLD: 2,
    RecommendationLevel.SKIP: 3,
}


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
            or case.system_preference_score is not None
            or case.system_qualification_score is not None
        ):
            raise ValueError("pre-populated system result is not allowed")

    source_snapshots = [case.model_dump(mode="json") for case in cases]
    started = time.monotonic()
    reported: list[EvaluationCase] = []
    completed_cases: list[EvaluationCase] = []
    first_results: dict[
        str, tuple[RecommendationLevel, list[GapKind], float | None, float | None]
    ] = {}
    failure_ids: list[str] = []
    unsupported_counts = {
        "fact": 0,
        "signal": 0,
        "gap": 0,
        "contribution": 0,
    }

    for case in cases:
        try:
            record = service.analyze_job(case.job_description)
            recommendation, gaps, preference_score, qualification_score = (
                _result_values(record)
            )
        except Exception:
            failure_ids.append(case.id)
            reported.append(case.model_copy(deep=True))
            continue
        first_results[case.id] = (
            recommendation,
            gaps,
            preference_score,
            qualification_score,
        )
        for category, count in _unsupported_sources(record).items():
            unsupported_counts[category] += count
        evaluated_case = case.model_copy(
            update={
                "system_recommendation": recommendation,
                "system_gap_labels": gaps,
                "system_preference_score": preference_score,
                "system_qualification_score": qualification_score,
            },
            deep=True,
        )
        completed_cases.append(evaluated_case)
        reported.append(evaluated_case)

    if [case.model_dump(mode="json") for case in cases] != source_snapshots:
        raise ValueError("source evaluation case changed during analysis")

    stable = 0
    rerun_attempted_count = 0
    rerun_completed_count = 0
    rerun_failure_ids: list[str] = []
    for case in cases:
        first = first_results.get(case.id)
        if first is None:
            continue
        rerun_attempted_count += 1
        try:
            second = _result_values(service.analyze_job(case.job_description))
        except Exception:
            rerun_failure_ids.append(case.id)
            continue
        rerun_completed_count += 1
        if second == first:
            stable += 1

    completed = len(completed_cases)
    recommendation_agreement = (
        sum(
            case.human_recommendation is case.system_recommendation
            for case in completed_cases
        )
        / completed
        if completed
        else None
    )
    predicted_blockers = sum(
        GapKind.HARD_BLOCKER in (case.system_gap_labels or [])
        for case in completed_cases
    )
    true_blockers = sum(
        GapKind.HARD_BLOCKER in case.human_gap_labels
        and GapKind.HARD_BLOCKER in (case.system_gap_labels or [])
        for case in completed_cases
    )
    top_five_cases = _ranked_top_five(completed_cases)
    unsupported_claim_count = sum(unsupported_counts.values())
    rerun_failed_count = len(rerun_failure_ids)

    return EvaluationReport(
        cases=reported,
        attempted_count=len(cases),
        completed_count=completed,
        failed_count=len(failure_ids),
        rerun_attempted_count=rerun_attempted_count,
        rerun_completed_count=rerun_completed_count,
        rerun_failed_count=rerun_failed_count,
        run_complete=not failure_ids and not rerun_failure_ids,
        recommendation_agreement=recommendation_agreement,
        blocker_precision=(
            true_blockers / predicted_blockers if predicted_blockers else None
        ),
        gap_type_confusion_counts=_gap_confusion(completed_cases),
        unsupported_fact_count=unsupported_counts["fact"],
        unsupported_signal_count=unsupported_counts["signal"],
        unsupported_gap_count=unsupported_counts["gap"],
        unsupported_contribution_count=unsupported_counts["contribution"],
        unsupported_claim_count=unsupported_claim_count,
        rerun_stability=(
            stable / rerun_completed_count
            if rerun_completed_count and not rerun_failure_ids
            else None
        ),
        gap_label_agreement=_gap_label_agreement(completed_cases),
        top_five_case_ids=[case.id for case in top_five_cases],
        evaluation_runtime_seconds=time.monotonic() - started,
        top_five_usefulness=_top_five_usefulness(top_five_cases),
        review_time_seconds=_complete_human_total(
            completed_cases, "human_review_time_seconds"
        ),
        failure_case_ids=failure_ids,
        rerun_failure_case_ids=rerun_failure_ids,
    )


def _result_values(
    record: object,
) -> tuple[RecommendationLevel, list[GapKind], float | None, float | None]:
    result = getattr(record, "result", None)
    if result is None:
        raise ValueError("analysis did not return a result")
    recommendation = RecommendationLevel(result.recommendation)
    gaps = [GapKind(gap.kind) for gap in result.gaps]
    return (
        recommendation,
        gaps,
        getattr(result, "preference_score", None),
        getattr(result, "qualification_score", None),
    )


def _unsupported_sources(record: object) -> dict[str, int]:
    result = getattr(record, "result", None)
    if result is None:
        return {"fact": 0, "signal": 0, "gap": 0, "contribution": 0}
    fact_count = sum(
        not getattr(fact, "is_unknown", False) and not getattr(fact, "evidence", [])
        for fact in getattr(record, "facts", [])
    )
    signal_count = sum(
        _is_available(signal)
        and not all(
            getattr(signal, field, None)
            for field in ("provider", "model", "question_id", "question_version")
        )
        for signal in getattr(record, "signals", [])
    )
    gap_count = sum(
        not gap.job_evidence
        and not gap.candidate_evidence
        and not getattr(gap, "missing_evidence", False)
        for gap in result.gaps
    )
    contributions = [
        *getattr(result, "preference_contributions", []),
        *getattr(result, "qualification_contributions", []),
    ]
    contribution_count = sum(
        not contribution.job_evidence
        and not contribution.candidate_evidence
        and not getattr(contribution, "audit_support", [])
        for contribution in contributions
    )
    return {
        "fact": fact_count,
        "signal": signal_count,
        "gap": gap_count,
        "contribution": contribution_count,
    }


def _is_available(signal: object) -> bool:
    status = getattr(signal, "status", None)
    return getattr(status, "value", status) == "available"


def _gap_confusion(
    cases: list[EvaluationCase],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for case in cases:
        human_labels = set(case.human_gap_labels)
        system_labels = set(case.system_gap_labels or [])
        for label in sorted(human_labels & system_labels):
            _increment(counts, label, label)
        for label in sorted(human_labels - system_labels):
            _increment(counts, label, "__missing_system__")
        for label in sorted(system_labels - human_labels):
            _increment(counts, "__missing_human__", label)
    return counts


def _increment(counts: dict[str, dict[str, int]], row: str, column: str) -> None:
    counts.setdefault(row, {}).setdefault(column, 0)
    counts[row][column] += 1


def _gap_label_agreement(cases: list[EvaluationCase]) -> float | None:
    if not cases:
        return None
    scores = []
    for case in cases:
        predicted = set(case.system_gap_labels or [])
        expected = set(case.human_gap_labels)
        union = predicted | expected
        scores.append(len(predicted & expected) / len(union) if union else 1.0)
    return sum(scores) / len(scores)


def _ranked_top_five(cases: list[EvaluationCase]) -> list[EvaluationCase]:
    def rank(case: EvaluationCase) -> tuple[int, bool, float, bool, float, str]:
        preference = case.system_preference_score
        qualification = case.system_qualification_score
        return (
            _RECOMMENDATION_RANK[case.system_recommendation],  # type: ignore[index]
            preference is None,
            -(preference or 0),
            qualification is None,
            -(qualification or 0),
            case.id,
        )

    return sorted(cases, key=rank)[:5]


def _top_five_usefulness(cases: list[EvaluationCase]) -> float | None:
    values = [case.human_top_five_usefulness for case in cases]
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def _complete_human_total(cases: list[EvaluationCase], field: str) -> float | None:
    values = [getattr(case, field) for case in cases]
    if not values or any(value is None for value in values):
        return None
    return sum(values)  # type: ignore[arg-type]


def _load_cases(path: Path) -> list[EvaluationCase]:
    cases = []
    first_lines: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationInputError("could not read UTF-8 evaluation input") from None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvaluationInputError(f"line {line_number}: blank line is invalid")
        try:
            case = EvaluationCase.model_validate_json(line)
        except (ValidationError, ValueError, json.JSONDecodeError):
            raise EvaluationInputError(
                f"line {line_number}: invalid evaluation case"
            ) from None
        if case.id in first_lines:
            raise EvaluationInputError(
                f"duplicate case ID: line {line_number}; "
                f"first line {first_lines[case.id]}"
            )
        first_lines[case.id] = line_number
        cases.append(case)
    return cases


def _paths_share_input(source: Path, destination: Path) -> bool:
    if destination.exists() or destination.is_symlink():
        try:
            return source.samefile(destination)
        except OSError:
            raise EvaluationInputError("report path could not be validated") from None
    return source.resolve() == destination.resolve()


def _write_report_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
        if _paths_share_input(args.cases, args.report):
            raise EvaluationInputError("report path must differ from input path")
        cases = _load_cases(args.cases)
        active_service = service or (OfflineSyntheticService() if args.offline else None)
        if active_service is None:
            raise EvaluationInputError(
                "no analysis service configured; use dependency injection or --offline"
            )
        report = run_evaluation(cases, active_service)  # type: ignore[arg-type]
        _write_report_atomic(
            args.report,
            report.model_dump_json(indent=2) + "\n",
        )
    except (EvaluationInputError, OSError, ValueError):
        print("Evaluation failed: " + _safe_error_message(sys.exc_info()[1]), file=sys.stderr)
        return 2
    if not report.run_complete:
        print(f"Partial evaluation report written: {args.report}", file=sys.stderr)
        return 1
    print(f"Evaluation report written: {args.report}")
    return 0


def _safe_error_message(error: BaseException | None) -> str:
    if isinstance(error, EvaluationInputError):
        return str(error)
    return "evaluation could not be completed"


if __name__ == "__main__":
    raise SystemExit(main())
