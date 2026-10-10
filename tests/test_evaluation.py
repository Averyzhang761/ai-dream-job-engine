from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_dream_job_engine.models import (
    EvaluationCase,
    GapKind,
    RecommendationLevel,
)
from scripts.run_evaluation import main, run_evaluation


ROOT = Path(__file__).parents[1]
SYNTHETIC_FIXTURE = ROOT / "eval" / "synthetic_cases.jsonl"


def _case(**updates: object) -> EvaluationCase:
    values = {
        "id": "invented-case-1",
        "job_description": "Invented role at Northstar Atelier building local tools.",
        "human_recommendation": RecommendationLevel.APPLY_AFTER_REVIEW,
        "human_gap_labels": [GapKind.EVIDENCE_GAP],
    }
    values.update(updates)
    return EvaluationCase(**values)


def _record(
    recommendation: RecommendationLevel,
    gaps: list[GapKind],
    *,
    unsupported_claims: int = 0,
) -> SimpleNamespace:
    result_gaps = [
        SimpleNamespace(
            kind=kind,
            job_evidence=[] if index < unsupported_claims else ["jd-ref"],
            candidate_evidence=[],
            missing_evidence=False,
        )
        for index, kind in enumerate(gaps)
    ]
    return SimpleNamespace(
        result=SimpleNamespace(recommendation=recommendation, gaps=result_gaps),
        facts=[],
        signals=[],
    )


class SequenceService:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = iter(records)
        self.seen: list[str] = []

    def analyze_job(self, job_description: str) -> SimpleNamespace:
        self.seen.append(job_description)
        return next(self.records)


def test_case_requires_human_judgment_before_system_result() -> None:
    service = SequenceService(
        [
            _record(RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]),
            _record(RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]),
        ]
    )
    contaminated = _case(system_recommendation=RecommendationLevel.PRIORITY_APPLY)

    with pytest.raises(ValueError, match="pre-populated system result"):
        run_evaluation([contaminated], service)

    clean = _case()
    before = clean.model_dump(mode="json")
    report = run_evaluation([clean], service)

    assert clean.model_dump(mode="json") == before
    assert report.cases[0].system_recommendation is RecommendationLevel.APPLY_AFTER_REVIEW


def test_report_separates_agreement_from_accuracy() -> None:
    cases = [
        _case(id="agree"),
        _case(id="disagree", human_recommendation=RecommendationLevel.SKIP),
    ]
    records = [
        _record(RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]),
        _record(RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]),
    ] * 2

    report = run_evaluation(cases, SequenceService(records))
    payload = report.model_dump(mode="json")

    assert report.recommendation_agreement == 0.5
    assert "accuracy" not in json.dumps(payload).lower()


def test_report_counts_gap_classification() -> None:
    cases = [
        _case(
            id="blocker-match",
            human_gap_labels=[GapKind.HARD_BLOCKER, GapKind.EVIDENCE_GAP],
        ),
        _case(id="false-blocker", human_gap_labels=[GapKind.CAPABILITY_GAP]),
    ]
    first_run = [
        _record(
            RecommendationLevel.APPLY_AFTER_REVIEW,
            [GapKind.HARD_BLOCKER, GapKind.EVIDENCE_GAP],
        ),
        _record(RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.HARD_BLOCKER]),
    ]

    report = run_evaluation(cases, SequenceService(first_run + first_run))

    assert report.blocker_precision == 0.5
    assert report.gap_type_confusion_counts[GapKind.HARD_BLOCKER][GapKind.HARD_BLOCKER] == 1
    assert report.gap_type_confusion_counts[GapKind.CAPABILITY_GAP][GapKind.HARD_BLOCKER] == 1

    no_predicted_blockers = run_evaluation(
        [_case(human_gap_labels=[GapKind.HARD_BLOCKER])],
        SequenceService(
            [
                _record(RecommendationLevel.APPLY_AFTER_REVIEW, []),
                _record(RecommendationLevel.APPLY_AFTER_REVIEW, []),
            ]
        ),
    )
    assert no_predicted_blockers.blocker_precision is None


def test_real_eval_paths_are_gitignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "eval/private/" in gitignore
    assert "eval/reports/" in gitignore
    assert not any(
        SYNTHETIC_FIXTURE.match(pattern)
        for pattern in ("eval/", "eval/*.jsonl", "eval/synthetic_cases.jsonl")
        if pattern in gitignore
    )


def test_synthetic_fixture_contains_no_personal_data() -> None:
    text = SYNTHETIC_FIXTURE.read_text(encoding="utf-8")
    cases = [EvaluationCase.model_validate_json(line) for line in text.splitlines()]

    assert cases
    assert all("invented" in case.notes.lower() for case in cases if case.notes)
    assert "@" not in text
    assert not any(token in text.lower() for token in ("resume", "api_key", "secret", "linkedin"))


def test_cli_writes_only_report_and_redacts_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "cases.jsonl"
    report_path = tmp_path / "report.json"
    raw_jd = "PRIVATE-JD-MARKER"
    source.write_text(json.dumps({"id": "bad", "job_description": raw_jd}) + "\n")

    exit_code = main([str(source), str(report_path), "--offline"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert "line 1" in captured.err
    assert raw_jd not in captured.err + captured.out
    assert source.read_text().endswith("\n")
    assert not report_path.exists()
