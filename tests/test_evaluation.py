from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_dream_job_engine.models import (
    EvaluationCase,
    GapKind,
    RecommendationLevel,
)
import scripts.run_evaluation as evaluation_module
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
    preference_score: float | None = 0.5,
    qualification_score: float | None = 0.5,
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
        result=SimpleNamespace(
            recommendation=recommendation,
            gaps=result_gaps,
            preference_score=preference_score,
            qualification_score=qualification_score,
            preference_contributions=[],
            qualification_contributions=[],
        ),
        facts=[],
        signals=[],
    )


class SequenceService:
    def __init__(self, records: list[SimpleNamespace | BaseException]) -> None:
        self.records = iter(records)
        self.seen: list[str] = []

    def analyze_job(self, job_description: str) -> SimpleNamespace:
        self.seen.append(job_description)
        outcome = next(self.records)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FailingService:
    def analyze_job(self, job_description: str) -> SimpleNamespace:
        raise RuntimeError("PRIVATE-JD-MARKER secret@example.test")


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

    contaminated_score = _case(system_preference_score=0.9)
    with pytest.raises(ValueError, match="pre-populated system result"):
        run_evaluation([contaminated_score], service)

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
    assert report.gap_type_confusion_counts[GapKind.CAPABILITY_GAP]["__missing_system__"] == 1
    assert report.gap_type_confusion_counts["__missing_human__"][GapKind.HARD_BLOCKER] == 1

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


def test_partial_report_retains_all_cases_and_nullable_zero_denominators() -> None:
    cases = [_case(id="first"), _case(id="second")]
    success = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]
    )
    service = SequenceService([success, RuntimeError("private failure"), success])

    report = run_evaluation(cases, service)

    assert [case.id for case in report.cases] == ["first", "second"]
    assert report.cases[1].human_gap_labels == [GapKind.EVIDENCE_GAP]
    assert report.cases[1].system_recommendation is None
    assert report.cases[1].system_gap_labels is None
    assert report.attempted_count == 2
    assert report.completed_count == 1
    assert report.failed_count == 1
    assert report.run_complete is False

    failed = run_evaluation([_case()], FailingService())
    assert failed.recommendation_agreement is None
    assert failed.rerun_stability is None


def test_confusion_matrix_counts_all_labels_with_missing_sentinels() -> None:
    first = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW,
        [GapKind.HARD_BLOCKER, GapKind.CAPABILITY_GAP, GapKind.PREFERENCE_MISMATCH],
    )
    report = run_evaluation(
        [
            _case(
                human_gap_labels=[GapKind.EVIDENCE_GAP, GapKind.HARD_BLOCKER],
            )
        ],
        SequenceService([first, first]),
    )

    confusion = report.gap_type_confusion_counts
    assert confusion[GapKind.HARD_BLOCKER][GapKind.HARD_BLOCKER] == 1
    assert confusion[GapKind.EVIDENCE_GAP]["__missing_system__"] == 1
    assert confusion["__missing_human__"][GapKind.CAPABILITY_GAP] == 1
    assert confusion["__missing_human__"][GapKind.PREFERENCE_MISMATCH] == 1
    assert sum(sum(row.values()) for row in confusion.values()) == 4


def test_human_review_metrics_are_optional_and_never_inferred() -> None:
    record = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]
    )
    unreviewed = run_evaluation([_case()], SequenceService([record, record]))

    assert unreviewed.gap_label_agreement == 1.0
    assert unreviewed.evaluation_runtime_seconds >= 0
    assert unreviewed.top_five_usefulness is None
    assert unreviewed.review_time_seconds is None

    reviewed_cases = [
        _case(
            id="one",
            human_top_five_usefulness=0.5,
            human_review_time_seconds=30,
        ),
        _case(
            id="two",
            human_top_five_usefulness=1.0,
            human_review_time_seconds=45,
        ),
    ]
    reviewed = run_evaluation(
        reviewed_cases, SequenceService([record, record, record, record])
    )
    assert reviewed.top_five_usefulness == 0.75
    assert reviewed.review_time_seconds == 75


def test_rerun_failure_marks_run_incomplete_but_preserves_first_pass() -> None:
    first = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW,
        [GapKind.EVIDENCE_GAP],
        preference_score=0.8,
        qualification_score=0.7,
    )
    report = run_evaluation(
        [_case()], SequenceService([first, RuntimeError("private rerun failure")])
    )

    assert report.completed_count == 1
    assert report.failed_count == 0
    assert report.recommendation_agreement == 1.0
    assert report.cases[0].system_recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert report.cases[0].system_preference_score == 0.8
    assert report.cases[0].system_qualification_score == 0.7
    assert report.rerun_attempted_count == 1
    assert report.rerun_completed_count == 0
    assert report.rerun_failed_count == 1
    assert report.rerun_failure_case_ids == ["invented-case-1"]
    assert report.rerun_stability is None
    assert report.run_complete is False


def test_nondeterministic_completed_rerun_is_not_a_rerun_failure() -> None:
    first = _record(RecommendationLevel.PRIORITY_APPLY, [])
    changed = _record(RecommendationLevel.HOLD, [GapKind.CAPABILITY_GAP])

    report = run_evaluation([_case()], SequenceService([first, changed]))

    assert report.rerun_attempted_count == 1
    assert report.rerun_completed_count == 1
    assert report.rerun_failed_count == 0
    assert report.rerun_failure_case_ids == []
    assert report.rerun_stability == 0.0
    assert report.run_complete is True


def test_top_five_usefulness_uses_ranked_jobs_and_persisted_scores() -> None:
    cases = [
        _case(id="z-skip"),
        _case(id="b-priority", human_top_five_usefulness=0.8),
        _case(id="a-priority", human_top_five_usefulness=1.0),
        _case(id="c-priority", human_top_five_usefulness=0.6),
        _case(id="d-apply", human_top_five_usefulness=0.4),
        _case(id="e-hold", human_top_five_usefulness=0.2),
    ]
    first_run = [
        _record(RecommendationLevel.SKIP, [], preference_score=1, qualification_score=1),
        _record(RecommendationLevel.PRIORITY_APPLY, [], preference_score=0.9, qualification_score=0.5),
        _record(RecommendationLevel.PRIORITY_APPLY, [], preference_score=0.9, qualification_score=0.5),
        _record(RecommendationLevel.PRIORITY_APPLY, [], preference_score=0.8, qualification_score=1),
        _record(RecommendationLevel.APPLY_AFTER_REVIEW, [], preference_score=1, qualification_score=1),
        _record(RecommendationLevel.HOLD, [], preference_score=1, qualification_score=1),
    ]

    report = run_evaluation(cases, SequenceService(first_run + first_run))

    assert [case.id for case in report.cases] == [case.id for case in cases]
    by_id = {case.id: case for case in report.cases}
    assert by_id["a-priority"].system_preference_score == 0.9
    assert by_id["a-priority"].system_qualification_score == 0.5
    assert report.top_five_case_ids == [
        "a-priority",
        "b-priority",
        "c-priority",
        "d-apply",
        "e-hold",
    ]
    assert report.top_five_usefulness == pytest.approx(0.6)

    missing_selected_rating = [case.model_copy(deep=True) for case in cases]
    missing_selected_rating[2].human_top_five_usefulness = None
    missing = run_evaluation(
        missing_selected_rating, SequenceService(first_run + first_run)
    )
    assert missing.top_five_usefulness is None


def test_unsupported_source_audit_counts_each_defined_class() -> None:
    unsupported_gap = SimpleNamespace(
        kind=GapKind.CAPABILITY_GAP,
        job_evidence=[],
        candidate_evidence=[],
        missing_evidence=False,
    )
    explicit_missing_gap = SimpleNamespace(
        kind=GapKind.EVIDENCE_GAP,
        job_evidence=[],
        candidate_evidence=[],
        missing_evidence=True,
    )
    unsupported_contribution = SimpleNamespace(
        job_evidence=[], candidate_evidence=[], audit_support=[]
    )
    supported_contribution = SimpleNamespace(
        job_evidence=[], candidate_evidence=[], audit_support=["configured-default:fact"]
    )
    record = SimpleNamespace(
        facts=[
            SimpleNamespace(is_unknown=False, evidence=[]),
            SimpleNamespace(is_unknown=True, evidence=[]),
            SimpleNamespace(is_unknown=False, evidence=["jd-ref"]),
        ],
        signals=[
            SimpleNamespace(
                status="available",
                provider=None,
                model=None,
                question_id="question",
                question_version="v1",
                evidence=[],
            ),
            SimpleNamespace(
                status="available",
                provider="provider",
                model="model",
                question_id="question",
                question_version="v1",
                evidence=[],
            ),
        ],
        result=SimpleNamespace(
            recommendation=RecommendationLevel.APPLY_AFTER_REVIEW,
            gaps=[unsupported_gap, explicit_missing_gap],
            preference_score=0.5,
            qualification_score=0.5,
            preference_contributions=[unsupported_contribution],
            qualification_contributions=[supported_contribution],
        ),
    )

    report = run_evaluation([_case()], SequenceService([record, record]))

    assert report.unsupported_fact_count == 1
    assert report.unsupported_signal_count == 1
    assert report.unsupported_gap_count == 1
    assert report.unsupported_contribution_count == 1
    assert report.unsupported_claim_count == 4


def test_real_eval_paths_are_gitignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "eval/private/" in gitignore
    assert "eval/reports/" in gitignore
    assert "*.sqlite" in gitignore
    assert "*.sqlite-wal" in gitignore
    assert "*.sqlite-shm" in gitignore
    assert ".cache/" in gitignore
    assert "private-data/**" in gitignore
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


def test_cli_writes_partial_report_atomically_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "cases.jsonl"
    destination = tmp_path / "report.json"
    cases = [_case(id="success"), _case(id="failure")]
    source.write_text(
        "\n".join(case.model_dump_json() for case in cases) + "\n",
        encoding="utf-8",
    )
    success = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]
    )
    service = SequenceService([success, RuntimeError("PRIVATE-JD-MARKER"), success])

    exit_code = main([str(source), str(destination)], service=service)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert exit_code != 0
    assert payload["run_complete"] is False
    assert payload["attempted_count"] == 2
    assert payload["completed_count"] == 1
    assert payload["failed_count"] == 1
    assert len(payload["cases"]) == 2
    assert "PRIVATE-JD-MARKER" not in captured.out + captured.err
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_cli_writes_report_and_exits_nonzero_on_rerun_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cases.jsonl"
    destination = tmp_path / "report.json"
    source.write_text(_case().model_dump_json() + "\n", encoding="utf-8")
    first = _record(
        RecommendationLevel.APPLY_AFTER_REVIEW, [GapKind.EVIDENCE_GAP]
    )

    exit_code = main(
        [str(source), str(destination)],
        service=SequenceService([first, RuntimeError("private rerun failure")]),
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert exit_code != 0
    assert payload["completed_count"] == 1
    assert payload["recommendation_agreement"] == 1.0
    assert payload["rerun_attempted_count"] == 1
    assert payload["rerun_completed_count"] == 0
    assert payload["rerun_failed_count"] == 1
    assert payload["rerun_stability"] is None
    assert payload["run_complete"] is False


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_cli_rejects_existing_destination_with_same_inode(
    tmp_path: Path, link_kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "cases.jsonl"
    destination = tmp_path / "report.json"
    original = _case().model_dump_json() + "\n"
    source.write_text(original, encoding="utf-8")
    if link_kind == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source)

    exit_code = main([str(source), str(destination)], service=FailingService())

    assert exit_code != 0
    assert source.read_text(encoding="utf-8") == original
    assert "PRIVATE-JD-MARKER" not in capsys.readouterr().err


def test_atomic_write_preserves_destination_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text("old report\n", encoding="utf-8")

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(evaluation_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        evaluation_module._write_report_atomic(destination, "new report\n")

    assert destination.read_text(encoding="utf-8") == "old report\n"
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_duplicate_ids_report_both_line_numbers_without_private_content(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "cases.jsonl"
    destination = tmp_path / "report.json"
    private_id = "PRIVATE-ID-MARKER"
    private_jd = "PRIVATE-JD-MARKER"
    case = _case(id=private_id, job_description=private_jd)
    source.write_text(case.model_dump_json() + "\n" + case.model_dump_json() + "\n")

    exit_code = main([str(source), str(destination)], service=FailingService())
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "line 2" in captured.err
    assert "first line 1" in captured.err
    assert private_id not in captured.err
    assert private_jd not in captured.err
    assert not destination.exists()


def test_readme_documents_clean_startup_and_local_database_path() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    eval_readme = (ROOT / "eval" / "README.md").read_text(encoding="utf-8")

    assert "python -m ai_dream_job_engine.mcp_server" in readme
    assert "~/Library/Application Support/ai-dream-job-engine" in readme
    assert "XDG_DATA_HOME" in readme
    assert "AI_DREAM_JOB_DATA_PATH" in readme
    assert "rerun_failed_count" in readme
    assert "system_preference_score" in eval_readme
    assert "system_qualification_score" in eval_readme
    assert "Candidate raw claims" in eval_readme
