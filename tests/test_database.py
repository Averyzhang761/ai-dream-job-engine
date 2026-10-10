import sqlite3
from datetime import datetime, timezone

import pytest

from ai_dream_job_engine.database import LocalStore, StoreError
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimSource,
    DecisionTrace,
    Rubric,
)


def test_initialize_is_idempotent(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    store = LocalStore.open(database_path)

    store.initialize()
    store.initialize()

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert tables == {"profiles", "rubrics", "analyses"}
    assert user_version == 1


def test_profile_round_trip_preserves_claim_sources(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    profile = CandidateProfile(
        claims=[
            CandidateClaim(
                id="claim-1",
                claim_type="skill",
                value="Python",
                source=ClaimSource.AGENT_SUGGESTED,
                provenance="memory:career",
                source_agent="career-agent",
            )
        ]
    )

    store.save_profile(profile)

    assert store.load_profile() == profile
    assert store.load_profile().claims[0].source is ClaimSource.AGENT_SUGGESTED


def test_rubric_round_trip_preserves_version(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    rubric = Rubric(version="ai-job-v7")

    store.save_rubric(rubric)

    loaded = store.load_rubric()
    assert loaded == rubric
    assert loaded.version == "ai-job-v7"


def test_analysis_round_trip_preserves_decision_trace(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    later = AnalysisRecord(
        id="analysis-b",
        input_hash="sha256:later",
        status="success",
        latency_ms=10,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        trace=DecisionTrace(
            analysis_id="analysis-b",
            input_ids=["job-2", "profile-default"],
            evidence=["Requires Python"],
            providers={"facts": "openai"},
            rubric_version="ai-job-v1",
            prompt_versions={"facts": "prompt-v2"},
            result_version="1",
        ),
    )
    earlier = AnalysisRecord(
        id="analysis-a",
        input_hash="sha256:earlier",
        status="success",
        latency_ms=5,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    store.save_analysis(later)
    store.save_analysis(earlier)

    loaded = store.list_analyses()
    assert [record.id for record in loaded] == ["analysis-a", "analysis-b"]
    assert loaded[1] == later
    assert loaded[1].trace.prompt_versions == {"facts": "prompt-v2"}


def test_invalid_saved_payload_raises_store_error(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    store = LocalStore.open(database_path)
    store.initialize()

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO profiles
                (id, schema_version, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("active", 1, "not-json", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )

    with pytest.raises(StoreError) as exc_info:
        store.load_profile()

    assert exc_info.value.__cause__ is not None
