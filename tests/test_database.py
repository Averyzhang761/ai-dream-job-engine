import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import ai_dream_job_engine.database as database_module
from ai_dream_job_engine.database import LocalStore, StoreError
from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateClaim,
    CandidateProfile,
    ClaimDecision,
    ClaimSource,
    DecisionTrace,
    JobReviewRecord,
    JobReviewStatus,
    Rubric,
)
from ai_dream_job_engine.profile import confirm_claims, disconnect_memory_source


def _create_version_one_schema(connection, *, malformed_profiles=False) -> None:
    for table in ("profiles", "rubrics", "analyses"):
        payload_constraint = (
            "" if table == "profiles" and malformed_profiles else "NOT NULL"
        )
        connection.execute(
            f"""
            CREATE TABLE {table} (
                id TEXT PRIMARY KEY NOT NULL,
                schema_version INTEGER NOT NULL,
                payload TEXT {payload_constraint},
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
    connection.execute("PRAGMA user_version = 1")


def _analysis(analysis_id: str, created_at: datetime, input_hash: str | None = None):
    return AnalysisRecord(
        id=analysis_id,
        input_hash=input_hash or f"sha256:{analysis_id}",
        status="success",
        latency_ms=0,
        created_at=created_at,
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

    assert tables == {"profiles", "rubrics", "analyses", "job_reviews"}
    assert user_version == 2


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


def test_profile_round_trip_preserves_import_receipt_and_disconnects(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    suggestion = CandidateClaim(
        id="memory-1",
        claim_type="location",
        value="Remote",
        source=ClaimSource.AGENT_SUGGESTED,
        provenance="agent-memory://career-agent/memory-1",
        source_agent="career-agent",
    )
    profile = confirm_claims(
        CandidateProfile(claims=[suggestion]),
        [ClaimDecision(claim_id="memory-1", action="accept")],
    )

    store.save_profile(profile)
    loaded = store.load_profile()

    assert loaded == profile
    assert len(loaded.import_receipts) == 1
    disconnected = disconnect_memory_source(loaded, "career-agent", True)
    assert [claim.id for claim in disconnected.claims] == ["memory-1"]
    assert disconnected.import_receipts == []


def test_rubric_round_trip_preserves_version(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    rubric = Rubric(version="ai-job-v7")

    store.save_rubric(rubric)

    loaded = store.load_rubric()
    assert loaded == rubric
    assert loaded.version == "ai-job-v7"


def test_profile_compare_and_swap_rejects_stale_writer_across_stores(tmp_path) -> None:
    path = tmp_path / "store.sqlite3"
    first = LocalStore.open(path)
    second = LocalStore.open(path)
    first.initialize()
    first.save_profile(CandidateProfile())
    original = first.load_profile()
    assert original is not None

    winner = original.model_copy(update={"storage_revision": 1})
    loser = original.model_copy(update={"storage_revision": 1})

    assert first.compare_and_swap_profile(winner, expected_storage_revision=0)
    assert not second.compare_and_swap_profile(loser, expected_storage_revision=0)
    assert second.load_profile().storage_revision == 1


def test_rubric_compare_and_swap_rejects_stale_writer_across_stores(tmp_path) -> None:
    path = tmp_path / "store.sqlite3"
    first = LocalStore.open(path)
    second = LocalStore.open(path)
    first.initialize()
    first.save_rubric(Rubric(version="ai-job-v1"))
    original = first.load_rubric()
    assert original is not None

    winner = original.model_copy(update={"storage_revision": 1})
    loser = original.model_copy(update={"storage_revision": 1})

    assert first.compare_and_swap_rubric(winner, expected_storage_revision=0)
    assert not second.compare_and_swap_rubric(loser, expected_storage_revision=0)
    assert second.load_rubric().storage_revision == 1


def test_version_one_database_migrates_to_review_state_schema(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    profile = CandidateProfile()
    rubric = Rubric(version="ai-job-v1")
    analysis = _analysis("analysis-1", datetime(2026, 1, 1, tzinfo=timezone.utc))
    created_at = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(database_path) as connection:
        _create_version_one_schema(connection)
        connection.execute(
            "INSERT INTO profiles VALUES (?, ?, ?, ?, ?)",
            ("active", 1, profile.model_dump_json(), created_at, created_at),
        )
        connection.execute(
            "INSERT INTO rubrics VALUES (?, ?, ?, ?, ?)",
            ("active", 1, rubric.model_dump_json(), created_at, created_at),
        )
        connection.execute(
            "INSERT INTO analyses VALUES (?, ?, ?, ?, ?)",
            (analysis.id, 1, analysis.model_dump_json(), created_at, created_at),
        )

    store = LocalStore.open(database_path)
    store.initialize()

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        preserved_versions = {
            table: connection.execute(
                f"SELECT schema_version FROM {table}"
            ).fetchone()[0]
            for table in ("profiles", "rubrics", "analyses")
        }

    assert tables == {"profiles", "rubrics", "analyses", "job_reviews"}
    assert user_version == 2
    assert preserved_versions == {"profiles": 1, "rubrics": 1, "analyses": 1}
    assert store.load_profile() == profile
    assert store.load_rubric() == rubric
    assert store.list_analyses() == [analysis]


def test_job_review_round_trip(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    review = JobReviewRecord(
        analysis_id="analysis-1",
        status=JobReviewStatus.KEPT,
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    store.save_job_review(review)

    assert store.load_job_review("analysis-1") == review
    assert store.list_job_reviews() == [review]


def test_job_review_upsert_preserves_one_row_per_analysis(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    original = JobReviewRecord(
        analysis_id="analysis-1",
        status=JobReviewStatus.KEPT,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    replacement = JobReviewRecord(
        analysis_id="analysis-1",
        status=JobReviewStatus.READY_TO_APPLY,
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    store.save_job_review(original)
    store.save_job_review(replacement)

    assert store.list_job_reviews() == [replacement]


def test_job_review_order_uses_record_updated_at(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    later = JobReviewRecord(
        analysis_id="analysis-later",
        status=JobReviewStatus.KEPT,
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    earlier = JobReviewRecord(
        analysis_id="analysis-earlier",
        status=JobReviewStatus.KEPT,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    store.save_job_review(later)
    store.save_job_review(earlier)

    assert store.list_job_reviews() == [earlier, later]


def test_job_review_rejects_version_one_payload(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    store = LocalStore.open(database_path)
    store.initialize()
    review = JobReviewRecord(
        analysis_id="analysis-1",
        status=JobReviewStatus.KEPT,
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO job_reviews VALUES (?, ?, ?, ?, ?)",
            (
                review.analysis_id,
                1,
                review.model_dump_json(),
                review.updated_at.isoformat(),
                review.updated_at.isoformat(),
            ),
        )

    with pytest.raises(StoreError, match="invalid persisted job review payload"):
        store.load_job_review(review.analysis_id)


def test_review_state_schema_rejects_unexpected_objects(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    store = LocalStore.open(database_path)
    store.initialize()
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(StoreError) as exc_info:
        store.initialize()

    assert exc_info.value.__cause__ is not None


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


def test_version_zero_with_existing_objects_is_rejected_without_mutation(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(StoreError) as exc_info:
        LocalStore.open(database_path).initialize()

    assert exc_info.value.__cause__ is not None
    with sqlite3.connect(database_path) as connection:
        objects = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert objects == [("unrelated",)]
    assert user_version == 0


def test_malformed_version_one_schema_is_rejected_without_mutation(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    with sqlite3.connect(database_path) as connection:
        _create_version_one_schema(connection, malformed_profiles=True)

    with pytest.raises(StoreError) as exc_info:
        LocalStore.open(database_path).initialize()

    assert exc_info.value.__cause__ is not None
    with sqlite3.connect(database_path) as connection:
        payload_column = next(
            row
            for row in connection.execute("PRAGMA table_info(profiles)")
            if row[1] == "payload"
        )
    assert payload_column[3] == 0


def test_current_version_with_extra_table_is_rejected_without_mutation(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    store = LocalStore.open(database_path)
    store.initialize()
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")

    with pytest.raises(StoreError) as exc_info:
        store.initialize()

    assert exc_info.value.__cause__ is not None
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert tables == {
        "profiles",
        "rubrics",
        "analyses",
        "job_reviews",
        "unrelated",
    }


def test_unsupported_database_version_is_rejected(tmp_path) -> None:
    database_path = tmp_path / "store.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA user_version = 3")

    with pytest.raises(StoreError) as exc_info:
        LocalStore.open(database_path).initialize()

    assert exc_info.value.__cause__ is not None


def test_analysis_order_uses_stable_id_to_break_creation_time_ties(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store.save_analysis(_analysis("analysis-b", created_at))
    store.save_analysis(_analysis("analysis-a", created_at))

    assert [record.id for record in store.list_analyses()] == [
        "analysis-a",
        "analysis-b",
    ]


def test_analysis_order_normalizes_timezone_offsets(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    earlier = _analysis(
        "analysis-earlier",
        datetime(2026, 1, 1, 1, tzinfo=timezone(timedelta(hours=1))),
    )
    later = _analysis(
        "analysis-later",
        datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
    )
    store.save_analysis(later)
    store.save_analysis(earlier)

    assert [record.id for record in store.list_analyses()] == [
        "analysis-earlier",
        "analysis-later",
    ]


def test_analysis_replacement_uses_replacement_creation_time(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    store.save_analysis(
        _analysis("analysis-a", datetime(2026, 1, 3, tzinfo=timezone.utc))
    )
    store.save_analysis(
        _analysis("analysis-b", datetime(2026, 1, 2, tzinfo=timezone.utc))
    )
    replacement = _analysis(
        "analysis-a",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        input_hash="sha256:replacement",
    )

    store.save_analysis(replacement)

    loaded = store.list_analyses()
    assert [record.id for record in loaded] == ["analysis-a", "analysis-b"]
    assert loaded[0] == replacement


def test_analysis_insert_rejects_duplicate_without_overwrite(tmp_path) -> None:
    store = LocalStore.open(tmp_path / "store.sqlite3")
    store.initialize()
    original = _analysis(
        "analysis-a",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        input_hash="sha256:original",
    )
    duplicate = _analysis(
        "analysis-a",
        datetime(2026, 1, 2, tzinfo=timezone.utc),
        input_hash="sha256:duplicate",
    )
    store.insert_analysis(original)

    with pytest.raises(StoreError, match="analysis insert failed"):
        store.insert_analysis(duplicate)

    assert store.list_analyses() == [original]


def test_connections_close_on_success_and_exception(tmp_path, monkeypatch) -> None:
    database_path = tmp_path / "store.sqlite3"
    unsupported_path = tmp_path / "unsupported.sqlite3"
    store = LocalStore.open(database_path)
    store.initialize()
    real_connect = sqlite3.connect
    with real_connect(unsupported_path) as connection:
        connection.execute("PRAGMA user_version = 3")
    tracked_connections = []

    class TrackingConnection(sqlite3.Connection):
        def close(self) -> None:
            tracked_connections.append(self)
            super().close()

    def tracked_connect(path):
        return real_connect(path, factory=TrackingConnection)

    monkeypatch.setattr(database_module.sqlite3, "connect", tracked_connect)
    profile = CandidateProfile()
    rubric = Rubric(version="ai-job-v1")
    analysis = _analysis("analysis-1", datetime(2026, 1, 1, tzinfo=timezone.utc))

    store.initialize()
    store.save_profile(profile)
    assert store.load_profile() == profile
    store.save_rubric(rubric)
    assert store.load_rubric() == rubric
    store.save_analysis(analysis)
    assert store.list_analyses() == [analysis]

    with real_connect(database_path) as connection:
        connection.execute(
            "UPDATE profiles SET payload = 'not-json' WHERE id = 'active'"
        )
    with pytest.raises(StoreError):
        store.load_profile()
    with pytest.raises(StoreError):
        LocalStore.open(unsupported_path).initialize()

    assert len(tracked_connections) == 9
    for connection in tracked_connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
