"""Versioned local SQLite persistence for validated domain models."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ai_dream_job_engine.models import (
    AnalysisRecord,
    CandidateProfile,
    JobReviewRecord,
    Rubric,
)


SCHEMA_VERSION = 2
_ACTIVE_ID = "active"
_VERSION_ONE_TABLES = ("profiles", "rubrics", "analyses")
_TABLES = (*_VERSION_ONE_TABLES, "job_reviews")
_EXPECTED_COLUMNS = [
    ("id", "TEXT", 1, None, 1),
    ("schema_version", "INTEGER", 1, None, 0),
    ("payload", "TEXT", 1, None, 0),
    ("created_at", "TEXT", 1, None, 0),
    ("updated_at", "TEXT", 1, None, 0),
]
_Model = TypeVar("_Model", bound=BaseModel)


class StoreError(Exception):
    """Raised when persisted data is malformed or schema-incompatible."""


class LocalStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def open(cls, path: Path) -> LocalStore:
        return cls(path)

    def initialize(self) -> None:
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                current_version = connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
                if current_version not in (0, 1, SCHEMA_VERSION):
                    raise ValueError(
                        f"unsupported database version {current_version}; "
                        f"expected {SCHEMA_VERSION}"
                    )

                objects = self._application_objects(connection)
                if current_version == 0:
                    if objects:
                        raise ValueError(
                            "version-0 database contains pre-existing objects"
                        )
                    for table in _TABLES:
                        self._create_table(connection, table)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                elif current_version == 1:
                    self._validate_schema(connection, _VERSION_ONE_TABLES)
                    self._create_table(connection, "job_reviews")
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

                self._validate_schema(connection)
        except (sqlite3.DatabaseError, ValueError) as error:
            raise StoreError("incompatible database schema") from error

    def save_profile(self, profile: CandidateProfile) -> None:
        self._save_singleton("profiles", profile)

    def load_profile(self) -> CandidateProfile | None:
        return self._load_singleton("profiles", CandidateProfile)

    def compare_and_swap_profile(
        self, profile: CandidateProfile, expected_storage_revision: int
    ) -> bool:
        return self._compare_and_swap_singleton(
            "profiles", profile, expected_storage_revision
        )

    def save_rubric(self, rubric: Rubric) -> None:
        self._save_singleton("rubrics", rubric)

    def load_rubric(self) -> Rubric | None:
        return self._load_singleton("rubrics", Rubric)

    def compare_and_swap_rubric(
        self, rubric: Rubric, expected_storage_revision: int
    ) -> bool:
        return self._compare_and_swap_singleton(
            "rubrics", rubric, expected_storage_revision
        )

    def save_analysis(self, record: AnalysisRecord) -> None:
        self._save(
            "analyses",
            record.id,
            record,
            self._utc_timestamp(record.created_at),
            replace_created_at=True,
        )

    def compare_and_swap_analysis(
        self, record: AnalysisRecord, expected_decision_revision: int
    ) -> bool:
        if record.decision_revision != expected_decision_revision + 1:
            raise StoreError("invalid analysis revision update")
        now = datetime.now(timezone.utc).isoformat()
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT schema_version, payload
                    FROM analyses
                    WHERE id = ?
                    """,
                    (record.id,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return False
                current = self._validate_payload(
                    "analysis", row, AnalysisRecord
                )
                if current.decision_revision != expected_decision_revision:
                    connection.rollback()
                    return False
                connection.execute(
                    """
                    UPDATE analyses
                    SET schema_version = ?, payload = ?, created_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        SCHEMA_VERSION,
                        record.model_dump_json(),
                        self._utc_timestamp(record.created_at),
                        now,
                        record.id,
                    ),
                )
                connection.commit()
                return True
        except sqlite3.DatabaseError as error:
            raise StoreError("analysis compare-and-swap failed") from error

    def insert_analysis(self, record: AnalysisRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO analyses
                        (id, schema_version, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        SCHEMA_VERSION,
                        record.model_dump_json(),
                        self._utc_timestamp(record.created_at),
                        now,
                    ),
                )
        except sqlite3.DatabaseError as error:
            raise StoreError("analysis insert failed") from error

    def list_analyses(self) -> list[AnalysisRecord]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute(
                """
                SELECT schema_version, payload
                FROM analyses
                ORDER BY created_at, id
                """
            ).fetchall()
        return [
            self._validate_payload("analysis", row, AnalysisRecord) for row in rows
        ]

    def save_job_review(self, record: JobReviewRecord) -> None:
        self._save(
            "job_reviews",
            record.analysis_id,
            record,
            self._utc_timestamp(record.updated_at),
            updated_at=self._utc_timestamp(record.updated_at),
        )

    def load_job_review(self, analysis_id: str) -> JobReviewRecord | None:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                """
                SELECT schema_version, payload
                FROM job_reviews
                WHERE id = ?
                """,
                (analysis_id,),
            ).fetchone()
        if row is None:
            return None
        return self._validate_payload("job review", row, JobReviewRecord)

    def list_job_reviews(self) -> list[JobReviewRecord]:
        with closing(sqlite3.connect(self._path)) as connection:
            rows = connection.execute(
                """
                SELECT schema_version, payload
                FROM job_reviews
                ORDER BY updated_at, id
                """
            ).fetchall()
        return [
            self._validate_payload("job review", row, JobReviewRecord) for row in rows
        ]

    def _save_singleton(self, table: str, model: BaseModel) -> None:
        self._save(table, _ACTIVE_ID, model, datetime.now(timezone.utc).isoformat())

    def _compare_and_swap_singleton(
        self,
        table: str,
        model: CandidateProfile | Rubric,
        expected_storage_revision: int,
    ) -> bool:
        if model.storage_revision != expected_storage_revision + 1:
            raise StoreError("invalid singleton revision update")
        now = datetime.now(timezone.utc).isoformat()
        model_type = CandidateProfile if table == "profiles" else Rubric
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    f"SELECT schema_version, payload FROM {table} WHERE id = ?",
                    (_ACTIVE_ID,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return False
                current = self._validate_payload(table[:-1], row, model_type)
                if current.storage_revision != expected_storage_revision:
                    connection.rollback()
                    return False
                connection.execute(
                    f"""
                    UPDATE {table}
                    SET schema_version = ?, payload = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (SCHEMA_VERSION, model.model_dump_json(), now, _ACTIVE_ID),
                )
                connection.commit()
                return True
        except sqlite3.DatabaseError as error:
            raise StoreError(f"{table[:-1]} compare-and-swap failed") from error

    def _save(
        self,
        table: str,
        stable_id: str,
        model: BaseModel,
        created_at: str,
        *,
        replace_created_at: bool = False,
        updated_at: str | None = None,
    ) -> None:
        updated_at = updated_at or datetime.now(timezone.utc).isoformat()
        created_at_update = (
            ",\n                    created_at = excluded.created_at"
            if replace_created_at
            else ""
        )
        with closing(sqlite3.connect(self._path)) as connection, connection:
            connection.execute(
                f"""
                INSERT INTO {table}
                    (id, schema_version, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at{created_at_update}
                """,
                (
                    stable_id,
                    SCHEMA_VERSION,
                    model.model_dump_json(),
                    created_at,
                    updated_at,
                ),
            )

    def _load_singleton(self, table: str, model_type: type[_Model]) -> _Model | None:
        with closing(sqlite3.connect(self._path)) as connection:
            row = connection.execute(
                f"SELECT schema_version, payload FROM {table} WHERE id = ?",
                (_ACTIVE_ID,),
            ).fetchone()
        if row is None:
            return None
        return self._validate_payload(table[:-1], row, model_type)

    @staticmethod
    def _application_objects(connection: sqlite3.Connection) -> set[tuple[str, str]]:
        return set(
            connection.execute(
                """
                SELECT type, name
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
        )

    @staticmethod
    def _create_table(connection: sqlite3.Connection, table: str) -> None:
        connection.execute(
            f"""
            CREATE TABLE {table} (
                id TEXT PRIMARY KEY NOT NULL,
                schema_version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    @classmethod
    def _validate_schema(
        cls,
        connection: sqlite3.Connection,
        tables: tuple[str, ...] = _TABLES,
    ) -> None:
        expected_objects = {("table", table) for table in tables}
        if cls._application_objects(connection) != expected_objects:
            raise ValueError("database objects do not match the application schema")

        for table in tables:
            columns = [
                (row[1], row[2].upper(), row[3], row[4], row[5])
                for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            if columns != _EXPECTED_COLUMNS:
                raise ValueError(f"table {table} has an incompatible schema")

            indexes = connection.execute(f"PRAGMA index_list({table})").fetchall()
            if len(indexes) != 1 or indexes[0][2:] != (1, "pk", 0):
                raise ValueError(f"table {table} has incompatible constraints")
            indexed_columns = connection.execute(
                f"PRAGMA index_info({indexes[0][1]})"
            ).fetchall()
            if [row[2] for row in indexed_columns] != ["id"]:
                raise ValueError(f"table {table} has an incompatible primary key")

    @staticmethod
    def _utc_timestamp(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _validate_payload(
        record_type: str, row: tuple[int, str], model_type: type[_Model]
    ) -> _Model:
        schema_version, payload = row
        try:
            legacy_model = model_type in (CandidateProfile, Rubric, AnalysisRecord)
            if schema_version != SCHEMA_VERSION and not (
                schema_version == 1 and legacy_model
            ):
                raise ValueError(
                    f"unsupported payload version {schema_version}"
                )
            return model_type.model_validate_json(payload)
        except (TypeError, ValueError) as error:
            raise StoreError(f"invalid persisted {record_type} payload") from error
