"""Versioned local SQLite persistence for validated domain models."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from ai_dream_job_engine.models import AnalysisRecord, CandidateProfile, Rubric


SCHEMA_VERSION = 1
_ACTIVE_ID = "active"
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
        with sqlite3.connect(self._path) as connection:
            current_version = connection.execute("PRAGMA user_version").fetchone()[0]
            if current_version not in (0, SCHEMA_VERSION):
                error = ValueError(
                    f"unsupported database version {current_version}; "
                    f"expected {SCHEMA_VERSION}"
                )
                raise StoreError("incompatible database schema") from error

            for table in ("profiles", "rubrics", "analyses"):
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def save_profile(self, profile: CandidateProfile) -> None:
        self._save_singleton("profiles", profile)

    def load_profile(self) -> CandidateProfile | None:
        return self._load_singleton("profiles", CandidateProfile)

    def save_rubric(self, rubric: Rubric) -> None:
        self._save_singleton("rubrics", rubric)

    def load_rubric(self) -> Rubric | None:
        return self._load_singleton("rubrics", Rubric)

    def save_analysis(self, record: AnalysisRecord) -> None:
        self._save("analyses", record.id, record, record.created_at)

    def list_analyses(self) -> list[AnalysisRecord]:
        with sqlite3.connect(self._path) as connection:
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

    def _save_singleton(self, table: str, model: BaseModel) -> None:
        self._save(table, _ACTIVE_ID, model, datetime.now(timezone.utc))

    def _save(
        self, table: str, stable_id: str, model: BaseModel, created_at: datetime
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                f"""
                INSERT INTO {table}
                    (id, schema_version, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    stable_id,
                    SCHEMA_VERSION,
                    model.model_dump_json(),
                    created_at.isoformat(),
                    now,
                ),
            )

    def _load_singleton(self, table: str, model_type: type[_Model]) -> _Model | None:
        with sqlite3.connect(self._path) as connection:
            row = connection.execute(
                f"SELECT schema_version, payload FROM {table} WHERE id = ?",
                (_ACTIVE_ID,),
            ).fetchone()
        if row is None:
            return None
        return self._validate_payload(table[:-1], row, model_type)

    @staticmethod
    def _validate_payload(
        record_type: str, row: tuple[int, str], model_type: type[_Model]
    ) -> _Model:
        schema_version, payload = row
        try:
            if schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported payload version {schema_version}; "
                    f"expected {SCHEMA_VERSION}"
                )
            return model_type.model_validate_json(payload)
        except (TypeError, ValueError) as error:
            raise StoreError(f"invalid persisted {record_type} payload") from error
