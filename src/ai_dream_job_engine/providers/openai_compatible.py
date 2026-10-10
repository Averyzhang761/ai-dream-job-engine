"""Fact extraction through OpenAI-compatible structured responses."""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from ai_dream_job_engine.evidence import build_evidence_chunks, validate_quote
from ai_dream_job_engine.models import (
    ExtractedFact,
    FactDataType,
    FactDefinition,
    Rubric,
)
from ai_dream_job_engine.providers.base import ProviderError


PROMPT_VERSION = "job-fact-extraction-v1"
SCHEMA_VERSION = "extracted-fact-v1"


class _ProviderFact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fact_id: str = Field(min_length=1)
    value: Any | None = None
    is_unknown: StrictBool = False
    evidence: list[str] = Field(default_factory=list)


class _ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    facts: list[_ProviderFact]


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        provider: Literal["openai", "openrouter"],
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        if provider not in {"openai", "openrouter"}:
            raise ValueError("provider must be openai or openrouter")
        if not api_key or not api_key.strip():
            raise ProviderError("provider API key is required")
        if not model:
            raise ValueError("model is required")

        self.provider = provider
        self.model = model
        if client is not None:
            self.client = client
        else:
            initialization_failed = False
            try:
                if provider == "openrouter":
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                        max_retries=0,
                    )
                else:
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.openai.com/v1",
                        max_retries=0,
                    )
            except Exception:
                initialization_failed = True
            if initialization_failed:
                raise ProviderError("provider client initialization failed")

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"model={self.model!r})"
        )

    def extract(self, job_description: str, rubric: Rubric) -> list[ExtractedFact]:
        request = {
            "job_description": job_description,
            "facts": [
                {
                    "id": fact.id,
                    "name": fact.name,
                    "definition": fact.definition,
                    "data_type": fact.data_type.value,
                    "extraction_instructions": fact.extraction_instructions,
                    "requires_evidence": fact.requires_evidence,
                }
                for fact in rubric.facts
            ],
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        request_failed = False
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=(
                    "Extract only the requested facts from the supplied public job "
                    "description. Evidence must be an exact, verbatim quote."
                ),
                input=json.dumps(request),
                text_format=_ExtractionResponse,
                store=False,
            )
        except Exception:
            request_failed = True
        if request_failed:
            raise ProviderError("job fact provider request failed")
        parsed = self._parse_response(response)
        return self._validated_facts(parsed, job_description, rubric)

    @staticmethod
    def _parse_response(response: Any) -> _ExtractionResponse:
        parsing_failed = False
        try:
            raw = getattr(response, "output_parsed", None)
            if raw is None:
                raw = getattr(response, "output_text", None)
            if isinstance(raw, str):
                parsed = _ExtractionResponse.model_validate_json(raw)
            else:
                parsed = _ExtractionResponse.model_validate(raw)
        except Exception:
            parsing_failed = True
        if parsing_failed:
            raise ProviderError("provider returned malformed structured output")
        return parsed

    def _validated_facts(
        self,
        response: _ExtractionResponse,
        job_description: str,
        rubric: Rubric,
    ) -> list[ExtractedFact]:
        definitions = {fact.id: fact for fact in rubric.facts}
        if len(definitions) != len(rubric.facts):
            raise ProviderError("provider fact definitions are not unique")

        returned_ids = [fact.fact_id for fact in response.facts]
        unknown_ids = set(returned_ids) - definitions.keys()
        if unknown_ids:
            raise ProviderError("provider returned an unknown fact")
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != set(
            definitions
        ):
            raise ProviderError("provider returned an invalid fact set")

        returned_by_id = {fact.fact_id: fact for fact in response.facts}
        document = build_evidence_chunks(job_description)
        facts = []
        for definition in rubric.facts:
            returned = returned_by_id[definition.id]
            if returned.is_unknown:
                if returned.value is not None or returned.evidence:
                    raise ProviderError(
                        "provider returned a contradictory unknown fact"
                    )
            elif not self._valid_fact_value(definition, returned.value):
                raise ProviderError("provider returned an invalid fact value")
            if (
                definition.requires_evidence
                and not returned.is_unknown
                and not returned.evidence
            ):
                raise ProviderError("provider returned a fact without evidence")
            evidence_failed = False
            try:
                for quote in returned.evidence:
                    validate_quote(document, quote)
                facts.append(
                    ExtractedFact(
                        fact_id=returned.fact_id,
                        value=returned.value,
                        is_unknown=returned.is_unknown,
                        evidence=returned.evidence,
                        provider=self.provider,
                        model=self.model,
                    )
                )
            except (ValueError, ValidationError):
                evidence_failed = True
            if evidence_failed:
                raise ProviderError("provider returned invalid evidence")
        return facts

    @staticmethod
    def _valid_fact_value(definition: FactDefinition, value: Any) -> bool:
        if definition.data_type is FactDataType.BOOLEAN:
            return type(value) is bool
        if definition.data_type is FactDataType.NUMBER:
            if type(value) is int:
                return True
            return type(value) is float and math.isfinite(value)
        if definition.data_type is FactDataType.TEXT:
            return type(value) is str
        if definition.data_type is FactDataType.ENUM:
            return type(value) is str and value in definition.enum_values
        if definition.data_type is FactDataType.LIST:
            return type(value) is list
        return False
