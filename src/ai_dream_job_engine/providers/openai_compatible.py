"""Fact extraction through OpenAI-compatible structured responses."""

from __future__ import annotations

import json
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_dream_job_engine.evidence import build_evidence_chunks, validate_quote
from ai_dream_job_engine.models import ExtractedFact, Rubric
from ai_dream_job_engine.providers.base import ProviderError


PROMPT_VERSION = "job-fact-extraction-v1"
SCHEMA_VERSION = "extracted-fact-v1"


class _ProviderFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_id: str = Field(min_length=1)
    value: Any | None = None
    is_unknown: bool = False
    evidence: list[str] = Field(default_factory=list)


class _ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        elif provider == "openrouter":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                max_retries=0,
            )
        else:
            self.client = OpenAI(api_key=api_key, max_retries=0)

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
            parsed = self._parse_response(response)
            return self._validated_facts(parsed, job_description, rubric)
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("job fact provider request failed") from None

    @staticmethod
    def _parse_response(response: Any) -> _ExtractionResponse:
        raw = getattr(response, "output_parsed", None)
        if raw is None:
            raw = getattr(response, "output_text", None)
        try:
            if isinstance(raw, str):
                return _ExtractionResponse.model_validate_json(raw)
            return _ExtractionResponse.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            raise ProviderError("provider returned malformed structured output") from None

    def _validated_facts(
        self,
        response: _ExtractionResponse,
        job_description: str,
        rubric: Rubric,
    ) -> list[ExtractedFact]:
        definitions = {fact.id: fact for fact in rubric.facts}
        document = build_evidence_chunks(job_description)
        facts = []
        for returned in response.facts:
            definition = definitions.get(returned.fact_id)
            if definition is None:
                raise ProviderError("provider returned an unknown fact")
            if (
                definition.requires_evidence
                and not returned.is_unknown
                and not returned.evidence
            ):
                raise ProviderError("provider returned a fact without evidence")
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
                raise ProviderError("provider returned invalid evidence") from None
        return facts
