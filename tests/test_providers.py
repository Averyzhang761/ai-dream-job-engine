import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ai_dream_job_engine.evidence import build_evidence_chunks, validate_quote
from ai_dream_job_engine.models import (
    DecisionQuestion,
    FactDefinition,
    Rubric,
    SignalStatus,
)
from ai_dream_job_engine.providers.base import ProviderError
from ai_dream_job_engine.providers.jev import JevSignalProvider
from ai_dream_job_engine.providers.openai_compatible import OpenAICompatibleProvider


class FakeResponses:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def parse(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if isinstance(self.output, str):
            return SimpleNamespace(output_parsed=None, output_text=self.output)
        return SimpleNamespace(output_parsed=self.output, output_text="")


class FakeOpenAIClient:
    def __init__(self, output: Any) -> None:
        self.responses = FakeResponses(output)


def rubric() -> Rubric:
    return Rubric(
        version="rubric-v1",
        facts=[
            FactDefinition(
                id="location",
                name="Location",
                definition="Where the role is based.",
                data_type="text",
                extraction_instructions="Return the stated work location.",
            )
        ],
    )


def questions() -> list[DecisionQuestion]:
    return [
        DecisionQuestion(
            id="applied_ai",
            version="question-v1",
            question="Does this role build applied AI products?",
        ),
        DecisionQuestion(
            id="product_facing",
            version="question-v2",
            question="Is this role product-facing?",
        ),
    ]


def test_openai_direct_uses_official_base_url() -> None:
    provider = OpenAICompatibleProvider(
        provider="openai", api_key="test-key", model="test-model"
    )

    assert str(provider.client.base_url) == "https://api.openai.com/v1/"


def test_openrouter_uses_openrouter_base_url() -> None:
    provider = OpenAICompatibleProvider(
        provider="openrouter", api_key="test-key", model="test-model"
    )

    assert str(provider.client.base_url) == "https://openrouter.ai/api/v1/"


def test_job_extraction_payload_excludes_candidate_profile() -> None:
    client = FakeOpenAIClient(
        {
            "facts": [
                {
                    "fact_id": "location",
                    "value": "Remote",
                    "evidence": ["Remote"],
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=client,
    )

    provider.extract("This role is Remote.", rubric())

    payload = json.dumps(client.responses.calls[0], default=str).lower()
    assert "this role is remote" in payload
    assert "location" in payload
    for forbidden in (
        "candidate_profile",
        "resume",
        "agent memory",
        "email",
        "recommendation",
        "test-key",
    ):
        assert forbidden not in payload


def test_provider_returns_validated_job_facts() -> None:
    client = FakeOpenAIClient(
        {
            "facts": [
                {
                    "fact_id": "location",
                    "value": "Remote",
                    "is_unknown": False,
                    "evidence": ["Remote"],
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=client,
    )

    facts = provider.extract("This role is Remote.", rubric())

    assert facts[0].fact_id == "location"
    assert facts[0].value == "Remote"
    assert facts[0].evidence == ["Remote"]
    assert facts[0].provider == "openai"
    assert facts[0].model == "test-model"


def test_quote_not_in_jd_is_rejected() -> None:
    document = build_evidence_chunks("This role is Remote.")
    reference = validate_quote(document, "Remote")

    assert document == build_evidence_chunks("This role is Remote.")
    assert document.text[reference.start_offset : reference.end_offset] == "Remote"
    assert reference.chunk_id == "chunk-0001"

    with pytest.raises(ValueError, match="not present"):
        validate_quote(document, "Hybrid")

    with pytest.raises(ValueError, match="empty"):
        validate_quote(document, "")

    client = FakeOpenAIClient(
        {
            "facts": [
                {
                    "fact_id": "location",
                    "value": "Hybrid",
                    "evidence": ["Hybrid"],
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(ProviderError, match="evidence"):
        provider.extract("This role is Remote.", rubric())


def test_malformed_structured_output_raises_provider_error() -> None:
    client = FakeOpenAIClient("not-json")
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(ProviderError, match="structured output"):
        provider.extract("This role is Remote.", rubric())


def test_jev_payload_contains_only_jd_and_questions() -> None:
    captured: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"answers": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider = JevSignalProvider(api_key="test-key", client=client)
        provider.judge("Public JD text", questions())

    request = captured[0]
    payload = json.loads(request.content)
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert payload == {
        "model": "jev-latest",
        "input": {
            "job_description": "Public JD text",
            "questions": [question.model_dump() for question in questions()],
        },
    }
    assert request.headers["authorization"] == "Bearer test-key"
    serialized = json.dumps(payload).lower()
    for forbidden in ("candidate", "profile", "resume", "memory", "email"):
        assert forbidden not in serialized


def test_jev_maps_noul_probability() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [
                    {"question_id": "applied_ai", "noul": 0.82},
                    {"question_id": "product_facing", "noul": 0.67},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        signals = JevSignalProvider(api_key="test-key", client=client).judge(
            "Public JD text", questions()
        )

    assert [signal.probability for signal in signals] == [0.82, 0.67]
    assert {signal.status for signal in signals} == {SignalStatus.AVAILABLE}
    assert {signal.provider for signal in signals} == {"jev"}
    assert {signal.model for signal in signals} == {"jev-latest"}


def test_jev_missing_answer_returns_invalid_signal() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [
                    {"question_id": "applied_ai", "noul": "NaN"},
                    {"question_id": "unknown", "noul": 2},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        signals = JevSignalProvider(api_key="test-key", client=client).judge(
            "Public JD text", questions()
        )

    assert [signal.status for signal in signals] == [
        SignalStatus.INVALID,
        SignalStatus.INVALID,
    ]
    assert [signal.probability for signal in signals] == [None, None]


def test_missing_api_key_fails_before_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network transport must not be called")

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    with pytest.raises(ProviderError, match="API key"):
        JevSignalProvider(client=client)
    with pytest.raises(ProviderError, match="API key"):
        OpenAICompatibleProvider(provider="openai", api_key="", model="test-model")
    client.close()

    assert calls == 0
