import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import ai_dream_job_engine.providers.jev as jev_module
import ai_dream_job_engine.providers.openai_compatible as openai_module
from ai_dream_job_engine.evidence import build_evidence_chunks, validate_quote
from ai_dream_job_engine.models import (
    DecisionQuestion,
    EvidenceChunk,
    EvidenceDocument,
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


class RaisingResponses:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def parse(self, **_: Any) -> Any:
        raise self.error


class RaisingOpenAIClient:
    def __init__(self, error: BaseException) -> None:
        self.responses = RaisingResponses(error)


class RaisingHttpClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def post(self, *_: Any, **__: Any) -> Any:
        raise self.error


class BrokenParsedResponse:
    @property
    def output_parsed(self) -> Any:
        raise RuntimeError("SECRET_KEY JD SECRET")


class BrokenDecoderClient:
    class Responses:
        def parse(self, **_: Any) -> BrokenParsedResponse:
            return BrokenParsedResponse()

    responses = Responses()


def raise_nested_secret_error() -> None:
    try:
        raise ValueError("INNER SECRET")
    except ValueError as inner:
        raise RuntimeError("OUTER SECRET") from inner


class NestedErrorOpenAIClient:
    class Responses:
        def parse(self, **_: Any) -> Any:
            raise_nested_secret_error()

    responses = Responses()


class NestedErrorHttpClient:
    def post(self, *_: Any, **__: Any) -> Any:
        raise_nested_secret_error()


def assert_no_exception_chain(error: ProviderError) -> None:
    assert error.__context__ is None
    assert error.__cause__ is None


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


def provider_with_output(output: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=FakeOpenAIClient(output),
    )


def test_openai_direct_uses_official_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://hostile.invalid/v1")
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


@pytest.mark.parametrize("is_unknown", ["false", "true", 0, 1])
def test_job_fact_is_unknown_requires_strict_boolean(is_unknown: Any) -> None:
    output = {
        "facts": [
            {
                "fact_id": "location",
                "value": "Remote",
                "is_unknown": is_unknown,
                "evidence": ["Remote"],
            }
        ]
    }

    with pytest.raises(ProviderError, match="malformed structured output"):
        provider_with_output(output).extract("Remote", rubric())


@pytest.mark.parametrize(
    ("data_type", "value", "enum_values"),
    [
        ("boolean", "yes", []),
        ("number", True, []),
        ("number", float("nan"), []),
        ("number", float("inf"), []),
        ("text", {"private": "shape"}, []),
        ("enum", "hybrid", ["remote", "onsite"]),
        ("list", "remote", []),
    ],
)
def test_job_fact_values_are_strictly_validated(
    data_type: str, value: Any, enum_values: list[str]
) -> None:
    definition = FactDefinition(
        id="fact",
        name="Fact",
        definition="A strictly typed fact.",
        data_type=data_type,
        extraction_instructions="Extract the fact.",
        enum_values=enum_values,
    )
    result = {
        "facts": [
            {
                "fact_id": "fact",
                "value": value,
                "evidence": ["Remote"],
            }
        ]
    }

    with pytest.raises(ProviderError, match="invalid fact value"):
        provider_with_output(result).extract(
            "Remote", Rubric(version="v1", facts=[definition])
        )


@pytest.mark.parametrize(
    "facts",
    [
        [],
        [
            {"fact_id": "location", "value": "Remote", "evidence": ["Remote"]},
            {"fact_id": "location", "value": "Remote", "evidence": ["Remote"]},
        ],
    ],
)
def test_job_fact_set_must_be_complete_and_unique(facts: list[dict[str, Any]]) -> None:
    with pytest.raises(ProviderError, match="fact set"):
        provider_with_output({"facts": facts}).extract("Remote", rubric())


@pytest.mark.parametrize(
    "contradiction",
    [
        {"value": "Remote", "evidence": []},
        {"value": None, "evidence": ["Remote"]},
    ],
)
def test_unknown_job_fact_cannot_have_value_or_evidence(
    contradiction: dict[str, Any],
) -> None:
    output = {
        "facts": [
            {"fact_id": "location", "is_unknown": True, **contradiction}
        ]
    }

    with pytest.raises(ProviderError, match="unknown fact"):
        provider_with_output(output).extract("Remote", rubric())


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

    with pytest.raises(ProviderError, match="evidence") as error:
        provider.extract("This role is Remote.", rubric())
    assert_no_exception_chain(error.value)


@pytest.mark.parametrize(
    "chunks",
    [
        [
            EvidenceChunk(id="same", text="abc", start_offset=0, end_offset=3),
            EvidenceChunk(id="same", text="def", start_offset=3, end_offset=6),
        ],
        [EvidenceChunk(id="wrong", text="xxxxxx", start_offset=0, end_offset=6)],
        [EvidenceChunk(id="gap", text="def", start_offset=3, end_offset=6)],
        [
            EvidenceChunk(id="one", text="abcd", start_offset=0, end_offset=4),
            EvidenceChunk(id="two", text="def", start_offset=3, end_offset=6),
        ],
    ],
)
def test_evidence_document_rejects_invalid_chunk_layout(
    chunks: list[EvidenceChunk],
) -> None:
    with pytest.raises(ValidationError):
        EvidenceDocument(text="abcdef", chunks=chunks)


def test_evidence_chunk_rejects_empty_span() -> None:
    with pytest.raises(ValidationError):
        EvidenceChunk(id="empty", text="", start_offset=0, end_offset=0)


def test_validate_quote_checks_the_selected_chunk_text() -> None:
    document = EvidenceDocument.model_construct(
        text="secret Remote text",
        chunks=[
            EvidenceChunk(
                id="wrong",
                text="x" * 18,
                start_offset=0,
                end_offset=18,
            )
        ],
    )

    with pytest.raises(ValueError, match="chunk"):
        validate_quote(document, "Remote")


def test_malformed_structured_output_raises_provider_error() -> None:
    client = FakeOpenAIClient("not-json")
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=client,
    )

    with pytest.raises(ProviderError, match="structured output") as error:
        provider.extract("This role is Remote.", rubric())
    assert_no_exception_chain(error.value)


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
                    {
                        "question_id": "product_facing",
                        "question_version": "question-v2",
                        "noul": 0.67,
                    },
                    {
                        "question_id": "applied_ai",
                        "question_version": "question-v1",
                        "noul": 0.82,
                    },
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
                    {
                        "question_id": "applied_ai",
                        "question_version": "question-v1",
                    },
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


def test_jev_rejects_duplicate_input_ids_before_request() -> None:
    calls = 0

    def respond(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"answers": []})

    duplicate_questions = [
        DecisionQuestion(id="same", version="v1", question="First?"),
        DecisionQuestion(id="same", version="v2", question="Second?"),
    ]
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError, match="duplicate question ID"):
            JevSignalProvider(api_key="test-key", client=client).judge(
                "Public JD", duplicate_questions
            )

    assert calls == 0


@pytest.mark.parametrize(
    "answer",
    [
        {"question_id": "unknown", "question_version": "v1", "noul": 0.5},
        {
            "question_id": "applied_ai",
            "question_version": "wrong",
            "noul": 0.5,
        },
        {
            "question_id": "applied_ai",
            "question_version": "question-v1",
            "noul": 0.5,
            "extra": "not allowed",
        },
    ],
)
def test_jev_rejects_unbound_or_malformed_answers(answer: dict[str, Any]) -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": [answer]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError, match="invalid response") as error:
            JevSignalProvider(api_key="test-key", client=client).judge(
                "Public JD", questions()
            )
    assert_no_exception_chain(error.value)


def test_jev_rejects_duplicate_returned_answers() -> None:
    answer = {
        "question_id": "applied_ai",
        "question_version": "question-v1",
        "noul": 0.5,
    }

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": [answer, answer]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError, match="invalid response"):
            JevSignalProvider(api_key="test-key", client=client).judge(
                "Public JD", questions()
            )


def test_provider_client_exceptions_are_redacted() -> None:
    leaked = "SECRET_KEY JD SECRET"
    openai_provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=RaisingOpenAIClient(RuntimeError(leaked)),
    )

    with pytest.raises(ProviderError) as openai_error:
        openai_provider.extract("JD SECRET", rubric())
    assert str(openai_error.value) == "job fact provider request failed"
    assert leaked not in repr(openai_error.value)
    assert_no_exception_chain(openai_error.value)

    decoding_provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=BrokenDecoderClient(),
    )
    with pytest.raises(ProviderError) as decoding_error:
        decoding_provider.extract("JD SECRET", rubric())
    assert str(decoding_error.value) == "provider returned malformed structured output"
    assert leaked not in repr(decoding_error.value)
    assert_no_exception_chain(decoding_error.value)

    with pytest.raises(ProviderError) as jev_error:
        JevSignalProvider(
            api_key="SECRET_KEY",
            client=RaisingHttpClient(RuntimeError(leaked)),  # type: ignore[arg-type]
        ).judge("JD SECRET", questions())
    assert str(jev_error.value) == "JEV provider request failed"
    assert leaked not in repr(jev_error.value)
    assert_no_exception_chain(jev_error.value)


def test_sanitized_provider_errors_have_no_exception_chain() -> None:
    openai_provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=NestedErrorOpenAIClient(),
    )
    with pytest.raises(ProviderError) as openai_error:
        openai_provider.extract("JD SECRET", rubric())
    assert_no_exception_chain(openai_error.value)

    with pytest.raises(ProviderError) as jev_error:
        JevSignalProvider(
            api_key="SECRET_KEY",
            client=NestedErrorHttpClient(),  # type: ignore[arg-type]
        ).judge("JD SECRET", questions())
    assert_no_exception_chain(jev_error.value)


def test_provider_client_initialization_exceptions_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked = "SECRET_KEY"

    def fail_client(**_: Any) -> Any:
        raise RuntimeError(leaked)

    monkeypatch.setattr(openai_module, "OpenAI", fail_client)
    with pytest.raises(ProviderError) as openai_error:
        OpenAICompatibleProvider(
            provider="openai", api_key=leaked, model="test-model"
        )
    assert str(openai_error.value) == "provider client initialization failed"
    assert leaked not in repr(openai_error.value)
    assert_no_exception_chain(openai_error.value)

    monkeypatch.setattr(jev_module.httpx, "Client", fail_client)
    with pytest.raises(ProviderError) as jev_error:
        JevSignalProvider(api_key=leaked)
    assert str(jev_error.value) == "JEV client initialization failed"
    assert leaked not in repr(jev_error.value)
    assert_no_exception_chain(jev_error.value)


@pytest.mark.parametrize(
    "control_error", [KeyboardInterrupt(), SystemExit(), GeneratorExit()]
)
def test_provider_process_control_exceptions_escape(
    control_error: BaseException,
) -> None:
    provider = OpenAICompatibleProvider(
        provider="openai",
        api_key="test-key",
        model="test-model",
        client=RaisingOpenAIClient(control_error),
    )

    with pytest.raises(type(control_error)):
        provider.extract("Public JD", rubric())

    with pytest.raises(type(control_error)):
        JevSignalProvider(
            api_key="test-key",
            client=RaisingHttpClient(control_error),  # type: ignore[arg-type]
        ).judge("Public JD", questions())


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
