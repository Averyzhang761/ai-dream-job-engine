from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ai_dream_job_engine.decision import decide_job
from ai_dream_job_engine.models import (
    CandidateClaim,
    CandidateProfile,
    ClaimSource,
    DecisionQuestion,
    DecisionSignal,
    ExtractedFact,
    FactDefinition,
    GapKind,
    RecommendationLevel,
    Rubric,
    ValueDimension,
)
from ai_dream_job_engine.rubric import load_default_rubric


def _rubric() -> Rubric:
    return Rubric(
        version="test-v1",
        facts=[
            FactDefinition(
                id="preferred_role",
                name="Preferred role",
                definition="Whether the role matches the candidate's preference.",
                data_type="boolean",
                extraction_instructions="Extract the preference match.",
                requires_evidence=False,
            ),
            FactDefinition(
                id="python_required",
                name="Python required",
                definition="Whether Python capability is required.",
                data_type="boolean",
                extraction_instructions="Extract the Python requirement.",
                candidate_comparison="exact",
                qualification_weight=1,
            ),
            FactDefinition(
                id="work_authorized",
                name="Work authorization",
                definition="Whether the required authorization must be held.",
                data_type="boolean",
                extraction_instructions="Extract the authorization requirement.",
                candidate_comparison="exact",
                qualification_weight=1,
            ),
        ],
        dimensions=[
            ValueDimension(
                id="preference",
                name="Preference",
                definition="Candidate preference fit.",
                fact_ids=["preferred_role"],
                signal_ids=["preference_signal"],
                weight=10,
                scoring_direction="higher_is_better",
                threshold=0.5,
            ),
            ValueDimension(
                id="python",
                name="Python",
                definition="Python requirement.",
                fact_ids=["python_required"],
                weight=1,
                scoring_direction="higher_is_better",
                threshold=0.5,
            ),
            ValueDimension(
                id="authorization",
                name="Authorization",
                definition="Work authorization constraint.",
                fact_ids=["work_authorized"],
                weight=1,
                scoring_direction="higher_is_better",
                hard_constraint=True,
            ),
        ],
        questions=[
            DecisionQuestion(
                id="preference_signal",
                version="1",
                question="Does the role match the candidate's preference?",
            )
        ],
        recommendation_thresholds={
            RecommendationLevel.PRIORITY_APPLY: 0.8,
            RecommendationLevel.APPLY_AFTER_REVIEW: 0.6,
            RecommendationLevel.HOLD: 0.4,
            RecommendationLevel.SKIP: 0,
        },
    )


def _fact(fact_id: str, value: object, evidence: str) -> ExtractedFact:
    return ExtractedFact(fact_id=fact_id, value=value, evidence=[evidence])


def _claim(
    claim_id: str,
    claim_type: str,
    value: object,
    source: ClaimSource = ClaimSource.RESUME_EVIDENCE,
    *,
    confirmed: bool = True,
) -> CandidateClaim:
    return CandidateClaim(
        id=claim_id,
        claim_type=claim_type,
        value=value,
        source=source,
        confirmed=confirmed,
        source_quote=f"evidence:{claim_id}",
    )


def _facts(*, preferred: bool = True) -> list[ExtractedFact]:
    return [
        _fact("preferred_role", preferred, "JD role evidence"),
        _fact("python_required", True, "JD Python requirement"),
        _fact("work_authorized", True, "JD authorization requirement"),
    ]


def _signals(probability: float = 1) -> list[DecisionSignal]:
    return [
        DecisionSignal(
            question_id="preference_signal",
            question_version="1",
            probability=probability,
            evidence=["classifier preference evidence"],
        )
    ]


def _strong_claims() -> list[CandidateClaim]:
    return [
        _claim("claim-python", "python_required", True),
        _claim(
            "claim-auth",
            "work_authorized",
            True,
            ClaimSource.USER_CONFIRMED,
        ),
    ]


def _with_fact_type(fact: FactDefinition, data_type: str) -> FactDefinition:
    payload = fact.model_dump(mode="python")
    payload["data_type"] = data_type
    return FactDefinition.model_validate(payload)


def _with_candidate_policy(
    fact: FactDefinition,
    candidate_comparison: str,
    qualification_weight: float,
) -> FactDefinition:
    payload = fact.model_dump(mode="python")
    payload.update(
        candidate_comparison=candidate_comparison,
        qualification_weight=qualification_weight,
    )
    return FactDefinition.model_validate(payload)


@pytest.mark.parametrize(
    ("case", "preferred", "claims", "recommendation", "gap_kind"),
    [
        (
            "high preference and confirmed evidence",
            True,
            _strong_claims(),
            RecommendationLevel.PRIORITY_APPLY,
            None,
        ),
        (
            "missing resume evidence",
            True,
            [_strong_claims()[1]],
            RecommendationLevel.APPLY_AFTER_REVIEW,
            GapKind.EVIDENCE_GAP,
        ),
        (
            "confirmed missing capability",
            True,
            [
                _claim("claim-python", "python_required", False),
                _strong_claims()[1],
            ],
            RecommendationLevel.HOLD,
            GapKind.CAPABILITY_GAP,
        ),
        (
            "low preference despite strong capability",
            False,
            _strong_claims(),
            RecommendationLevel.SKIP,
            GapKind.PREFERENCE_MISMATCH,
        ),
        (
            "confirmed authorization conflict",
            True,
            [
                _strong_claims()[0],
                _claim(
                    "claim-auth-conflict",
                    "work_authorized",
                    False,
                    ClaimSource.USER_CONFIRMED,
                ),
            ],
            RecommendationLevel.SKIP,
            GapKind.HARD_BLOCKER,
        ),
    ],
)
def test_decision_table(
    case: str,
    preferred: bool,
    claims: list[CandidateClaim],
    recommendation: RecommendationLevel,
    gap_kind: GapKind | None,
) -> None:
    result = decide_job(
        _facts(preferred=preferred),
        _signals(float(preferred)),
        CandidateProfile(claims=claims),
        _rubric(),
    )

    assert result.recommendation is recommendation, case
    assert [gap.kind for gap in result.gaps] == (
        [] if gap_kind is None else [gap_kind]
    )
    assert all(gap.input_ids for gap in result.gaps)
    assert all(
        gap.job_evidence or gap.candidate_evidence or gap.missing_evidence
        for gap in result.gaps
    )


def test_unconfirmed_agent_suggestion_changes_neither_score_nor_gaps() -> None:
    baseline_profile = CandidateProfile(claims=[_strong_claims()[1]])
    suggested_profile = baseline_profile.model_copy(
        update={
            "claims": [
                *baseline_profile.claims,
                _claim(
                    "agent-python",
                    "python_required",
                    True,
                    ClaimSource.AGENT_SUGGESTED,
                    confirmed=False,
                ),
            ]
        }
    )

    baseline = decide_job(_facts(), _signals(), baseline_profile, _rubric())
    suggested = decide_job(_facts(), _signals(), suggested_profile, _rubric())

    assert suggested.preference_score == baseline.preference_score
    assert suggested.qualification_score == baseline.qualification_score
    assert suggested.gaps == baseline.gaps


def test_unknown_required_fact_requires_review_without_counting_as_zero() -> None:
    facts = _facts()
    facts[1] = ExtractedFact(
        fact_id="python_required",
        is_unknown=True,
        evidence=["JD does not state whether Python is required"],
    )

    result = decide_job(
        facts,
        _signals(),
        CandidateProfile(claims=_strong_claims()),
        _rubric(),
    )

    assert result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert result.needs_review is True
    assert result.preference_score == 1
    assert result.qualification_score == 1
    assert [gap.kind for gap in result.gaps] == [GapKind.EVIDENCE_GAP]


def test_normalized_input_order_produces_byte_identical_result() -> None:
    facts = _facts()
    signals = _signals()
    claims = _strong_claims()

    first = decide_job(
        facts,
        signals,
        CandidateProfile(claims=claims),
        _rubric(),
    )
    second = decide_job(
        list(reversed(facts)),
        list(reversed(signals)),
        CandidateProfile(claims=list(reversed(claims))),
        _rubric(),
    )

    assert first.model_dump_json() == second.model_dump_json()
    assert first.preference_contributions
    assert first.qualification_contributions
    assert all(item.input_ids for item in first.preference_contributions)
    assert all(
        item.job_evidence or item.candidate_evidence
        for item in [
            *first.preference_contributions,
            *first.qualification_contributions,
        ]
    )


def test_result_is_byte_identical_across_processes() -> None:
    script = """
import runpy
ns = runpy.run_path('tests/test_decision.py')
result = ns['decide_job'](
    ns['_facts'](),
    ns['_signals'](),
    ns['CandidateProfile'](claims=ns['_strong_claims']()),
    ns['_rubric'](),
)
print(result.model_dump_json())
"""
    root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for _ in range(2)
    ]

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("candidate_value", ["unknown", "unavailable", "invalid"])
def test_unknown_candidate_text_requires_review_not_capability_gap(
    candidate_value: str,
) -> None:
    rubric = _rubric()
    rubric.facts[1] = _with_fact_type(rubric.facts[1], "text")
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(
        update={"positive_keywords": ["python"]}
    )
    facts = _facts()
    facts[1] = _fact("python_required", "python", "JD Python requirement")
    claims = [
        _claim("claim-python", "python_required", candidate_value),
        _strong_claims()[1],
    ]

    result = decide_job(facts, _signals(), CandidateProfile(claims=claims), rubric)

    assert result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert result.qualification_score == 1
    assert [gap.kind for gap in result.gaps] == [GapKind.EVIDENCE_GAP]


@pytest.mark.parametrize(
    ("data_type", "job_value", "candidate_value"),
    [
        ("text", ["python"], "python"),
        ("list", "python", ["python"]),
        ("text", "python", ["python"]),
        ("list", ["python"], "python"),
    ],
)
def test_text_and_list_values_must_match_fact_definition(
    data_type: str,
    job_value: object,
    candidate_value: object,
) -> None:
    rubric = _rubric()
    rubric.facts[1] = _with_fact_type(rubric.facts[1], data_type)
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(
        update={"positive_keywords": ["python"]}
    )
    facts = _facts()
    facts[1] = _fact("python_required", job_value, "JD Python requirement")
    claims = [
        _claim("claim-python", "python_required", candidate_value),
        _strong_claims()[1],
    ]

    with pytest.raises(ValueError, match="python_required"):
        decide_job(facts, _signals(), CandidateProfile(claims=claims), rubric)


def test_absent_source_evidence_is_explicit_and_never_fabricated() -> None:
    facts = _facts()
    facts[1] = ExtractedFact(fact_id="python_required", is_unknown=True)
    claims = [
        CandidateClaim(
            id="claim-auth",
            claim_type="work_authorized",
            value=True,
            source=ClaimSource.USER_CONFIRMED,
            confirmed=True,
        )
    ]

    result = decide_job(facts, _signals(), CandidateProfile(claims=claims), _rubric())
    python_gap = next(gap for gap in result.gaps if gap.subject_id == "python_required")

    assert python_gap.missing_evidence is True
    assert python_gap.job_evidence == []
    assert python_gap.candidate_evidence == []
    assert python_gap.input_ids == ["python_required"]
    assert all(
        not evidence.startswith(("fact:", "signal:", "user_confirmed:"))
        for gap in result.gaps
        for evidence in [*gap.job_evidence, *gap.candidate_evidence]
    )


def test_confirmed_claim_without_source_evidence_requires_review() -> None:
    claims = [
        CandidateClaim(
            id="claim-python",
            claim_type="python_required",
            value=True,
            source=ClaimSource.RESUME_EVIDENCE,
            confirmed=True,
        ),
        _strong_claims()[1],
    ]

    result = decide_job(_facts(), _signals(), CandidateProfile(claims=claims), _rubric())
    gap = next(gap for gap in result.gaps if gap.subject_id == "python_required")

    assert gap.kind is GapKind.EVIDENCE_GAP
    assert gap.missing_evidence is True
    assert gap.candidate_evidence == []
    assert result.qualification_score == 1


def test_duplicate_candidate_claim_ids_are_rejected_before_selection() -> None:
    claims = [
        _claim("duplicate", "python_required", True),
        _claim("duplicate", "work_authorized", True),
    ]

    with pytest.raises(ValueError, match="duplicate candidate claim id 'duplicate'"):
        decide_job(_facts(), _signals(), CandidateProfile(claims=claims), _rubric())


def test_unreferenced_required_fact_still_requires_review() -> None:
    rubric = _rubric()
    rubric.facts.append(
        FactDefinition(
            id="unreferenced_required",
            name="Unreferenced requirement",
            definition="A required fact outside all dimensions.",
            data_type="boolean",
            extraction_instructions="Extract the requirement.",
            candidate_comparison="exact",
            qualification_weight=1,
        )
    )

    result = decide_job(_facts(), _signals(), CandidateProfile(claims=_strong_claims()), rubric)
    gap = next(gap for gap in result.gaps if gap.subject_id == "unreferenced_required")

    assert result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert result.needs_review is True
    assert gap.kind is GapKind.EVIDENCE_GAP
    assert gap.missing_evidence is True


def test_dimension_threshold_equality_passes_and_above_boundary_fails() -> None:
    facts = _facts(preferred=False)
    signals = _signals(1)
    at_boundary = _rubric()
    above_boundary = _rubric()
    above_boundary.dimensions[0] = above_boundary.dimensions[0].model_copy(
        update={"threshold": 0.500001}
    )

    passing = decide_job(
        facts, signals, CandidateProfile(claims=_strong_claims()), at_boundary
    )
    failing = decide_job(
        facts, signals, CandidateProfile(claims=_strong_claims()), above_boundary
    )

    passing_preference = next(
        item for item in passing.preference_contributions if item.subject_id == "preference"
    )
    failing_preference = next(
        item for item in failing.preference_contributions if item.subject_id == "preference"
    )
    assert passing_preference.score == 1
    assert passing.recommendation is RecommendationLevel.PRIORITY_APPLY
    assert failing_preference.score == 0
    assert failing.recommendation is RecommendationLevel.SKIP


def test_zero_weight_dimension_is_disabled() -> None:
    rubric = _rubric()
    rubric.dimensions[0] = rubric.dimensions[0].model_copy(update={"weight": 0})
    facts = [fact for fact in _facts() if fact.fact_id != "preferred_role"]

    result = decide_job(
        facts, [], CandidateProfile(claims=_strong_claims()), rubric
    )

    assert result.recommendation is RecommendationLevel.PRIORITY_APPLY
    assert all(item.subject_id != "preference" for item in result.preference_contributions)
    assert all(gap.subject_id not in {"preferred_role", "preference_signal"} for gap in result.gaps)


def test_zero_weight_fact_without_qualification_policy_is_disabled() -> None:
    rubric = _rubric()
    rubric.facts[1] = _with_candidate_policy(rubric.facts[1], "none", 0)
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(update={"weight": 0})
    facts = [fact for fact in _facts() if fact.fact_id != "python_required"]
    claims = [_strong_claims()[1]]

    result = decide_job(facts, _signals(), CandidateProfile(claims=claims), rubric)

    assert all(gap.subject_id != "python_required" for gap in result.gaps)
    assert all(
        item.subject_id != "python_required"
        for item in result.qualification_contributions
    )


def test_decision_rejects_hard_constraint_without_candidate_comparison() -> None:
    rubric = _rubric()
    rubric.facts[2] = _with_candidate_policy(rubric.facts[2], "none", 0)

    with pytest.raises(ValueError, match="hard_constraint.*candidate_comparison"):
        decide_job(_facts(), _signals(), CandidateProfile(claims=_strong_claims()), rubric)


def test_explicit_qualification_survives_zero_weight_dimension() -> None:
    rubric = _rubric()
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(update={"weight": 0})
    claims = [
        _claim("claim-python", "python_required", False),
        _strong_claims()[1],
    ]

    result = decide_job(_facts(), _signals(), CandidateProfile(claims=claims), rubric)

    assert any(gap.kind is GapKind.CAPABILITY_GAP for gap in result.gaps)


def test_stale_signal_question_version_requires_review() -> None:
    stale_signal = DecisionSignal(
        question_id="preference_signal",
        question_version="stale",
        probability=1,
        evidence=["stale classifier output"],
    )

    result = decide_job(
        _facts(),
        [stale_signal],
        CandidateProfile(claims=_strong_claims()),
        _rubric(),
    )

    gap = next(gap for gap in result.gaps if gap.subject_id == "preference_signal")
    assert result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert gap.job_evidence == ["stale classifier output"]
    assert all(
        "preference_signal" not in contribution.input_ids
        for contribution in result.preference_contributions
    )


def test_inputs_are_not_mutated() -> None:
    facts = _facts()
    signals = _signals()
    profile = CandidateProfile(claims=_strong_claims())
    rubric = _rubric()
    before = deepcopy((facts, signals, profile, rubric))

    decide_job(facts, signals, profile, rubric)

    assert (facts, signals, profile, rubric) == before


def test_complete_favorable_default_rubric_fixture_is_priority_apply() -> None:
    rubric = load_default_rubric()
    values = {
        "role_family": "applied AI",
        "ai_scope": "LLM evaluation",
        "product_customer_scope": "customers",
        "ownership_scope": "own launch",
        "infrastructure_scope": "model serving",
        "work_arrangement_terms": ["remote"],
        "compensation_terms": "salary range",
        "work_authorization_requirement": ["sponsorship available"],
        "required_skills": ["Python", "RAG"],
        "minimum_experience_years": 5,
        "applied_ai_requirements": ["evaluation", "agents"],
    }
    facts = [
        _fact(fact.id, values[fact.id], f"JD evidence:{fact.id}")
        for fact in rubric.facts
    ]
    claims = [
        _claim(f"claim:{fact.id}", fact.id, values[fact.id])
        for fact in rubric.facts
    ]
    signals = [
        DecisionSignal(
            question_id=question.id,
            question_version=question.version,
            probability=0 if question.id == "pure_infrastructure" else 1,
            evidence=[f"signal evidence:{question.id}"],
        )
        for question in rubric.questions
    ]

    result = decide_job(facts, signals, CandidateProfile(claims=claims), rubric)

    assert result.recommendation is RecommendationLevel.PRIORITY_APPLY
    assert result.preference_score == 1
    assert result.qualification_score == 1
    assert result.gaps == []


def test_available_version_matched_signal_needs_no_quote() -> None:
    rubric = _rubric()
    result = decide_job(
        _facts(),
        [
            DecisionSignal(
                question_id="preference_signal",
                question_version="1",
                probability=0.9,
                provider="jev",
                model="jev-latest",
            )
        ],
        CandidateProfile(claims=_strong_claims()),
        rubric,
    )

    contribution = next(
        item for item in result.preference_contributions if item.subject_id == "preference"
    )
    assert contribution.score == 1
    assert contribution.job_evidence == ["JD role evidence"]
    assert contribution.audit_support == [
        "signal:jev:jev-latest:preference_signal:1"
    ]


@pytest.mark.parametrize("unknown", [False, True])
def test_configured_default_is_typed_and_auditable(unknown: bool) -> None:
    rubric = _rubric()
    rubric.facts[0] = rubric.facts[0].model_copy(
        update={"missing_policy": "default", "default_value": True}
    )
    facts = [item for item in _facts() if item.fact_id != "preferred_role"]
    if unknown:
        facts.append(ExtractedFact(fact_id="preferred_role", is_unknown=True))

    result = decide_job(
        facts,
        _signals(),
        CandidateProfile(claims=_strong_claims()),
        rubric,
    )

    contribution = next(
        item for item in result.preference_contributions if item.subject_id == "preference"
    )
    assert contribution.score == 1
    assert contribution.job_evidence == ["classifier preference evidence"]
    assert contribution.audit_support == ["configured-default:preferred_role"]


def test_default_rubric_material_qualifications_report_missing_and_conflict() -> None:
    rubric = load_default_rubric()
    values = {
        "role_family": "applied AI",
        "ai_scope": "LLM evaluation",
        "product_customer_scope": "customers",
        "ownership_scope": "own launch",
        "infrastructure_scope": "model serving",
        "work_arrangement_terms": ["remote"],
        "compensation_terms": "salary range",
        "work_authorization_requirement": [],
        "required_skills": ["Python", "RAG"],
        "minimum_experience_years": 5,
        "applied_ai_requirements": ["evaluation", "agents"],
    }
    facts = [
        _fact(fact.id, values[fact.id], f"JD evidence:{fact.id}")
        for fact in rubric.facts
    ]
    profile = CandidateProfile(
        claims=[
            _claim("skills", "required_skills", ["Python"]),
            _claim("experience", "minimum_experience_years", 3),
            _claim("applied-ai", "applied_ai_requirements", ["evaluation"]),
        ]
    )

    result = decide_job(facts, [], profile, rubric)

    assert {
        gap.subject_id
        for gap in result.gaps
        if gap.kind is GapKind.CAPABILITY_GAP
    } == {
        "required_skills",
        "minimum_experience_years",
        "applied_ai_requirements",
    }

    missing = decide_job(facts, [], CandidateProfile(), rubric)
    assert {
        gap.subject_id
        for gap in missing.gaps
        if gap.kind is GapKind.EVIDENCE_GAP
    } >= {
        "required_skills",
        "minimum_experience_years",
        "applied_ai_requirements",
    }


def test_default_preference_and_synonym_differences_are_not_capability_gaps() -> None:
    rubric = load_default_rubric()
    values = {
        "role_family": "applied AI",
        "ai_scope": "LLM evaluation",
        "product_customer_scope": "customers",
        "ownership_scope": "own launch",
        "infrastructure_scope": "model serving",
        "work_arrangement_terms": ["remote"],
        "compensation_terms": "salary range",
        "work_authorization_requirement": ["sponsorship available"],
        "required_skills": ["Python", "RAG"],
        "minimum_experience_years": 5,
        "applied_ai_requirements": ["evaluation", "agents"],
    }
    candidate_values = {
        **values,
        "role_family": "machine learning engineer",
        "work_arrangement_terms": ["onsite"],
        "compensation_terms": "different compensation preference",
    }
    facts = [
        _fact(fact.id, values[fact.id], f"JD evidence:{fact.id}")
        for fact in rubric.facts
    ]
    claims = [
        _claim(f"claim:{fact.id}", fact.id, candidate_values[fact.id])
        for fact in rubric.facts
    ]
    signals = [
        DecisionSignal(
            question_id=question.id,
            question_version=question.version,
            probability=0 if question.id == "pure_infrastructure" else 1,
            evidence=[f"signal evidence:{question.id}"],
        )
        for question in rubric.questions
    ]

    result = decide_job(facts, signals, CandidateProfile(claims=claims), rubric)

    assert result.recommendation is RecommendationLevel.PRIORITY_APPLY
    assert all(gap.kind is not GapKind.CAPABILITY_GAP for gap in result.gaps)


def test_empty_default_authorization_list_is_unrestricted() -> None:
    rubric = load_default_rubric()
    fact = _fact(
        "work_authorization_requirement",
        [],
        "JD states no authorization restriction",
    )
    other_facts = [
        ExtractedFact(fact_id=item.id, is_unknown=True, evidence=["not needed here"])
        for item in rubric.facts
        if item.id != fact.fact_id
    ]

    result = decide_job(
        [fact, *other_facts],
        [],
        CandidateProfile(),
        rubric,
    )

    authorization = next(
        item
        for item in result.qualification_contributions
        if item.subject_id == "work_authorization_requirement"
    )
    assert authorization.score == 1
    assert authorization.input_ids == ["work_authorization_requirement"]
    assert authorization.job_evidence == ["JD states no authorization restriction"]
    assert authorization.candidate_evidence == []
    assert all(
        gap.subject_id != "work_authorization_requirement" for gap in result.gaps
    )


def test_nonempty_allowed_authorization_requires_candidate_overlap() -> None:
    rubric = load_default_rubric()
    facts = [
        ExtractedFact(fact_id=item.id, is_unknown=True, evidence=["not needed here"])
        for item in rubric.facts
    ]
    auth_index = next(
        index
        for index, fact in enumerate(facts)
        if fact.fact_id == "work_authorization_requirement"
    )
    facts[auth_index] = _fact(
        "work_authorization_requirement",
        ["citizen"],
        "JD requires citizenship",
    )
    claim = _claim(
        "claim-auth",
        "work_authorization_requirement",
        ["requires sponsorship"],
        ClaimSource.USER_CONFIRMED,
    )

    result = decide_job(facts, [], CandidateProfile(claims=[claim]), rubric)

    assert result.recommendation is RecommendationLevel.SKIP
    assert any(gap.kind is GapKind.HARD_BLOCKER for gap in result.gaps)


def test_hard_blocker_preserves_simultaneous_evidence_gap_and_review_state() -> None:
    claims = [
        _claim(
            "claim-auth",
            "work_authorized",
            False,
            ClaimSource.USER_CONFIRMED,
        )
    ]

    result = decide_job(_facts(), _signals(), CandidateProfile(claims=claims), _rubric())

    assert result.recommendation is RecommendationLevel.SKIP
    assert result.needs_review is True
    assert {gap.kind for gap in result.gaps} == {
        GapKind.EVIDENCE_GAP,
        GapKind.HARD_BLOCKER,
    }


def test_empty_confirmed_list_is_an_explicit_capability_gap() -> None:
    rubric = _rubric()
    rubric.facts[1] = _with_fact_type(rubric.facts[1], "list")
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(
        update={"positive_keywords": ["python"]}
    )
    facts = _facts()
    facts[1] = _fact("python_required", ["python"], "JD Python requirement")
    claims = [
        _claim("claim-python", "python_required", []),
        _strong_claims()[1],
    ]

    result = decide_job(facts, _signals(), CandidateProfile(claims=claims), rubric)

    assert any(gap.kind is GapKind.CAPABILITY_GAP for gap in result.gaps)


def test_list_with_unknown_candidate_value_requires_review() -> None:
    rubric = _rubric()
    rubric.facts[1] = _with_fact_type(rubric.facts[1], "list")
    rubric.dimensions[1] = rubric.dimensions[1].model_copy(
        update={"positive_keywords": ["python"]}
    )
    facts = _facts()
    facts[1] = _fact("python_required", ["python"], "JD Python requirement")
    claims = [
        _claim("claim-python", "python_required", ["python", "unknown"]),
        _strong_claims()[1],
    ]

    result = decide_job(facts, _signals(), CandidateProfile(claims=claims), rubric)

    assert result.recommendation is RecommendationLevel.APPLY_AFTER_REVIEW
    assert any(gap.kind is GapKind.EVIDENCE_GAP for gap in result.gaps)
