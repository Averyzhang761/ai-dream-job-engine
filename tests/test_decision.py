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
            ),
            FactDefinition(
                id="work_authorized",
                name="Work authorization",
                definition="Whether the required authorization must be held.",
                data_type="boolean",
                extraction_instructions="Extract the authorization requirement.",
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
    assert all(gap.job_evidence or gap.candidate_evidence for gap in result.gaps)


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
