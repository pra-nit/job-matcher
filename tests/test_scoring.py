"""
Scoring + verdict tests (sections 17-19): deterministic final score, hard
rejection precedence, Qwen guardrails, explainability.
"""
from __future__ import annotations

import pytest

from llm.schemas import JobAnalysis, JobRequirements
from matching import skill_taxonomy
from matching.scoring import (
    compute_final_score,
    determine_verdict,
    reconcile,
    score_job,
)
from models import (
    ExperienceEvaluation,
    ExperienceFit,
    JobMatchResult,
    JobRecord,
    Rejection,
    RejectionReason,
    Verdict,
)


def _job(**kwargs) -> JobRecord:
    defaults = dict(
        company="TestCorp", title="Edge AI Engineer", location="Bangalore, India",
        url="https://example.com/1", source="linkedin", sources=["linkedin"],
        description="Deploy models with ONNX and TensorRT. 3-5 years experience.",
        content_hash="abc",
    )
    defaults.update(kwargs)
    return JobRecord(**defaults)


def _requirements(**kwargs) -> JobRequirements:
    defaults = dict(
        min_experience_years=3, max_experience_years=5,
        experience_domain="model_deployment",
        required_skills=["Python", "ONNX", "TensorRT", "Quantization"],
        preferred_skills=["CUDA"],
        seniority="senior",
        extraction_method="regex",
    )
    defaults.update(kwargs)
    return JobRequirements(**defaults)


def _experience(fit=ExperienceFit.STRONG_MATCH, **kwargs) -> ExperienceEvaluation:
    defaults = dict(
        effective_candidate_years=4.0, basis="direct:model_deployment",
        job_min_years=3, job_max_years=5, gap=1.0, ratio=1.0, confidence=0.95,
    )
    defaults.update(kwargs)
    return ExperienceEvaluation(fit=fit, **defaults)


class TestFinalScore:
    def test_weighted_sum(self):
        components = {
            "experience": 100, "skills": 100, "embedding": 100,
            "responsibility": 100, "domain": 100, "career_alignment": 100,
        }
        weights = {"experience": 0.30, "skills": 0.25, "embedding": 0.20,
                   "responsibility": 0.10, "domain": 0.10, "career_alignment": 0.05}
        assert compute_final_score(components, weights) == 100.0

    def test_weights_normalised_automatically(self):
        score = compute_final_score(
            {"experience": 100, "skills": 0, "embedding": 0, "responsibility": 0,
             "domain": 0, "career_alignment": 0},
            {"experience": 0.60, "skills": 0.50, "embedding": 0.40,
             "responsibility": 0.20, "domain": 0.20, "career_alignment": 0.10},
        )
        # experience share of the (normalised) weights is 0.60/2.00 = 0.30
        assert score == pytest.approx(30.0)

    def test_spec_formula(self, cfg, profile_10y):
        requirements = _requirements()
        skills = skill_taxonomy.match_skills(
            requirements.required_skills + requirements.preferred_skills,
            profile_10y.core_skills,
            required=requirements.required_skills,
            preferred=requirements.preferred_skills,
        )
        result = score_job(
            profile_10y, _job(), requirements,
            _experience(), skills,
            embedding_similarity=0.75, rejections=[], qwen_analysis=None, cfg=cfg,
        )
        # strong spec-31 job A scenario: must be high scoring and a MATCH
        assert result.final_score >= 75
        assert result.verdict == Verdict.MATCH
        assert result.embedding_score == 80.0  # 0.75 cosine on 0.35-0.85 scale


class TestHardRejectionPrecedence:
    def test_high_score_cannot_rescue_hard_rejection(self, cfg, profile_10y):
        """RULE 3: Qwen (or the score) cannot override deterministic rejection."""
        requirements = _requirements()
        skills = skill_taxonomy.match_skills([], profile_10y.core_skills)
        result = score_job(
            profile_10y, _job(), requirements, _experience(), skills,
            embedding_similarity=0.95,
            rejections=[Rejection(RejectionReason.UNDERQUALIFIED, "4 < 8 required")],
            qwen_analysis=JobAnalysis(match_score=100, reasoning="Excellent candidate!"),
            cfg=cfg,
        )
        assert result.hard_rejected
        assert result.verdict == Verdict.REJECTED

    def test_overqualified_rejection_not_rescued(self, cfg, profile_10y):
        result = score_job(
            profile_10y, _job(), _requirements(),
            _experience(fit=ExperienceFit.OVERQUALIFIED, effective_candidate_years=12,
                        job_min_years=3, job_max_years=4, gap=8, ratio=3.0),
            skill_taxonomy.match_skills([], []),
            embedding_similarity=0.9,
            rejections=[Rejection(RejectionReason.OVERQUALIFIED, "12 vs 3-4 years")],
            qwen_analysis=JobAnalysis(match_score=95),
            cfg=cfg,
        )
        assert result.verdict == Verdict.REJECTED


class TestVerdictThresholds:
    def _result(self, final_score, fit=ExperienceFit.STRONG_MATCH, rejected=False,
                qwen=None) -> JobMatchResult:
        result = JobMatchResult(
            job=_job(), requirements=_requirements(), experience=_experience(fit),
            final_score=final_score, hard_rejected=rejected, qwen_analysis=qwen,
        )
        return result

    def test_match_threshold(self, cfg):
        assert determine_verdict(self._result(75.0), cfg) == Verdict.MATCH
        assert determine_verdict(self._result(74.9), cfg) == Verdict.BORDERLINE

    def test_borderline_threshold(self, cfg):
        assert determine_verdict(self._result(60.0), cfg) == Verdict.BORDERLINE
        assert determine_verdict(self._result(59.9), cfg) == Verdict.REJECTED

    def test_marginal_experience_caps_at_borderline(self, cfg):
        assert determine_verdict(self._result(90.0, fit=ExperienceFit.MARGINAL), cfg) \
            == Verdict.BORDERLINE

    def test_potentially_overqualified_caps_at_borderline(self, cfg):
        assert determine_verdict(
            self._result(88.0, fit=ExperienceFit.POTENTIALLY_OVERQUALIFIED), cfg
        ) == Verdict.BORDERLINE

    def test_qwen_low_score_caps_at_borderline(self, cfg):
        qwen = JobAnalysis(match_score=10)
        assert determine_verdict(self._result(90.0, qwen=qwen), cfg) == Verdict.BORDERLINE


class TestQwenGuardrails:
    def test_hallucinated_score_replaced_by_deterministic(self):
        assert reconcile(98.0, 20.0, tolerance=40.0, label="domain") == 20.0

    def test_reasonable_qwen_score_blended(self):
        blended = reconcile(60.0, 40.0, tolerance=40.0)
        assert blended == 50.0

    def test_missing_qwen_uses_deterministic(self):
        assert reconcile(None, 42.0, tolerance=40.0) == 42.0


class TestExplainability:
    def test_rejected_job_has_reason(self, cfg, profile_10y):
        result = score_job(
            profile_10y, _job(), _requirements(),
            _experience(fit=ExperienceFit.UNDERQUALIFIED, effective_candidate_years=2,
                        job_min_years=5, gap=-3),
            skill_taxonomy.match_skills([], []),
            embedding_similarity=0.5,
            rejections=[Rejection(RejectionReason.UNDERQUALIFIED, "2 < 5 required")],
            qwen_analysis=None, cfg=cfg,
        )
        assert "UNDERQUALIFIED" in result.reasoning
        assert result.concerns

    def test_matched_job_has_strengths(self, cfg, profile_10y):
        requirements = _requirements()
        skills = skill_taxonomy.match_skills(
            requirements.required_skills, profile_10y.core_skills,
            required=requirements.required_skills,
        )
        result = score_job(
            profile_10y, _job(), requirements, _experience(), skills,
            embedding_similarity=0.7, rejections=[], qwen_analysis=None, cfg=cfg,
        )
        assert result.strengths
        assert any("relevant years" in s for s in result.strengths)

    def test_low_career_alignment_rejects_generic_role(self, cfg, profile_10y):
        """Spec Job C: generic Software Engineer role for an AI-target candidate."""
        requirements = _requirements(
            min_experience_years=2, max_experience_years=4,
            experience_domain="software_engineering", required_skills=[],
        )
        result = score_job(
            profile_10y,
            _job(title="Software Engineer",
                 description="Build web services. 2-4 years experience."),
            requirements,
            _experience(fit=ExperienceFit.STRONG_MATCH, effective_candidate_years=10,
                        basis="total-assumed", job_min_years=2, job_max_years=4),
            skill_taxonomy.match_skills([], profile_10y.core_skills),
            embedding_similarity=0.55, rejections=[], qwen_analysis=None, cfg=cfg,
        )
        assert result.verdict == Verdict.REJECTED
        assert RejectionReason.LOW_CAREER_ALIGNMENT in {r.reason for r in result.rejections}
