"""
Experience MATCHING tests — the spec's own decision table (sections 6-8, 31).

All of these must pass with the DEFAULT (hard) experience policy.
"""
from __future__ import annotations

import pytest

from config import ExperiencePolicy
from llm.schemas import CandidateProfile, JobRequirements
from matching.experience import evaluate_experience
from models import ExperienceDomain, ExperienceFit


def _profile(total=10.0, relevant=4.0, domains=None, **extra) -> CandidateProfile:
    if domains is None:
        domains = {
            "machine_learning": 4, "computer_vision": 4,
            "model_deployment": 3, "quantization": 2,
        }
    return CandidateProfile(
        total_experience_years=total,
        relevant_experience_years=relevant,
        experience_by_domain=domains,
    )


def _req(min_years=None, max_years=None, domain=ExperienceDomain.GENERAL) -> JobRequirements:
    return JobRequirements(
        min_experience_years=min_years,
        max_experience_years=max_years,
        experience_domain=domain,
        extraction_method="regex",
    )


def _fit(profile, req, policy):
    return evaluate_experience(profile, req, policy).fit


class TestSpecDecisionTable:
    """The exact cases from spec sections 6, 7 and 31."""

    def test_10_total_4_ml_vs_3_4_ml(self, policy):
        # GOOD FIT
        assert _fit(_profile(), _req(3, 4, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.STRONG_MATCH

    def test_10_total_4_ml_vs_3_4_general(self, policy):
        # GOOD FIT (general requirement -> relevant years = 4)
        assert _fit(_profile(), _req(3, 4), policy) == ExperienceFit.STRONG_MATCH

    def test_10_total_4_ml_vs_5_7_ml(self, policy):
        # UNDERQUALIFIED
        assert _fit(_profile(), _req(5, 7, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.UNDERQUALIFIED

    def test_10_total_10_ml_vs_3_4(self, policy):
        # OVERQUALIFIED
        profile = _profile(total=10, relevant=10, machine_learning=10)
        assert _fit(profile, _req(3, 4), policy) == ExperienceFit.OVERQUALIFIED

    def test_10_total_6_ml_vs_3_4(self, policy):
        # Potentially OVERQUALIFIED
        profile = _profile(relevant=6, machine_learning=6)
        assert _fit(profile, _req(3, 4), policy) == ExperienceFit.POTENTIALLY_OVERQUALIFIED

    def test_10_vs_3_no_max(self, policy):
        # gap 7, ratio 3.33 -> OVERQUALIFIED
        profile = _profile(total=10, relevant=10, machine_learning=10)
        assert _fit(profile, _req(3), policy) == ExperienceFit.OVERQUALIFIED

    def test_5_vs_4(self, policy):
        # gap 1, ratio 1.25 -> ACCEPTABLE
        profile = _profile(total=5, relevant=5, machine_learning=5)
        assert _fit(profile, _req(4), policy) == ExperienceFit.ACCEPTABLE

    def test_8_vs_4(self, policy):
        # gap 4, ratio 2 -> OVERQUALIFIED
        profile = _profile(total=8, relevant=8, machine_learning=8)
        assert _fit(profile, _req(4), policy) == ExperienceFit.OVERQUALIFIED

    def test_4_ml_vs_5_ml(self, policy):
        assert _fit(_profile(), _req(5, None, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.UNDERQUALIFIED

    def test_4_cv_vs_5_ml(self, policy):
        # CV experience must not count as direct ML experience
        profile = _profile(relevant=4, machine_learning=4, computer_vision=4)
        assert _fit(profile, _req(5, None, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.UNDERQUALIFIED

    def test_unknown_job_experience(self, policy):
        assert _fit(_profile(), _req(None, None), policy) == ExperienceFit.UNKNOWN


class TestSoftExperienceMode:
    def test_soft_mode_near_miss_is_marginal(self):
        policy = ExperiencePolicy(soft_experience_mode=True, soft_experience_gap_tolerance=1.0)
        assert _fit(_profile(), _req(5, None, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.MARGINAL

    def test_soft_mode_big_gap_still_underqualified(self):
        policy = ExperiencePolicy(soft_experience_mode=True, soft_experience_gap_tolerance=1.0)
        assert _fit(_profile(), _req(8, None, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.UNDERQUALIFIED

    def test_hard_mode_near_miss_is_underqualified(self, policy):
        assert _fit(_profile(), _req(5, None, ExperienceDomain.MACHINE_LEARNING), policy) \
            == ExperienceFit.UNDERQUALIFIED


class TestDomainAwareness:
    def test_cv_requirement_uses_cv_years(self, policy):
        profile = _profile(relevant=4, machine_learning=2, computer_vision=4)
        evaluation = evaluate_experience(
            profile, _req(3, 5, ExperienceDomain.COMPUTER_VISION), policy
        )
        assert evaluation.fit == ExperienceFit.STRONG_MATCH
        assert evaluation.effective_candidate_years == 4.0
        assert evaluation.basis.startswith("direct")

    def test_swe_requirement_uses_total(self, policy):
        profile = _profile(total=10, relevant=4, software_engineering=0,
                           machine_learning=4)
        evaluation = evaluate_experience(
            profile, _req(3, 6, ExperienceDomain.SOFTWARE_ENGINEERING), policy
        )
        assert evaluation.effective_candidate_years == 10.0
        assert evaluation.basis == "total-assumed"

    def test_ml_requirement_vs_networking_candidate(self, policy):
        # a pure networking candidate has no ML years: adjacent credit only
        profile = _profile(total=6, relevant=6,
                           domains={"networking": 6, "software_engineering": 1})
        evaluation = evaluate_experience(
            profile, _req(5, None, ExperienceDomain.MACHINE_LEARNING), policy
        )
        assert evaluation.effective_candidate_years == 0.0
        assert evaluation.fit == ExperienceFit.UNDERQUALIFIED

    def test_adjacent_credit_when_no_direct_domain(self, policy):
        # CV-only candidate applying to an ML-titled job with 3y requirement
        profile = _profile(total=4, relevant=4, domains={"computer_vision": 4})
        policy = ExperiencePolicy(adjacent_domain_credit=0.75)
        evaluation = evaluate_experience(
            profile, _req(3, None, ExperienceDomain.MACHINE_LEARNING), policy
        )
        assert evaluation.effective_candidate_years == pytest.approx(3.0)
        assert evaluation.basis.startswith("adjacent")
        assert evaluation.fit == ExperienceFit.STRONG_MATCH

    def test_no_domain_breakdown_falls_back_to_relevant(self, policy):
        profile = _profile(total=10, relevant=5, domains={})
        evaluation = evaluate_experience(
            profile, _req(4, 6, ExperienceDomain.MACHINE_LEARNING), policy
        )
        assert evaluation.effective_candidate_years == 5.0
        assert evaluation.basis == "relevant-fallback"


class TestEvaluationFields:
    def test_flags_and_gap(self, policy):
        profile = _profile(total=10, relevant=10, machine_learning=10)
        evaluation = evaluate_experience(profile, _req(3), policy)
        assert evaluation.overqualified is True
        assert evaluation.underqualified is False
        assert evaluation.gap == 7.0  # effective - min
        assert evaluation.ratio == pytest.approx(3.33, abs=0.01)

    def test_underqualified_flag(self, policy):
        evaluation = evaluate_experience(
            _profile(), _req(8, None, ExperienceDomain.MACHINE_LEARNING), policy
        )
        assert evaluation.underqualified is True
        assert evaluation.overqualified is False
        assert evaluation.gap == -4.0

    def test_reasons_are_populated(self, policy):
        evaluation = evaluate_experience(
            _profile(), _req(5, 7, ExperienceDomain.MACHINE_LEARNING), policy
        )
        assert evaluation.reasons, "every evaluation must be explainable"
        assert any("4" in r for r in evaluation.reasons)

    def test_seniority_inference_when_no_explicit_requirement(self, policy):
        req = JobRequirements(min_experience_years=None, max_experience_years=None,
                              seniority="senior")
        evaluation = evaluate_experience(_profile(), req, policy)
        # 4 ML years vs inferred 5-10 senior range
        assert evaluation.fit == ExperienceFit.UNDERQUALIFIED
        assert evaluation.requirement_source == "seniority_inference"


class TestConfigurableOverqualification:
    def test_relaxed_thresholds_accept_8_vs_4(self):
        policy = ExperiencePolicy(
            overqualification_absolute_gap=5.0, overqualification_ratio=2.5
        )
        profile = _profile(total=8, relevant=8, machine_learning=8)
        assert _fit(profile, _req(4), policy) == ExperienceFit.ACCEPTABLE

    def test_stricter_thresholds_flag_earlier(self):
        policy = ExperiencePolicy(
            overqualification_absolute_gap=1.0, overqualification_ratio=1.2
        )
        profile = _profile(total=6, relevant=6, machine_learning=6)
        assert _fit(profile, _req(4), policy) == ExperienceFit.OVERQUALIFIED
