"""
Rule-engine tests: seniority, location, roles, must-haves, freshness
(spec sections 9, 10, 23, 24).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from config import AppConfig
from llm.schemas import CandidateProfile
from matching.rules import (
    RuleEngine,
    candidate_seniority,
    detect_employment_type,
    detect_work_mode,
    infer_seniority,
    is_indian_location,
    location_matches,
    normalize_location,
    seniority_from_years,
)
from models import JobRecord, RejectionReason, Seniority


class TestSeniorityNormalization:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Junior ML Engineer", Seniority.JUNIOR),
            ("ML Engineer", Seniority.MID),
            ("Machine Learning Engineer", Seniority.MID),
            ("Senior ML Engineer", Seniority.SENIOR),
            ("Sr. Software Engineer", Seniority.SENIOR),
            ("Staff ML Engineer", Seniority.STAFF),
            ("Principal ML Engineer", Seniority.PRINCIPAL),
            ("Lead ML Engineer", Seniority.LEAD),
            ("Engineering Manager, ML", Seniority.MANAGER),
            ("Data Science Intern", Seniority.INTERN),
            ("ML Engineer II", Seniority.MID),
            ("Software Engineer III", Seniority.SENIOR),
            ("AI Research Scientist", Seniority.MID),
        ],
    )
    def test_title_to_seniority(self, title, expected):
        assert infer_seniority(title) == expected

    def test_years_to_seniority(self):
        assert seniority_from_years(0.5) == Seniority.ENTRY
        assert seniority_from_years(3) == Seniority.MID
        assert seniority_from_years(6) == Seniority.SENIOR
        assert seniority_from_years(10) == Seniority.LEAD
        assert seniority_from_years(13) == Seniority.STAFF

    def test_candidate_seniority_prefers_title_marker(self, profile_10y):
        assert candidate_seniority(profile_10y) == Seniority.SENIOR

    def test_candidate_seniority_falls_back_to_years(self):
        profile = CandidateProfile(
            current_title="Machine Learning Engineer",
            total_experience_years=8,
            relevant_experience_years=6,
        )
        assert candidate_seniority(profile) == Seniority.SENIOR


class TestLocation:
    def test_bengaluru_normalizes_to_bangalore(self):
        assert normalize_location("Bengaluru, Karnataka, India") == "bangalore"

    def test_gurugram_normalizes_to_delhi(self):
        assert normalize_location("Gurugram, Haryana") == "delhi"

    def test_remote_detection(self):
        assert normalize_location("Remote — anywhere in India") == "remote"
        assert normalize_location("Work From Home") == "remote"

    def test_india_passthrough(self):
        assert normalize_location("India") == "india"

    def test_matching_cities(self):
        ok, _ = location_matches("Hyderabad, Telangana, India", ["Bangalore", "Hyderabad"])
        assert ok

    def test_india_allows_any_indian_city(self):
        ok, _ = location_matches("Pune, Maharashtra, India", ["India"])
        assert ok

    def test_mismatch(self):
        ok, detail = location_matches("Mumbai, India", ["Bangalore", "Hyderabad"])
        assert not ok

    def test_remote_allowed(self):
        ok, _ = location_matches("Remote", ["Bangalore", "Remote"])
        assert ok

    def test_remote_job_matches_remote_filter_via_work_mode(self):
        ok, _ = location_matches("Bengaluru (Remote)", ["Remote"], "remote")
        assert ok

    def test_unknown_location_not_rejected(self):
        ok, _ = location_matches("", ["Bangalore"])
        assert ok


class TestDetection:
    def test_work_mode(self):
        assert detect_work_mode("This is a remote position").value == "remote"
        assert detect_work_mode("Hybrid: 3 days from office").value == "hybrid"
        assert detect_work_mode("On-site role in our Pune office").value == "onsite"
        assert detect_work_mode("Great team").value == "unknown"

    def test_employment_type(self):
        assert detect_employment_type("Full-time employment").value == "full_time"
        assert detect_employment_type("6-month contract").value == "contract"
        assert detect_employment_type("Internship for 6 months").value == "internship"


class TestRoleRelevance:
    def _engine(self, cfg: AppConfig, profile: CandidateProfile) -> RuleEngine:
        return RuleEngine(profile, cfg)

    def test_telecom_job_rejected_as_irrelevant(self, cfg, profile_10y):
        job = JobRecord(
            company="BSNL", title="Telecom Network Engineer",
            location="Bangalore, India", url="https://x/1",
            description="Routing, switching, LTE network operations. 5 years experience.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert not outcome.passed
        reasons = {r.reason for r in outcome.rejections}
        assert RejectionReason.IRRELEVANT_ROLE in reasons

    def test_network_engineer_rejected_despite_networking_history(self, cfg, profile_10y):
        # candidate WITH networking history must still not get networking jobs
        profile_10y.experience_by_domain.update(
            {"machine_learning": 4, "networking": 5, "software_engineering": 6}
        )
        job = JobRecord(
            company="Cisco", title="Network Engineer",
            location="Chennai, India", url="https://x/2",
            description="Enterprise routing and switching. 4+ years experience.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert not outcome.passed

    def test_edge_ai_job_passes_rules(self, cfg, profile_10y):
        job = JobRecord(
            company="Qualcomm", title="Senior Edge AI Engineer",
            location="Bangalore, India", url="https://x/3",
            description=("Optimize and deploy CNN models on edge devices with ONNX and "
                         "TensorRT. 3-5 years experience. INT8 quantization required."),
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert outcome.passed, [str(r.detail) for r in outcome.rejections]

    def test_ml_job_5_8_years_rejected_underqualified(self, cfg, profile_10y):
        job = JobRecord(
            company="Startup", title="Machine Learning Engineer",
            location="Bangalore, India", url="https://x/4",
            description="Build recommendation models. 5-8 years of ML experience required.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert not outcome.passed
        reasons = {r.reason for r in outcome.rejections}
        assert RejectionReason.UNDERQUALIFIED in reasons

    def test_location_mismatch_rejection(self, cfg, profile_10y):
        cfg = cfg.__class__(**{**cfg.__dict__, "locations": ("Bangalore",)})
        job = JobRecord(
            company="X", title="ML Engineer", location="Mumbai, India",
            url="https://x/5", description="ML models. 3 years experience.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        reasons = {r.reason for r in outcome.rejections}
        assert RejectionReason.LOCATION_MISMATCH in reasons

    def test_stale_job_rejected(self, cfg, profile_10y):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        job = JobRecord(
            company="X", title="ML Engineer", location="Bangalore, India",
            url="https://x/6", description="ML models. 3 years experience.",
            posted_date=old,
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert not outcome.passed
        assert RejectionReason.STALE_JOB in {r.reason for r in outcome.rejections}

    def test_freshness_unknown_not_rejected(self, cfg, profile_10y):
        job = JobRecord(
            company="X", title="ML Engineer", location="Bangalore, India",
            url="https://x/7", description="ML models. 3 years experience.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        assert outcome.passed
        assert any("freshness unknown" in n for n in outcome.notes)

    def test_must_have_skill_filter(self, profile_10y):
        # job-involvement semantics: the job must mention the must-have skill
        cfg = AppConfig(must_have_skills=("CUDA",))
        job = JobRecord(
            company="X", title="ML Engineer", location="Bangalore, India",
            url="https://x/8", description="ML models. 3 years experience.",
        )
        outcome = RuleEngine(profile_10y, cfg).apply(job)
        assert not outcome.passed
        assert RejectionReason.MISSING_MUST_HAVE_SKILL in {
            r.reason for r in outcome.rejections
        }

    def test_must_have_skill_satisfied_when_job_mentions_it(self, profile_10y):
        cfg = AppConfig(must_have_skills=("CUDA",))
        job = JobRecord(
            company="X", title="ML Engineer", location="Bangalore, India",
            url="https://x/8b", description="ML models with CUDA kernels. 3 years experience.",
        )
        outcome = RuleEngine(profile_10y, cfg).apply(job)
        assert RejectionReason.MISSING_MUST_HAVE_SKILL not in {
            r.reason for r in outcome.rejections
        }

    def test_seniority_mismatch_rejected(self, cfg, profile_10y):
        job = JobRecord(
            company="X", title="Principal ML Engineer", location="Bangalore, India",
            url="https://x/9", description="Lead research. 12+ years experience.",
        )
        outcome = self._engine(cfg, profile_10y).apply(job)
        reasons = {r.reason for r in outcome.rejections}
        assert RejectionReason.SENIORITY_MISMATCH in reasons or \
            RejectionReason.UNDERQUALIFIED in reasons


class TestIndeedStateCodes:
    def test_state_code_normalizes(self):
        assert normalize_location("KA, IN") == "karnataka"
        assert normalize_location("TS, IN") == "telangana"
        assert normalize_location("MH, IN") == "maharashtra"

    def test_state_code_is_indian(self):
        assert is_indian_location("KA, IN")
        assert is_indian_location("TN, IN")

    def test_india_filter_accepts_state_code_locations(self):
        ok, detail = location_matches("KA, IN", ["India"])
        assert ok, detail

    def test_city_filter_still_strict_against_state_only(self):
        # a job that only says "Karnataka" is not provably in Bangalore
        ok, _ = location_matches("KA, IN", ["Bangalore"])
        assert not ok

    def test_state_name_matches(self):
        ok, _ = location_matches("Karnataka, India", ["India"])
        assert ok
