"""
End-to-end pipeline tests with a FAKE Ollama client and the offline hashing
embedding provider — no network, no torch, no LLM server required.

These exercise the full orchestration: scrape-normalised job dicts ->
rules -> embeddings -> (fake) Qwen -> scoring -> CSV output, replicating
main.run() stage by stage.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from config import AppConfig, load_config
from database.sqlite_store import SqliteStore
from llm.schemas import CandidateProfile, JobAnalysis, JobRequirements
from matching import skill_taxonomy
from matching.embeddings import HashingProvider, rescale_similarity
from matching.rules import RuleEngine
from matching.scoring import score_job
from matching.semantic_matcher import build_job_text, build_resume_text, rank_jobs
from models import JobMatchResult, PipelineStats, Verdict
from output import csv_writer
from scraping.normalizer import deduplicate, normalize_job


# ---------------------------------------------------------------------------
# Fixtures: sample jobs (spec section 31 Jobs A-D + extras)
# ---------------------------------------------------------------------------
JOB_A = {  # Senior Edge AI Engineer — should MATCH
    "site": "linkedin", "title": "Senior Edge AI Engineer", "company": "Qualcomm",
    "job_url": "https://www.linkedin.com/jobs/view/1001/?trk=abc",
    "location": "Bengaluru, Karnataka, India",
    "description": (
        "Optimize and deploy deep learning models on embedded edge platforms. "
        "Requirements: 3-5 years of experience in edge AI deployment. "
        "Skills: Python, C++, ONNX, TensorRT, Quantization (INT8). "
        "Responsibilities: model optimization, inference acceleration, "
        "hardware software co-design. Nice to have CUDA."
    ),
    "date_posted": (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d"),
}

JOB_B = {  # ML Engineer 5-8 years — UNDERQUALIFIED for the 4y ML candidate
    "site": "indeed", "title": "Machine Learning Engineer", "company": "StartupIo",
    "job_url": "https://in.indeed.com/viewjob?jk=b222",
    "location": "Hyderabad, Telangana, India",
    "description": (
        "Build recommendation systems end to end. "
        "Requirements: 5-8 years of machine learning experience. "
        "Skills: Python, PyTorch, Kubernetes, MLOps."
    ),
    "date_posted": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
}

JOB_C = {  # generic Software Engineer — LOW_CAREER_ALIGNMENT
    "site": "indeed", "title": "Software Engineer", "company": "WebCorp",
    "job_url": "https://in.indeed.com/viewjob?jk=c333",
    "location": "Pune, Maharashtra, India",
    "description": (
        "Build REST APIs and web services. 2-4 years of software development "
        "experience. Skills: Java, Spring, SQL, Microservices."
    ),
    "date_posted": (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d"),
}

JOB_D = {  # Telecom — IRRELEVANT_ROLE (career-direction rule)
    "site": "indeed", "title": "Telecom Network Engineer", "company": "BSNL",
    "job_url": "https://in.indeed.com/viewjob?jk=d444",
    "location": "Chennai, Tamil Nadu, India",
    "description": (
        "Maintain LTE network operations. 5 years of telecom experience. "
        "Skills: routing, switching, TCP/IP, firewall configuration."
    ),
    "date_posted": (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d"),
}

JOB_E = {  # Computer Vision Engineer 3-4y — strong MATCH
    "site": "linkedin", "title": "Computer Vision Engineer", "company": "DriveAI",
    "job_url": "https://www.linkedin.com/jobs/view/1005/",
    "location": "Remote",
    "description": (
        "Develop 3D perception (BEV, Lift-Splat-Shoot) for autonomous driving. "
        "Requirements: 3-4 years of computer vision experience. "
        "Skills: PyTorch, Object Detection, Semantic Segmentation, ONNX."
    ),
    "date_posted": (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d"),
}

ALL_JOBS = [JOB_A, JOB_B, JOB_C, JOB_D, JOB_E]


class FakeOllamaClient:
    """Deterministic stand-in for the real client; returns canned JSON."""

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.calls: list[str] = []

    def chat_json(self, system: str, user: str, schema_model=None):
        self.calls.append(system[:40])
        if schema_model is JobRequirements:
            data = self._requirements(user)
        elif schema_model is JobAnalysis:
            data = {
                "match_score": 82, "skill_match_score": 80,
                "responsibility_match_score": 78, "domain_match_score": 85,
                "career_alignment_score": 88,
                "matched_skills": ["Python", "ONNX"],
                "related_skills": ["TensorRT"],
                "missing_required_skills": [],
                "missing_preferred_skills": ["CUDA"],
                "strengths": ["deep edge deployment experience"],
                "concerns": ["TensorRT not directly listed"],
                "reasoning": "Candidate's ONNX/quantization background maps well.",
                "semantic_verdict": "good_match",
            }
        elif schema_model is CandidateProfile:
            data = {}
        else:
            data = {}
        if schema_model is None:
            return data
        return schema_model.model_validate(data)

    @staticmethod
    def _requirements(user: str) -> dict:
        if "Machine Learning Engineer" in user and "StartupIo" in user:
            return {
                "min_experience_years": 5, "max_experience_years": 8,
                "experience_domain": "machine_learning",
                "experience_confidence": 0.9, "required_skills": ["Python", "PyTorch"],
                "preferred_skills": ["Kubernetes"], "seniority": "senior",
            }
        if "Edge AI" in user:
            return {
                "min_experience_years": 3, "max_experience_years": 5,
                "experience_domain": "model_deployment",
                "experience_confidence": 0.9,
                "required_skills": ["Python", "C++", "ONNX", "TensorRT"],
                "preferred_skills": ["CUDA", "Quantization"], "seniority": "senior",
            }
        if "Computer Vision" in user:
            return {
                "min_experience_years": 3, "max_experience_years": 4,
                "experience_domain": "computer_vision",
                "experience_confidence": 0.9,
                "required_skills": ["PyTorch", "Object Detection"],
                "preferred_skills": ["ONNX"], "seniority": "mid",
            }
        return {"min_experience_years": None, "max_experience_years": None,
                "experience_confidence": 0.2, "seniority": "unknown"}


@pytest.fixture
def pipeline_cfg(tmp_path: Path) -> AppConfig:
    return load_config(overrides={
        "paths.database": str(tmp_path / "jobs.db"),
        "paths.output_dir": str(tmp_path / "out"),
        "embedding_model": "hashing-test",
        "matching.top_k_embedding": 10,
        "matching.embedding_threshold": 0.05,  # hashing provider: low sim values
        "matching.embedding_score_floor": 0.0,
        "matching.embedding_score_ceiling": 0.5,
        "locations": ("India",),
    })


def _run_pipeline(cfg: AppConfig, profile: CandidateProfile, jobs: list[dict],
                  client: Optional[FakeOllamaClient]):
    """Mirror of main.run() for testing (no scraping, no real LLM)."""
    stats = PipelineStats()
    records = [normalize_job(row) for row in jobs]
    records = [r for r in records if r is not None]
    unique, dups = deduplicate(records)
    stats.jobs_scraped = len(unique)
    stats.duplicates_removed = dups

    engine = RuleEngine(profile, cfg)
    eligible, rejected = [], []
    for job in unique:
        outcome = engine.apply(job)
        if outcome.passed:
            eligible.append((job, outcome.requirements))
        else:
            rejected.append(job)

    stats.embedding_candidates = len(eligible)
    provider = HashingProvider()
    top = rank_jobs(profile, eligible, provider, cfg)

    candidate_skills = list(profile.core_skills) + list(profile.skills)
    results: list[JobMatchResult] = []
    for job, _regex_req, similarity in top:
        requirements = (
            client.chat_json("", f"{job.title} {job.company}", schema_model=JobRequirements)
            if client else engine.extract_requirements(job)
        )
        outcome = engine.apply(job, requirements)
        skills = skill_taxonomy.match_skills(
            list(requirements.required_skills) + list(requirements.preferred_skills),
            candidate_skills,
            required=requirements.required_skills,
            preferred=requirements.preferred_skills,
        )
        analysis = (
            client.chat_json("", job.title, schema_model=JobAnalysis) if client else None
        )
        results.append(score_job(
            profile, job, requirements, outcome.experience, skills,
            embedding_similarity=similarity,
            rejections=outcome.rejections, qwen_analysis=analysis, cfg=cfg,
        ))
    for job in rejected:
        outcome = engine.apply(job)
        results.append(score_job(
            profile, job, outcome.requirements, outcome.experience,
            skill_taxonomy.match_skills([], []),
            embedding_similarity=0.0, rejections=outcome.rejections,
            qwen_analysis=None, cfg=cfg,
        ))

    stats.final_matches = sum(1 for r in results if r.verdict == Verdict.MATCH)
    stats.final_borderline = sum(1 for r in results if r.verdict == Verdict.BORDERLINE)
    stats.final_rejected = sum(1 for r in results if r.verdict == Verdict.REJECTED)
    return results, stats


class TestEndToEnd:
    def test_spec31_scenario(self, pipeline_cfg, profile_10y):
        results, stats = _run_pipeline(pipeline_cfg, profile_10y, ALL_JOBS,
                                       FakeOllamaClient())
        by_title = {r.job.title: r for r in results}

        # Job A: strong MATCH
        job_a = by_title["Senior Edge AI Engineer"]
        assert job_a.verdict in (Verdict.MATCH, Verdict.BORDERLINE)
        assert job_a.experience.fit.value == "strong_match"
        assert job_a.final_score >= 60

        # Job B: UNDERQUALIFIED (4 ML years vs 5-8 required)
        job_b = by_title["Machine Learning Engineer"]
        assert job_b.verdict == Verdict.REJECTED
        assert job_b.experience.fit.value == "underqualified"

        # Job C: rejected via career alignment (generic SWE role)
        job_c = by_title["Software Engineer"]
        assert job_c.verdict == Verdict.REJECTED

        # Job D: IRRELEVANT_ROLE despite the candidate's networking history
        job_d = by_title["Telecom Network Engineer"]
        assert job_d.verdict == Verdict.REJECTED
        assert "IRRELEVANT_ROLE" in {r.reason.value for r in job_d.rejections}

        # Job E: CV job should also be a good match
        job_e = by_title["Computer Vision Engineer"]
        assert job_e.verdict in (Verdict.MATCH, Verdict.BORDERLINE)

        assert stats.final_matches >= 2
        assert stats.final_rejected >= 2

    def test_soft_experience_mode_rescues_job_b(self, pipeline_cfg, profile_10y):
        pipeline_cfg = load_config(overrides={
            "paths.database": pipeline_cfg.paths.database,
            "embedding_model": "hashing-test",
            "matching.top_k_embedding": 10,
            "matching.embedding_threshold": 0.05,
            "matching.embedding_score_floor": 0.0,
            "matching.embedding_score_ceiling": 0.5,
            "locations": ("India",),
            "experience.soft_experience_mode": True,
        })
        results, _ = _run_pipeline(pipeline_cfg, profile_10y, ALL_JOBS,
                                   FakeOllamaClient())
        job_b = next(r for r in results if r.job.title == "Machine Learning Engineer")
        # 4 ML years vs 5 minimum: within 1-year tolerance -> borderline-ish
        assert job_b.experience.fit.value in ("marginal", "underqualified")
        if job_b.experience.fit.value == "marginal":
            assert job_b.verdict in (Verdict.BORDERLINE, Verdict.REJECTED)

    def test_csv_outputs_written(self, pipeline_cfg, profile_10y, tmp_path):
        results, _ = _run_pipeline(pipeline_cfg, profile_10y, ALL_JOBS,
                                   FakeOllamaClient())
        matched_path = tmp_path / "matched_jobs.csv"
        rejected_path = tmp_path / "rejected_jobs.csv"
        csv_writer.write_matched_csv(results, profile_10y, matched_path)
        csv_writer.write_rejected_csv(results, profile_10y, rejected_path)

        with open(matched_path, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "matched CSV must contain rows"
        assert "Final Match Score" in rows[0]
        assert "Verdict" in rows[0]
        assert "Job Link" in rows[0]
        # sorted: eligible first, then score desc
        verdicts = [r["Verdict"] for r in rows]
        assert verdicts == sorted(verdicts, key=lambda v: {"MATCH": 0, "BORDERLINE": 1}[v])

        with open(rejected_path, encoding="utf-8-sig") as fh:
            rejected_rows = list(csv.DictReader(fh))
        assert rejected_rows
        reasons = {r["Rejection Reason"] for r in rejected_rows}
        assert "IRRELEVANT_ROLE" in reasons
        assert "UNDERQUALIFIED" in reasons
        assert all(r["Rejection Detail"] for r in rejected_rows)

    def test_no_llm_offline_mode_still_produces_verdicts(self, pipeline_cfg, profile_10y):
        results, _ = _run_pipeline(pipeline_cfg, profile_10y, ALL_JOBS, client=None)
        assert results, "offline mode must still classify every job"
        job_a = next(r for r in results if "Edge AI" in r.job.title)
        assert job_a.verdict in (Verdict.MATCH, Verdict.BORDERLINE, Verdict.REJECTED)
        assert job_a.reasoning  # always explainable


class TestSqliteStore:
    def test_roundtrip(self, tmp_path, profile_10y):
        store = SqliteStore(tmp_path / "test.db")
        try:
            assert store.sync_skills() > 100
            record = normalize_job(JOB_A)
            persisted, new = store.upsert_jobs([record])
            assert new == 1
            job_id = persisted[0].job_id

            # re-upsert same job: no new insert, last_seen updated
            persisted2, new2 = store.upsert_jobs(persisted)
            assert new2 == 0
            assert persisted2[0].job_id == job_id

            requirements = JobRequirements(
                min_experience_years=3, max_experience_years=5,
                experience_domain="model_deployment",
                required_skills=["ONNX"], preferred_skills=["CUDA"],
                seniority="senior",
            )
            store.save_requirements(job_id, requirements, "hash-1")
            loaded, req_hash = store.load_requirements(job_id)
            assert loaded.min_experience_years == 3
            assert req_hash == "hash-1"
            assert loaded.experience_domain.value == "model_deployment"

            # profile cache roundtrip
            fingerprint = "abc123"
            store.save_profile(fingerprint, "qwen3:4b", profile_10y)
            loaded_profile = store.load_profile(fingerprint)
            assert loaded_profile.name == "Rahul Sharma"
            assert loaded_profile.total_experience_years == 10

            # full result roundtrip
            result = JobMatchResult(
                job=persisted[0], requirements=requirements,
                final_score=88.5, verdict=Verdict.MATCH, hard_rejected=False,
                strengths=["s1"], concerns=["c1"], reasoning="because",
            )
            store.save_match(result, "config-hash-1")
            restored = store.load_full_results("config-hash-1")
            assert len(restored) == 1
            assert restored[0].final_score == 88.5
            assert restored[0].verdict == Verdict.MATCH
            assert restored[0].requirements.min_experience_years == 3
            assert restored[0].job.company == "Qualcomm"

            # embedding cache roundtrip
            store.save_embedding("key1", "hashing-test", [0.1, 0.2, 0.3])
            vector = store.load_embedding("key1")
            assert vector is not None and abs(float(vector[0]) - 0.1) < 1e-6
        finally:
            store.close()

    def test_incremental_requirements_cache(self, tmp_path):
        store = SqliteStore(tmp_path / "inc.db")
        try:
            record = normalize_job(JOB_E)
            persisted, _ = store.upsert_jobs([record])
            requirements = JobRequirements(min_experience_years=3, seniority="mid")
            store.save_requirements(persisted[0].job_id, requirements, "hash-A")
            # same hash -> cache hit; different hash -> caller re-extracts
            assert store.load_requirements(persisted[0].job_id)[1] == "hash-A"
        finally:
            store.close()


class TestEmbeddingProviders:
    def test_hashing_similarity_orders_related_texts(self):
        provider = HashingProvider()
        resume = "ML engineer ONNX quantization edge deployment TensorRT INT8"
        close = "Edge AI engineer model optimization ONNX TensorRT INT8 quantization deploy"
        far = "Recruiter for a civil engineering construction company sales team"
        vectors = provider.encode([resume, close, far])
        close_sim = provider.similarity(vectors[0], vectors[1])
        far_sim = provider.similarity(vectors[0], vectors[2])
        assert close_sim > far_sim

    def test_rescale(self):
        assert rescale_similarity(0.5, 0.35, 0.85) == 30.0
        assert rescale_similarity(0.0, 0.35, 0.85) == 0.0
        assert rescale_similarity(1.0, 0.35, 0.85) == 100.0

    def test_structured_texts_built(self, profile_10y):
        from models import JobRecord

        resume_text = build_resume_text(profile_10y)
        assert "TITLE:" in resume_text
        assert "ONNX" in resume_text
        job = JobRecord(company="X", title="Edge AI Engineer",
                        description="Deploy models. Requirements: 3-5 years experience.")
        job_text = build_job_text(job, None, max_description_chars=200)
        assert "TITLE: Edge AI Engineer" in job_text
        assert "DESCRIPTION:" in job_text

    def test_long_description_truncated_for_embedding(self):
        from matching.semantic_matcher import extract_relevant_sections

        description = ("About us: we are a great company. " * 20) + \
                      ("Requirements: Python, ONNX, 5 years experience. " * 5)
        trimmed = extract_relevant_sections(description, 200)
        assert len(trimmed) <= 250
        assert "Requirements" in trimmed
