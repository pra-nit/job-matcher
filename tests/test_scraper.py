"""JobSpy wrapper tests (mocked — no network access)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from llm.schemas import CandidateProfile  # noqa: E402
from scraping import jobspy_scraper  # noqa: E402


@pytest.fixture
def cfg():
    return load_config(overrides={
        "scraping.sites": ("linkedin", "indeed"),
        "scraping.results_wanted": 5,
        "scraping.max_scrape_calls": 4,
        "locations": ("Bangalore",),
        "max_job_age_days": 14,
    })


class TestQueryGeneration:
    def test_queries_from_target_roles(self, cfg):
        profile = CandidateProfile(
            target_roles=["ML Engineer", "Computer Vision Engineer", "AI Engineer"],
            core_skills=["ONNX", "TensorRT"],
        )
        queries = jobspy_scraper.generate_search_queries(profile, cfg)
        assert "ML Engineer" in queries
        assert "Computer Vision Engineer" in queries
        assert len(queries) <= cfg.scraping.max_search_queries

    def test_core_skills_seed_queries(self, cfg):
        profile = CandidateProfile(
            target_roles=["AI Engineer"], core_skills=["Quantization", "ONNX"],
        )
        queries = jobspy_scraper.generate_search_queries(profile, cfg)
        assert any("Quantization" in q or "ONNX" in q for q in queries)

    def test_fallback_queries(self, cfg):
        queries = jobspy_scraper.generate_search_queries(CandidateProfile(), cfg)
        assert queries, "must always produce at least one query"


class TestScrapeWrapper:
    def test_scrape_collects_rows_and_isolates_failures(self, cfg, monkeypatch):
        import pandas as pd

        calls = []

        def fake_jobspy(**kwargs):
            calls.append(kwargs)
            if kwargs["search_term"] == "fail query":
                raise RuntimeError("rate limited")
            return pd.DataFrame([
                {"site": "linkedin", "job_url": "https://linkedin.com/jobs/view/1",
                 "title": "ML Engineer", "company": "X", "description": "d"},
                {"site": "indeed", "job_url": "https://in.indeed.com/viewjob?jk=2",
                 "title": "AI Engineer", "company": "Y", "description": "d"},
                # duplicate URL must be de-duplicated by the wrapper
                {"site": "indeed", "job_url": "https://in.indeed.com/viewjob?jk=2",
                 "title": "AI Engineer", "company": "Y", "description": "d"},
            ])

        # patch the symbol the wrapper imports inside the function body
        import jobspy as real_jobspy
        monkeypatch.setattr(real_jobspy, "scrape_jobs", fake_jobspy)

        rows = jobspy_scraper.scrape_jobs(
            ["ML Engineer", "fail query", "AI Engineer"], cfg, locations=["Bangalore"]
        )
        urls = [r["job_url"] for r in rows]
        assert len(urls) == len(set(urls)), "duplicate URLs must not repeat"
        assert len(rows) == 2
        assert len(calls) == 3  # the failing call was isolated, not fatal
        assert all(c["hours_old"] == 14 * 24 for c in calls)

    def test_rows_limited_by_max_scrape_calls(self, cfg, monkeypatch):
        import pandas as pd

        def fake_jobspy(**kwargs):
            return pd.DataFrame([
                {"site": "linkedin", "job_url": f"https://x/{kwargs['search_term']}",
                 "title": "T", "company": "C", "description": "d"},
            ])

        import jobspy as real_jobspy
        monkeypatch.setattr(real_jobspy, "scrape_jobs", fake_jobspy)

        rows = jobspy_scraper.scrape_jobs(
            ["q1", "q2", "q3", "q4", "q5", "q6"], cfg, locations=["Bangalore", "Remote"]
        )
        # capped at max_scrape_calls = 4
        assert len(rows) <= 4
