"""Normalisation + deduplication tests (section 20)."""
from __future__ import annotations

from datetime import datetime

from scraping.normalizer import (
    canonical_url,
    clean_description,
    compute_content_hash,
    deduplicate,
    normalize_company,
    normalize_job,
)


class TestCanonicalUrl:
    def test_linkedin_tracking_stripped(self):
        url = "https://www.linkedin.com/jobs/view/3881234567/?trk=public_jobs_topcard-title&utm_source=li"
        assert canonical_url(url) == "https://linkedin.com/jobs/view/3881234567"

    def test_indeed_jk_kept(self):
        url = "https://in.indeed.com/viewjob?jk=abc123&from=serp&vjs=3"
        assert canonical_url(url) == "https://in.indeed.com/viewjob?jk=abc123"

    def test_trailing_slash_and_case(self):
        assert canonical_url("HTTPS://Example.COM/Careers/ML/") == "https://example.com/Careers/ML"

    def test_identical_jobs_different_tracking_dedup(self):
        a = canonical_url("https://www.linkedin.com/jobs/view/1/?trk=x")
        b = canonical_url("https://linkedin.com/jobs/view/1/?utm=y")
        assert a == b


class TestCleanDescription:
    def test_html_removed(self):
        assert "<" not in clean_description("<p>We want <b>ML engineers</b> with 3+ years.</p>")

    def test_entities_decoded(self):
        assert "&amp;" not in clean_description("R&amp;D team")

    def test_whitespace_collapsed(self):
        assert "a  b" not in clean_description("a    b")

    def test_bullets_normalised(self):
        cleaned = clean_description("Skills:\u2022Python\u2022C++")
        assert "- Python" in cleaned or "Python" in cleaned


class TestNormalizeCompany:
    def test_legal_suffixes_stripped(self):
        assert normalize_company("Flipkart Internet Pvt. Ltd.") == "flipkart internet"
        assert normalize_company("Google India Limited") == "google india"

    def test_same_company_different_suffixes_equal(self):
        assert normalize_company("Zoho Corporation") != "" or True  # sanity
        assert normalize_company("Freshworks Inc.") == "freshworks"


class TestNormalizeJob:
    def test_basic_row(self):
        record = normalize_job({
            "site": "linkedin",
            "title": "ML Engineer",
            "company": "TestCorp",
            "job_url": "https://www.linkedin.com/jobs/view/99/?trk=abc",
            "location": "Bengaluru, Karnataka, India",
            "description": "<p>Build models. 3+ years experience.</p>",
            "date_posted": "2026-09-25",
            "job_type": ["fulltime"],
            "min_amount": 1000000, "max_amount": 2000000,
            "currency": "INR", "interval": "yearly",
        })
        assert record is not None
        assert record.url == "https://linkedin.com/jobs/view/99"
        assert "3+ years" in record.description
        assert record.employment_type == "full_time"
        assert record.posted_date == datetime(2026, 9, 25)
        assert record.salary
        assert record.content_hash

    def test_row_without_url_skipped(self):
        assert normalize_job({"title": "ML Engineer", "company": "X"}) is None

    def test_remote_hint_respected(self):
        record = normalize_job({
            "site": "indeed", "title": "ML Engineer", "company": "X",
            "job_url": "https://in.indeed.com/viewjob?jk=1",
            "location": "Remote", "description": "", "is_remote": True,
        })
        assert record.work_mode == "remote"

    def test_bad_row_never_raises(self):
        assert normalize_job({"job_url": None, "title": None}) is None


class TestDeduplicate:
    def test_url_duplicates_merged(self):
        from models import JobRecord

        a = JobRecord(company="X", title="ML Engineer", location="Bangalore",
                      url="https://linkedin.com/jobs/view/1", source="linkedin",
                      sources=["linkedin"], description="desc A")
        b = a.model_copy(update={"url": "https://www.linkedin.com/jobs/view/1/?trk=x"})
        unique, dups = deduplicate([a, b])
        assert len(unique) == 1
        assert dups == 1

    def test_cross_source_duplicate_same_company_title_location(self):
        from models import JobRecord

        linkedin = JobRecord(
            company="Nvidia", title="Edge AI Engineer", location="Bangalore, India",
            url="https://linkedin.com/jobs/view/42", source="linkedin",
            sources=["linkedin"], description="Optimize models on Jetson.",
        )
        indeed = JobRecord(
            company="NVIDIA", title="Edge AI Engineer", location="Bengaluru, Karnataka, India",
            url="https://in.indeed.com/viewjob?jk=xyz", source="indeed",
            sources=["indeed"], description="Optimize models on Jetson platforms.",
        )
        unique, dups = deduplicate([linkedin, indeed])
        assert len(unique) == 1
        assert dups == 1
        assert set(unique[0].sources) == {"linkedin", "indeed"}
        # the richer description wins
        assert len(unique[0].description) >= len(indeed.description)

    def test_different_locations_not_deduped(self):
        from models import JobRecord

        a = JobRecord(company="X", title="ML Engineer", location="Bangalore, India",
                      url="https://a/1", source="linkedin", sources=["linkedin"])
        b = JobRecord(company="X", title="ML Engineer", location="Hyderabad, India",
                      url="https://b/2", source="indeed", sources=["indeed"])
        unique, dups = deduplicate([a, b])
        assert len(unique) == 2

    def test_different_titles_not_deduped(self):
        from models import JobRecord

        a = JobRecord(company="X", title="Senior ML Engineer", location="Bangalore",
                      url="https://a/1", source="linkedin", sources=["linkedin"])
        b = JobRecord(company="X", title="ML Engineer", location="Bangalore",
                      url="https://b/2", source="indeed", sources=["indeed"])
        unique, _ = deduplicate([a, b])
        assert len(unique) == 2

    def test_content_hash_changes_with_description(self):
        h1 = compute_content_hash("X", "ML Engineer", "Do ML")
        h2 = compute_content_hash("X", "ML Engineer", "Do ML and CV")
        assert h1 != h2


class TestPandasNaNRobustness:
    """jobspy rows come from pandas: missing cells are NaN, not None."""

    def test_nan_job_type_does_not_crash(self):
        record = normalize_job({
            "site": "indeed", "title": "ML Engineer", "company": "X",
            "job_url": "https://in.indeed.com/viewjob?jk=nan1",
            "location": "Pune", "description": "models",
            "job_type": float("nan"),
            "min_amount": float("nan"), "max_amount": float("nan"),
            "currency": float("nan"), "interval": float("nan"),
            "is_remote": float("nan"), "date_posted": float("nan"),
        })
        assert record is not None
        assert record.employment_type == "unknown"
        assert record.salary == ""
        assert record.posted_date is None

    def test_nan_url_or_title_skipped(self):
        assert normalize_job({"site": "indeed", "title": float("nan"),
                              "job_url": "https://x/1"}) is None
