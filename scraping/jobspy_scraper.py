"""
JobSpy scraping wrapper (LinkedIn + Indeed + ...).

* one reusable function per (query, location) call, failures isolated
* lazy import of jobspy (heavy pandas dependency)
* search queries generated deterministically from the candidate profile
* returns raw row dicts; normalisation/dedup happens in scraping.normalizer

JobSpy notes:
    * the returned DataFrame always contains the columns
      site, job_url, title, company, location, date_posted, description, ...
    * LinkedIn guest scraping is rate-limited; set LINKEDIN_USERNAME /
      LINKEDIN_PASSWORD env vars for authenticated scraping when needed
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from config import AppConfig
from llm.schemas import CandidateProfile

logger = logging.getLogger(__name__)

# fallback queries when the profile gives us nothing specific
_FALLBACK_QUERIES = ["Machine Learning Engineer", "AI Engineer"]


def generate_search_queries(profile: CandidateProfile, cfg: AppConfig) -> List[str]:
    """Deterministic query generation from the candidate profile (section 38)."""
    queries: List[str] = []
    for role in profile.target_roles:
        if role and role.strip() and role.strip() not in queries:
            queries.append(role.strip())
    # add the strongest core skills as query seeds when we have room
    for skill in profile.core_skills:
        if len(queries) >= cfg.scraping.max_search_queries:
            break
        query = f"{skill} Engineer"
        if query.lower() not in {q.lower() for q in queries}:
            queries.append(query)
    if not queries:
        queries = list(_FALLBACK_QUERIES)
    return queries[: cfg.scraping.max_search_queries]


def scrape_jobs(
    queries: List[str],
    cfg: AppConfig,
    locations: Optional[List[str]] = None,
    stats=None,
) -> List[Dict[str, Any]]:
    """
    Run JobSpy for every (query, location) combination (bounded by
    max_scrape_calls) and yield raw job rows.  Never raises: per-call errors
    are logged and skipped.
    """
    try:
        from jobspy import scrape_jobs as jobspy_scrape
    except ImportError as exc:
        raise RuntimeError(
            "jobspy is not installed. Install requirements.txt "
            "(pip install python-jobspy) or run with a pre-populated database."
        ) from exc

    locations = [l for l in (locations or list(cfg.locations) or ["India"])]
    sites = list(cfg.scraping.sites) or ["indeed"]
    hours_old = max(1, cfg.max_job_age_days * 24)

    calls: List[tuple] = []
    for location in locations:
        for query in queries:
            calls.append((query, location))
    calls = calls[: cfg.scraping.max_scrape_calls]

    logger.info(
        "Scraping %d call(s): sites=%s queries=%s locations=%s (max %d results each)",
        len(calls), sites, queries, locations, cfg.scraping.results_wanted,
    )

    rows: List[Dict[str, Any]] = []
    seen_urls = set()

    for index, (query, location) in enumerate(calls, start=1):
        logger.info("[%d/%d] Scraping '%s' in '%s'", index, len(calls), query, location)
        try:
            frame = jobspy_scrape(
                site_name=sites,
                search_term=query,
                location=location,
                results_wanted=cfg.scraping.results_wanted,
                hours_old=hours_old,
                country_indeed=cfg.scraping.country_indeed,
                linkedin_fetch_description=cfg.scraping.linkedin_fetch_description,
            )
        except Exception as exc:
            logger.warning("Scrape call failed for '%s'/%s: %s", query, location, exc)
            continue

        count = 0
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                raw = row.to_dict()
                url = str(raw.get("job_url") or raw.get("job_url_direct") or "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                rows.append(raw)
                count += 1
        logger.info("[%d/%d]   -> %d new jobs", index, len(calls), count)
        if stats is not None:
            stats.jobs_scraped += count
        if index < len(calls) and cfg.scraping.request_delay_seconds > 0:
            time.sleep(cfg.scraping.request_delay_seconds)

    logger.info("Scraping finished: %d raw jobs collected", len(rows))
    return rows
