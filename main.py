#!/usr/bin/env python3
"""
job_matcher — local job-search & job-matching pipeline.

    Resume PDF -> Qwen profile -> JobSpy scrape -> normalise/dedup -> SQLite
    -> deterministic rule engine -> embeddings (top-K) -> Qwen deep analysis
    -> deterministic scoring -> matched_jobs.csv / rejected_jobs.csv

Examples
--------
Full run:
    python main.py --resume resume.pdf --sites linkedin indeed \
        --location "Bangalore,Hyderabad,Remote" --model qwen3:4b \
        --embedding-model BAAI/bge-small-en-v1.5 --max-age-days 14 \
        --top-k 50 --output data/matched_jobs.csv

Scrape only (fill the database):
    python main.py --resume resume.pdf --scrape-only

Match using jobs already in the database:
    python main.py --resume resume.pdf --match-only

Re-export CSVs from cached results:
    python main.py --export

Offline (no Ollama; heuristic profile + regex requirements):
    python main.py --resume resume.pdf --match-only --offline
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import AppConfig, config_fingerprint, load_config
from database.sqlite_store import SqliteStore
from llm.ollama_client import OllamaClient
from llm.prompts import (
    JOB_ANALYSIS_SYSTEM,
    JOB_REQUIREMENTS_SYSTEM,
    PROMPT_VERSION,
    job_analysis_user,
    job_requirements_user,
)
from llm.schemas import CandidateProfile, JobAnalysis, JobRequirements
from matching import skill_taxonomy
from matching.experience import merge_requirements
from matching.rules import RuleEngine
from matching.scoring import score_job
from matching.semantic_matcher import rank_jobs
from models import JobMatchResult, JobRecord, PipelineStats, RejectionReason
from output import csv_writer
from resume.parser import extract_resume_text
from resume.profile_extractor import extract_profile
from scraping.jobspy_scraper import generate_search_queries, scrape_jobs as run_scrape
from scraping.normalizer import deduplicate, normalize_job

logger = logging.getLogger("job_matcher")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job_matcher",
        description="Local job-search and job-matching pipeline "
                    "(JobSpy + rules + embeddings + Ollama/Qwen).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resume", type=Path, help="Path to resume PDF/TXT")
    parser.add_argument("--profile-json", type=Path,
                        help="Use a pre-extracted candidate profile JSON instead of --resume")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"),
                        help="YAML configuration file")
    parser.add_argument("--sites", nargs="+", default=["linkedin", "indeed"],
                        choices=["linkedin", "indeed", "glassdoor", "zip_recruiter",
                                 "google", "naukri", "bayt"],
                        help="Job sites to scrape")
    parser.add_argument("--location", "--locations", dest="locations",
                        default="India",
                        help="Comma-separated locations, e.g. 'Bangalore,Hyderabad,Remote'")
    parser.add_argument("--model", default=None, help="Ollama model (e.g. qwen3:4b)")
    parser.add_argument("--embedding-model", default=None,
                        help="Sentence-transformers model (e.g. BAAI/bge-small-en-v1.5)")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Maximum job age in days (posted_date filter)")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-K jobs after embedding ranking sent to Qwen")
    parser.add_argument("--max-qwen-jobs", type=int, default=None,
                        help="Hard cap on Qwen analysis calls")
    parser.add_argument("--output", type=Path, default=None,
                        help="Matched jobs CSV path (default data/matched_jobs.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for matched/rejected CSVs")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path")
    parser.add_argument("--results-wanted", type=int, default=None,
                        help="Results wanted per scrape call")
    parser.add_argument("--must-have", nargs="*", default=None,
                        help="Skills every recommended job must involve")
    parser.add_argument("--soft-experience", action="store_true", default=None,
                        help="Enable soft experience mode (near-misses become BORDERLINE)")
    parser.add_argument("--offline", action="store_true",
                        help="Skip all LLM calls (heuristic profile + regex extraction)")
    parser.add_argument("--scrape-only", action="store_true",
                        help="Scrape and store jobs; skip matching")
    parser.add_argument("--match-only", action="store_true",
                        help="Match jobs already in the database; skip scraping")
    parser.add_argument("--export", action="store_true",
                        help="Re-generate CSVs from cached results; no scraping/LLM")
    parser.add_argument("--reprocess", action="store_true",
                        help="Ignore cached Qwen results and re-analyze")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def cli_overrides(args: argparse.Namespace) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    if args.model:
        overrides["qwen.model"] = args.model
    if args.embedding_model:
        overrides["embedding_model"] = args.embedding_model
    if args.max_age_days is not None:
        overrides["max_job_age_days"] = args.max_age_days
    if args.top_k is not None:
        overrides["matching.top_k_embedding"] = args.top_k
    if args.max_qwen_jobs is not None:
        overrides["matching.max_qwen_jobs"] = args.max_qwen_jobs
    if args.results_wanted is not None:
        overrides["scraping.results_wanted"] = args.results_wanted
    if args.sites:
        overrides["scraping.sites"] = tuple(args.sites)
    if args.soft_experience is not None:
        overrides["experience.soft_experience_mode"] = args.soft_experience
    if args.must_have is not None:
        overrides["must_have_skills"] = tuple(args.must_have)
    if args.locations:
        locations = tuple(
            loc.strip() for loc in str(args.locations).split(",") if loc.strip()
        )
        if locations:
            overrides["locations"] = locations
    return overrides


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for noisy in ("httpx", "urllib3", "sentence_transformers", "jobspy"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def resolve_profile(
    args: argparse.Namespace,
    cfg: AppConfig,
    store: SqliteStore,
    client: Optional[OllamaClient],
) -> CandidateProfile:
    if args.profile_json:
        profile = CandidateProfile.model_validate_json(
            Path(args.profile_json).read_text(encoding="utf-8")
        )
        logger.info("Loaded candidate profile from %s", args.profile_json)
        return profile

    if not args.resume:
        cached = store.latest_profile()
        if cached is not None:
            logger.info("No --resume given; using the latest cached profile")
            return cached
        raise SystemExit(
            "No candidate profile available. Provide --resume resume.pdf "
            "(or --profile-json), or run once with a resume to cache the profile."
        )

    resume_text = extract_resume_text(args.resume)
    return extract_profile(resume_text, client, cfg, store=store)


def scrape_stage(
    profile: CandidateProfile,
    cfg: AppConfig,
    store: SqliteStore,
    stats: PipelineStats,
) -> List[JobRecord]:
    """JobSpy -> normalise -> dedup -> SQLite. Returns persisted records."""
    queries = generate_search_queries(profile, cfg)
    logger.info("Search queries: %s", queries)
    raw_rows = run_scrape(queries, cfg, locations=list(cfg.locations), stats=stats)

    normalized: List[JobRecord] = []
    for row in raw_rows:
        record = normalize_job(row)
        if record is not None:
            normalized.append(record)
    logger.info("Normalized %d/%d raw job(s)", len(normalized), len(raw_rows))

    unique, duplicates = deduplicate(normalized)
    stats.duplicates_removed += duplicates
    logger.info("Deduplicated: %d unique job(s) (%d duplicates removed)",
                len(unique), duplicates)

    persisted, _new = store.upsert_jobs(unique)
    return persisted


def rules_stage(
    profile: CandidateProfile,
    cfg: AppConfig,
    jobs: List[JobRecord],
    stats: PipelineStats,
) -> Tuple[List[Tuple[JobRecord, JobRequirements]], List[JobMatchResult]]:
    """Deterministic hard filters before any embedding work."""
    engine = RuleEngine(profile, cfg)
    eligible: List[Tuple[JobRecord, JobRequirements]] = []
    rejected: List[JobMatchResult] = []

    total = len(jobs)
    for index, job in enumerate(jobs, start=1):
        if total > 20 and index % 25 == 0:
            logger.info("[rules %d/%d] filtering...", index, total)
        outcome = engine.apply(job)
        if outcome.passed:
            eligible.append((job, outcome.requirements))
        else:
            for rejection in outcome.rejections:
                _bump_rejection_stat(stats, rejection.reason)
            rejected.append(
                _stub_result(job, outcome.requirements, outcome, rejected=True)
            )
    logger.info(
        "Rules: %d eligible, %d rejected (experience=%d, role=%d, location=%d)",
        len(eligible), len(rejected),
        stats.rejected_by_experience, stats.rejected_by_role, stats.rejected_by_location,
    )
    return eligible, rejected


def _bump_rejection_stat(stats: PipelineStats, reason: RejectionReason) -> None:
    mapping = {
        RejectionReason.UNDERQUALIFIED: "rejected_by_experience",
        RejectionReason.OVERQUALIFIED: "rejected_by_experience",
        RejectionReason.IRRELEVANT_ROLE: "rejected_by_role",
        RejectionReason.EXCLUDED_ROLE: "rejected_by_role",
        RejectionReason.LOW_CAREER_ALIGNMENT: "rejected_by_role",
        RejectionReason.LOCATION_MISMATCH: "rejected_by_location",
        RejectionReason.STALE_JOB: "rejected_by_freshness",
        RejectionReason.SENIORITY_MISMATCH: "rejected_by_seniority",
        RejectionReason.EMPLOYMENT_TYPE_MISMATCH: "rejected_by_employment",
        RejectionReason.WORK_MODE_MISMATCH: "rejected_by_work_mode",
        RejectionReason.MISSING_MUST_HAVE_SKILL: "rejected_by_must_have",
    }
    field = mapping.get(reason)
    if field:
        setattr(stats, field, getattr(stats, field) + 1)


def _stub_result(job, requirements, outcome, rejected: bool) -> JobMatchResult:
    from models import Verdict

    result = JobMatchResult(
        job=job,
        requirements=requirements,
        experience=outcome.experience,
        hard_rejected=rejected,
        rejections=outcome.rejections,
        verdict=Verdict.REJECTED,
    )
    result.reasoning = "; ".join(
        f"{r.reason.value}: {r.detail}" for r in outcome.rejections
    ) or "rejected by rule engine"
    return result


# ---------------------------------------------------------------------------
# Qwen stages (top-K only)
# ---------------------------------------------------------------------------
def requirements_for_job(
    job: JobRecord,
    engine: RuleEngine,
    client: Optional[OllamaClient],
    cfg: AppConfig,
    store: SqliteStore,
    reprocess: bool,
    stats: PipelineStats,
) -> Tuple[JobRequirements, bool]:
    """Qwen requirement extraction with SQLite cache (section 22)."""
    req_hash = hashlib.sha256(
        f"{PROMPT_VERSION}|{cfg.qwen.model}|{job.content_hash}".encode("utf-8")
    ).hexdigest()

    if not reprocess and job.job_id is not None:
        cached = store.load_requirements(job.job_id)
        if cached is not None and cached[1] == req_hash:
            stats.qwen_cache_hits += 1
            return cached[0], True

    regex_req = engine.extract_requirements(job)
    llm_req: Optional[JobRequirements] = None
    if client is not None:
        try:
            llm_req = client.chat_json(
                JOB_REQUIREMENTS_SYSTEM, job_requirements_user(job), schema_model=JobRequirements
            )
        except Exception as exc:
            logger.warning("Qwen requirements failed for '%s': %s", job.title[:60], exc)
        if llm_req is None:
            stats.qwen_failures += 1
    merged = merge_requirements(regex_req, llm_req)

    if job.job_id is not None:
        store.save_requirements(job.job_id, merged, req_hash)
    return merged, False


def analyze_job(
    profile: CandidateProfile,
    job: JobRecord,
    requirements: JobRequirements,
    client: Optional[OllamaClient],
    engine: RuleEngine,
    cfg: AppConfig,
    store: SqliteStore,
    config_hash: str,
    reprocess: bool,
    stats: PipelineStats,
) -> Optional[JobAnalysis]:
    """Qwen deep analysis with cache; None when unavailable (deterministic fallback used)."""
    cached_match = None
    if not reprocess and job.job_id is not None:
        cached_match = store.load_match(job.job_id, config_hash)
        if cached_match is not None and cached_match.get("content_hash") == job.content_hash \
                and cached_match.get("analysis_json"):
            try:
                stats.qwen_cache_hits += 1
                return JobAnalysis.model_validate_json(cached_match["analysis_json"])
            except Exception:
                pass

    if client is None:
        return None

    outcome = engine.apply(job, requirements)
    skills = skill_taxonomy.match_skills(
        list(requirements.required_skills) + list(requirements.preferred_skills),
        list(profile.core_skills) + list(profile.skills) + list(profile.secondary_skills),
        required=requirements.required_skills,
        preferred=requirements.preferred_skills,
    )
    try:
        analysis = client.chat_json(
            JOB_ANALYSIS_SYSTEM,
            job_analysis_user(profile, job, requirements, outcome.experience, skills),
            schema_model=JobAnalysis,
        )
    except Exception as exc:
        logger.warning("Qwen analysis failed for '%s': %s", job.title[:60], exc)
        analysis = None
    if analysis is None:
        stats.qwen_failures += 1
    else:
        stats.qwen_analyzed += 1
    return analysis


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    cfg = load_config(args.config, cli_overrides(args))
    config_hash = config_fingerprint(cfg)

    db_path = args.db or cfg.paths.database
    store = SqliteStore(db_path)
    synced_skills = store.sync_skills()
    logger.debug("Synced %d taxonomy rows into the skills table", synced_skills)

    stats = PipelineStats()
    client: Optional[OllamaClient] = None

    try:
        if args.export:
            return export_stage(args, cfg, store, config_hash)

        # ---- client / profile --------------------------------------------
        if not args.offline:
            client = OllamaClient(cfg.qwen)
            if not client.ping():
                logger.warning(
                    "Ollama not reachable or model missing at %s (model '%s'). "
                    "Continuing in OFFLINE mode — install/start Ollama and "
                    "'ollama pull %s' for full quality.",
                    cfg.qwen.base_url, cfg.qwen.model, cfg.qwen.model,
                )
                client = None

        profile = resolve_profile(args, cfg, store, client)
        logger.info(
            "Candidate: %s | %.1fy total, %.1fy relevant | seniority=%s | targets=%s",
            profile.name or "unknown", profile.total_experience_years,
            profile.relevant_experience_years, profile.inferred_seniority.value,
            ", ".join(profile.target_roles[:4]),
        )

        run_id = store.start_run(
            sites=list(cfg.scraping.sites), queries=[], locations=list(cfg.locations),
            model=cfg.qwen.model, embedding_model=cfg.embedding_model,
        )

        # ---- scraping ------------------------------------------------------
        if args.match_only:
            jobs = store.get_all_jobs()
            logger.info("Match-only mode: loaded %d job(s) from %s", len(jobs), db_path)
            stats.jobs_scraped = len(jobs)
        else:
            jobs = scrape_stage(profile, cfg, store, stats)

        if args.scrape_only:
            store.finish_run(run_id, stats.__dict__, status="scrape_only")
            logger.info("Scrape-only mode: %d job(s) stored in %s", store.count_jobs(), db_path)
            return 0

        # ---- rules ---------------------------------------------------------
        eligible, rejected = rules_stage(profile, cfg, jobs, stats)
        stats.embedding_candidates = len(eligible)

        # persist deterministic rejections so --export reproduces them
        for stub in rejected:
            if stub.job.job_id is not None:
                store.save_match(stub, config_hash)

        if not eligible:
            logger.warning("No eligible jobs after rule filtering; nothing to analyse")
            _finish(args, cfg, store, profile, rejected, stats, run_id)
            return 0

        # ---- embeddings ------------------------------------------------------
        from matching.embeddings import get_provider

        provider = get_provider(cfg.embedding_model, allow_fallback=args.offline)
        top_jobs = rank_jobs(profile, eligible, provider, cfg, store=store)
        logger.info("Embedding shortlist: %d job(s) proceed to Qwen", len(top_jobs))
        if not top_jobs:
            logger.warning("No jobs passed the embedding similarity threshold")

        # ---- Qwen analysis + deterministic scoring ---------------------------
        engine = RuleEngine(profile, cfg)
        candidate_skills = (
            list(profile.core_skills) + list(profile.skills) + list(profile.secondary_skills)
        )
        max_qwen = cfg.matching.max_qwen_jobs
        results: List[JobMatchResult] = list(rejected)

        for index, (job, _regex_requirements, _similarity) in enumerate(top_jobs[:max_qwen], start=1):
            logger.info("[Qwen %d/%d] %s @ %s", index, min(len(top_jobs), max_qwen),
                        job.title[:60], job.company)
            requirements, from_cache = requirements_for_job(
                job, engine, client, cfg, store, args.reprocess, stats
            )
            if not from_cache:
                stats.qwen_analyzed += 0  # requirements call counted separately

            analysis = analyze_job(
                profile, job, requirements, client, engine, cfg, store,
                config_hash, args.reprocess, stats,
            )

            # re-run rules with refined requirements: Qwen can only ADD
            # rejections here, never remove a deterministic one (RULE 3)
            outcome = engine.apply(job, requirements)
            post_qwen_rejections = outcome.rejections
            if post_qwen_rejections:
                stats.hard_rejected_after_qwen += 1
                for rejection in post_qwen_rejections:
                    _bump_rejection_stat(stats, rejection.reason)
                logger.info(
                    "  -> Qwen-refined requirements triggered rejection: %s",
                    post_qwen_rejections[0].reason.value,
                )

            skills = skill_taxonomy.match_skills(
                list(requirements.required_skills) + list(requirements.preferred_skills),
                candidate_skills,
                required=requirements.required_skills,
                preferred=requirements.preferred_skills,
            )
            similarity = next(
                (sim for j, _r, sim in top_jobs if j.url == job.url), 0.0
            )
            result = score_job(
                profile, job, requirements, outcome.experience, skills,
                embedding_similarity=similarity,
                rejections=post_qwen_rejections,
                qwen_analysis=analysis,
                cfg=cfg,
            )
            results.append(result)
            if job.job_id is not None:
                store.save_match(result, config_hash)

        # jobs beyond max_qwen keep their rule-stage evaluation only
        overflow = [job for job, _, _ in top_jobs[max_qwen:]]
        for job in overflow:
            requirements = engine.extract_requirements(job)
            outcome = engine.apply(job, requirements)
            skills = skill_taxonomy.match_skills(
                list(requirements.required_skills) + list(requirements.preferred_skills),
                candidate_skills,
                required=requirements.required_skills,
                preferred=requirements.preferred_skills,
            )
            result = score_job(
                profile, job, requirements, outcome.experience, skills,
                embedding_similarity=0.0,
                rejections=outcome.rejections,
                qwen_analysis=None,
                cfg=cfg,
            )
            results.append(result)
            if job.job_id is not None:
                store.save_match(result, config_hash)

        _finish(args, cfg, store, profile, results, stats, run_id)
        return 0
    finally:
        store.close()


def _finish(
    args: argparse.Namespace,
    cfg: AppConfig,
    store: SqliteStore,
    profile: CandidateProfile,
    results: List[JobMatchResult],
    stats: PipelineStats,
    run_id: int,
) -> None:
    stats.final_matches = sum(1 for r in results if r.verdict.value == "MATCH")
    stats.final_borderline = sum(1 for r in results if r.verdict.value == "BORDERLINE")
    stats.final_rejected = sum(1 for r in results if r.verdict.value == "REJECTED")
    store.finish_run(run_id, stats.__dict__)

    output_dir = args.output_dir or cfg.paths.output_dir
    matched_path = args.output or (output_dir / "matched_jobs.csv")
    rejected_path = output_dir / "rejected_jobs.csv"
    csv_writer.write_matched_csv(results, profile, matched_path)
    csv_writer.write_rejected_csv(results, profile, rejected_path)
    stats.log_summary(logger)
    logger.info("Matched jobs:  %s", matched_path)
    logger.info("Rejected jobs: %s", rejected_path)


def export_stage(
    args: argparse.Namespace,
    cfg: AppConfig,
    store: SqliteStore,
    config_hash: str,
) -> int:
    results = store.load_full_results(config_hash) or store.load_full_results()
    if not results:
        logger.warning("No cached results found in %s; run the pipeline first", store.path)
        return 1
    profile = store.latest_profile()
    if profile is None:
        from llm.schemas import CandidateProfile as _P

        profile = _P()
    logger.info("Exporting %d cached result(s)", len(results))
    output_dir = args.output_dir or cfg.paths.output_dir
    matched_path = args.output or (output_dir / "matched_jobs.csv")
    csv_writer.write_matched_csv(results, profile, matched_path)
    csv_writer.write_rejected_csv(results, profile, output_dir / "rejected_jobs.csv")
    stats = PipelineStats(
        final_matches=sum(1 for r in results if r.verdict.value == "MATCH"),
        final_borderline=sum(1 for r in results if r.verdict.value == "BORDERLINE"),
        final_rejected=sum(1 for r in results if r.verdict.value == "REJECTED"),
    )
    stats.log_summary(logger)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(sys.argv) == 1 and argv is None:
        parser.print_help()
        return 0
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
