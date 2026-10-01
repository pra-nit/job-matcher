"""
SQLite persistence (sections 21-22, 28).

Tables:
    candidate_profile  — cached Qwen resume extractions (resume hash + model)
    jobs               — deduplicated postings with first_seen/last_seen/sources
    job_requirements   — extracted requirements, cached per job content
    job_matches        — final results, invalidated via config fingerprint
    skills             — the skill taxonomy (populated at start-up)
    search_runs        — one row per pipeline run with stats JSON

All writes are idempotent upserts; all reads are parameterised.  The store
implements incremental processing: unchanged jobs (same content_hash) keep
their cached requirements/matches unless the config fingerprint or the
--reprocess flag says otherwise.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm.schemas import CandidateProfile, JobRequirements
from models import JobRecord, JobMatchResult
from matching import skill_taxonomy

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS candidate_profile (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    model TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(fingerprint, model)
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    company TEXT,
    title TEXT,
    location TEXT,
    work_mode TEXT,
    employment_type TEXT,
    description TEXT,
    salary TEXT,
    source TEXT,
    sources_json TEXT,
    posted_date TEXT,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    content_hash TEXT,
    job_level_raw TEXT,
    experience_range_raw TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_content_hash ON jobs(content_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_last_seen ON jobs(last_seen);
CREATE TABLE IF NOT EXISTS job_requirements (
    job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    min_experience REAL,
    max_experience REAL,
    experience_text TEXT,
    experience_domain TEXT,
    experience_confidence REAL,
    required_skills_json TEXT,
    preferred_skills_json TEXT,
    seniority TEXT,
    employment_type TEXT,
    work_mode TEXT,
    responsibilities_json TEXT,
    extraction_method TEXT,
    requirements_hash TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS job_matches (
    job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    embedding_score REAL,
    experience_score REAL,
    skill_score REAL,
    domain_score REAL,
    responsibility_score REAL,
    career_alignment_score REAL,
    final_score REAL,
    experience_fit TEXT,
    verdict TEXT,
    reasoning TEXT,
    strengths_json TEXT,
    concerns_json TEXT,
    rejections_json TEXT,
    analysis_json TEXT,
    result_json TEXT,
    content_hash TEXT,
    config_hash TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_config ON job_matches(config_hash);
CREATE TABLE IF NOT EXISTS embeddings_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    category TEXT,
    is_alias INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS search_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    sites_json TEXT,
    queries_json TEXT,
    locations_json TEXT,
    model TEXT,
    embedding_model TEXT,
    stats_json TEXT,
    status TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SqliteStore:
    """Thin, explicit persistence layer. No ORM, no global state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()
        logger.debug("Opened SQLite store at %s", self.path)

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except sqlite3.Error:  # pragma: no cover
            pass

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _commit(self) -> None:
        self._conn.commit()

    # -- candidate profile ----------------------------------------------------
    def save_profile(self, fingerprint: str, model: str, profile: CandidateProfile) -> None:
        self._conn.execute(
            """
            INSERT INTO candidate_profile (fingerprint, model, profile_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint, model) DO UPDATE SET
                profile_json=excluded.profile_json, updated_at=excluded.updated_at
            """,
            (fingerprint, model, profile.model_dump_json(), _now(), _now()),
        )
        self._commit()

    def load_profile(self, fingerprint: str, model: Optional[str] = None) -> Optional[CandidateProfile]:
        if model:
            row = self._conn.execute(
                "SELECT profile_json FROM candidate_profile WHERE fingerprint=? AND model=?",
                (fingerprint, model),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT profile_json FROM candidate_profile WHERE fingerprint=? ORDER BY id DESC",
                (fingerprint,),
            ).fetchone()
        if row is None:
            return None
        try:
            return CandidateProfile.model_validate_json(row["profile_json"])
        except Exception as exc:
            logger.warning("Cached profile failed validation: %s", exc)
            return None

    def latest_profile(self) -> Optional[CandidateProfile]:
        row = self._conn.execute(
            "SELECT profile_json FROM candidate_profile ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        try:
            return CandidateProfile.model_validate_json(row["profile_json"])
        except Exception:
            return None

    # -- jobs -----------------------------------------------------------------
    def upsert_jobs(self, jobs: Iterable[JobRecord]) -> Tuple[List[JobRecord], int]:
        """
        Insert new jobs / update last_seen (+sources) for existing ones.
        Returns (persisted_records, new_count).  Persisted records carry their
        database id and the stored first_seen values.
        """
        now = _now()
        persisted: List[JobRecord] = []
        new_count = 0
        for job in jobs:
            if not job.url:
                continue
            existing = self._conn.execute(
                "SELECT id, first_seen, sources_json, description, content_hash, posted_date FROM jobs WHERE url=?",
                (job.url,),
            ).fetchone()
            if existing is None:
                sources = json.dumps(job.sources or ([job.source] if job.source else []))
                cursor = self._conn.execute(
                    """
                    INSERT INTO jobs (url, company, title, location, work_mode, employment_type,
                                      description, salary, source, sources_json, posted_date,
                                      first_seen, last_seen, content_hash, job_level_raw,
                                      experience_range_raw)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.url, job.company, job.title, job.location, job.work_mode,
                        job.employment_type, job.description, job.salary, job.source, sources,
                        job.posted_date.isoformat() if job.posted_date else None,
                        now, now, job.content_hash, job.job_level_raw, job.experience_range_raw,
                    ),
                )
                job.job_id = cursor.lastrowid
                job.first_seen = datetime.now(timezone.utc)
                job.last_seen = job.first_seen
                new_count += 1
            else:
                merged_sources = self._merge_sources(existing["sources_json"], job)
                description = job.description or existing["description"] or ""
                content_hash = job.content_hash or existing["content_hash"]
                posted = job.posted_date.isoformat() if job.posted_date else existing["posted_date"]
                self._conn.execute(
                    """
                    UPDATE jobs SET last_seen=?, sources_json=?, company=?, title=?, location=?,
                        work_mode=COALESCE(NULLIF(?, ''), work_mode),
                        employment_type=COALESCE(NULLIF(?, ''), employment_type),
                        description=?, salary=COALESCE(NULLIF(?, ''), salary),
                        posted_date=COALESCE(?, posted_date), content_hash=?
                    WHERE id=?
                    """,
                    (
                        now, json.dumps(merged_sources), job.company or "", job.title or "",
                        job.location or "", job.work_mode or "", job.employment_type or "",
                        description, job.salary or "", posted, content_hash, existing["id"],
                    ),
                )
                job.job_id = existing["id"]
                first_seen = existing["first_seen"]
                job.first_seen = datetime.fromisoformat(first_seen) if first_seen else None
                job.last_seen = datetime.now(timezone.utc)
                job.sources = merged_sources
            persisted.append(job)
        self._commit()
        logger.info("Persisted %d job(s) (%d new)", len(persisted), new_count)
        return persisted, new_count

    @staticmethod
    def _merge_sources(existing_json: Optional[str], job: JobRecord) -> List[str]:
        try:
            existing = json.loads(existing_json) if existing_json else []
        except json.JSONDecodeError:
            existing = []
        merged = list(existing)
        for src in (job.sources or []) + ([job.source] if job.source else []):
            if src and src not in merged:
                merged.append(src)
        return merged

    def get_all_jobs(self, max_age_days: Optional[int] = None) -> List[JobRecord]:
        """All stored jobs (optionally only those seen recently)."""
        query = "SELECT * FROM jobs"
        params: tuple = ()
        if max_age_days is not None:
            query += " WHERE last_seen >= datetime('now', ?)"
            params = (f"-{int(max_age_days)} days",)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_job(row) for row in rows]

    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        try:
            sources = json.loads(row["sources_json"] or "[]")
        except json.JSONDecodeError:
            sources = []
        posted = None
        if row["posted_date"]:
            try:
                posted = datetime.fromisoformat(row["posted_date"])
            except ValueError:
                posted = None
        first_seen = None
        if row["first_seen"]:
            try:
                first_seen = datetime.fromisoformat(row["first_seen"])
            except ValueError:
                first_seen = None
        return JobRecord(
            job_id=row["id"],
            company=row["company"] or "",
            title=row["title"] or "",
            location=row["location"] or "",
            description=row["description"] or "",
            url=row["url"],
            source=row["source"] or "",
            sources=sources,
            salary=row["salary"] or "",
            employment_type=row["employment_type"] or "unknown",
            work_mode=row["work_mode"] or "unknown",
            posted_date=posted,
            first_seen=first_seen,
            last_seen=first_seen,
            content_hash=row["content_hash"] or "",
            job_level_raw=row["job_level_raw"] or "",
            experience_range_raw=row["experience_range_raw"] or "",
        )

    # -- job requirements -----------------------------------------------------
    def save_requirements(self, job_id: int, requirements: JobRequirements, requirements_hash: str) -> None:
        self._conn.execute(
            """
            INSERT INTO job_requirements (job_id, min_experience, max_experience, experience_text,
                                          experience_domain, experience_confidence, required_skills_json,
                                          preferred_skills_json, seniority, employment_type, work_mode,
                                          responsibilities_json, extraction_method, requirements_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                min_experience=excluded.min_experience, max_experience=excluded.max_experience,
                experience_text=excluded.experience_text, experience_domain=excluded.experience_domain,
                experience_confidence=excluded.experience_confidence,
                required_skills_json=excluded.required_skills_json,
                preferred_skills_json=excluded.preferred_skills_json, seniority=excluded.seniority,
                employment_type=excluded.employment_type, work_mode=excluded.work_mode,
                responsibilities_json=excluded.responsibilities_json,
                extraction_method=excluded.extraction_method,
                requirements_hash=excluded.requirements_hash, updated_at=excluded.updated_at
            """,
            (
                job_id, requirements.min_experience_years, requirements.max_experience_years,
                requirements.experience_text, requirements.experience_domain.value,
                requirements.experience_confidence,
                json.dumps(requirements.required_skills), json.dumps(requirements.preferred_skills),
                requirements.seniority.value, requirements.employment_type, requirements.work_mode,
                json.dumps(requirements.responsibilities), requirements.extraction_method,
                requirements_hash, _now(),
            ),
        )
        self._commit()

    def load_requirements(self, job_id: int) -> Optional[Tuple[JobRequirements, str]]:
        row = self._conn.execute(
            "SELECT * FROM job_requirements WHERE job_id=?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            requirements = JobRequirements(
                min_experience_years=row["min_experience"],
                max_experience_years=row["max_experience"],
                experience_text=row["experience_text"] or "",
                experience_domain=row["experience_domain"] or "general",
                experience_confidence=row["experience_confidence"] or 0.0,
                required_skills=json.loads(row["required_skills_json"] or "[]"),
                preferred_skills=json.loads(row["preferred_skills_json"] or "[]"),
                seniority=row["seniority"] or "unknown",
                employment_type=row["employment_type"] or "unknown",
                work_mode=row["work_mode"] or "unknown",
                responsibilities=json.loads(row["responsibilities_json"] or "[]"),
                extraction_method=row["extraction_method"] or "none",
            )
            return requirements, row["requirements_hash"] or ""
        except Exception as exc:
            logger.warning("Could not load cached requirements for job %d: %s", job_id, exc)
            return None

    # -- job matches ----------------------------------------------------------
    def save_match(self, result: JobMatchResult, config_hash: str) -> None:
        from models import match_result_to_dict

        self._conn.execute(
            """
            INSERT INTO job_matches (job_id, embedding_score, experience_score, skill_score,
                                     domain_score, responsibility_score, career_alignment_score,
                                     final_score, experience_fit, verdict, reasoning,
                                     strengths_json, concerns_json, rejections_json, analysis_json,
                                     result_json, content_hash, config_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                embedding_score=excluded.embedding_score, experience_score=excluded.experience_score,
                skill_score=excluded.skill_score, domain_score=excluded.domain_score,
                responsibility_score=excluded.responsibility_score,
                career_alignment_score=excluded.career_alignment_score,
                final_score=excluded.final_score, experience_fit=excluded.experience_fit,
                verdict=excluded.verdict, reasoning=excluded.reasoning,
                strengths_json=excluded.strengths_json, concerns_json=excluded.concerns_json,
                rejections_json=excluded.rejections_json, analysis_json=excluded.analysis_json,
                result_json=excluded.result_json, content_hash=excluded.content_hash,
                config_hash=excluded.config_hash, created_at=excluded.created_at
            """,
            (
                result.job.job_id, result.embedding_score, result.experience_score,
                result.skill_score, result.domain_score, result.responsibility_score,
                result.career_alignment_score, result.final_score,
                result.experience.fit.value, result.verdict.value, result.reasoning,
                json.dumps(result.strengths), json.dumps(result.concerns),
                json.dumps(
                    [
                        {"reason": r.reason.value, "detail": r.detail, "stage": r.stage}
                        for r in result.rejections
                    ]
                ),
                result.qwen_analysis.model_dump_json() if result.qwen_analysis else None,
                json.dumps(match_result_to_dict(result)),
                result.job.content_hash,
                config_hash, _now(),
            ),
        )
        self._commit()

    def load_match(self, job_id: int, config_hash: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM job_matches WHERE job_id=? AND config_hash=?",
            (job_id, config_hash),
        ).fetchone()
        return dict(row) if row else None

    def load_full_results(self, config_hash: Optional[str] = None) -> List[JobMatchResult]:
        """Reconstructed results for --export (newest per job)."""
        from models import match_result_from_dict

        if config_hash:
            rows = self._conn.execute(
                "SELECT job_id, result_json FROM job_matches WHERE config_hash=? "
                "ORDER BY created_at DESC",
                (config_hash,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT job_id, result_json FROM job_matches ORDER BY created_at DESC"
            ).fetchall()
        results: List[JobMatchResult] = []
        seen_jobs = set()
        for row in rows:
            if row["job_id"] in seen_jobs or not row["result_json"]:
                continue
            seen_jobs.add(row["job_id"])
            try:
                results.append(match_result_from_dict(json.loads(row["result_json"])))
            except Exception as exc:
                logger.warning("Could not reconstruct cached result for job %d: %s", row["job_id"], exc)
        return results

    # -- embedding cache --------------------------------------------------------
    def save_embedding(self, cache_key: str, model: str, vector) -> None:
        import numpy as np

        vector = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            """
            INSERT INTO embeddings_cache (cache_key, model, dim, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                model=excluded.model, dim=excluded.dim, vector=excluded.vector,
                created_at=excluded.created_at
            """,
            (cache_key, model, int(vector.shape[0]), vector.tobytes(), _now()),
        )
        self._commit()

    def load_embedding(self, cache_key: str, dim: Optional[int] = None):
        import numpy as np

        row = self._conn.execute(
            "SELECT dim, vector FROM embeddings_cache WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        if dim is not None and row["dim"] != dim:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32)

    def load_match_results(self, config_hash: Optional[str] = None) -> List[Dict[str, Any]]:
        if config_hash:
            rows = self._conn.execute(
                "SELECT m.*, j.company, j.title, j.url, j.location, j.sources_json, j.work_mode, "
                "j.employment_type, j.posted_date, j.description, j.salary "
                "FROM job_matches m JOIN jobs j ON j.id = m.job_id WHERE m.config_hash=?",
                (config_hash,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT m.*, j.company, j.title, j.url, j.location, j.sources_json, j.work_mode, "
                "j.employment_type, j.posted_date, j.description, j.salary "
                "FROM job_matches m JOIN jobs j ON j.id = m.job_id"
            ).fetchall()
        return [dict(r) for r in rows]

    # -- skills ---------------------------------------------------------------
    def sync_skills(self) -> int:
        rows = skill_taxonomy.taxonomy_rows()
        self._conn.executemany(
            "INSERT OR IGNORE INTO skills (name, category, is_alias) VALUES (?, ?, ?)",
            [(r["name"], r["category"], r["is_alias"]) for r in rows],
        )
        self._commit()
        return len(rows)

    # -- search runs ----------------------------------------------------------
    def start_run(self, sites: List[str], queries: List[str], locations: List[str],
                  model: str, embedding_model: str) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO search_runs (started_at, sites_json, queries_json, locations_json,
                                     model, embedding_model, status)
            VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
            (_now(), json.dumps(sites), json.dumps(queries), json.dumps(locations),
             model, embedding_model),
        )
        self._commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int, stats: Dict[str, Any], status: str = "finished") -> None:
        self._conn.execute(
            "UPDATE search_runs SET finished_at=?, stats_json=?, status=? WHERE id=?",
            (_now(), json.dumps(stats, default=str), status, run_id),
        )
        self._commit()

    # -- maintenance ----------------------------------------------------------
    def count_jobs(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"]
