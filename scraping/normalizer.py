"""
Job normalisation and deduplication (spec sections 20-21).

Raw rows from JobSpy (LinkedIn / Indeed / ...) are turned into
:class:`models.JobRecord` objects with:

    * cleaned descriptions (HTML stripped, whitespace collapsed)
    * normalised work mode / employment type
    * canonical URLs (tracking parameters removed) used as the primary key
    * content hashes (company+title+description) for change detection
    * deterministic deduplication: canonical URL first, then
      (company, normalised title, normalised location)
"""
from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from models import EmploymentType, JobRecord, WorkMode
from matching.rules import detect_employment_type, detect_work_mode, normalize_location

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_BULLET_RE = re.compile(r"[\u2022\u25cf\u25aa\u2023\u2043]")


def clean_description(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r", "\n")
    text = _BULLET_RE.sub("\n- ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# URL canonicalisation
# ---------------------------------------------------------------------------
# query params that identify a job (kept) — everything else is tracking noise
_KEEP_PARAMS = {"jk", "jv", "currentjobid", "jobid", "id"}


def canonical_url(url: str) -> str:
    """Strip tracking parameters and normalise host/case for dedup keys."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip()
    if not parsed.scheme or not parsed.netloc:
        return url.strip()
    host = parsed.netloc.lower()
    # treat www.linkedin.com and linkedin.com as identical
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    kept = [(k, v) for k, v in parse_qsl(parsed.query) if k.lower() in _KEEP_PARAMS]
    query = urlencode(kept) if kept else ""
    return urlunparse((parsed.scheme.lower(), host, path, "", query, ""))


# ---------------------------------------------------------------------------
# Company / title normalisation
# ---------------------------------------------------------------------------
_COMPANY_SUFFIXES = re.compile(
    r"\b(pvt\.? ?ltd\.?|private limited|ltd\.?|limited|llc|inc\.?|llp|"
    r"gmbh|s\.?a\.?|pvt|technologies|technology|solutions|labs?)\b\.?",
    re.I,
)


def normalize_company(company: str) -> str:
    """Normalised company name used for dedup (not for display)."""
    name = (company or "").lower().strip()
    name = _COMPANY_SUFFIXES.sub(" ", name)
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9+ ]+", " ", (title or "").lower()).strip()


# ---------------------------------------------------------------------------
# Raw job -> JobRecord
# ---------------------------------------------------------------------------
def _is_missing(value) -> bool:
    """None, empty string or pandas NaN."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _pick(row: Dict, *names: str, default=None):
    """Defensively read a value from a raw scraped row (handles pandas NaN)."""
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in lowered and not _is_missing(lowered[name]):
            return lowered[name]
    return default


def _parse_date(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        from datetime import date

        if isinstance(value, date):
            return datetime(value.year, value.month, value.day)
    except Exception:  # pragma: no cover
        pass
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _format_salary(row: Dict) -> str:
    min_amount = _pick(row, "min_amount")
    max_amount = _pick(row, "max_amount")
    currency = _pick(row, "currency", default="")
    interval = _pick(row, "interval", default="")
    if min_amount is None and max_amount is None:
        return ""
    if isinstance(min_amount, float) and min_amount != min_amount:
        min_amount = None
    if isinstance(max_amount, float) and max_amount != max_amount:
        max_amount = None
    if min_amount is None and max_amount is None:
        return ""
    min_s = f"{min_amount:,.0f}" if isinstance(min_amount, (int, float)) else str(min_amount or "")
    max_s = f"{max_amount:,.0f}" if isinstance(max_amount, (int, float)) else str(max_amount or "")
    if min_s and max_s and min_s != max_s:
        amount = f"{min_s}-{max_s}"
    else:
        amount = min_s or max_s
    parts = [str(currency or ""), amount, f"{interval}" if interval else ""]
    return " ".join(p for p in parts if p).strip()


def _normalize_employment(raw_job_types) -> str:
    """JobSpy returns a list of JobType enums, strings, or pandas NaN."""
    if _is_missing(raw_job_types):
        return EmploymentType.UNKNOWN.value
    if isinstance(raw_job_types, (str, int, float)):
        raw_job_types = [raw_job_types]
    if not isinstance(raw_job_types, (list, tuple, set)):
        return EmploymentType.UNKNOWN.value
    values = []
    for jt in raw_job_types:
        value = getattr(jt, "value", None) or str(jt)
        if isinstance(value, tuple):  # JobType enum values are tuples of aliases
            value = value[0] if value else None
        if value:
            values.append(str(value).lower())
    joined = " ".join(values)
    mapping = {
        "fulltime": EmploymentType.FULL_TIME.value,
        "full-time": EmploymentType.FULL_TIME.value,
        "parttime": EmploymentType.PART_TIME.value,
        "contract": EmploymentType.CONTRACT.value,
        "temporary": EmploymentType.TEMPORARY.value,
        "internship": EmploymentType.INTERNSHIP.value,
    }
    for key, mapped in mapping.items():
        if key in joined:
            return mapped
    return EmploymentType.UNKNOWN.value


def normalize_job(row: Dict) -> Optional[JobRecord]:
    """
    Convert one raw scraped row into a JobRecord.  Returns None for rows
    without a usable URL or title (logged, never raises).
    """
    try:
        url = str(_pick(row, "job_url", "job_url_direct", "link", "url", default="") or "").strip()
        title = str(_pick(row, "title", "job_title", default="") or "").strip()
        company = str(_pick(row, "company", "company_name", default="") or "").strip()
        if not url or not title:
            logger.debug("Skipping row without url/title: %s", str(row)[:120])
            return None

        description = str(_pick(row, "description", "job_description", default="") or "")
        description = clean_description(description)

        location = str(_pick(row, "location", default="") or "").strip()
        site = str(_pick(row, "site", "source", default="") or "").strip().lower()
        posted = _parse_date(_pick(row, "date_posted", "posted_date"))
        salary = _format_salary(row)
        is_remote = _pick(row, "is_remote", default=None)

        employment = _normalize_employment(_pick(row, "job_type", default=None))
        if employment == EmploymentType.UNKNOWN.value:
            employment = detect_employment_type(title, description[:400]).value

        work_mode = detect_work_mode(
            "remote" if is_remote else "", title, location, description[:400]
        ).value

        content_hash = compute_content_hash(company, title, description)

        return JobRecord(
            company=company,
            title=title,
            location=location,
            description=description,
            url=canonical_url(url),
            source=site,
            sources=[site] if site else [],
            salary=salary,
            employment_type=employment,
            work_mode=work_mode,
            posted_date=posted,
            content_hash=content_hash,
            job_level_raw=str(_pick(row, "job_level", default="") or ""),
            experience_range_raw=str(_pick(row, "experience_range", default="") or ""),
        )
    except Exception as exc:  # never let one bad row kill the run
        logger.warning("Failed to normalise job row: %s (%s)", exc, str(row)[:120])
        return None


def compute_content_hash(company: str, title: str, description: str) -> str:
    normalised = re.sub(r"\s+", " ", f"{company}|{title}|{description}").strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def dedup_key_url(job: JobRecord) -> str:
    return job.url


def dedup_key_fuzzy(job: JobRecord) -> str:
    """company + normalised title + normalised location."""
    return "|".join(
        [
            normalize_company(job.company),
            normalize_title_key(job.title),
            normalize_location(job.location),
        ]
    )


def deduplicate(jobs: Iterable[JobRecord]) -> Tuple[List[JobRecord], int]:
    """
    Deduplicate a batch of jobs (section 20).

    Primary key: canonical URL.  Secondary key: (normalised company, normalised
    title, normalised location) — when two URLs point at the same logical
    posting (e.g. LinkedIn + Indeed) the records are merged: sources are
    combined and the richer description wins.

    Returns (unique_jobs, duplicate_count).
    """
    unique: Dict[str, JobRecord] = {}
    fuzzy_seen: Dict[str, str] = {}
    duplicates = 0

    for job in jobs:
        url_key = dedup_key_url(job)
        if url_key in unique:
            duplicates += 1
            _merge(unique[url_key], job)
            continue
        fuzzy = dedup_key_fuzzy(job)
        if fuzzy and fuzzy in fuzzy_seen:
            duplicates += 1
            _merge(unique[fuzzy_seen[fuzzy]], job)
            continue
        unique[url_key] = job
        if fuzzy:
            fuzzy_seen[fuzzy] = url_key

    return list(unique.values()), duplicates


def _merge(primary: JobRecord, duplicate: JobRecord) -> None:
    """Merge duplicate info into the primary record (in place)."""
    sources = list(primary.sources or [])
    for src in [duplicate.source] + (duplicate.sources or []):
        if src and src not in sources:
            sources.append(src)
    primary.sources = sources
    if len(duplicate.description) > len(primary.description):
        primary.description = duplicate.description
        primary.content_hash = duplicate.content_hash
    if primary.posted_date is None and duplicate.posted_date is not None:
        primary.posted_date = duplicate.posted_date
    if not primary.salary and duplicate.salary:
        primary.salary = duplicate.salary
    if primary.work_mode == WorkMode.UNKNOWN.value and duplicate.work_mode != WorkMode.UNKNOWN.value:
        primary.work_mode = duplicate.work_mode
    if not primary.location and duplicate.location:
        primary.location = duplicate.location
