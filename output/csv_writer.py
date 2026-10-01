"""
CSV output (sections 25-26).

Two files are produced:

    matched_jobs.csv  — MATCH + BORDERLINE rows, hard-eligible first, then
                        final score descending
    rejected_jobs.csv — everything rejected, with the human-readable reason

Encoding is UTF-8 with BOM so Excel on Windows opens them correctly.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from llm.schemas import CandidateProfile
from models import JobMatchResult, Rejection, Verdict

logger = logging.getLogger(__name__)

MATCHED_COLUMNS = [
    "Company",
    "Job Title",
    "Platform",
    "Location",
    "Work Mode",
    "Employment Type",
    "Posted Date",
    "Candidate Total Experience",
    "Candidate Relevant Experience",
    "Job Minimum Experience",
    "Job Maximum Experience",
    "Experience Domain",
    "Experience Fit",
    "Experience Gap",
    "Overqualification Flag",
    "Underqualification Flag",
    "Required Skills",
    "Preferred Skills",
    "Matched Skills",
    "Related Skills",
    "Missing Required Skills",
    "Missing Preferred Skills",
    "Embedding Score",
    "Skill Match Score",
    "Experience Match Score",
    "Domain Match Score",
    "Responsibility Match Score",
    "Career Alignment Score",
    "Final Match Score",
    "Verdict",
    "Strengths",
    "Concerns",
    "Reasoning",
    "Job Link",
]

REJECTED_COLUMNS = [
    "Company",
    "Job Title",
    "Platform",
    "Location",
    "Job Experience",
    "Candidate Experience",
    "Rejection Reason",
    "Rejection Stage",
    "Rejection Detail",
    "Final Match Score",
    "Job Link",
]


def _join(values: Iterable) -> str:
    return "; ".join(str(v) for v in values if v not in (None, ""))


def _fmt_years(value) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def _fmt_date(value) -> str:
    if value is None:
        return "Unknown"
    try:
        return value.strftime("%Y-%m-%d")
    except AttributeError:
        return str(value)[:10]


def sort_results(results: Sequence[JobMatchResult]) -> List[JobMatchResult]:
    """Eligible first, then final score descending (section 25)."""
    return sorted(results, key=lambda r: (r.verdict_rank, -r.final_score))


def write_matched_csv(
    results: Sequence[JobMatchResult],
    profile: CandidateProfile,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sort_results([r for r in results if r.verdict in (Verdict.MATCH, Verdict.BORDERLINE)])

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=MATCHED_COLUMNS)
        writer.writeheader()
        for result in rows:
            req = result.requirements
            exp = result.experience
            job = result.job
            analysis = result.qwen_analysis
            writer.writerow({
                "Company": job.company,
                "Job Title": job.title,
                "Platform": job.platform,
                "Location": job.location,
                "Work Mode": job.work_mode,
                "Employment Type": job.employment_type,
                "Posted Date": _fmt_date(job.posted_date),
                "Candidate Total Experience": _fmt_years(profile.total_experience_years),
                "Candidate Relevant Experience": _fmt_years(profile.relevant_experience_years),
                "Job Minimum Experience": _fmt_years(exp.job_min_years),
                "Job Maximum Experience": _fmt_years(exp.job_max_years),
                "Experience Domain": exp.domain.value,
                "Experience Fit": exp.fit.value,
                "Experience Gap": _fmt_years(exp.gap) if exp.gap is not None else "",
                "Overqualification Flag": "Yes" if exp.overqualified else "No",
                "Underqualification Flag": "Yes" if exp.underqualified else "No",
                "Required Skills": _join(req.required_skills if req else []),
                "Preferred Skills": _join(req.preferred_skills if req else []),
                "Matched Skills": _join(
                    [m.job_skill for m in result.skills.exact]
                    + ([s for s in (analysis.matched_skills if analysis else [])])
                ),
                "Related Skills": _join(
                    [m.job_skill for m in result.skills.related + result.skills.transferable]
                    + ([s for s in (analysis.related_skills if analysis else [])])
                ),
                "Missing Required Skills": _join(
                    [m.job_skill for m in result.skills.missing]
                    + ([s for s in (analysis.missing_required_skills if analysis else [])])
                ),
                "Missing Preferred Skills": _join(
                    [s for s in (analysis.missing_preferred_skills if analysis else [])]
                ),
                "Embedding Score": f"{result.embedding_score:.1f}",
                "Skill Match Score": f"{result.skill_score:.1f}",
                "Experience Match Score": f"{result.experience_score:.1f}",
                "Domain Match Score": f"{result.domain_score:.1f}",
                "Responsibility Match Score": f"{result.responsibility_score:.1f}",
                "Career Alignment Score": f"{result.career_alignment_score:.1f}",
                "Final Match Score": f"{result.final_score:.1f}",
                "Verdict": result.verdict.value,
                "Strengths": _join(result.strengths),
                "Concerns": _join(result.concerns),
                "Reasoning": " ".join(str(result.reasoning).splitlines()),
                "Job Link": job.url,
            })
    logger.info("Wrote %d matched/borderline job(s) -> %s", len(rows), path)
    return path


def write_rejected_csv(
    results: Sequence[JobMatchResult],
    profile: CandidateProfile,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rejected = [r for r in results if r.verdict == Verdict.REJECTED]

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=REJECTED_COLUMNS)
        writer.writeheader()
        for result in rejected:
            job = result.job
            exp = result.experience
            job_experience = _job_experience_text(result)
            candidate_experience = (
                f"{profile.relevant_experience_years:g} relevant "
                f"({profile.total_experience_years:g} total; "
                f"{exp.effective_candidate_years:g} in {exp.domain.value})"
            )
            rejections = result.rejections or [Rejection(reason=None, detail=result.reasoning)]
            primary = rejections[0]
            writer.writerow({
                "Company": job.company,
                "Job Title": job.title,
                "Platform": job.platform,
                "Location": job.location,
                "Job Experience": job_experience,
                "Candidate Experience": candidate_experience,
                "Rejection Reason": primary.reason.value if primary.reason else "LOW_SCORE",
                "Rejection Stage": primary.stage,
                "Rejection Detail": " | ".join(
                    f"{r.reason.value if r.reason else ''}: {r.detail}" for r in rejections[:3]
                ),
                "Final Match Score": f"{result.final_score:.1f}",
                "Job Link": job.url,
            })
    logger.info("Wrote %d rejected job(s) -> %s", len(rejected), path)
    return path


def _job_experience_text(result: JobMatchResult) -> str:
    req = result.requirements
    if req is None:
        return "Unknown"
    if req.min_experience_years is None and req.max_experience_years is None:
        return "Not specified"
    if req.max_experience_years is not None and req.min_experience_years is not None:
        return f"{req.min_experience_years:g}-{req.max_experience_years:g} years"
    if req.min_experience_years is not None:
        return f"{req.min_experience_years:g}+ years"
    return f"up to {req.max_experience_years:g} years"
