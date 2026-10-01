"""
Shared domain models for the job-matching pipeline.

This module is the single source of truth for enums and data records that are
used across every stage (scraping, rules, embeddings, LLM, storage, output).
It deliberately depends only on Pydantic / the standard library so that the
deterministic core of the pipeline can be imported and tested without Ollama,
torch or any scraping dependency installed.

Design rules honoured here (see README "Design principles"):
    * Relevant experience is modelled separately from total experience.
    * Career alignment is modelled separately from technical capability.
    * Every evaluation carries an explainable reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - only used for type hints
    from llm.schemas import JobAnalysis, JobRequirements


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Seniority(str, Enum):
    """Normalised seniority ladder (ordered junior -> senior)."""

    INTERN = "intern"
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    STAFF = "staff"
    PRINCIPAL = "principal"
    MANAGER = "manager"
    UNKNOWN = "unknown"


SENIORITY_ORDER: Dict[Seniority, int] = {
    Seniority.INTERN: 0,
    Seniority.ENTRY: 1,
    Seniority.JUNIOR: 2,
    Seniority.MID: 3,
    Seniority.SENIOR: 4,
    Seniority.LEAD: 5,
    Seniority.STAFF: 6,
    Seniority.PRINCIPAL: 7,
    Seniority.MANAGER: 8,
    Seniority.UNKNOWN: -1,
}

# Typical year ranges used to *infer* an experience requirement from a job's
# seniority when the description does not state one explicitly.  These are
# soft bounds only (low confidence), never a hard rejection on their own.
SENIORITY_YEARS: Dict[Seniority, tuple] = {
    Seniority.INTERN: (0.0, 1.0),
    Seniority.ENTRY: (0.0, 2.0),
    Seniority.JUNIOR: (1.0, 3.0),
    Seniority.MID: (2.0, 6.0),
    Seniority.SENIOR: (5.0, 10.0),
    Seniority.LEAD: (7.0, 12.0),
    Seniority.STAFF: (9.0, 20.0),
    Seniority.PRINCIPAL: (12.0, 20.0),
    Seniority.MANAGER: (8.0, 20.0),
}


class ExperienceDomain(str, Enum):
    """Domain that a job's experience requirement refers to."""

    GENERAL = "general"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    GENAI = "genai"
    MODEL_DEPLOYMENT = "model_deployment"
    MLOPS = "mlops"
    QUANTIZATION = "quantization"
    SOFTWARE_ENGINEERING = "software_engineering"
    NETWORKING = "networking"
    OTHER = "other"


class ExperienceFit(str, Enum):
    """Outcome of comparing candidate experience against a job requirement."""

    STRONG_MATCH = "strong_match"            # inside the required range
    ACCEPTABLE = "acceptable"                # mild over/under, well within tolerance
    MARGINAL = "marginal"                    # slightly under minimum (soft mode only)
    UNDERQUALIFIED = "underqualified"        # below minimum
    POTENTIALLY_OVERQUALIFIED = "potentially_overqualified"
    OVERQUALIFIED = "overqualified"          # severe overqualification
    UNKNOWN = "unknown"                      # job gives us nothing to reason about


class SkillMatchType(str, Enum):
    EXACT = "exact"
    RELATED = "related"            # same taxonomy category
    TRANSFERABLE = "transferable"  # adjacent taxonomy category
    MISSING = "missing"


class Verdict(str, Enum):
    MATCH = "MATCH"
    BORDERLINE = "BORDERLINE"
    REJECTED = "REJECTED"


class WorkMode(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    OTHER = "other"
    UNKNOWN = "unknown"


class RejectionReason(str, Enum):
    UNDERQUALIFIED = "UNDERQUALIFIED"
    OVERQUALIFIED = "OVERQUALIFIED"
    IRRELEVANT_ROLE = "IRRELEVANT_ROLE"
    EXCLUDED_ROLE = "EXCLUDED_ROLE"
    LOW_CAREER_ALIGNMENT = "LOW_CAREER_ALIGNMENT"
    MISSING_MUST_HAVE_SKILL = "MISSING_MUST_HAVE_SKILL"
    LOCATION_MISMATCH = "LOCATION_MISMATCH"
    WORK_MODE_MISMATCH = "WORK_MODE_MISMATCH"
    SENIORITY_MISMATCH = "SENIORITY_MISMATCH"
    EMPLOYMENT_TYPE_MISMATCH = "EMPLOYMENT_TYPE_MISMATCH"
    STALE_JOB = "STALE_JOB"
    LOW_EMBEDDING_SIMILARITY = "LOW_EMBEDDING_SIMILARITY"
    LOW_SCORE = "LOW_SCORE"


# ---------------------------------------------------------------------------
# Job record (normalised, post-scraping)
# ---------------------------------------------------------------------------
class JobRecord(BaseModel):
    """A normalised job posting, independent of the source site."""

    job_id: Optional[int] = None  # database id once persisted
    company: str = ""
    title: str = ""
    location: str = ""
    description: str = ""
    url: str = ""  # canonical URL used as primary key
    source: str = ""  # first source that reported the job
    sources: List[str] = Field(default_factory=list)
    salary: str = ""
    employment_type: str = EmploymentType.UNKNOWN.value
    work_mode: str = WorkMode.UNKNOWN.value
    posted_date: Optional[datetime] = None
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    content_hash: str = ""
    job_level_raw: str = ""  # raw seniority hint from the site (e.g. LinkedIn)
    experience_range_raw: str = ""  # raw experience hint (e.g. Naukri)

    @property
    def platform(self) -> str:
        return ", ".join(self.sources) if self.sources else self.source


# ---------------------------------------------------------------------------
# Experience evaluation
# ---------------------------------------------------------------------------
@dataclass
class ExperienceEvaluation:
    """Result of the domain-aware experience comparison (RULE 9)."""

    fit: ExperienceFit = ExperienceFit.UNKNOWN
    effective_candidate_years: float = 0.0
    basis: str = "none"  # direct | adjacent:<domain> | relevant | total-assumed | none
    job_min_years: Optional[float] = None
    job_max_years: Optional[float] = None
    domain: ExperienceDomain = ExperienceDomain.GENERAL
    gap: Optional[float] = None  # effective - job_min (negative => short)
    ratio: Optional[float] = None  # effective / reference
    overqualified: bool = False
    underqualified: bool = False
    confidence: float = 0.0
    requirement_source: str = "none"  # llm | regex | seniority_inference | none
    reasons: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Deterministic 0-100 experience score (RULE 1: no LLM involved)."""
        return experience_fit_score(self.fit, self.gap, self.job_min_years)


def experience_fit_score(
    fit: ExperienceFit,
    gap: Optional[float] = None,
    job_min_years: Optional[float] = None,
) -> float:
    """Map an experience fit to a 0-100 score."""
    if fit == ExperienceFit.STRONG_MATCH:
        return 100.0
    if fit == ExperienceFit.ACCEPTABLE:
        return 80.0
    if fit == ExperienceFit.POTENTIALLY_OVERQUALIFIED:
        return 65.0
    if fit == ExperienceFit.MARGINAL:
        return 55.0
    if fit == ExperienceFit.UNKNOWN:
        return 55.0
    if fit == ExperienceFit.UNDERQUALIFIED:
        # gap is (effective - job_min): negative gap == shortfall in years.
        if gap is not None:
            shortfall = max(0.0, -gap)
            return max(0.0, 30.0 - 10.0 * shortfall)
        return 25.0
    if fit == ExperienceFit.OVERQUALIFIED:
        return 30.0
    return 50.0


# ---------------------------------------------------------------------------
# Skill matching
# ---------------------------------------------------------------------------
@dataclass
class SkillMatch:
    job_skill: str
    candidate_skill: Optional[str]
    match_type: SkillMatchType
    category: Optional[str] = None


@dataclass
class SkillMatchResult:
    matches: List[SkillMatch] = field(default_factory=list)
    score: float = 50.0  # neutral when nothing to compare

    @property
    def exact(self) -> List[SkillMatch]:
        return [m for m in self.matches if m.match_type == SkillMatchType.EXACT]

    @property
    def related(self) -> List[SkillMatch]:
        return [m for m in self.matches if m.match_type == SkillMatchType.RELATED]

    @property
    def transferable(self) -> List[SkillMatch]:
        return [m for m in self.matches if m.match_type == SkillMatchType.TRANSFERABLE]

    @property
    def missing(self) -> List[SkillMatch]:
        return [m for m in self.matches if m.match_type == SkillMatchType.MISSING]

    def matched_names(self) -> List[str]:
        return [m.job_skill for m in self.matches if m.match_type != SkillMatchType.MISSING]


# ---------------------------------------------------------------------------
# Rejections / verdicts
# ---------------------------------------------------------------------------
@dataclass
class Rejection:
    reason: RejectionReason
    detail: str = ""
    stage: str = "rules"  # rules | embedding | post_qwen | scoring


# ---------------------------------------------------------------------------
# Full match result (one per analysed job)
# ---------------------------------------------------------------------------
@dataclass
class JobMatchResult:
    job: JobRecord
    requirements: Optional["JobRequirements"] = None
    experience: ExperienceEvaluation = field(default_factory=ExperienceEvaluation)
    skills: SkillMatchResult = field(default_factory=SkillMatchResult)
    embedding_score: float = 0.0        # 0-100 (rescaled cosine similarity)
    embedding_similarity: float = 0.0   # raw cosine similarity
    experience_score: float = 0.0
    skill_score: float = 0.0
    domain_score: float = 0.0
    responsibility_score: float = 0.0
    career_alignment_score: float = 0.0
    final_score: float = 0.0
    verdict: Verdict = Verdict.REJECTED
    hard_rejected: bool = True
    rejections: List[Rejection] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    reasoning: str = ""
    qwen_analysis: Optional["JobAnalysis"] = None

    @property
    def verdict_rank(self) -> int:
        return {Verdict.MATCH: 0, Verdict.BORDERLINE: 1, Verdict.REJECTED: 2}[self.verdict]


# ---------------------------------------------------------------------------
# (De)serialisation of full results for the SQLite cache / CSV export
# ---------------------------------------------------------------------------
def match_result_to_dict(result: JobMatchResult) -> Dict[str, Any]:
    """JSON-safe dict of a full match result (stored in job_matches.result_json)."""
    return {
        "job": result.job.model_dump(mode="json"),
        "requirements": result.requirements.model_dump(mode="json") if result.requirements else None,
        "experience": {
            "fit": result.experience.fit.value,
            "effective_candidate_years": result.experience.effective_candidate_years,
            "basis": result.experience.basis,
            "job_min_years": result.experience.job_min_years,
            "job_max_years": result.experience.job_max_years,
            "domain": result.experience.domain.value,
            "gap": result.experience.gap,
            "ratio": result.experience.ratio,
            "overqualified": result.experience.overqualified,
            "underqualified": result.experience.underqualified,
            "confidence": result.experience.confidence,
            "requirement_source": result.experience.requirement_source,
            "reasons": result.experience.reasons,
        },
        "skills": {
            "score": result.skills.score,
            "matches": [
                {
                    "job_skill": m.job_skill,
                    "candidate_skill": m.candidate_skill,
                    "match_type": m.match_type.value,
                    "category": m.category,
                }
                for m in result.skills.matches
            ],
        },
        "scores": {
            "embedding_score": result.embedding_score,
            "embedding_similarity": result.embedding_similarity,
            "experience_score": result.experience_score,
            "skill_score": result.skill_score,
            "domain_score": result.domain_score,
            "responsibility_score": result.responsibility_score,
            "career_alignment_score": result.career_alignment_score,
            "final_score": result.final_score,
        },
        "verdict": result.verdict.value,
        "hard_rejected": result.hard_rejected,
        "rejections": [
            {"reason": r.reason.value, "detail": r.detail, "stage": r.stage}
            for r in result.rejections
        ],
        "strengths": result.strengths,
        "concerns": result.concerns,
        "reasoning": result.reasoning,
        "qwen_analysis": result.qwen_analysis.model_dump(mode="json") if result.qwen_analysis else None,
    }


def match_result_from_dict(data: Dict[str, Any]) -> JobMatchResult:
    """Inverse of :func:`match_result_to_dict`."""
    from llm.schemas import JobAnalysis, JobRequirements

    exp = data.get("experience", {})
    skills_data = data.get("skills", {})
    scores = data.get("scores", {})

    experience = ExperienceEvaluation(
        fit=ExperienceFit(exp.get("fit", "unknown")),
        effective_candidate_years=float(exp.get("effective_candidate_years", 0) or 0),
        basis=exp.get("basis", "none"),
        job_min_years=exp.get("job_min_years"),
        job_max_years=exp.get("job_max_years"),
        domain=ExperienceDomain(exp.get("domain", "general")),
        gap=exp.get("gap"),
        ratio=exp.get("ratio"),
        overqualified=bool(exp.get("overqualified", False)),
        underqualified=bool(exp.get("underqualified", False)),
        confidence=float(exp.get("confidence", 0) or 0),
        requirement_source=exp.get("requirement_source", "none"),
        reasons=list(exp.get("reasons", [])),
    )
    skills = SkillMatchResult(
        matches=[
            SkillMatch(
                job_skill=m.get("job_skill", ""),
                candidate_skill=m.get("candidate_skill"),
                match_type=SkillMatchType(m.get("match_type", "missing")),
                category=m.get("category"),
            )
            for m in skills_data.get("matches", [])
        ],
        score=float(skills_data.get("score", 50) or 50),
    )
    requirements = None
    if data.get("requirements"):
        requirements = JobRequirements.model_validate(data["requirements"])
    analysis = None
    if data.get("qwen_analysis"):
        try:
            analysis = JobAnalysis.model_validate(data["qwen_analysis"])
        except Exception:
            analysis = None
    return JobMatchResult(
        job=JobRecord.model_validate(data.get("job", {})),
        requirements=requirements,
        experience=experience,
        skills=skills,
        embedding_score=float(scores.get("embedding_score", 0) or 0),
        embedding_similarity=float(scores.get("embedding_similarity", 0) or 0),
        experience_score=float(scores.get("experience_score", 0) or 0),
        skill_score=float(scores.get("skill_score", 0) or 0),
        domain_score=float(scores.get("domain_score", 0) or 0),
        responsibility_score=float(scores.get("responsibility_score", 0) or 0),
        career_alignment_score=float(scores.get("career_alignment_score", 0) or 0),
        final_score=float(scores.get("final_score", 0) or 0),
        verdict=Verdict(data.get("verdict", "REJECTED")),
        hard_rejected=bool(data.get("hard_rejected", True)),
        rejections=[
            Rejection(
                reason=RejectionReason(r.get("reason", "LOW_SCORE")),
                detail=r.get("detail", ""),
                stage=r.get("stage", "rules"),
            )
            for r in data.get("rejections", [])
        ],
        strengths=list(data.get("strengths", [])),
        concerns=list(data.get("concerns", [])),
        reasoning=data.get("reasoning", ""),
        qwen_analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Pipeline statistics (section 30 of the spec)
# ---------------------------------------------------------------------------
@dataclass
class PipelineStats:
    jobs_scraped: int = 0
    duplicates_removed: int = 0
    rejected_by_role: int = 0
    rejected_by_location: int = 0
    rejected_by_freshness: int = 0
    rejected_by_seniority: int = 0
    rejected_by_employment: int = 0
    rejected_by_work_mode: int = 0
    rejected_by_must_have: int = 0
    rejected_by_experience: int = 0
    embedding_candidates: int = 0
    qwen_analyzed: int = 0
    qwen_failures: int = 0
    qwen_cache_hits: int = 0
    hard_rejected_after_qwen: int = 0
    final_matches: int = 0
    final_borderline: int = 0
    final_rejected: int = 0

    def log_summary(self, logger: Any) -> None:
        logger.info("=" * 62)
        logger.info("Pipeline summary")
        logger.info("=" * 62)
        logger.info("Jobs scraped:            %d", self.jobs_scraped)
        logger.info("Duplicates removed:      %d", self.duplicates_removed)
        logger.info("Rejected by role:        %d", self.rejected_by_role)
        logger.info("Rejected by location:    %d", self.rejected_by_location)
        logger.info("Rejected by freshness:   %d", self.rejected_by_freshness)
        logger.info("Rejected by seniority:   %d", self.rejected_by_seniority)
        logger.info("Rejected by employment:  %d", self.rejected_by_employment)
        logger.info("Rejected by work mode:   %d", self.rejected_by_work_mode)
        logger.info("Rejected by must-have:   %d", self.rejected_by_must_have)
        logger.info("Rejected by experience:  %d", self.rejected_by_experience)
        logger.info("Embedding candidates:    %d", self.embedding_candidates)
        logger.info("Qwen analyzed:           %d", self.qwen_analyzed)
        logger.info("Qwen failures:           %d", self.qwen_failures)
        logger.info("Qwen cache hits:         %d", self.qwen_cache_hits)
        logger.info("Hard rejected post-Qwen: %d", self.hard_rejected_after_qwen)
        logger.info("Final MATCH:             %d", self.final_matches)
        logger.info("Final BORDERLINE:        %d", self.final_borderline)
        logger.info("Final REJECTED:          %d", self.final_rejected)
        logger.info("=" * 62)
