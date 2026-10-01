"""
Pydantic schemas for every LLM (Qwen) input/output contract.

These models are *lenient by design*: a small local model will occasionally
return numbers as strings, nulls where lists belong, or out-of-range scores.
``mode="before"`` validators coerce and clamp such values so a single bad
field never crashes the pipeline (section 29).

The models are also imported by the deterministic rule engine (JobRequirements
is the shared contract between the regex extractor and Qwen), which is why
this module must stay free of any runtime LLM dependency.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from models import EmploymentType, ExperienceDomain, Seniority, WorkMode


def _to_float(v: Any) -> Optional[float]:
    """Coerce LLM-ish values ('4+', '3.5', 4, None) to float."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _to_int_score(v: Any) -> int:
    f = _to_float(v)
    if f is None:
        return 0
    return int(max(0, min(100, round(f))))


def _to_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = re.split(r"[;,•\n]", v)
        return [p.strip() for p in parts if p.strip()]
    if isinstance(v, (list, tuple, set)):
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(v).strip()] if str(v).strip() else []


def _clamp_unit(v: Any) -> float:
    f = _to_float(v)
    if f is None:
        return 0.0
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Candidate profile (section 4)
# ---------------------------------------------------------------------------
class CandidateProfile(BaseModel):
    name: str = ""
    current_title: str = ""
    total_experience_years: float = 0.0
    relevant_experience_years: float = 0.0
    experience_by_domain: Dict[str, float] = Field(default_factory=dict)
    skills: List[str] = Field(default_factory=list)
    core_skills: List[str] = Field(default_factory=list)
    secondary_skills: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    preferred_work_modes: List[str] = Field(default_factory=list)
    excluded_roles: List[str] = Field(default_factory=list)
    summary: str = ""
    # Filled deterministically after extraction (never by the LLM):
    inferred_seniority: Seniority = Seniority.UNKNOWN

    @field_validator(
        "total_experience_years", "relevant_experience_years", mode="before"
    )
    @classmethod
    def _float_years(cls, v: Any) -> float:
        return _to_float(v) or 0.0

    @field_validator("experience_by_domain", mode="before")
    @classmethod
    def _domain_map(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        out: Dict[str, float] = {}
        for key, value in v.items():
            f = _to_float(value)
            if f is None or f < 0:
                continue
            key_clean = str(key).strip().lower().replace(" ", "_").replace("-", "_")
            if not key_clean:
                continue
            try:
                ExperienceDomain(key_clean)
            except ValueError:
                # unknown domain key: still keep it, harmless
                pass
            out[key_clean] = f
        return out

    @field_validator(
        "skills",
        "core_skills",
        "secondary_skills",
        "domains",
        "target_roles",
        "education",
        "certifications",
        "locations",
        "preferred_work_modes",
        "excluded_roles",
        mode="before",
    )
    @classmethod
    def _str_lists(cls, v: Any) -> List[str]:
        return _to_str_list(v)

    def domain_years(self, domain: ExperienceDomain) -> float:
        """Years of experience in a domain, checking common alias keys."""
        key = domain.value
        aliases = {
            ExperienceDomain.MACHINE_LEARNING: {"machine_learning", "ml", "machinelearning"},
            ExperienceDomain.DEEP_LEARNING: {"deep_learning", "dl"},
            ExperienceDomain.COMPUTER_VISION: {"computer_vision", "cv"},
            ExperienceDomain.NLP: {"nlp", "natural_language_processing"},
            ExperienceDomain.GENAI: {"genai", "generative_ai", "gen_ai"},
            ExperienceDomain.MODEL_DEPLOYMENT: {"model_deployment", "deployment", "model_deployment_mlops"},
            ExperienceDomain.MLOPS: {"mlops", "ml_operations"},
            ExperienceDomain.QUANTIZATION: {"quantization", "quantisation"},
            ExperienceDomain.SOFTWARE_ENGINEERING: {"software_engineering", "swe", "software_development"},
            ExperienceDomain.NETWORKING: {"networking", "network_engineering", "telecom"},
        }
        keys = aliases.get(domain, {key})
        best = 0.0
        for k in keys:
            best = max(best, float(self.experience_by_domain.get(k, 0.0) or 0.0))
        return best

    def has_domain_experience(self) -> bool:
        return any((v or 0) > 0 for v in self.experience_by_domain.values())


# ---------------------------------------------------------------------------
# Experience requirement extracted from a job (sections 5 & 8)
# ---------------------------------------------------------------------------
class ExperienceRequirement(BaseModel):
    min_experience_years: Optional[float] = None
    max_experience_years: Optional[float] = None
    experience_text: str = ""
    experience_domain: ExperienceDomain = ExperienceDomain.GENERAL
    experience_confidence: float = 0.0
    extraction_method: str = "none"  # llm | regex | seniority_inference | none

    @field_validator("min_experience_years", "max_experience_years", mode="before")
    @classmethod
    def _years(cls, v: Any) -> Optional[float]:
        f = _to_float(v)
        if f is None or f < 0 or f > 45:
            return None
        return f

    @field_validator("experience_confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_unit(v)

    @field_validator("experience_domain", mode="before")
    @classmethod
    def _domain(cls, v: Any) -> ExperienceDomain:
        if isinstance(v, ExperienceDomain):
            return v
        s = str(v or "general").strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return ExperienceDomain(s)
        except ValueError:
            return ExperienceDomain.GENERAL


# ---------------------------------------------------------------------------
# Full job requirements (section 21: job_requirements table)
# ---------------------------------------------------------------------------
class JobRequirements(BaseModel):
    min_experience_years: Optional[float] = None
    max_experience_years: Optional[float] = None
    experience_text: str = ""
    experience_domain: ExperienceDomain = ExperienceDomain.GENERAL
    experience_confidence: float = 0.0
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    seniority: Seniority = Seniority.UNKNOWN
    employment_type: str = EmploymentType.UNKNOWN.value
    work_mode: str = WorkMode.UNKNOWN.value
    responsibilities: List[str] = Field(default_factory=list)
    extraction_method: str = "none"  # llm | regex | merged

    @field_validator("min_experience_years", "max_experience_years", mode="before")
    @classmethod
    def _years(cls, v: Any) -> Optional[float]:
        f = _to_float(v)
        if f is None or f < 0 or f > 45:
            return None
        return f

    @field_validator("experience_confidence", mode="before")
    @classmethod
    def _conf(cls, v: Any) -> float:
        return _clamp_unit(v)

    @field_validator("required_skills", "preferred_skills", "responsibilities", mode="before")
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return _to_str_list(v)

    @field_validator("experience_domain", mode="before")
    @classmethod
    def _domain(cls, v: Any) -> ExperienceDomain:
        if isinstance(v, ExperienceDomain):
            return v
        s = str(v or "general").strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return ExperienceDomain(s)
        except ValueError:
            return ExperienceDomain.GENERAL

    @field_validator("seniority", mode="before")
    @classmethod
    def _seniority(cls, v: Any) -> Seniority:
        if isinstance(v, Seniority):
            return v
        s = str(v or "").strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return Seniority(s)
        except ValueError:
            return Seniority.UNKNOWN


# ---------------------------------------------------------------------------
# Qwen deep job analysis (section 16)
# ---------------------------------------------------------------------------
class JobAnalysis(BaseModel):
    match_score: int = 0
    skill_match_score: int = 0
    responsibility_match_score: int = 0
    domain_match_score: int = 0
    career_alignment_score: int = 0
    matched_skills: List[str] = Field(default_factory=list)
    related_skills: List[str] = Field(default_factory=list)
    missing_required_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    reasoning: str = ""
    semantic_verdict: str = ""

    @field_validator(
        "match_score",
        "skill_match_score",
        "responsibility_match_score",
        "domain_match_score",
        "career_alignment_score",
        mode="before",
    )
    @classmethod
    def _scores(cls, v: Any) -> int:
        return _to_int_score(v)

    @field_validator(
        "matched_skills",
        "related_skills",
        "missing_required_skills",
        "missing_preferred_skills",
        "strengths",
        "concerns",
        mode="before",
    )
    @classmethod
    def _lists(cls, v: Any) -> List[str]:
        return _to_str_list(v)
