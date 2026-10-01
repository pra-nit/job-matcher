"""
Domain-aware experience extraction and matching (spec sections 5-9).

This is the deterministic heart of the pipeline:

    * ``extract_experience_regex``  — multi-pattern regex parsing of job text
      ("3+ years", "3-5 years", "2 to 4 years", "minimum 4 years",
       "5 years preferred", ...).  Used standalone and as fallback when Qwen
      fails or returns low-confidence output.
    * ``classify_experience_domain`` — deterministic keyword classification of
      what the experience requirement refers to (ML vs CV vs networking vs ...).
    * ``effective_candidate_years`` — the candidate's *relevant* years for the
      job's domain (RULE 9: relevant beats total).
    * ``evaluate_experience`` — the full ladder:
          UNDERQUALIFIED / MARGINAL / STRONG_MATCH / ACCEPTABLE /
          POTENTIALLY_OVERQUALIFIED / OVERQUALIFIED / UNKNOWN
      using *both* an absolute gap and a ratio for overqualification so a
      single metric never triggers an aggressive rejection on its own.

No LLM and no embeddings are involved anywhere in this module (RULEs 1 & 2).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import ExperiencePolicy
from llm.schemas import CandidateProfile, JobRequirements
from models import (
    SENIORITY_YEARS,
    ExperienceDomain,
    ExperienceEvaluation,
    ExperienceFit,
    Seniority,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for experience extraction (ordered by specificity)
# ---------------------------------------------------------------------------
_YEARS = r"(?:years?|yrs?)"  # grouped: interpolated into larger patterns

# (priority, pattern, kind)  -- higher priority wins when several match.
# Range patterns capture (min, max); others capture a single number.
_PATTERNS: List[Tuple[float, re.Pattern, str]] = [
    # "2 to 4 years", "2-4 years", "3–5 years of experience"
    (
        4.0,
        re.compile(
            rf"(\d{{1,2}})\s*(?:years?|yrs?)?\s*(?:to|through|[-–—])\s*(\d{{1,2}})\s*(?:years?|yrs?)",
            re.I,
        ),
        "range",
    ),
    # "8+ years", "8 + years", "8+ yrs of experience"
    (3.5, re.compile(rf"(\d{{1,2}})\s*\+\s*{_YEARS}", re.I), "min_plus"),
    # "minimum 4 years", "at least 3 years", "min. 5 yrs"
    (
        3.0,
        re.compile(rf"(?:minimum|min|at\s*least|atleast|minimum\s*of)\s*(\d{{1,2}})\s*{_YEARS}", re.I),
        "min_word",
    ),
    # "5 years preferred", "5+ years required"
    (
        2.8,
        re.compile(
            rf"(\d{{1,2}})\s*\+?\s*{_YEARS}[^.\n]{{0,40}}?(?:preferred|required|expected)",
            re.I,
        ),
        "min_qualifier",
    ),
    # "5 years of experience", "4 years relevant experience", "3 yrs experience",
    # "10 years of total professional experience"
    (
        2.5,
        re.compile(
            rf"(\d{{1,2}})\s*\+?\s*{_YEARS}\s*(?:of\s+)?"
            rf"(?:relevant\s+|related\s+|hands-on\s+|professional\s+|work\s+|industry\s+|"
            rf"total\s+|overall\s+|cumulative\s+|combined\s+|full[- ]time\s+)?"
            rf"(?:experience|expertise)",
            re.I,
        ),
        "min_experience",
    ),
    # "experience of/over 5 years", "experience: 5+ years"
    (
        2.2,
        re.compile(
            rf"experience[^.\n]{{0,25}}?(\d{{1,2}})\s*\+?\s*{_YEARS}",
            re.I,
        ),
        "min_experience_reversed",
    ),
    # bare "5 years" in a sentence that mentions experience/expertise
    (1.0, re.compile(rf"(\d{{1,2}})\s*\+?\s*{_YEARS}", re.I), "bare"),
]

_EXPERIENCE_CONTEXT_WORDS = re.compile(r"experience|expertise|professional|career|worked", re.I)

# Domain keyword classification (checked specific -> general).
_DOMAIN_KEYWORDS: List[Tuple[ExperienceDomain, re.Pattern]] = [
    (ExperienceDomain.COMPUTER_VISION, re.compile(r"computer vision|image processing|vision model|perception|imagery", re.I)),
    (ExperienceDomain.QUANTIZATION, re.compile(r"quantiz", re.I)),
    (ExperienceDomain.MODEL_DEPLOYMENT, re.compile(r"deploy\w*|edge ai|inference|productionis|productioniz|embedded", re.I)),
    (ExperienceDomain.MLOPS, re.compile(r"mlops|ml operations|machine learning operations", re.I)),
    (ExperienceDomain.NLP, re.compile(r"\bnlp\b|natural language", re.I)),
    (ExperienceDomain.GENAI, re.compile(r"generative ai|genai|gen ai|llm|large language", re.I)),
    (ExperienceDomain.DEEP_LEARNING, re.compile(r"deep learning|neural network|\bdl\b", re.I)),
    (ExperienceDomain.MACHINE_LEARNING, re.compile(r"machine learning|\bml\b|artificial intelligence|\bai\b", re.I)),
    (ExperienceDomain.NETWORKING, re.compile(r"network\w*|telecom|routing|switching|\b5g\b|\blte\b", re.I)),
    (ExperienceDomain.SOFTWARE_ENGINEERING, re.compile(r"software (?:development|engineering)|programming|developing software|software development", re.I)),
]

# Adjacent domains used when the candidate has *zero* direct years in the
# required domain (credit factor is configurable, default 0.5).
DOMAIN_ADJACENCY = {
    ExperienceDomain.MACHINE_LEARNING: [
        ExperienceDomain.DEEP_LEARNING,
        ExperienceDomain.COMPUTER_VISION,
        ExperienceDomain.NLP,
        ExperienceDomain.GENAI,
    ],
    ExperienceDomain.DEEP_LEARNING: [
        ExperienceDomain.MACHINE_LEARNING,
        ExperienceDomain.COMPUTER_VISION,
    ],
    ExperienceDomain.COMPUTER_VISION: [
        ExperienceDomain.MACHINE_LEARNING,
        ExperienceDomain.DEEP_LEARNING,
    ],
    ExperienceDomain.NLP: [
        ExperienceDomain.MACHINE_LEARNING,
        ExperienceDomain.DEEP_LEARNING,
        ExperienceDomain.GENAI,
    ],
    ExperienceDomain.GENAI: [
        ExperienceDomain.DEEP_LEARNING,
        ExperienceDomain.NLP,
        ExperienceDomain.MACHINE_LEARNING,
    ],
    ExperienceDomain.MODEL_DEPLOYMENT: [
        ExperienceDomain.MACHINE_LEARNING,
        ExperienceDomain.DEEP_LEARNING,
        ExperienceDomain.QUANTIZATION,
        ExperienceDomain.SOFTWARE_ENGINEERING,
    ],
    ExperienceDomain.QUANTIZATION: [
        ExperienceDomain.MODEL_DEPLOYMENT,
        ExperienceDomain.DEEP_LEARNING,
    ],
    ExperienceDomain.MLOPS: [
        ExperienceDomain.MODEL_DEPLOYMENT,
        ExperienceDomain.SOFTWARE_ENGINEERING,
        ExperienceDomain.MACHINE_LEARNING,
    ],
    ExperienceDomain.NETWORKING: [],
    ExperienceDomain.SOFTWARE_ENGINEERING: [],
    ExperienceDomain.GENERAL: [],
    ExperienceDomain.OTHER: [],
}


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------
def classify_experience_domain(
    experience_context: str,
    title: str = "",
    description: str = "",
) -> ExperienceDomain:
    """
    Deterministically decide which domain an experience requirement refers to.

    Priority: the sentence containing the requirement > the job title > the
    first domain keyword found near the top of the description.
    """
    for text in (experience_context, title, description[:800]):
        if not text:
            continue
        for domain, pattern in _DOMAIN_KEYWORDS:
            if pattern.search(text):
                return domain
    return ExperienceDomain.GENERAL


# ---------------------------------------------------------------------------
# Regex extraction
# ---------------------------------------------------------------------------
@dataclass
class _RegexMatch:
    min_years: float
    max_years: Optional[float]
    kind: str
    score: float
    sentence: str


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+|•|\u2022", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def extract_experience_regex(text: str) -> Optional[_RegexMatch]:
    """
    Scan text with multiple patterns and return the most plausible
    experience requirement, or None if nothing credible is found.

    Plausibility scoring: pattern specificity + bonus when the containing
    sentence actually talks about experience.
    """
    if not text:
        return None
    best: Optional[_RegexMatch] = None
    for sentence in _sentences(text):
        for priority, pattern, kind in _PATTERNS:
            m = pattern.search(sentence)
            if not m:
                continue
            try:
                first = float(m.group(1))
            except (ValueError, IndexError):
                continue
            if first > 45:  # not a plausible experience number
                continue
            if kind == "range":
                second = float(m.group(2))
                if second < first or second > 45:
                    continue
                min_y, max_y = first, second
            else:
                min_y, max_y = first, None
            score = priority
            if _EXPERIENCE_CONTEXT_WORDS.search(sentence):
                score += 2.0
            candidate = _RegexMatch(min_y, max_y, kind, score, sentence)
            if best is None or candidate.score > best.score:
                best = candidate
    if best is not None and best.kind == "bare" and best.score < 3.0:
        # a bare number with no experience context is too risky
        return None
    return best


def extract_experience_requirement(
    description: str,
    title: str = "",
) -> JobRequirements:
    """Deterministic experience + domain extraction from raw job text."""
    match = extract_experience_regex(description)
    if match is None:
        return JobRequirements(extraction_method="none")
    domain = classify_experience_domain(match.sentence, title, description)
    return JobRequirements(
        min_experience_years=match.min_years,
        max_experience_years=match.max_years,
        experience_text=match.sentence.strip()[:300],
        experience_domain=domain,
        experience_confidence=0.7 if match.kind != "bare" else 0.5,
        extraction_method="regex",
    )


# ---------------------------------------------------------------------------
# Effective candidate years (RULE 9)
# ---------------------------------------------------------------------------
@dataclass
class EffectiveYears:
    years: float = 0.0
    basis: str = "none"
    confidence: float = 0.0
    detail: str = ""


def effective_candidate_years(
    profile: CandidateProfile,
    domain: ExperienceDomain,
    policy: ExperiencePolicy,
) -> EffectiveYears:
    """
    Compute the candidate's experience that is *relevant* to this job's domain.

    Order of preference:
        1. direct years in the required domain
        2. (software_engineering) total years — software development is
           assumed to accumulate across a technical career
        3. credited years from an adjacent domain (configurable factor)
        4. relevant years when the resume gave no domain breakdown at all
        5. relevant years for general requirements (fallback: total)
    """
    total = float(profile.total_experience_years or 0.0)
    relevant = float(profile.relevant_experience_years or 0.0)

    if domain in (ExperienceDomain.GENERAL, ExperienceDomain.OTHER):
        if relevant > 0:
            return EffectiveYears(relevant, "relevant", 0.9, "relevant experience for a general requirement")
        return EffectiveYears(total, "total", 0.6, "total experience (no relevant figure available)")

    direct = profile.domain_years(domain)
    if direct > 0:
        return EffectiveYears(
            direct,
            f"direct:{domain.value}",
            0.95,
            f"{direct:g} years directly in {domain.value}",
        )

    # No domain breakdown at all on the resume: fall back to relevant years.
    if not profile.has_domain_experience() and relevant > 0:
        return EffectiveYears(
            relevant,
            "relevant-fallback",
            0.5,
            "resume had no domain breakdown; using relevant years",
        )

    if (
        domain == ExperienceDomain.SOFTWARE_ENGINEERING
        and policy.software_engineering_uses_total
        and total > 0
    ):
        return EffectiveYears(
            total,
            "total-assumed",
            0.5,
            "software development assumed to span the whole career",
        )

    adjacent = DOMAIN_ADJACENCY.get(domain, [])
    best_adjacent: Optional[ExperienceDomain] = None
    best_years = 0.0
    for adj in adjacent:
        years = profile.domain_years(adj)
        if years > best_years:
            best_years, best_adjacent = years, adj
    if best_adjacent is not None and policy.adjacent_domain_credit > 0:
        credited = best_years * policy.adjacent_domain_credit
        return EffectiveYears(
            credited,
            f"adjacent:{best_adjacent.value}",
            0.5,
            f"{best_years:g} years in adjacent domain {best_adjacent.value} credited at "
            f"{policy.adjacent_domain_credit:g}x",
        )

    return EffectiveYears(0.0, "none", 0.3, f"no {domain.value} experience on resume")


# ---------------------------------------------------------------------------
# The experience ladder
# ---------------------------------------------------------------------------
def evaluate_experience(
    profile: CandidateProfile,
    requirements: JobRequirements,
    policy: ExperiencePolicy,
) -> ExperienceEvaluation:
    """
    Compare candidate experience against a job's requirement.

    Ladder (in evaluation order):
        1. no requirement & no seniority to infer from -> UNKNOWN
        2. effective < min -> MARGINAL (soft mode, within tolerance) else UNDERQUALIFIED
        3. effective > reference (max, or min when no max) ->
             gap >= ABS and ratio >= RATIO -> OVERQUALIFIED
             gap >= ABS or  ratio >= RATIO -> POTENTIALLY_OVERQUALIFIED
             otherwise                      -> ACCEPTABLE
        4. inside range -> STRONG_MATCH
    """
    job_min = requirements.min_experience_years
    job_max = requirements.max_experience_years
    domain = requirements.experience_domain
    source = requirements.extraction_method

    effective = effective_candidate_years(profile, domain, policy)
    result = ExperienceEvaluation(
        effective_candidate_years=effective.years,
        basis=effective.basis,
        job_min_years=job_min,
        job_max_years=job_max,
        domain=domain,
        confidence=effective.confidence,
        requirement_source=source,
    )
    result.reasons.append(effective.detail)

    if job_min is None and job_max is None:
        inferred = _infer_from_seniority(requirements, policy)
        if inferred is not None:
            job_min, job_max, source = inferred
            result.job_min_years, result.job_max_years = job_min, job_max
            result.requirement_source = source
            result.reasons.append(
                f"requirement inferred from seniority '{requirements.seniority.value}' "
                f"({job_min:g}-{job_max:g} years, low confidence)"
            )
        else:
            result.fit = ExperienceFit.UNKNOWN
            result.reasons.append("job states no experience requirement")
            return result

    if job_min is not None:
        result.gap = round(effective.years - job_min, 2)
        reference = job_max if job_max is not None else job_min
        result.ratio = round(effective.years / reference, 2) if reference > 0 else None

        if effective.years < job_min:
            shortfall = job_min - effective.years
            if policy.soft_experience_mode and shortfall <= policy.soft_experience_gap_tolerance:
                result.fit = ExperienceFit.MARGINAL
                result.reasons.append(
                    f"{effective.years:g} relevant years is {shortfall:g} short of the "
                    f"{job_min:g}-year minimum (within soft tolerance)"
                )
            else:
                result.fit = ExperienceFit.UNDERQUALIFIED
                result.underqualified = True
                result.reasons.append(
                    f"{effective.years:g} relevant years ({effective.basis}) is below the "
                    f"{job_min:g}-year minimum ({domain.value} requirement)"
                )
            return result

        if effective.years > reference:
            gap = effective.years - reference
            ratio = (effective.years / reference) if reference > 0 else 99.0
            if (
                gap >= policy.overqualification_absolute_gap
                and ratio >= policy.overqualification_ratio
            ):
                result.fit = ExperienceFit.OVERQUALIFIED
                result.overqualified = True
                result.reasons.append(
                    f"{effective.years:g} relevant years vs {reference:g}-year ceiling "
                    f"(gap {gap:g}y, ratio {ratio:.2f}) -> overqualified"
                )
            elif (
                gap >= policy.overqualification_absolute_gap
                or ratio >= policy.overqualification_ratio
            ):
                result.fit = ExperienceFit.POTENTIALLY_OVERQUALIFIED
                result.overqualified = True
                result.reasons.append(
                    f"{effective.years:g} relevant years vs {reference:g}-year ceiling "
                    f"(gap {gap:g}y, ratio {ratio:.2f}) -> potentially overqualified"
                )
            else:
                result.fit = ExperienceFit.ACCEPTABLE
                result.reasons.append(
                    f"{effective.years:g} relevant years exceeds the {reference:g}-year "
                    "reference but stays within overqualification tolerances"
                )
            return result

        result.fit = ExperienceFit.STRONG_MATCH
        result.reasons.append(
            f"{effective.years:g} relevant years ({effective.basis}) fits the "
            f"{job_min:g}" + (f"-{job_max:g}" if job_max is not None else "+")
            + f" year requirement ({domain.value})"
        )
        return result

    # job_min is None but job_max exists ("up to N years") -> rare
    result.fit = ExperienceFit.UNKNOWN
    result.reasons.append("only a maximum experience bound was found; treating as unknown")
    return result


def _infer_from_seniority(
    requirements: JobRequirements, policy: ExperiencePolicy
) -> Optional[Tuple[float, Optional[float], str]]:
    if not policy.infer_experience_from_seniority:
        return None
    if requirements.seniority in (Seniority.UNKNOWN, None):
        return None
    lo, hi = SENIORITY_YEARS.get(requirements.seniority, (0.0, 20.0))
    return lo, hi, "seniority_inference"


# ---------------------------------------------------------------------------
# Merging Qwen + deterministic extraction
# ---------------------------------------------------------------------------
def merge_requirements(
    regex_req: JobRequirements,
    llm_req: Optional[JobRequirements],
) -> JobRequirements:
    """
    Merge LLM extraction with the deterministic one.

    Rules:
        * experience bounds: prefer the LLM's when it produced any, else regex
        * domain: prefer the LLM's unless it answered "general" while the
          deterministic classifier found something specific
        * skills: union, LLM first (deterministic scan runs later anyway)
        * seniority: prefer the deterministic title-based inference
    """
    if llm_req is None:
        return regex_req
    merged = llm_req.model_copy()
    updates = {}

    if merged.min_experience_years is None and regex_req.min_experience_years is not None:
        updates["min_experience_years"] = regex_req.min_experience_years
        updates["max_experience_years"] = regex_req.max_experience_years
        updates["experience_confidence"] = regex_req.experience_confidence
        updates["experience_text"] = regex_req.experience_text
    elif merged.min_experience_years is not None:
        updates["experience_confidence"] = max(
            merged.experience_confidence, regex_req.experience_confidence * 0.9
        )

    if (
        merged.experience_domain == ExperienceDomain.GENERAL
        and regex_req.experience_domain != ExperienceDomain.GENERAL
    ):
        updates["experience_domain"] = regex_req.experience_domain

    if regex_req.seniority != Seniority.UNKNOWN:
        updates["seniority"] = regex_req.seniority

    merged_req = merged.model_copy(update=updates)
    merged_req.extraction_method = "merged"
    return merged_req


def requirement_fingerprint(requirements: JobRequirements) -> str:
    """Stable fingerprint used to detect requirement changes (section 22)."""
    import hashlib

    blob = requirements.model_dump_json(exclude={"responsibilities"}).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
