"""
Deterministic final scoring and verdicts (sections 17-19, 31-33).

The final score is ALWAYS computed by this module from component scores:

    final = 0.30*experience + 0.25*skills + 0.20*embedding
          + 0.10*responsibility + 0.10*domain + 0.05*career_alignment

Component sources (and who is allowed to influence what):
    experience      -> deterministic only (Qwen cannot touch it, RULE 3)
    skills          -> deterministic taxonomy score, optionally blended with
                       Qwen's skill score
    embedding       -> deterministic cosine rescale
    responsibility  -> Qwen when available, else deterministic fallback
    domain          -> Qwen when available, else deterministic fallback
    career_align.   -> deterministic role/skill logic + Qwen blending

Guardrails:
    * a hard rejection always wins: verdict = REJECTED regardless of score
    * Qwen scores may only *downgrade* (never rescue a rejected job)
    * MARGINAL / POTENTIALLY_OVERQUALIFIED experience caps the verdict at
      BORDERLINE even when the numeric score is high
    * when |qwen - deterministic| > tolerance the deterministic value wins
      (a single hallucinated score cannot dominate)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from config import AppConfig, normalize_weights
from llm.schemas import CandidateProfile, JobAnalysis, JobRequirements
from matching.embeddings import rescale_similarity
from models import (
    ExperienceDomain,
    ExperienceEvaluation,
    ExperienceFit,
    JobMatchResult,
    JobRecord,
    Rejection,
    RejectionReason,
    SkillMatchResult,
    Verdict,
)

logger = logging.getLogger(__name__)

_DOMAIN_SYNONYM_GROUPS = [
    {"machine_learning", "deep_learning", "computer_vision", "nlp", "genai",
     "model_deployment", "quantization", "mlops"},
    {"software_engineering"},
    {"networking"},
]


# ---------------------------------------------------------------------------
# Deterministic fallback components (used without/blended with Qwen)
# ---------------------------------------------------------------------------
def deterministic_domain_score(profile: CandidateProfile, requirements: JobRequirements) -> float:
    """Overlap between the job's domain and the candidate's domain experience."""
    job_domain = requirements.experience_domain
    if job_domain in (ExperienceDomain.GENERAL, ExperienceDomain.OTHER):
        return 55.0
    if profile.domain_years(job_domain) > 0:
        return 85.0
    # adjacent domains count partially
    from matching.experience import DOMAIN_ADJACENCY

    for adjacent in DOMAIN_ADJACENCY.get(job_domain, []):
        if profile.domain_years(adjacent) > 0:
            return 65.0
    # textual overlap of candidate domains with the job domain
    for group in _DOMAIN_SYNONYM_GROUPS:
        if job_domain.value in group:
            for candidate_domain in profile.domains:
                if candidate_domain.lower().replace(" ", "_") in group:
                    return 60.0
    return 25.0


def deterministic_responsibility_score(
    profile: CandidateProfile,
    requirements: JobRequirements,
    skills: SkillMatchResult,
) -> float:
    """Proxy: skill coverage weighted toward required skills."""
    if not requirements.required_skills and not requirements.preferred_skills:
        return 55.0
    return skills.score


def deterministic_career_alignment(
    profile: CandidateProfile,
    job: JobRecord,
    requirements: JobRequirements,
    semantic_score: float,
) -> Tuple[float, str]:
    """
    Career direction is NOT technical capability (section 32):
        * title directly matches a target role        -> 90
        * title mentions a target keyword             -> 70
        * generic software title for an AI/ML target  -> capped at 30
        * otherwise blend semantic relevance          -> 20-60
    """
    from matching.rules import _GENERIC_SOFTWARE_ROLES

    title = (job.title or "").lower()
    targets = [t.lower() for t in profile.target_roles if t.strip()]

    for target in targets:
        if target in title:
            return 90.0, f"job title matches target role '{target}'"

    # keyword overlap between title and target roles (e.g. 'vision' in both),
    # ignoring generic role words like "engineer" that match everything
    generic_words = {
        "engineer", "developer", "scientist", "analyst", "manager",
        "architect", "senior", "junior", "lead", "staff", "principal",
        "intern", "associate", "specialist", "consultant", "technical",
    }
    target_words = set()
    for target in targets:
        target_words.update(
            w
            for w in target.replace("/", " ").replace("-", " ").split()
            if len(w) > 2 and w not in generic_words
        )
    title_words = set(title.replace("-", " ").replace("/", " ").split())
    if target_words and target_words & title_words:
        return 70.0, "job title shares key terms with the candidate's target roles"

    # generic software roles are a career step away from an AI/ML target
    ml_targeted = any(
        re_search(target, r"ml|machine learning|ai|deep learning|vision|learning")
        for target in targets
    )
    if ml_targeted and _GENERIC_SOFTWARE_ROLES.search(title):
        return 30.0, (
            "generic software engineering role while the candidate targets AI/ML roles"
        )

    blended = 25.0 + 0.35 * max(0.0, min(100.0, semantic_score))
    return blended, "career alignment estimated from semantic relevance"


def re_search(text: str, pattern: str) -> bool:
    import re

    return bool(re.search(pattern, text, re.I))


# ---------------------------------------------------------------------------
# Qwen reconciliation
# ---------------------------------------------------------------------------
def reconcile(
    qwen_value: Optional[float],
    deterministic_value: float,
    tolerance: float,
    label: str = "",
) -> float:
    """Blend Qwen's nuanced score with the deterministic one; sanity-clamped."""
    if qwen_value is None:
        return deterministic_value
    if abs(qwen_value - deterministic_value) > tolerance:
        logger.debug(
            "Qwen %s score %.0f deviates > %.0f from deterministic %.0f; "
            "keeping deterministic value",
            label, qwen_value, tolerance, deterministic_value,
        )
        return deterministic_value
    return round(0.5 * qwen_value + 0.5 * deterministic_value, 1)


# ---------------------------------------------------------------------------
# Final scoring
# ---------------------------------------------------------------------------
def compute_final_score(components: dict, weights: dict) -> float:
    total_weight = sum(weights.values()) or 1.0
    score = sum(components.get(key, 0.0) * weight for key, weight in weights.items())
    return round(score / total_weight, 1)


def score_job(
    profile: CandidateProfile,
    job: JobRecord,
    requirements: JobRequirements,
    experience_eval: ExperienceEvaluation,
    skills: SkillMatchResult,
    embedding_similarity: float,
    rejections: List[Rejection],
    qwen_analysis: Optional[JobAnalysis],
    cfg: AppConfig,
) -> JobMatchResult:
    """Assemble the full deterministic result for one job."""
    embedding = round(
        rescale_similarity(
            embedding_similarity,
            cfg.matching.embedding_score_floor,
            cfg.matching.embedding_score_ceiling,
        ),
        1,
    )

    det_domain = deterministic_domain_score(profile, requirements)
    det_resp = deterministic_responsibility_score(profile, requirements, skills)
    det_alignment, alignment_reason = deterministic_career_alignment(
        profile, job, requirements, embedding
    )

    if qwen_analysis is not None:
        domain_score = reconcile(
            qwen_analysis.domain_match_score, det_domain,
            cfg.matching.qwen_deterministic_tolerance, "domain",
        )
        responsibility_score = reconcile(
            qwen_analysis.responsibility_match_score, det_resp,
            cfg.matching.qwen_deterministic_tolerance, "responsibility",
        )
        alignment_score = reconcile(
            qwen_analysis.career_alignment_score, det_alignment,
            cfg.matching.qwen_deterministic_tolerance, "career-alignment",
        )
        skill_score = reconcile(
            qwen_analysis.skill_match_score, skills.score,
            cfg.matching.qwen_deterministic_tolerance, "skills",
        )
    else:
        domain_score = det_domain
        responsibility_score = det_resp
        alignment_score = det_alignment
        skill_score = skills.score

    experience_score = experience_eval.score

    components = {
        "experience": experience_score,
        "skills": skill_score,
        "embedding": embedding,
        "responsibility": responsibility_score,
        "domain": domain_score,
        "career_alignment": alignment_score,
    }
    final_score = compute_final_score(components, normalize_weights(cfg))

    # hard rejections dominate the numeric score (RULE 3 + section 18)
    hard_rejected = bool(rejections)
    if alignment_score < cfg.matching.min_career_alignment:
        rejections = list(rejections) + [
            Rejection(
                RejectionReason.LOW_CAREER_ALIGNMENT,
                f"career alignment {alignment_score:.0f} below minimum "
                f"{cfg.matching.min_career_alignment:.0f} ({alignment_reason})",
                stage="scoring",
            )
        ]
        hard_rejected = True

    result = JobMatchResult(
        job=job,
        requirements=requirements,
        experience=experience_eval,
        skills=skills,
        embedding_score=embedding,
        embedding_similarity=round(embedding_similarity, 4),
        experience_score=round(experience_score, 1),
        skill_score=round(skill_score, 1),
        domain_score=round(domain_score, 1),
        responsibility_score=round(responsibility_score, 1),
        career_alignment_score=round(alignment_score, 1),
        final_score=final_score,
        hard_rejected=hard_rejected,
        rejections=rejections,
        qwen_analysis=qwen_analysis,
    )
    result.verdict = determine_verdict(result, cfg)
    result.strengths, result.concerns, result.reasoning = build_explanations(
        result, profile, alignment_reason
    )
    return result


def determine_verdict(result: JobMatchResult, cfg: AppConfig) -> Verdict:
    """Verdict = hard filters first, then thresholds, then experience caps."""
    # 1. hard rejections can never be rescued by a high score
    if result.hard_rejected:
        return Verdict.REJECTED

    # 2. Qwen may downgrade, never upgrade (configurable safety cap)
    if (
        result.qwen_analysis is not None
        and result.qwen_analysis.match_score <= cfg.matching.qwen_low_score_cap
    ):
        return Verdict.BORDERLINE

    # 3. experience quality caps: an ambiguous experience situation cannot
    #    produce a full MATCH even with perfect skills
    if result.experience.fit in (ExperienceFit.MARGINAL, ExperienceFit.POTENTIALLY_OVERQUALIFIED):
        if result.final_score >= cfg.matching.borderline_threshold:
            return Verdict.BORDERLINE

    # 4. score thresholds
    if result.final_score >= cfg.matching.match_threshold:
        return Verdict.MATCH
    if result.final_score >= cfg.matching.borderline_threshold:
        return Verdict.BORDERLINE
    return Verdict.REJECTED


# ---------------------------------------------------------------------------
# Explainability (section 33)
# ---------------------------------------------------------------------------
def build_explanations(
    result: JobMatchResult,
    profile: CandidateProfile,
    alignment_reason: str,
) -> Tuple[List[str], List[str], str]:
    """Deterministic, human-readable why-matched / why-not text."""
    strengths: List[str] = []
    concerns: List[str] = []
    exp = result.experience

    # --- experience story ---------------------------------------------------
    if exp.fit == ExperienceFit.STRONG_MATCH:
        strengths.append(
            f"{exp.effective_candidate_years:g} relevant years ({exp.basis}) fits the "
            f"{_fmt_range(exp.job_min_years, exp.job_max_years)} year requirement"
        )
    elif exp.fit == ExperienceFit.ACCEPTABLE:
        strengths.append(
            f"{exp.effective_candidate_years:g} relevant years is acceptable for the "
            f"{_fmt_range(exp.job_min_years, exp.job_max_years)} year requirement"
        )
    elif exp.fit == ExperienceFit.MARGINAL:
        concerns.append(
            f"slightly under the {exp.job_min_years:g}-year requirement "
            f"({exp.effective_candidate_years:g} relevant years)"
        )
    elif exp.fit == ExperienceFit.UNDERQUALIFIED:
        concerns.append(
            f"underqualified on experience: {exp.effective_candidate_years:g} relevant years "
            f"({exp.basis}) vs {exp.job_min_years:g} required in {exp.domain.value}"
        )
    elif exp.fit in (ExperienceFit.POTENTIALLY_OVERQUALIFIED,):
        concerns.append(
            f"potentially overqualified: {exp.effective_candidate_years:g} relevant years vs "
            f"{_fmt_range(exp.job_min_years, exp.job_max_years)} wanted"
        )
    elif exp.fit == ExperienceFit.OVERQUALIFIED:
        concerns.append(
            f"overqualified: {exp.effective_candidate_years:g} relevant years vs "
            f"{_fmt_range(exp.job_min_years, exp.job_max_years)} wanted"
        )
    elif exp.fit == ExperienceFit.UNKNOWN:
        strengths.append("job states no explicit experience requirement")

    # --- skills story --------------------------------------------------------
    exact = [m.job_skill for m in result.skills.exact]
    related = [m.job_skill for m in result.skills.related]
    transferable = [m.job_skill for m in result.skills.transferable]
    missing = [m.job_skill for m in result.skills.missing]
    if exact:
        strengths.append(f"exact skill matches: {', '.join(exact[:8])}")
    if related:
        strengths.append(f"related skill matches: {', '.join(related[:6])}")
    if transferable:
        concerns.append(
            f"no direct experience with: {', '.join(transferable[:6])} "
            "(related experience exists)"
        )
    if missing:
        concerns.append(f"missing skills: {', '.join(missing[:8])}")

    # --- semantic / alignment story ------------------------------------------
    if result.embedding_score >= 70:
        strengths.append("strong semantic similarity between resume and role")
    if result.career_alignment_score >= 70:
        strengths.append(f"aligns with target career: {alignment_reason}")
    elif result.career_alignment_score < 40:
        concerns.append(f"limited career alignment: {alignment_reason}")

    # --- merge Qwen's narrative ----------------------------------------------
    if result.qwen_analysis is not None:
        for s in result.qwen_analysis.strengths[:4]:
            if s not in strengths:
                strengths.append(s)
        for c in result.qwen_analysis.concerns[:4]:
            if c not in concerns:
                concerns.append(c)

    strengths = strengths[:6]
    concerns = concerns[:6]

    # --- final reasoning line -------------------------------------------------
    if result.hard_rejected:
        reasons = "; ".join(r.reason.value + ": " + r.detail for r in result.rejections[:3])
        reasoning = f"REJECTED — {reasons}"
    else:
        parts = [
            f"final score {result.final_score:.1f} ({result.verdict.value})",
            f"experience {result.experience_score:.0f}/100 ({exp.fit.value})",
            f"skills {result.skill_score:.0f}/100",
            f"semantic {result.embedding_score:.0f}/100",
            f"career alignment {result.career_alignment_score:.0f}/100",
        ]
        if result.qwen_analysis is not None and result.qwen_analysis.reasoning:
            parts.append("Qwen: " + result.qwen_analysis.reasoning[:300])
        reasoning = " | ".join(parts)

    return strengths, concerns, reasoning


def _fmt_range(min_years: Optional[float], max_years: Optional[float]) -> str:
    if min_years is None and max_years is None:
        return "unspecified"
    if min_years is not None and max_years is not None:
        return f"{min_years:g}-{max_years:g}"
    if min_years is not None:
        return f"{min_years:g}+"
    return f"up to {max_years:g}"
