"""
Semantic retrieval stage (sections 13-15).

Builds structured, *truncated* text representations of the resume and of
each job (never blindly embedding a huge description), computes cosine
similarity in one batch, and selects the top-K jobs that proceed to Qwen.

Embeddings decide semantic relevance only — eligibility was already settled
by the rule engine, and final scoring stays deterministic.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple

from config import AppConfig
from llm.schemas import CandidateProfile, JobRequirements
from matching.embeddings import EmbeddingProvider, rescale_similarity
from models import JobRecord

logger = logging.getLogger(__name__)

_REQUIREMENT_SECTION_RE = re.compile(
    r"(requirements?|qualifications?|what you.{0,20}bring|skills? needed|"
    r"responsibilit(?:y|ies)|what you.{0,20}do|about the role|your profile)",
    re.I,
)


def extract_relevant_sections(description: str, max_chars: int) -> str:
    """
    Prefer requirement/responsibility sections over the generic description
    when the posting is long.  Falls back to the head of the text.
    """
    if not description:
        return ""
    if len(description) <= max_chars:
        return description
    marker = _REQUIREMENT_SECTION_RE.search(description)
    if marker is not None and marker.start() < len(description) - 100:
        # keep a little role context plus the requirement block onward
        head = description[max(0, marker.start() - max_chars // 4) : marker.start()]
        tail = description[marker.start() : marker.start() + max_chars]
        return (head + "\n" + tail)[:max_chars + max_chars // 4]
    return description[:max_chars]


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------
def build_resume_text(profile: CandidateProfile) -> str:
    domain_years = "; ".join(
        f"{domain.replace('_', ' ')}: {years:g} years"
        for domain, years in sorted(profile.experience_by_domain.items())
        if (years or 0) > 0
    )
    return (
        f"TITLE: {profile.current_title}\n"
        f"TARGET ROLES: {', '.join(profile.target_roles)}\n"
        f"SUMMARY: {profile.summary}\n"
        f"CORE SKILLS: {', '.join(profile.core_skills)}\n"
        f"SKILLS: {', '.join(profile.skills[:40])}\n"
        f"DOMAINS: {', '.join(profile.domains)}\n"
        f"DOMAIN EXPERIENCE: {domain_years}\n"
        f"TOTAL EXPERIENCE: {profile.total_experience_years:g} years, "
        f"RELEVANT: {profile.relevant_experience_years:g} years"
    )


def build_job_text(
    job: JobRecord,
    requirements: Optional[JobRequirements],
    max_description_chars: int,
) -> str:
    req = requirements or JobRequirements()
    if req.min_experience_years is not None and req.max_experience_years is not None:
        experience = f"{req.min_experience_years:g}-{req.max_experience_years:g} years"
    elif req.min_experience_years is not None:
        experience = f"{req.min_experience_years:g}+ years"
    elif req.max_experience_years is not None:
        experience = f"up to {req.max_experience_years:g} years"
    else:
        experience = "not specified"

    description = extract_relevant_sections(job.description, max_description_chars)
    responsibilities = "; ".join(req.responsibilities[:6])
    skills = ", ".join(list(req.required_skills) + list(req.preferred_skills))

    return (
        f"TITLE: {job.title}\n"
        f"COMPANY: {job.company}\n"
        f"LOCATION: {job.location} | WORK MODE: {job.work_mode}\n"
        f"SKILLS: {skills}\n"
        f"EXPERIENCE: {experience} in {req.experience_domain.value.replace('_', ' ')}\n"
        f"RESPONSIBILITIES: {responsibilities}\n"
        f"DOMAIN: {req.experience_domain.value.replace('_', ' ')}\n"
        f"DESCRIPTION: {description}"
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def rank_jobs(
    profile: CandidateProfile,
    jobs: Sequence[Tuple[JobRecord, Optional[JobRequirements]]],
    provider: EmbeddingProvider,
    cfg: AppConfig,
    store=None,  # database.sqlite_store.SqliteStore (embedding cache)
) -> List[Tuple[JobRecord, Optional[JobRequirements], float]]:
    """
    Embed resume + jobs (single batch, with SQLite vector cache), compute
    cosine similarity and return the top-K jobs above the embedding
    threshold, most similar first.
    """
    if not jobs:
        return []

    resume_text = build_resume_text(profile)
    resume_vector = provider.encode([resume_text], cfg.embedding_batch_size)[0]

    job_texts = [
        build_job_text(job, requirements, cfg.matching.max_description_chars_embedding)
        for job, requirements in jobs
    ]
    logger.info("Embedding %d job(s) with '%s' ...", len(jobs), provider.name)

    vectors = [None] * len(jobs)
    missing_indices: List[int] = []
    for i, (job, _req) in enumerate(jobs):
        vector = None
        if store is not None and job.content_hash:
            key = _embedding_cache_key(provider.name, job.content_hash)
            try:
                vector = store.load_embedding(key)
            except Exception:  # pragma: no cover
                vector = None
        if vector is None:
            missing_indices.append(i)
        else:
            vectors[i] = vector
    if missing_indices:
        encoded = provider.encode([job_texts[i] for i in missing_indices], cfg.embedding_batch_size)
        for j, i in enumerate(missing_indices):
            vectors[i] = encoded[j]
            job = jobs[i][0]
            if store is not None and job.content_hash:
                try:
                    store.save_embedding(
                        _embedding_cache_key(provider.name, job.content_hash), provider.name, encoded[j]
                    )
                except Exception:  # pragma: no cover
                    pass

    scored: List[Tuple[JobRecord, Optional[JobRequirements], float]] = []
    for i, (job, requirements) in enumerate(jobs):
        similarity = provider.similarity(resume_vector, vectors[i])
        scored.append((job, requirements, similarity))

    scored.sort(key=lambda item: item[2], reverse=True)

    threshold = cfg.matching.embedding_threshold
    top_k = cfg.matching.top_k_embedding
    selected = [item for item in scored if item[2] >= threshold][:top_k]
    dropped_by_threshold = sum(1 for item in scored if item[2] < threshold)
    if dropped_by_threshold:
        logger.info(
            "Embedding stage: %d job(s) below similarity threshold %.2f",
            dropped_by_threshold, threshold,
        )
    if len(scored) > top_k:
        logger.info(
            "Embedding stage: keeping top %d of %d eligible job(s) "
            "(similarity %.3f - %.3f)",
            len(selected), len(scored),
            selected[-1][2] if selected else 0.0, selected[0][2] if selected else 0.0,
        )
    if selected:
        logger.info(
            "Top similarity: %.3f ('%s' @ %s)", selected[0][2], selected[0][0].title, selected[0][0].company
        )
    return selected


def _embedding_cache_key(model: str, content_hash: str) -> str:
    return f"{model}|{content_hash}"


def embedding_score(similarity: float, cfg: AppConfig) -> float:
    return round(
        rescale_similarity(
            similarity,
            cfg.matching.embedding_score_floor,
            cfg.matching.embedding_score_ceiling,
        ),
        1,
    )
