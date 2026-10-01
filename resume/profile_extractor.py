"""
Candidate-profile extraction: Qwen first, deterministic fallback always.

Pipeline:
    1. cache lookup (resume text hash + model + prompt version)
    2. Qwen structured extraction (strict JSON -> CandidateProfile)
    3. heuristic fallback (regex + skill taxonomy scan) when Qwen is
       unavailable or returns garbage — degraded but functional (--offline)

After extraction the profile is post-processed deterministically:
    * skills are normalised through the skill taxonomy
    * seniority is inferred from the title (never by the LLM)
    * target roles get a sane default from core skills when missing
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Optional

from config import AppConfig
from llm.ollama_client import OllamaClient
from llm.prompts import PROMPT_VERSION, RESUME_PROFILE_SYSTEM, resume_profile_user
from llm.schemas import CandidateProfile
from matching import skill_taxonomy
from matching.rules import candidate_seniority

logger = logging.getLogger(__name__)

# default domain roles used when a resume gives no explicit target
_DEFAULT_TARGET_ROLES = [
    "Machine Learning Engineer",
    "AI Engineer",
    "Deep Learning Engineer",
]


def resume_fingerprint(resume_text: str, model: str) -> str:
    blob = f"{PROMPT_VERSION}|{model}|{hashlib.sha256(resume_text.encode('utf-8')).hexdigest()}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_profile(
    resume_text: str,
    client: Optional[OllamaClient],
    cfg: AppConfig,
    store=None,  # database.sqlite_store.SqliteStore (avoided import cycle)
) -> CandidateProfile:
    fingerprint = resume_fingerprint(resume_text, cfg.qwen.model)

    # 1. cache
    if store is not None:
        cached = store.load_profile(fingerprint)
        if cached is not None:
            logger.info("Using cached candidate profile (fingerprint %s...)", fingerprint[:12])
            return cached

    # 2. Qwen extraction
    profile: Optional[CandidateProfile] = None
    if client is not None:
        logger.info("Extracting candidate profile with %s ...", cfg.qwen.model)
        try:
            profile = client.chat_json(
                RESUME_PROFILE_SYSTEM, resume_profile_user(resume_text), schema_model=CandidateProfile
            )
        except Exception as exc:
            logger.warning("Profile extraction via LLM failed: %s", exc)
        if profile is None:
            logger.warning("Qwen profile extraction failed; falling back to heuristic parsing")
    else:
        logger.info("Running in offline mode: heuristic profile extraction")

    # 3. heuristic fallback
    if profile is None:
        profile = heuristic_profile(resume_text)

    profile = postprocess(profile)

    if store is not None:
        store.save_profile(fingerprint, cfg.qwen.model, profile)
        logger.info("Cached candidate profile for fingerprint %s...", fingerprint[:12])
    return profile


# ---------------------------------------------------------------------------
# Deterministic post-processing
# ---------------------------------------------------------------------------
def postprocess(profile: CandidateProfile) -> CandidateProfile:
    """Normalise skills, infer seniority, apply sane defaults."""
    data = profile.model_dump()

    skills = _normalize_skill_list(data.get("skills", []))
    core = _normalize_skill_list(data.get("core_skills", []))
    secondary = _normalize_skill_list(data.get("secondary_skills", []))

    # keep core ⊆ skills; fill skills from core+secondary when empty
    for s in core:
        if s not in skills:
            skills.append(s)
    if not skills:
        skills = core + secondary
    if not core and skills:
        core = skills[:10]

    target_roles = [r.strip() for r in data.get("target_roles", []) if r and r.strip()]
    if not target_roles:
        target_roles = _infer_target_roles(skills, data.get("current_title", ""))

    excluded = [r.strip() for r in data.get("excluded_roles", []) if r and r.strip()]

    relevant = float(data.get("relevant_experience_years") or 0)
    total = float(data.get("total_experience_years") or 0)
    if relevant <= 0:
        relevant = total
    if total <= 0:
        total = relevant
    if relevant > total > 0:
        relevant = total

    profile = CandidateProfile(
        **{**data,
           "skills": skills,
           "core_skills": core,
           "secondary_skills": secondary,
           "target_roles": target_roles,
           "excluded_roles": excluded,
           "total_experience_years": total,
           "relevant_experience_years": relevant,
           }
    )
    profile.inferred_seniority = candidate_seniority(profile)
    return profile


def _normalize_skill_list(skills: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in skills or []:
        if not raw or not str(raw).strip():
            continue
        canonical, _ = skill_taxonomy.resolve_skill(str(raw))
        canonical = canonical or str(raw).strip()
        key = canonical.lower()
        if key not in seen:
            seen.add(key)
            out.append(canonical)
    return out


def _infer_target_roles(skills: List[str], current_title: str) -> List[str]:
    """Derive target roles from current title / dominant skill categories."""
    roles: List[str] = []
    if current_title.strip():
        roles.append(current_title.strip())
    categories = {skill_taxonomy.category_of(s) for s in skills}
    category_roles = {
        "MODEL_DEPLOYMENT": "ML Deployment Engineer",
        "COMPUTER_VISION": "Computer Vision Engineer",
        "DEEP_LEARNING": "Deep Learning Engineer",
        "NLP": "NLP Engineer",
        "GENAI": "AI Engineer",
        "MLOPS": "MLOps Engineer",
        "CLASSICAL_ML": "Machine Learning Engineer",
    }
    for category, role in category_roles.items():
        if category in categories and role not in roles:
            roles.append(role)
    if not roles:
        roles = list(_DEFAULT_TARGET_ROLES)
    return roles[:6]


# ---------------------------------------------------------------------------
# Heuristic (offline) extraction
# ---------------------------------------------------------------------------
_SUMMARY_SECTION = re.compile(
    r"(?:professional\s+)?summary|about\s+me|profile|objective", re.I
)
_SKILLS_SECTION = re.compile(r"^(?:technical\s+)?skills?(?:\s*&\s*tools)?\s*:?\s*$", re.I | re.M)
_SECTION_HEADER = re.compile(r"^[A-Z][A-Z /&+,-]{3,40}$", re.M)


def heuristic_profile(resume_text: str) -> CandidateProfile:
    """
    Deterministic best-effort profile used when Qwen is unavailable.
    Handles the common resume layout: name, title, summary, skills list.
    """
    from resume.parser import guess_name_from_text

    text = resume_text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = guess_name_from_text(text) or ""
    current_title = _guess_title(lines, name)

    # experience: scan summary + whole text for "X+ years" statements,
    # preferring the largest credible number for total and the number in the
    # summary (usually relevant) for relevant years.
    total_years, relevant_years, domain_years = _heuristic_years(text)

    # skills
    skills = skill_taxonomy.extract_skills_from_text(text, max_skills=50)
    skills = _normalize_skill_list(skills)
    core = skills[:12]

    # target / excluded roles from dedicated sections if present
    target_roles, excluded_roles = _heuristic_roles(text)

    # locations: any known Indian city / remote mention
    locations = _heuristic_locations(text)

    summary = _heuristic_summary(lines)

    profile = CandidateProfile(
        name=name,
        current_title=current_title,
        total_experience_years=total_years,
        relevant_experience_years=relevant_years,
        experience_by_domain=domain_years,
        skills=skills,
        core_skills=core,
        secondary_skills=skills[12:30],
        domains=[],
        target_roles=target_roles or _infer_target_roles(skills, current_title),
        education=[],
        certifications=[],
        locations=locations,
        preferred_work_modes=[],
        excluded_roles=excluded_roles,
        summary=summary,
    )
    logger.info(
        "Heuristic profile: total=%.1fy relevant=%.1fy skills=%d",
        total_years, relevant_years, len(skills),
    )
    return profile


def _guess_title(lines: List[str], name: str) -> str:
    for line in lines[:6]:
        if line == name:
            continue
        if re.search(r"engineer|developer|scientist|analyst|manager|architect|consultant", line, re.I) and len(line) < 90:
            return line
    return ""


def _heuristic_years(text: str) -> tuple:
    """Total = largest credible 'years of experience' claim on the resume."""
    from matching.experience import _PATTERNS, _EXPERIENCE_CONTEXT_WORDS, _sentences

    # resumes wrap lines mid-phrase ("... in machine\nlearning"); join first
    flat_text = re.sub(r"\s+", " ", text or "")

    total = 0.0
    for sentence in _sentences(flat_text):
        if not _EXPERIENCE_CONTEXT_WORDS.search(sentence):
            continue
        for priority, pattern, kind in _PATTERNS:
            for m in pattern.finditer(sentence):
                try:
                    years = float(m.group(1))
                except (ValueError, IndexError):
                    continue
                if kind == "range":
                    if priority >= 4.0 and years <= 45:
                        total = max(total, years)
                    continue
                if 0 < years <= 45:
                    total = max(total, years)

    relevant = 0.0
    domain_years: dict = {}
    # look for explicit per-domain statements like "4 years in machine learning"
    domain_pattern = re.compile(
        r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)[^.\n]{0,30}?\b"
        r"(machine learning|deep learning|computer vision|cv|nlp|deployment|quantiz\w*|"
        r"mlops|software|network\w*|telecom)\b",
        re.I,
    )
    for m in domain_pattern.finditer(flat_text):
        years = float(m.group(1))
        raw_domain = m.group(2).lower()
        key = {
            "machine learning": "machine_learning",
            "deep learning": "deep_learning",
            "computer vision": "computer_vision",
            "cv": "computer_vision",
            "nlp": "nlp",
            "deployment": "model_deployment",
            "mlops": "mlops",
        }.get(raw_domain)
        if key is None:
            if raw_domain.startswith("quantiz"):
                key = "quantization"
            elif raw_domain.startswith("software"):
                key = "software_engineering"
            else:
                key = "networking"
        domain_years[key] = max(domain_years.get(key, 0.0), years)

    if domain_years:
        ml_keys = ["machine_learning", "deep_learning", "computer_vision", "nlp",
                   "genai", "model_deployment", "quantization", "mlops"]
        relevant = max((domain_years.get(k, 0.0) for k in ml_keys), default=0.0)
    if relevant <= 0:
        relevant = total
    if total <= 0:
        total = relevant
    if relevant > total:
        relevant = total
    return total, relevant, domain_years


def _heuristic_roles(text: str) -> tuple:
    target_roles: List[str] = []
    excluded: List[str] = []

    def _role_items(section_text: str) -> List[str]:
        """Split section lines into individual role names (commas included)."""
        items: List[str] = []
        for line in section_text.splitlines():
            line = line.strip().strip("-•* ")
            if not line or re.search(r"experience|education|skills|summary", line, re.I):
                break
            for part in line.split(","):
                part = part.strip()
                if 3 < len(part) < 60:
                    items.append(part)
        return items

    target_section = re.search(
        r"target\s+roles?\s*:?\s*\n((?:.*\n){0,8})", text, re.I
    )
    if target_section:
        for part in _role_items(target_section.group(1)):
            if re.search(r"engineer|developer|scientist|analyst|architect", part, re.I):
                target_roles.append(part)

    exclude_section = re.search(
        r"exclud\w+\s+roles?\s*:?\s*\n((?:.*\n){0,8})", text, re.I
    )
    if exclude_section:
        excluded = _role_items(exclude_section.group(1))
    return target_roles, excluded


def _heuristic_locations(text: str) -> List[str]:
    from matching.rules import normalize_location as _norm

    found = []
    for token in re.findall(r"[A-Za-z ]{4,25}", text):
        norm = _norm(token)
        if norm and norm not in ("remote", "india") and norm == token.strip().lower():
            if norm not in found:
                found.append(norm)
        if len(found) >= 3:
            break
    if re.search(r"\bremote\b|work from home", text, re.I):
        found.append("Remote")
    return found[:4]


def _heuristic_summary(lines: List[str]) -> str:
    for i, line in enumerate(lines[:20]):
        if _SUMMARY_SECTION.search(line):
            block = []
            for candidate_line in lines[i + 1 : i + 6]:
                if _SKILLS_SECTION.match(candidate_line) or _SECTION_HEADER.match(candidate_line):
                    break
                block.append(candidate_line)
            summary = " ".join(block).strip()
            if 20 < len(summary) < 600:
                return summary
    return lines[1] if len(lines) > 1 and len(lines[1]) > 40 else ""
