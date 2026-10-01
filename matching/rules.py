"""
Deterministic hard filters (spec sections 9, 10, 23, 24, 31-32).

Everything in this module runs *before* embeddings and before Qwen:

    * seniority normalisation from titles (with a numeric fallback)
    * location normalisation + matching (no embeddings — RULE: basic location
      matching is string/set logic)
    * work-mode and employment-type filters
    * job freshness (posted_date) filtering with "unknown" pass-through
    * role whitelist/blacklist (career direction vs historical experience)
    * experience hard filter (thin wrapper over matching.experience)
    * must-have skill filter (via the skill taxonomy)

The engine collects *all* applicable rejections for a job (not just the
first) so the rejected-jobs CSV can explain itself fully (RULE 8).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Pattern, Tuple

from config import AppConfig
from llm.schemas import CandidateProfile, JobRequirements
from matching import skill_taxonomy
from matching.experience import evaluate_experience, extract_experience_requirement
from models import (
    SENIORITY_ORDER,
    EmploymentType,
    ExperienceEvaluation,
    ExperienceFit,
    JobRecord,
    Rejection,
    RejectionReason,
    Seniority,
    WorkMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seniority normalisation (section 9)
# ---------------------------------------------------------------------------
# Ordered: the first matching pattern wins, so specific levels come first.
_SENIORITY_PATTERNS: List[Tuple[Seniority, Pattern]] = [
    (Seniority.PRINCIPAL, re.compile(r"\bprincipal\b|\bdistinguished\b|\barchitect\b", re.I)),
    (Seniority.MANAGER, re.compile(r"\bmanager\b|\bmanagement\b|\bhead of\b|\bdirector\b|\bvp\b|\bvice president\b", re.I)),
    (Seniority.STAFF, re.compile(r"\bstaff\b", re.I)),
    (Seniority.LEAD, re.compile(r"\blead\b|\bteam lead\b|\btech lead\b|\btechlead\b", re.I)),
    (Seniority.SENIOR, re.compile(r"\bsenior\b|\bseniority\b|\bsr\.?\b|\bsnr\b", re.I)),
    (Seniority.JUNIOR, re.compile(r"\bjunior\b|\bjr\.?\b", re.I)),
    (Seniority.ENTRY, re.compile(r"\bentry[- ]level\b|\bgraduate\b|\bfresher\b|\btrainee\b|\bassociate\b", re.I)),
    (Seniority.INTERN, re.compile(r"\bintern\b|\binternship\b|\bintern ship\b|\bco-?op\b|\bapprentice\b", re.I)),
]

# Roman-numeral leveling, e.g. "Engineer II" (applied when nothing above hit)
_ROMAN_LEVELS: List[Tuple[Pattern, Seniority]] = [
    (re.compile(r"\b(i{1,3}|iv)\s*$", re.I), None),  # resolved below
]
_ROMAN_MAP = {"i": Seniority.ENTRY, "ii": Seniority.MID, "iii": Seniority.SENIOR, "iv": Seniority.SENIOR}


def infer_seniority(title: str) -> Seniority:
    """Deterministic title -> seniority mapping."""
    if not title or not title.strip():
        return Seniority.UNKNOWN
    cleaned = _clean_title(title)
    for seniority, pattern in _SENIORITY_PATTERNS:
        if pattern.search(cleaned):
            return seniority
    roman = re.search(r"\b(i{1,3}|iv)\s*$", cleaned, re.I)
    if roman:
        return _ROMAN_MAP.get(roman.group(1).lower(), Seniority.MID)
    return Seniority.MID  # an unmarked professional title is mid-level


def _clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip())


def seniority_from_years(
    total_years: float, relevant_years: Optional[float] = None
) -> Seniority:
    """Fallback when the title carries no level marker (deliberately conservative)."""
    years = relevant_years if relevant_years and relevant_years > 0 else total_years
    if years >= 12:
        return Seniority.STAFF
    if years >= 8:
        return Seniority.LEAD
    if years >= 5:
        return Seniority.SENIOR
    if years >= 2.5:
        return Seniority.MID
    if years >= 1:
        return Seniority.JUNIOR
    if years > 0:
        return Seniority.ENTRY
    return Seniority.UNKNOWN


def candidate_seniority(profile: CandidateProfile) -> Seniority:
    """Title rules first; years-based inference only for unmarked titles."""
    if not (profile.current_title or "").strip():
        return seniority_from_years(
            profile.total_experience_years, profile.relevant_experience_years
        )
    from_title = infer_seniority(profile.current_title)
    if from_title != Seniority.MID:
        # an explicitly marked title always wins; MID is the "no marker" default
        return from_title
    return seniority_from_years(
        profile.total_experience_years, profile.relevant_experience_years
    )


def seniority_gap(candidate: Seniority, job: Seniority) -> int:
    if candidate == Seniority.UNKNOWN or job == Seniority.UNKNOWN:
        return 0
    return SENIORITY_ORDER[job] - SENIORITY_ORDER[candidate]


# ---------------------------------------------------------------------------
# Location normalisation + matching (section 24)
# ---------------------------------------------------------------------------
_LOCATION_ALIASES: Dict[str, List[str]] = {
    "bangalore": ["bengaluru", "bangalore", "bengalooru"],
    "hyderabad": ["hyderabad"],
    "pune": ["pune", "poona"],
    "chennai": ["chennai", "madras"],
    "mumbai": ["mumbai", "bombay"],
    "delhi": ["delhi", "new delhi", "gurgaon", "gurugram", "noida", "ncr",
              "national capital region", "faridabad", "ghaziabad"],
    "kolkata": ["kolkata", "calcutta"],
    "ahmedabad": ["ahmedabad"],
    "jaipur": ["jaipur"],
    "kochi": ["kochi", "cochin"],
    "coimbatore": ["coimbatore"],
    "indore": ["indore"],
    "remote": ["remote", "work from home", "wfh", "anywhere", "any location",
               "work from anywhere", "virtual"],
    "india": ["india", "bharat"],
}

_INDIAN_CITIES = [
    "bangalore", "hyderabad", "pune", "chennai", "mumbai", "delhi", "kolkata",
    "ahmedabad", "jaipur", "kochi", "coimbatore", "indore", "noida", "gurgaon",
    "gurugram", "bengaluru", "chandigarh", "bhubaneswar", "nagpur", "vizag",
    "visakhapatnam", "thiruvananthapuram", "trivandrum", "mysore", "mysuru",
    "vadodara", "surat", "lucknow", "kanpur", "patna", "guwahati",
]

# Indeed returns abbreviated locations like "KA, IN" (Karnataka, India)
_INDIAN_STATE_CODES = {
    "ka": "karnataka", "ts": "telangana", "tg": "telangana",
    "ap": "andhra pradesh", "mh": "maharashtra", "tn": "tamil nadu",
    "up": "uttar pradesh", "dl": "delhi", "hr": "haryana", "gj": "gujarat",
    "wb": "west bengal", "kl": "kerala", "pb": "punjab", "rj": "rajasthan",
    "mp": "madhya pradesh", "br": "bihar", "od": "odisha", "or": "odisha",
    "as": "assam", "jh": "jharkhand", "cg": "chhattisgarh",
    "ut": "uttarakhand", "uk": "uttarakhand", "hp": "himachal pradesh",
    "ga": "goa", "py": "puducherry", "ct": "chhattisgarh",
}


def normalize_location(location: str) -> str:
    """Canonicalise a location string: 'Bengaluru, Karnataka, India' -> 'bangalore'."""
    if not location:
        return ""
    loc = location.lower()
    if any(token in loc for token in _LOCATION_ALIASES["remote"]):
        return "remote"
    for canonical, aliases in _LOCATION_ALIASES.items():
        if canonical == "remote":
            continue
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", loc):
                return canonical
    if "india" in loc:
        return "india"
    # state codes / names, e.g. "KA, IN" -> karnataka
    for token in re.split(r"[,\s]+", loc):
        if token in _INDIAN_STATE_CODES:
            return _INDIAN_STATE_CODES[token]
    for state in set(_INDIAN_STATE_CODES.values()):
        if re.search(rf"\b{re.escape(state)}\b", loc):
            return state
    return re.sub(r"\s+", " ", loc.strip())


def is_indian_location(location: str) -> bool:
    loc = (location or "").lower()
    if "india" in loc:
        return True
    if any(city in loc for city in _INDIAN_CITIES):
        return True
    tokens = {t.strip(". ") for t in re.split(r"[,\s]+", loc)}
    if "in" in tokens:  # country code, e.g. "KA, IN"
        return True
    return any(t in _INDIAN_STATE_CODES for t in tokens)


def location_matches(
    job_location: str,
    allowed_locations: List[str],
    job_work_mode: str = "",
) -> Tuple[bool, str]:
    """
    Basic string/alias matching only (never embeddings).

    Returns (matched, detail).  Unknown job locations are never rejected
    (configurable); 'india' allows any Indian city.
    """
    if not allowed_locations:
        return True, "no location filter configured"

    normalized_job = normalize_location(job_location)
    is_remote_job = (
        normalized_job == "remote"
        or job_work_mode == WorkMode.REMOTE.value
        or (job_work_mode or "").lower() == "remote"
    )

    # a remote job is workable from ANY allowed location
    if is_remote_job:
        return True, "remote job: workable from any allowed location"

    for allowed in allowed_locations:
        allowed_norm = normalize_location(allowed)
        if allowed_norm == "remote":
            if is_remote_job:
                return True, "remote work allowed and job is remote"
            continue
        if allowed_norm == "india":
            if is_indian_location(job_location) or normalized_job == "india":
                return True, "job is in India"
            continue
        if normalized_job == allowed_norm:
            return True, f"job location matches '{allowed}'"
        # city match inside a compound string, e.g. "bangalore, karnataka"
        if re.search(rf"\b{re.escape(allowed_norm)}\b", (job_location or "").lower()):
            return True, f"job location mentions '{allowed}'"

    if not normalized_job and not (job_location or "").strip():
        return True, "job location unknown; not rejected"
    return False, f"job location '{job_location}' is not in allowed locations"


# ---------------------------------------------------------------------------
# Work mode / employment type detection
# ---------------------------------------------------------------------------
def detect_work_mode(*texts: str) -> WorkMode:
    combined = " ".join(t for t in texts if t).lower()
    if not combined:
        return WorkMode.UNKNOWN
    if re.search(r"\bremote\b|work from home|\bwfh\b|work from anywhere|anywhere in india", combined):
        return WorkMode.REMOTE
    if re.search(r"\bhybrid\b|partially remote|flexible work", combined):
        return WorkMode.HYBRID
    if re.search(r"\bon[- ]site\b|\bonsite\b|in[- ]office\b|work from office", combined):
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN


def detect_employment_type(*texts: str) -> EmploymentType:
    combined = " ".join(t for t in texts if t).lower()
    if not combined:
        return EmploymentType.UNKNOWN
    if re.search(r"\binternship\b|\bintern\b|\btrainee\b", combined):
        return EmploymentType.INTERNSHIP
    if re.search(r"\bpart[- ]time\b", combined):
        return EmploymentType.PART_TIME
    if re.search(r"\bcontract\b|\bcontractor\b|\bcontractual\b|freelance", combined):
        return EmploymentType.CONTRACT
    if re.search(r"\btemporary\b|\btemp\b", combined):
        return EmploymentType.TEMPORARY
    if re.search(r"\bfull[- ]time\b|\bpermanent\b|\bfulltime\b", combined):
        return EmploymentType.FULL_TIME
    return EmploymentType.UNKNOWN


# ---------------------------------------------------------------------------
# Role relevance (sections 10, 31, 32)
# ---------------------------------------------------------------------------
_ROLE_ALIASES = {
    "ml": ["machine learning", "ml"],
    "ai": ["ai", "artificial intelligence"],
    "cv": ["computer vision", "cv"],
    "dl": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language"],
}

_GENERIC_SOFTWARE_ROLES = re.compile(
    r"software engineer|software developer|software test|test engineer|"
    r"qa engineer|quality assurance|manual test|automation test|"
    r"full[- ]?stack|backend developer|frontend developer|web developer|"
    r"\bsde\b|programmer|application developer|integration engineer|"
    r"support engineer|business analyst|system analyst|product designer|"
    r"devops engineer|\bsdet\b",
    re.I,
)


def _expand_role_pattern(role: str) -> List[str]:
    """'ML Engineer' -> ['ml engineer', 'machine learning engineer']."""
    role = (role or "").strip().lower()
    if not role:
        return []
    expanded = [role]
    for short, longs in _ROLE_ALIASES.items():
        if re.search(rf"\b{short}\b", role):
            for long_form in longs:
                expanded.append(role.replace(short, long_form))
    return list(dict.fromkeys(expanded))


@dataclass
class RoleRules:
    whitelist_patterns: List[Pattern] = field(default_factory=list)
    blacklist_patterns: List[Pattern] = field(default_factory=list)

    def matches_whitelist(self, title: str) -> Optional[str]:
        for pattern in self.whitelist_patterns:
            m = pattern.search(title.lower())
            if m:
                return m.group(0)
        return None

    def matches_blacklist(self, title: str) -> Optional[str]:
        for pattern in self.blacklist_patterns:
            m = pattern.search(title.lower())
            if m:
                return m.group(0)
        return None


def build_role_rules(profile: CandidateProfile, cfg: AppConfig) -> RoleRules:
    """Combine candidate target roles + profile exclusions with config lists."""
    whitelist_terms: List[str] = list(cfg.role_whitelist)
    for role in list(profile.target_roles) + [profile.current_title]:
        if role and role.strip():
            whitelist_terms.extend(_expand_role_pattern(role))
    # keep meaningful terms only
    whitelist_terms = [t for t in dict.fromkeys(whitelist_terms) if len(t.strip()) >= 3]

    blacklist_terms: List[str] = [t for t in cfg.role_blacklist if t.strip()]
    for role in profile.excluded_roles:
        if role and role.strip():
            blacklist_terms.extend(_expand_role_pattern(role))
    blacklist_terms = list(dict.fromkeys(blacklist_terms))

    return RoleRules(
        whitelist_patterns=[re.compile(re.escape(t.lower())) for t in whitelist_terms],
        blacklist_patterns=[re.compile(re.escape(t.lower())) for t in blacklist_terms],
    )


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------
@dataclass
class RuleOutcome:
    passed: bool
    requirements: JobRequirements
    experience: ExperienceEvaluation
    rejections: List[Rejection] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def primary_reason(self) -> str:
        return self.rejections[0].reason.value if self.rejections else "PASSED"


class RuleEngine:
    """
    Stateless-per-run hard filter.  Construct once per pipeline with the
    candidate profile + config, then call :meth:`apply` per job.
    """

    def __init__(self, profile: CandidateProfile, cfg: AppConfig) -> None:
        self.profile = profile
        self.cfg = cfg
        self.policy = cfg.experience
        self.role_rules = build_role_rules(profile, cfg)
        self.candidate_seniority_ = candidate_seniority(profile)
        self._candidate_skills = list(
            dict.fromkeys(list(profile.core_skills) + list(profile.skills) + list(profile.secondary_skills))
        )
        self._reference_date = datetime.now(timezone.utc)

    # -- requirement extraction (deterministic only) ------------------------
    def extract_requirements(self, job: JobRecord) -> JobRequirements:
        """Deterministic requirements used for pre-embedding filtering."""
        base = extract_experience_requirement(job.description, job.title)
        seniority = infer_seniority(job.title)
        if job.job_level_raw and seniority == Seniority.MID:
            seniority = _map_raw_level(job.job_level_raw)
        employment = detect_employment_type(job.employment_type, job.title, job.description[:400])
        work_mode = detect_work_mode(job.work_mode, job.title, job.location, job.description[:400])
        scanned = skill_taxonomy.extract_skills_from_text(job.description, max_skills=25)
        return JobRequirements(
            min_experience_years=base.min_experience_years,
            max_experience_years=base.max_experience_years,
            experience_text=base.experience_text,
            experience_domain=base.experience_domain,
            experience_confidence=base.experience_confidence,
            required_skills=scanned[:15],
            preferred_skills=scanned[15:25],
            seniority=seniority,
            employment_type=employment.value,
            work_mode=work_mode.value,
            responsibilities=[],
            extraction_method="regex" if base.extraction_method == "regex" else "none",
        )

    # -- the filters ---------------------------------------------------------
    def apply(self, job: JobRecord, requirements: Optional[JobRequirements] = None) -> RuleOutcome:
        req = requirements or self.extract_requirements(job)
        rejections: List[Rejection] = []
        notes: List[str] = []

        # 1. explicit role exclusions / blacklist
        black = self.role_rules.matches_blacklist(job.title)
        if black:
            rejections.append(
                Rejection(RejectionReason.IRRELEVANT_ROLE,
                          f"job title matches excluded role pattern '{black}' (career target is "
                          f"{', '.join(self.profile.target_roles[:3]) or 'AI/ML'})")
            )

        # 2. location
        allowed = [l for l in self.cfg.locations]
        ok, detail = location_matches(job.location, allowed, job.work_mode)
        if not ok:
            rejections.append(Rejection(RejectionReason.LOCATION_MISMATCH, detail))
        else:
            notes.append(detail)

        # 3. work mode
        if self.cfg.work_modes:
            allowed_modes = {m.lower() for m in self.cfg.work_modes}
            job_mode = (job.work_mode or req.work_mode or "unknown").lower()
            if job_mode not in allowed_modes and job_mode != "unknown":
                rejections.append(
                    Rejection(RejectionReason.WORK_MODE_MISMATCH,
                              f"job work mode '{job_mode}' not in {sorted(allowed_modes)}")
                )

        # 4. employment type
        if self.cfg.employment_types:
            allowed_types = {t.lower() for t in self.cfg.employment_types}
            job_type = (req.employment_type or "unknown").lower()
            if job_type not in allowed_types and job_type != "unknown":
                rejections.append(
                    Rejection(RejectionReason.EMPLOYMENT_TYPE_MISMATCH,
                              f"employment type '{job_type}' not in {sorted(allowed_types)}")
                )

        # 5. freshness
        if job.posted_date is not None:
            if job.posted_date.tzinfo is None:
                posted = job.posted_date.replace(tzinfo=timezone.utc)
            else:
                posted = job.posted_date
            age_days = (self._reference_date - posted).total_seconds() / 86400.0
            if age_days > self.cfg.max_job_age_days:
                rejections.append(
                    Rejection(RejectionReason.STALE_JOB,
                              f"posted {age_days:.0f} days ago (max {self.cfg.max_job_age_days})")
                )
        else:
            notes.append("freshness unknown; not rejected")

        # 6. seniority
        gap = seniority_gap(self.candidate_seniority_, req.seniority)
        if abs(gap) > self.cfg.matching.seniority_gap_tolerance:
            detail = (
                f"job seniority '{req.seniority.value}' vs candidate "
                f"'{self.candidate_seniority_.value}' (gap {gap:+d})"
            )
            if gap > 0 or self.cfg.matching.reject_seniority_mismatch:
                rejections.append(Rejection(RejectionReason.SENIORITY_MISMATCH, detail))
            else:
                notes.append("seniority below candidate level (flagged, not rejected): " + detail)

        # 7. must-have skills (job-involvement semantics: every recommended
        #    job must actually involve these skills, e.g. "only deployment work")
        must_haves = list(self.cfg.must_have_skills)
        if must_haves:
            job_skills = {
                skill_taxonomy.resolve_skill(s)[0].lower()
                for s in list(req.required_skills) + list(req.preferred_skills)
            }
            job_skills.update(
                s.lower() for s in skill_taxonomy.extract_skills_from_text(job.description, max_skills=60)
            )
            missing = []
            for must_have in must_haves:
                canonical, category = skill_taxonomy.resolve_skill(must_have)
                key = canonical.lower()
                if key in job_skills:
                    continue
                if self.cfg.matching.must_have_match_level == "related" and category:
                    if any(
                        job_skill for job_skill in job_skills
                        if skill_taxonomy.category_of(job_skill) == category
                    ):
                        continue
                missing.append(canonical)
            if missing:
                rejections.append(
                    Rejection(RejectionReason.MISSING_MUST_HAVE_SKILL,
                              f"job does not involve configured must-have skill(s): "
                              f"{', '.join(missing)}")
                )

        # 8. experience (the deterministic ladder from section 6)
        evaluation = evaluate_experience(self.profile, req, self.policy)
        if evaluation.fit == ExperienceFit.UNDERQUALIFIED and self.policy.reject_underqualified:
            rejections.append(
                Rejection(RejectionReason.UNDERQUALIFIED, "; ".join(evaluation.reasons))
            )
        elif evaluation.fit == ExperienceFit.UNDERQUALIFIED:
            notes.append("underqualified but soft experience mode is enabled: "
                         + "; ".join(evaluation.reasons))
        if evaluation.fit == ExperienceFit.OVERQUALIFIED and self.policy.reject_severe_overqualification:
            rejections.append(
                Rejection(RejectionReason.OVERQUALIFIED, "; ".join(evaluation.reasons))
            )

        # 9. optional: reject when job-required skills are simply missing
        if self.cfg.matching.reject_on_missing_required_skills and req.required_skills:
            match = skill_taxonomy.match_skills(
                req.required_skills, self._candidate_skills,
                required=req.required_skills,
            )
            missing_required = [m.job_skill for m in match.missing]
            if missing_required:
                rejections.append(
                    Rejection(RejectionReason.MISSING_MUST_HAVE_SKILL,
                              f"missing job-required skills: {', '.join(missing_required)}")
                )

        return RuleOutcome(
            passed=not rejections,
            requirements=req,
            experience=evaluation,
            rejections=rejections,
            notes=notes,
        )


def _map_raw_level(raw: str) -> Seniority:
    raw = (raw or "").strip().lower()
    mapping = {
        "internship": Seniority.INTERN,
        "entry": Seniority.ENTRY,
        "associate": Seniority.JUNIOR,
        "mid-senior level": Seniority.SENIOR,
        "mid senior level": Seniority.SENIOR,
        "director": Seniority.MANAGER,
        "executive": Seniority.MANAGER,
    }
    return mapping.get(raw, Seniority.MID)
