"""
Prompt templates for the three Qwen stages.

Design notes:
    * Every prompt demands STRICT JSON with an explicit field contract.
    * Prompts are version-tagged; the version is part of the cache key so a
      prompt change invalidates cached extractions automatically.
    * The job-analysis prompt receives the deterministic verdicts (experience
      fit, matched/missing skills) and explicitly tells Qwen it cannot
      override them (RULE 3) — Qwen only adds nuanced scoring and reasoning.
"""
from __future__ import annotations

from llm.schemas import CandidateProfile, JobRequirements
from models import ExperienceEvaluation, JobRecord, SkillMatchResult

PROMPT_VERSION = "v3"

_STRICT_JSON_RULES = """
Respond with a SINGLE valid JSON object and nothing else.
- No markdown, no code fences, no explanations before or after the JSON.
- Use double quotes for all strings.
- Use null for unknown values, [] for unknown lists, 0 for unknown numbers.
- Never invent facts that are not in the provided text.
"""

# ---------------------------------------------------------------------------
# 1. Resume -> candidate profile
# ---------------------------------------------------------------------------
RESUME_PROFILE_SYSTEM = f"""You are a precise resume parsing engine for a job-matching system.
{ _STRICT_JSON_RULES }
Extract a structured candidate profile from the resume text.

Critical rules:
- total_experience_years = ALL professional experience since the first job.
- relevant_experience_years = experience relevant to the candidate's CURRENT
  career direction (recent ML/AI roles), NOT the total. If someone has 10
  years total but only 4 years in machine learning, set
  total_experience_years=10 and relevant_experience_years=4.
- experience_by_domain: split experience into these exact keys (years, use 0
  when not applicable): "machine_learning", "deep_learning",
  "computer_vision", "nlp", "genai", "model_deployment", "quantization",
  "mlops", "software_engineering", "networking".
  Domain years may overlap each other but must never exceed
  total_experience_years.
- skills: every technical skill mentioned. core_skills: the 5-12 skills the
  candidate is strongest in. secondary_skills: supporting skills.
- target_roles: job titles the candidate is actively aiming for, based on
  their most recent roles and stated objectives (max 6).
- excluded_roles: role families clearly behind the candidate's career
  (e.g. old telecom/networking jobs when they have moved to AI/ML).
- preferred_work_modes: values from "remote", "hybrid", "onsite".
- Do NOT include the candidate's phone/email/address in any field.

Return exactly this JSON shape:
{{
  "name": "",
  "current_title": "",
  "total_experience_years": 0,
  "relevant_experience_years": 0,
  "experience_by_domain": {{"machine_learning": 0, "deep_learning": 0, "computer_vision": 0, "nlp": 0, "genai": 0, "model_deployment": 0, "quantization": 0, "mlops": 0, "software_engineering": 0, "networking": 0}},
  "skills": [],
  "core_skills": [],
  "secondary_skills": [],
  "domains": [],
  "target_roles": [],
  "education": [],
  "certifications": [],
  "locations": [],
  "preferred_work_modes": [],
  "excluded_roles": [],
  "summary": "2-3 sentences describing the candidate's professional identity, strengths and career direction"
}}"""


def resume_profile_user(resume_text: str) -> str:
    return f"RESUME TEXT:\n\"\"\"\n{resume_text[:12000]}\n\"\"\"\n\nExtract the candidate profile JSON now."


# ---------------------------------------------------------------------------
# 2. Job -> requirements
# ---------------------------------------------------------------------------
JOB_REQUIREMENTS_SYSTEM = f"""You are a precise job-requirement extraction engine.
{ _STRICT_JSON_RULES }
Extract the structured requirements of a job posting.

Critical rules for experience:
- min_experience_years / max_experience_years: numbers only.
  "3+ years" -> min 3, max null. "3-5 years" -> min 3, max 5.
  "5 years of experience" -> min 5, max null. "2 to 4 years" -> min 2, max 4.
  If no experience requirement exists, use null for both.
- experience_domain: what the experience must be IN. One of:
  "general", "machine_learning", "deep_learning", "computer_vision", "nlp",
  "genai", "model_deployment", "mlops", "quantization",
  "software_engineering", "networking", "other".
  "5 years of ML experience" -> machine_learning. "5 years software
  development" -> software_engineering. A bare "5 years" next to an ML
  Engineer title -> machine_learning.
- experience_confidence: 0.0-1.0 (how sure you are about the numbers).
- required_skills: hard requirements only (max 15). preferred_skills: nice-to-have.
- seniority: one of "intern","entry","junior","mid","senior","lead","staff",
  "principal","manager","unknown" — judge from the title AND the experience
  requirement.
- responsibilities: up to 6 short imperative phrases from the posting.
- work_mode: "remote", "hybrid", "onsite" or "unknown".
- employment_type: "full_time","part_time","contract","temporary",
  "internship","other" or "unknown".

Return exactly this JSON shape:
{{
  "min_experience_years": null,
  "max_experience_years": null,
  "experience_text": "the exact sentence the numbers came from",
  "experience_domain": "general",
  "experience_confidence": 0.0,
  "required_skills": [],
  "preferred_skills": [],
  "seniority": "unknown",
  "employment_type": "unknown",
  "work_mode": "unknown",
  "responsibilities": []
}}"""


def job_requirements_user(job: JobRecord) -> str:
    description = job.description[:6000]
    return (
        f"JOB TITLE: {job.title}\n"
        f"COMPANY: {job.company}\n"
        f"LOCATION: {job.location}\n\n"
        f"JOB DESCRIPTION:\n\"\"\"\n{description}\n\"\"\"\n\n"
        "Extract the job requirements JSON now."
    )


# ---------------------------------------------------------------------------
# 3. Job -> deep analysis (top-K only)
# ---------------------------------------------------------------------------
JOB_ANALYSIS_SYSTEM = f"""You are an expert technical recruiter evaluating how well ONE candidate
fits ONE job. You will receive the candidate profile, the job posting and the
output of a deterministic rule engine.

{ _STRICT_JSON_RULES }

Scoring rules:
- All scores are integers 0-100.
- The rule engine's experience verdict is FINAL. You cannot override it: if
  the candidate is underqualified or overqualified on experience, reflect
  that in match_score (low) even if the skills look great, and never claim
  the experience fits when it does not.
- skill_match_score: overlap of required/preferred skills with candidate
  skills, counting related/transferable skills as partial matches.
- responsibility_match_score: how well the candidate's history matches the
  day-to-day responsibilities.
- domain_match_score: overlap of industry/domain (e.g. edge AI, computer
  vision, autonomous driving, NLP).
- career_alignment_score: does this job move the candidate toward their
  TARGET roles? A candidate CAN do a job that is a poor career move — score
  that low.
- match_score: your overall judgment of fit.
- matched_skills: job skills the candidate clearly has.
- related_skills: job skills the candidate covers with related experience
  (e.g. TensorRT covered by ONNX Runtime/TIDL/QNN/SNPE experience).
- strengths / concerns: 2-5 short concrete bullets each.
- reasoning: 2-4 sentences, factual, referencing actual skills and years.
- semantic_verdict: one of "strong_match", "good_match", "borderline",
  "weak_match", "not_a_match".

Return exactly this JSON shape:
{{
  "match_score": 0,
  "skill_match_score": 0,
  "responsibility_match_score": 0,
  "domain_match_score": 0,
  "career_alignment_score": 0,
  "matched_skills": [],
  "related_skills": [],
  "missing_required_skills": [],
  "missing_preferred_skills": [],
  "strengths": [],
  "concerns": [],
  "reasoning": "",
  "semantic_verdict": ""
}}"""


def job_analysis_user(
    profile: CandidateProfile,
    job: JobRecord,
    requirements: JobRequirements,
    experience_eval: ExperienceEvaluation,
    skills: SkillMatchResult,
) -> str:
    matched = ", ".join(s.job_skill for s in skills.exact + skills.related) or "none detected"
    missing_required = ", ".join(s.job_skill for s in skills.missing) or "none"
    transferable = ", ".join(f"{s.job_skill} (via {s.candidate_skill})" for s in skills.transferable) or "none"
    domain_years = ", ".join(
        f"{k}: {v:g}y" for k, v in profile.experience_by_domain.items() if (v or 0) > 0
    ) or "not specified"

    return f"""CANDIDATE PROFILE
- Current title: {profile.current_title}
- Total experience: {profile.total_experience_years:g} years
- Relevant experience: {profile.relevant_experience_years:g} years
- Experience by domain: {domain_years}
- Core skills: {', '.join(profile.core_skills) or ', '.join(profile.skills[:12]) or 'n/a'}
- All skills: {', '.join(profile.skills[:30])}
- Target roles: {', '.join(profile.target_roles) or 'n/a'}
- Summary: {profile.summary}

RULE ENGINE RESULTS (deterministic, cannot be overridden)
- Experience fit: {experience_eval.fit.value}
- Candidate effective relevant years for this job: {experience_eval.effective_candidate_years:g} ({experience_eval.basis})
- Job requires: {requirements.min_experience_years or 'n/a'}-{requirements.max_experience_years or 'n/a'} years in domain '{requirements.experience_domain.value}'
- Deterministic skill match: {skills.score:g}/100
- Exact/related matches: {matched}
- Transferable: {transferable}
- Missing: {missing_required}

JOB POSTING
- Title: {job.title}
- Company: {job.company}
- Location: {job.location} (work mode: {job.work_mode})
- Required skills: {', '.join(requirements.required_skills) or 'n/a'}
- Preferred skills: {', '.join(requirements.preferred_skills) or 'n/a'}
- Responsibilities: {'; '.join(requirements.responsibilities[:6]) or 'n/a'}

JOB DESCRIPTION:
\"\"\"
{job.description[:4000]}
\"\"\"

Evaluate the fit and return the analysis JSON now."""
