"""
Configuration for the job-matching pipeline.

Nothing in the matching policy is hard-coded at point of use: every threshold
lives here as a default, can be overridden by ``config.yaml`` and again by
CLI flags (dotted-key overrides), e.g.::

    load_config(
        Path("config.yaml"),
        overrides={"matching.top_k_embedding": 50, "qwen.model": "qwen3:4b"},
    )

The config objects are frozen dataclasses (no global mutable state): one
instance is created at start-up and passed explicitly to every component.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

try:  # PyYAML is optional at runtime; defaults are used when missing
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
# Policy sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperiencePolicy:
    """All knobs of the experience / (over|under)qualification engine."""

    candidate_experience_policy: str = "relevant"  # relevant | total
    soft_experience_mode: bool = False
    soft_experience_gap_tolerance: float = 1.0
    overqualification_absolute_gap: float = 2.0
    overqualification_ratio: float = 1.75
    underqualification_reject_gap: float = 1.0  # > this many years short => hard reject
    reject_underqualified: bool = True
    reject_severe_overqualification: bool = True
    infer_experience_from_seniority: bool = True
    adjacent_domain_credit: float = 0.5
    software_engineering_uses_total: bool = True


@dataclass(frozen=True)
class MatchingPolicy:
    match_threshold: float = 75.0
    borderline_threshold: float = 60.0
    embedding_threshold: float = 0.25      # raw cosine similarity floor
    embedding_score_floor: float = 0.35    # cosine mapped to 0
    embedding_score_ceiling: float = 0.85  # cosine mapped to 100
    top_k_embedding: int = 50
    max_qwen_jobs: int = 50
    min_career_alignment: float = 40.0
    seniority_gap_tolerance: int = 1
    reject_seniority_mismatch: bool = True
    # 'exact': a configured must-have skill must appear in the job itself;
    # 'related': a same-category skill in the job also satisfies it
    must_have_match_level: str = "exact"
    reject_on_missing_required_skills: bool = False
    qwen_low_score_cap: float = 20.0       # Qwen may downgrade, never upgrade
    qwen_deterministic_tolerance: float = 40.0
    max_description_chars_embedding: int = 1500
    max_description_chars_qwen: int = 4000
    score_weights: Dict[str, float] = field(default_factory=lambda: {
        "experience": 0.30,
        "skills": 0.25,
        "embedding": 0.20,
        "responsibility": 0.10,
        "domain": 0.10,
        "career_alignment": 0.05,
    })


@dataclass(frozen=True)
class ScrapePolicy:
    sites: Tuple[str, ...] = ("linkedin", "indeed")
    results_wanted: int = 30
    max_search_queries: int = 6
    max_scrape_calls: int = 12
    country_indeed: str = "india"
    linkedin_fetch_description: bool = True
    request_delay_seconds: float = 1.0


@dataclass(frozen=True)
class QwenSettings:
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    num_ctx: int = 4096
    max_retries: int = 2
    request_timeout: float = 300.0
    disable_thinking: bool = True  # qwen3: append /no_think and think=False
    num_predict: int = 1024


@dataclass(frozen=True)
class PathsConfig:
    database: Path = Path("data/jobs.db")
    output_dir: Path = Path("data")


@dataclass(frozen=True)
class AppConfig:
    experience: ExperiencePolicy = field(default_factory=ExperiencePolicy)
    matching: MatchingPolicy = field(default_factory=MatchingPolicy)
    scraping: ScrapePolicy = field(default_factory=ScrapePolicy)
    qwen: QwenSettings = field(default_factory=QwenSettings)
    paths: PathsConfig = field(default_factory=PathsConfig)

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    max_job_age_days: int = 14
    locations: Tuple[str, ...] = ()            # empty => no location filter
    work_modes: Tuple[str, ...] = ()           # empty => no work-mode filter
    employment_types: Tuple[str, ...] = ()     # empty => no employment filter
    role_whitelist: Tuple[str, ...] = ()
    role_blacklist: Tuple[str, ...] = (
        "telecom engineer",
        "telecom",
        "network engineer",
        "network operations",
        "sales engineer",
        "civil engineer",
        "electrical maintenance",
        "electrical engineer",
        "site engineer",
    )
    must_have_skills: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Loading / merging
# ---------------------------------------------------------------------------
def default_config() -> AppConfig:
    return AppConfig()


def _deep_merge(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _construct(cls_type: type, data: Mapping[str, Any]) -> Any:
    """Recursively build (frozen) dataclasses from nested dicts, ignoring unknown keys."""
    if not is_dataclass(cls_type):
        return data
    registry = {
        "ExperiencePolicy": ExperiencePolicy,
        "MatchingPolicy": MatchingPolicy,
        "ScrapePolicy": ScrapePolicy,
        "QwenSettings": QwenSettings,
        "PathsConfig": PathsConfig,
        "AppConfig": AppConfig,
    }
    known = {f.name for f in fields(cls_type)}
    kwargs: Dict[str, Any] = {}
    for f in fields(cls_type):
        if f.name not in data:
            continue
        value = data[f.name]
        # f.type is a string because of `from __future__ import annotations`
        nested_cls = registry.get(f.type if isinstance(f.type, str) else getattr(f.type, "__name__", ""))
        if nested_cls is not None and isinstance(value, Mapping):
            value = _construct(nested_cls, value)
        kwargs[f.name] = value
    unknown = set(data) - known
    if unknown:
        logger.warning("Ignoring unknown config keys: %s", sorted(unknown))
    return cls_type(**kwargs)


def _apply_dotted(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Turn {"matching.top_k_embedding": 50} into nested dicts."""
    nested: Dict[str, Any] = {}
    for dotted, value in overrides.items():
        node = nested
        parts = str(dotted).split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def load_config(
    yaml_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> AppConfig:
    """default config <- config.yaml <- dotted CLI overrides."""
    data: Dict[str, Any] = dataclasses.asdict(default_config())
    if yaml_path is not None and Path(yaml_path).exists():
        if yaml is None:
            logger.warning("PyYAML not installed; ignoring %s", yaml_path)
        else:
            with open(yaml_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            data = _deep_merge(data, loaded)
            logger.info("Loaded config from %s", yaml_path)
    elif yaml_path is not None:
        logger.warning("Config file %s not found; using defaults", yaml_path)
    if overrides:
        data = _deep_merge(data, _apply_dotted(overrides))
    cfg = _construct(AppConfig, data)
    cfg = _normalise_sequences(cfg)
    _validate(cfg)
    return cfg


def _normalise_sequences(cfg: AppConfig) -> AppConfig:
    """YAML lists for tuple-typed fields become tuples (immutability)."""
    updates: Dict[str, Any] = {}
    for f in fields(AppConfig):
        if f.name in dataclasses.asdict(cfg):
            value = getattr(cfg, f.name)
            if isinstance(value, list) and isinstance(f.type, str) and f.type.startswith("Tuple"):
                updates[f.name] = tuple(value)
    if updates:
        return dataclasses.replace(cfg, **updates)
    return cfg


def _validate(cfg: AppConfig) -> None:
    weights = cfg.matching.score_weights
    total = sum(weights.values())
    if abs(total - 1.0) > 0.001:
        logger.warning(
            "Score weights sum to %.3f (expected 1.0); they will be normalised", total
        )
    if cfg.matching.match_threshold <= cfg.matching.borderline_threshold:
        raise ValueError(
            "match_threshold must be greater than borderline_threshold "
            f"(got {cfg.matching.match_threshold} <= {cfg.matching.borderline_threshold})"
        )
    if cfg.matching.embedding_score_floor >= cfg.matching.embedding_score_ceiling:
        raise ValueError("embedding_score_floor must be < embedding_score_ceiling")
    if cfg.matching.must_have_match_level not in ("exact", "related"):
        raise ValueError("must_have_match_level must be 'exact' or 'related'")


def config_fingerprint(cfg: AppConfig) -> str:
    """
    Stable hash of everything that influences matching results.  Stored with
    each job_matches row so cached results are automatically invalidated when
    the policy changes (section 22: incremental processing).
    """
    payload = {
        "experience": dataclasses.asdict(cfg.experience),
        "matching": dataclasses.asdict(cfg.matching),
        "embedding_model": cfg.embedding_model,
        "qwen_model": cfg.qwen.model,
        "max_job_age_days": cfg.max_job_age_days,
        "locations": list(cfg.locations),
        "work_modes": list(cfg.work_modes),
        "employment_types": list(cfg.employment_types),
        "role_whitelist": list(cfg.role_whitelist),
        "role_blacklist": list(cfg.role_blacklist),
        "must_have_skills": list(cfg.must_have_skills),
        "schema_version": 3,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def normalize_weights(cfg: AppConfig) -> Dict[str, float]:
    weights = dict(cfg.matching.score_weights)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}
