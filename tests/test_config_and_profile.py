"""Configuration loading / fingerprinting + heuristic profile tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    AppConfig,
    config_fingerprint,
    default_config,
    load_config,
)
from models import ExperienceFit


class TestConfigLoading:
    def test_defaults(self):
        cfg = default_config()
        assert cfg.qwen.model == "qwen3:4b"
        assert cfg.embedding_model == "BAAI/bge-small-en-v1.5"
        assert cfg.matching.top_k_embedding == 50
        assert cfg.max_job_age_days == 14
        assert cfg.experience.overqualification_absolute_gap == 2.0
        assert cfg.experience.overqualification_ratio == 1.75
        assert abs(sum(cfg.matching.score_weights.values()) - 1.0) < 0.001

    def test_yaml_override(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "qwen:\n  model: qwen3:8b\nmatching:\n  top_k_embedding: 30\n"
            "max_job_age_days: 7\nlocations: [Bangalore, Remote]\n",
            encoding="utf-8",
        )
        cfg = load_config(yaml_file)
        assert cfg.qwen.model == "qwen3:8b"
        assert cfg.matching.top_k_embedding == 30
        assert cfg.max_job_age_days == 7
        assert cfg.locations == ("Bangalore", "Remote")
        # unspecified values keep defaults
        assert cfg.matching.match_threshold == 75.0

    def test_cli_overrides_beat_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("matching:\n  top_k_embedding: 30\n", encoding="utf-8")
        cfg = load_config(yaml_file, overrides={"matching.top_k_embedding": 12})
        assert cfg.matching.top_k_embedding == 12

    def test_unknown_keys_ignored(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("totally_unknown_section: 5\n", encoding="utf-8")
        cfg = load_config(yaml_file)  # must not raise
        assert isinstance(cfg, AppConfig)

    def test_invalid_thresholds_rejected(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "matching:\n  match_threshold: 50\n  borderline_threshold: 70\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_config(yaml_file)

    def test_fingerprint_changes_with_policy(self):
        base = default_config()
        changed = default_config()
        object.__setattr__(changed.matching, "match_threshold", 80.0)
        assert config_fingerprint(base) != config_fingerprint(changed)

    def test_fingerprint_stable(self):
        assert config_fingerprint(default_config()) == config_fingerprint(default_config())

    def test_missing_yaml_file_uses_defaults(self, tmp_path: Path):
        cfg = load_config(tmp_path / "does_not_exist.yaml")
        assert cfg.qwen.model == "qwen3:4b"


class TestHeuristicProfile:
    """The offline fallback extractor (degraded but must stay sane)."""

    RESUME = """
RAHUL SHARMA
Senior Machine Learning Engineer
Bangalore, India

SUMMARY
Machine learning engineer with 10 years of total professional experience,
including 4 years in machine learning, 4 years in computer vision,
3 years in model deployment and 2 years in model quantization.

SKILLS
Python, C++, PyTorch, ONNX, TIDL, QNN, SNPE, INT8, PTQ, QAT, Docker

TARGET ROLES
AI Engineer, ML Engineer, Edge AI Engineer
"""

    def test_heuristic_profile_fields(self):
        from resume.profile_extractor import heuristic_profile, postprocess

        profile = postprocess(heuristic_profile(self.RESUME))
        assert profile.name == "RAHUL SHARMA"
        assert profile.total_experience_years == 10.0
        assert profile.relevant_experience_years == 4.0
        assert profile.experience_by_domain.get("machine_learning") == 4.0
        assert profile.experience_by_domain.get("model_deployment") == 3.0
        assert profile.experience_by_domain.get("quantization") == 2.0
        assert "Python" in profile.skills
        assert "AI Engineer" in profile.target_roles
        assert profile.inferred_seniority.value == "senior"

    def test_heuristic_years_feed_experience_engine(self):
        from resume.profile_extractor import heuristic_profile, postprocess
        from config import ExperiencePolicy
        from llm.schemas import JobRequirements
        from matching.experience import evaluate_experience

        profile = postprocess(heuristic_profile(self.RESUME))
        # ML job wanting 5 years must reject the 4-year ML candidate
        evaluation = evaluate_experience(
            profile,
            JobRequirements(min_experience_years=5, max_experience_years=8,
                            experience_domain="machine_learning"),
            ExperiencePolicy(),
        )
        assert evaluation.fit == ExperienceFit.UNDERQUALIFIED

    def test_postprocess_normalises_skills(self):
        from resume.profile_extractor import postprocess
        from llm.schemas import CandidateProfile

        raw = CandidateProfile(skills=["pytorch", "K8s", "docker"], core_skills=["PyTorch"])
        profile = postprocess(raw)
        assert "PyTorch" in profile.skills
        assert "Kubernetes" in profile.skills  # alias resolved
        assert "Docker" in profile.skills

    def test_postprocess_defaults_target_roles(self):
        from resume.profile_extractor import postprocess
        from llm.schemas import CandidateProfile

        raw = CandidateProfile(skills=["PyTorch", "ONNX", "TensorRT"])
        profile = postprocess(raw)
        assert profile.target_roles, "target roles must never be empty"
