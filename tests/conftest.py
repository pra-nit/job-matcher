"""Pytest configuration: project root on sys.path + shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, ExperiencePolicy  # noqa: E402
from llm.schemas import CandidateProfile  # noqa: E402


@pytest.fixture
def policy() -> ExperiencePolicy:
    """The spec's default over/under-qualification policy."""
    return ExperiencePolicy(
        overqualification_absolute_gap=2.0,
        overqualification_ratio=1.75,
        soft_experience_mode=False,
        soft_experience_gap_tolerance=1.0,
    )


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture
def profile_10y() -> CandidateProfile:
    """The section-31 example candidate."""
    return CandidateProfile(
        name="Rahul Sharma",
        current_title="Senior Machine Learning Engineer",
        total_experience_years=10,
        relevant_experience_years=4,
        experience_by_domain={
            "machine_learning": 4,
            "computer_vision": 4,
            "model_deployment": 3,
            "quantization": 2,
            "software_engineering": 0,
            "networking": 0,
        },
        skills=[
            "Python", "C++", "PyTorch", "ONNX", "ONNX Runtime", "TIDL", "TI Jacinto",
            "QNN", "SNPE", "INT8", "INT16", "PTQ", "QAT", "Knowledge Distillation",
            "Docker", "Jenkins", "BEV", "LSS", "Object Detection",
        ],
        core_skills=[
            "Python", "C++", "PyTorch", "ONNX", "ONNX Runtime", "TIDL", "QNN",
            "SNPE", "INT8", "PTQ", "QAT",
        ],
        domains=["Edge AI", "Computer Vision"],
        target_roles=[
            "AI Engineer", "ML Engineer", "Deep Learning Engineer",
            "Computer Vision Engineer", "Edge AI Engineer", "ML Deployment Engineer",
        ],
        excluded_roles=["Telecom Engineer", "Network Engineer"],
        summary="ML engineer focused on edge deployment and quantization.",
    )
