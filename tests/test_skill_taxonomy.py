"""Skill taxonomy + skill matching tests (spec sections 11-12)."""
from __future__ import annotations

from matching import skill_taxonomy
from models import SkillMatchType


class TestSkillResolution:
    def test_alias_pytorch(self):
        canonical, category = skill_taxonomy.resolve_skill("pytorch")
        assert canonical == "PyTorch"
        assert category == "DEEP_LEARNING"

    def test_alias_k8s(self):
        canonical, category = skill_taxonomy.resolve_skill("k8s")
        assert canonical == "Kubernetes"
        assert category == "MLOPS"

    def test_alias_bev(self):
        canonical, category = skill_taxonomy.resolve_skill("BEV")
        assert canonical == "BEV"
        assert category == "COMPUTER_VISION"

    def test_alias_lss(self):
        canonical, category = skill_taxonomy.resolve_skill("lift-splat-shoot")
        assert canonical == "LSS"
        assert category == "COMPUTER_VISION"

    def test_qnn_maps_to_model_deployment(self):
        _, category = skill_taxonomy.resolve_skill("QNN")
        assert category == "MODEL_DEPLOYMENT"

    def test_snpe_maps_to_model_deployment(self):
        _, category = skill_taxonomy.resolve_skill("SNPE")
        assert category == "MODEL_DEPLOYMENT"

    def test_int8_maps_to_quantization(self):
        _, category = skill_taxonomy.resolve_skill("INT8")
        assert category == "QUANTIZATION"

    def test_unknown_skill_kept_without_category(self):
        canonical, category = skill_taxonomy.resolve_skill("Some Rare Framework")
        assert canonical == "Some Rare Framework"
        assert category is None

    def test_fuzzy_substring_resolution(self):
        # "CUDA kernel programming" should resolve to CUDA
        canonical, category = skill_taxonomy.resolve_skill("CUDA kernel programming")
        assert canonical == "CUDA"
        assert category == "MODEL_DEPLOYMENT"

    def test_extraction_from_text(self):
        text = ("Optimize models with ONNX Runtime, TensorRT and INT8 quantization; "
                "deploy on QNN and SNPE; Python and C++ required.")
        skills = skill_taxonomy.extract_skills_from_text(text)
        skill_names = {s.lower() for s in skills}
        for expected in ["onnx runtime", "tensorrt", "int8", "qnn", "snpe", "python", "c++"]:
            assert expected in skill_names, f"missing {expected} in {skills}"


CANDIDATE_SKILLS = [
    "ONNX Runtime", "TIDL", "QNN", "SNPE", "Python", "C++",
    "Docker", "Jenkins",
]


class TestSkillMatching:

    def test_exact_python(self):
        result = skill_taxonomy.match_skills(["Python"], CANDIDATE_SKILLS)
        assert result.matches[0].match_type == SkillMatchType.EXACT

    def test_tensorrt_is_related_via_deployment_family(self):
        # spec: job wants TensorRT, candidate has ONNX Runtime/TIDL/QNN/SNPE
        result = skill_taxonomy.match_skills(["TensorRT"], CANDIDATE_SKILLS)
        match = result.matches[0]
        assert match.match_type == SkillMatchType.RELATED
        assert match.candidate_skill in {"ONNX Runtime", "TIDL", "QNN", "SNPE"}

    def test_kubeflow_is_related_via_mlops_family(self):
        # spec: job wants Kubeflow, candidate has Docker + Jenkins
        result = skill_taxonomy.match_skills(["Kubeflow"], CANDIDATE_SKILLS)
        assert result.matches[0].match_type == SkillMatchType.RELATED

    def test_cuda_missing_when_no_deployment_experience(self):
        result = skill_taxonomy.match_skills(["CUDA"], ["Pandas", "NumPy", "Scikit-learn"])
        assert result.matches[0].match_type == SkillMatchType.MISSING

    def test_transferable_across_related_categories(self):
        # job: PTQ (QUANTIZATION); candidate: TensorRT (MODEL_DEPLOYMENT)
        result = skill_taxonomy.match_skills(["PTQ"], ["TensorRT"])
        assert result.matches[0].match_type == SkillMatchType.TRANSFERABLE

    def test_score_reflects_match_quality(self):
        all_exact = skill_taxonomy.match_skills(["Python", "C++"], CANDIDATE_SKILLS)
        all_missing = skill_taxonomy.match_skills(["Spark", "Hadoop"], CANDIDATE_SKILLS)
        assert all_exact.score == 100.0
        assert all_missing.score == 0.0

    def test_required_weighs_more_than_preferred(self):
        # missing a required skill hurts more than missing a preferred one
        missing_required = skill_taxonomy.match_skills(
            ["Spark"], CANDIDATE_SKILLS, required=["Spark"], preferred=["Python"]
        )
        missing_preferred = skill_taxonomy.match_skills(
            ["Python"], CANDIDATE_SKILLS, required=["Spark"], preferred=["Python"]
        )
        assert missing_preferred.score > missing_required.score

    def test_neutral_score_without_job_skills(self):
        result = skill_taxonomy.match_skills([], CANDIDATE_SKILLS)
        assert result.score == 50.0


class TestMustHaves:
    def test_related_satisfies_must_have_by_default(self):
        missing = skill_taxonomy.missing_must_haves(["TensorRT"], CANDIDATE_SKILLS)
        assert missing == []  # ONNX family counts as related

    def test_exact_mode_requires_exact_skill(self):
        missing = skill_taxonomy.missing_must_haves(
            ["TensorRT"], CANDIDATE_SKILLS, match_level="exact"
        )
        assert missing == ["TensorRT"]

    def test_missing_must_have_detected(self):
        missing = skill_taxonomy.missing_must_haves(["Kubernetes"], ["Pandas"])
        assert missing == ["Kubernetes"]
