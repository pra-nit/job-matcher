"""Qwen JSON reliability tests (section 29)."""
from __future__ import annotations

import json

from llm.ollama_client import _fix_common_issues, extract_json_block, parse_llm_json
from llm.schemas import CandidateProfile, JobAnalysis, JobRequirements


class TestExtractJsonBlock:
    def test_plain_json(self):
        assert extract_json_block('{"a": 1}') == '{"a": 1}'

    def test_markdown_fences(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
        assert extract_json_block(text) == '{"a": 1}'

    def test_bare_fences(self):
        assert extract_json_block('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_think_tags_removed(self):
        text = '<think>let me reason about this...</think>{"a": 1}'
        assert extract_json_block(text) == '{"a": 1}'

    def test_prose_around_json(self):
        text = 'Sure! The result is: {"a": 1, "b": [2]} as requested.'
        assert extract_json_block(text) == '{"a": 1, "b": [2]}'

    def test_nested_braces(self):
        text = '{"outer": {"inner": "has } brace"}, "n": 2}'
        assert extract_json_block(text) == text

    def test_unbalanced_returns_none(self):
        assert extract_json_block("no json here at all") is None
        assert extract_json_block('{"broken": ') is None

    def test_first_object_wins(self):
        text = 'junk {"a": 1} more {"b": 2}'
        assert extract_json_block(text) == '{"a": 1}'


class TestCommonFixes:
    def test_trailing_comma_fixed(self):
        assert json.loads(_fix_common_issues('{"a": [1,2,],}')) == {"a": [1, 2]}

    def test_smart_quotes_fixed(self):
        fixed = _fix_common_issues('{"a": “hello”}')
        assert json.loads(fixed) == {"a": "hello"}


class TestParseLlmJson:
    def test_valid(self):
        assert parse_llm_json('```json\n{"x": 5}\n```') == {"x": 5}

    def test_invalid_returns_none(self):
        assert parse_llm_json("COMPLETELY NOT JSON") is None
        assert parse_llm_json("") is None


class TestSchemaValidation:
    def test_profile_with_string_numbers(self):
        profile = CandidateProfile.model_validate({
            "name": "Test", "total_experience_years": "10",
            "relevant_experience_years": "4+",
            "skills": "Python, C++, PyTorch",
            "experience_by_domain": {"machine_learning": "4"},
        })
        assert profile.total_experience_years == 10.0
        assert profile.relevant_experience_years == 4.0
        assert profile.skills == ["Python", "C++", "PyTorch"]
        assert profile.experience_by_domain["machine_learning"] == 4.0

    def test_analysis_scores_clamped(self):
        analysis = JobAnalysis.model_validate({
            "match_score": "150", "skill_match_score": -5,
            "strengths": None, "concerns": "single string",
        })
        assert analysis.match_score == 100
        assert analysis.skill_match_score == 0
        assert analysis.strengths == []
        assert analysis.concerns == ["single string"]

    def test_requirements_lenient(self):
        req = JobRequirements.model_validate({
            "min_experience_years": "3", "max_experience_years": None,
            "experience_domain": "Machine Learning",  # non-enum value
            "required_skills": "Python; C++; TensorRT",
            "seniority": "Senior",
            "experience_confidence": 1.7,  # out of range -> clamped
        })
        assert req.min_experience_years == 3.0
        assert req.experience_domain.value == "machine_learning"
        assert req.required_skills == ["Python", "C++", "TensorRT"]
        assert req.seniority.value == "senior"
        assert req.experience_confidence == 1.0

    def test_absurd_years_rejected_to_none(self):
        req = JobRequirements.model_validate({"min_experience_years": 500})
        assert req.min_experience_years is None


class TestFallbackBehaviour:
    def test_none_analysis_defaults_are_safe(self):
        """A failed analysis must still produce a usable (zeroed) object."""
        analysis = JobAnalysis.model_validate({})
        assert analysis.match_score == 0
        assert analysis.strengths == []
        assert analysis.reasoning == ""
