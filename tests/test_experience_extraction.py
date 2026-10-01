"""
Experience EXTRACTION tests (spec section 35).

Covers: "3+ years", "3-5 years", "5 years of experience", "8+ years",
"2 to 4 years", "5 years preferred", "minimum 4 years", en-dashes,
seniority inference fallback, and the no-requirement case.
"""
from __future__ import annotations

from matching.experience import (
    classify_experience_domain,
    extract_experience_regex,
    extract_experience_requirement,
)
from models import ExperienceDomain


def _min_max(text: str):
    match = extract_experience_regex(text)
    if match is None:
        return None
    return match.min_years, match.max_years


class TestExperienceExtraction:
    def test_plus_years(self):
        assert _min_max("We need 3+ years experience in ML.") == (3.0, None)

    def test_plus_years_bare(self):
        assert _min_max("Candidates should have 8+ years.") == (8.0, None)

    def test_range_years(self):
        assert _min_max("3-5 years of relevant experience required.") == (3.0, 5.0)

    def test_range_years_to(self):
        assert _min_max("2 to 4 years of professional experience.") == (2.0, 4.0)

    def test_range_years_en_dash(self):
        assert _min_max("Requires 3–5 years experience building models.") == (3.0, 5.0)

    def test_bare_years_of_experience(self):
        assert _min_max("Bachelor's degree and 5 years of experience.") == (5.0, None)

    def test_years_preferred(self):
        assert _min_max("5 years preferred in a production environment.") == (5.0, None)

    def test_minimum_word(self):
        assert _min_max("Minimum 4 years of software development experience.") == (4.0, None)

    def test_at_least(self):
        assert _min_max("At least 3 years experience required.") == (3.0, None)

    def test_yrs_abbreviation(self):
        assert _min_max("6+ yrs building deep learning models.") == (6.0, None)

    def test_no_requirement(self):
        assert _min_max("Great team, latest tech stack, amazing office.") is None

    def test_salary_not_mistaken_for_experience(self):
        assert _min_max("Salary ranges from 30-50 LPA per year.") is None

    def test_prefers_experienced_sentence_over_bare_number(self):
        # "3 years" near 'experience' must beat a stray "10 years" elsewhere
        match = extract_experience_regex(
            "The project took 10 years. Requirements: 3 years of experience with Python."
        )
        assert match is not None
        assert match.min_years == 3.0

    def test_zero_to_two_years(self):
        assert _min_max("0-2 years of experience, entry level role.") == (0.0, 2.0)


class TestDomainClassification:
    def test_ml_domain(self):
        assert classify_experience_domain("5 years of machine learning experience") == \
            ExperienceDomain.MACHINE_LEARNING

    def test_cv_domain(self):
        assert classify_experience_domain("4 years in computer vision") == \
            ExperienceDomain.COMPUTER_VISION

    def test_networking_domain(self):
        assert classify_experience_domain("5 years of network engineering") == \
            ExperienceDomain.NETWORKING

    def test_swe_domain(self):
        assert classify_experience_domain("5 years of software development") == \
            ExperienceDomain.SOFTWARE_ENGINEERING

    def test_title_context_used(self):
        # bare "5 years" + ML Engineer title -> machine learning domain
        assert classify_experience_domain("5 years", "Machine Learning Engineer") == \
            ExperienceDomain.MACHINE_LEARNING

    def test_general_domain(self):
        assert classify_experience_domain("5 years", "Program Manager") == \
            ExperienceDomain.GENERAL


class TestRequirementObject:
    def test_full_extraction_object(self):
        req = extract_experience_requirement(
            "We are hiring. Requirements: 3-5 years experience in deep learning. "
            "Nice to have Kubernetes.",
            title="Deep Learning Engineer",
        )
        assert req.min_experience_years == 3.0
        assert req.max_experience_years == 5.0
        assert req.experience_domain == ExperienceDomain.DEEP_LEARNING
        assert req.extraction_method == "regex"
        assert req.experience_confidence > 0

    def test_no_requirement_object(self):
        req = extract_experience_requirement("No numbers here at all.", "Engineer")
        assert req.min_experience_years is None
        assert req.max_experience_years is None
        assert req.extraction_method == "none"
