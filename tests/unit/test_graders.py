"""
Unit tests for all three ProteinEnv graders.

No network access and no model load required — all graders are pure functions.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from graders.disease_grader import grade_disease, jaccard_similarity
from graders.family_grader import grade_family, is_valid_family_choice
from graders.go_grader import grade_go_terms, validate_go_id
from models import Pathogenicity


# ── Family Grader ──────────────────────────────────────────────────────────────


class TestGradeFamily:
    """Tests for the easy-tier family classification grader."""

    def test_exact_match_returns_1(self):
        """Identical strings → full score."""
        assert grade_family("Insulin family", "Insulin family") == 1.0

    def test_case_insensitive_exact_match(self):
        """Match is case-insensitive."""
        assert grade_family("insulin family", "Insulin family") == 1.0
        assert grade_family("INSULIN FAMILY", "Insulin family") == 1.0

    def test_stripped_whitespace_match(self):
        """Leading/trailing whitespace is stripped before comparison."""
        assert grade_family("  Insulin family  ", "Insulin family") == 1.0

    def test_superfamily_predicted_contains_truth(self):
        """Ground truth contained inside predicted → superfamily credit 0.3."""
        assert grade_family("Insulin superfamily group A", "Insulin") == 0.3

    def test_superfamily_truth_contains_predicted(self):
        """Predicted contained inside ground truth → superfamily credit 0.3."""
        assert grade_family("Insulin", "Insulin family") == 0.3

    def test_no_match_returns_0(self):
        """Completely unrelated strings → 0.0."""
        assert grade_family("Globin family", "Insulin family") == 0.0

    def test_empty_predicted_returns_0(self):
        """Empty predicted string cannot match or substring → 0.0."""
        assert grade_family("", "Insulin family") == 0.0

    def test_return_type_is_float(self):
        """Return value must always be a float."""
        result = grade_family("Insulin family", "Insulin family")
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "predicted,truth",
        [
            ("Insulin family", "Insulin family"),
            ("Insulin", "Insulin family"),
            ("Globin family", "Insulin family"),
            ("", "Insulin family"),
            ("Actin", "Globin family"),
        ],
    )
    def test_return_values_only_valid(self, predicted: str, truth: str):
        """Result must always be in {0.0, 0.3, 1.0}."""
        result = grade_family(predicted, truth)
        assert result in {0.0, 0.3, 1.0}


class TestIsValidFamilyChoice:
    """Tests for the hallucination-detection helper."""

    _CHOICES = ["Insulin family", "Globin family", "Actin family"]

    def test_valid_choice_returns_true(self):
        """Exact choice returns True."""
        assert is_valid_family_choice("Insulin family", self._CHOICES) is True

    def test_invalid_choice_returns_false(self):
        """Made-up family name returns False."""
        assert is_valid_family_choice("made up family", self._CHOICES) is False

    def test_case_insensitive_validation(self):
        """Comparison is case-insensitive."""
        assert is_valid_family_choice("insulin FAMILY", self._CHOICES) is True

    def test_empty_predicted_returns_false(self):
        """Empty string is not in any valid choice list."""
        assert is_valid_family_choice("", self._CHOICES) is False

    def test_empty_choices_returns_false(self):
        """Empty choices list means nothing is valid."""
        assert is_valid_family_choice("Insulin family", []) is False


# ── GO Term Grader ─────────────────────────────────────────────────────────────


class TestValidateGoId:
    """Tests for the GO term ID format validator."""

    def test_valid_format(self):
        """Well-formed GO ID passes validation."""
        assert validate_go_id("GO:0003700") is True

    def test_missing_prefix_invalid(self):
        """IDs without 'GO:' prefix are invalid."""
        assert validate_go_id("0003700") is False

    def test_wrong_digit_count_invalid(self):
        """IDs with fewer than 7 digits are invalid."""
        assert validate_go_id("GO:12345") is False

    def test_non_numeric_invalid(self):
        """IDs with non-digit characters after 'GO:' are invalid."""
        assert validate_go_id("GO:ABCDEFG") is False

    def test_empty_string_invalid(self):
        """Empty string is not a valid GO ID."""
        assert validate_go_id("") is False

    def test_eight_digits_invalid(self):
        """GO:00037001 has 8 digits — too many."""
        assert validate_go_id("GO:00037001") is False


class TestGradeGoTerms:
    """Tests for the medium-tier GO term prediction grader."""

    @pytest.fixture
    def ground_truth(self) -> dict:
        """Standard three-namespace ground truth used in most tests."""
        return {
            "molecular_function": ["GO:0003700", "GO:0005515"],
            "biological_process": ["GO:0006915"],
            "cellular_component": ["GO:0005634"],
        }

    def test_perfect_prediction_returns_1(self, ground_truth):
        """Predicting exactly all true IDs → macro F1 == 1.0."""
        score, _ = grade_go_terms(
            ["GO:0003700", "GO:0005515", "GO:0006915", "GO:0005634"],
            ground_truth,
        )
        assert abs(score - 1.0) < 1e-6

    def test_empty_prediction_returns_0(self, ground_truth):
        """Empty predicted list → macro F1 == 0.0."""
        score, _ = grade_go_terms([], ground_truth)
        assert score == 0.0

    def test_partial_prediction_between_0_and_1(self, ground_truth):
        """Predicting a subset of true IDs → score in (0, 1)."""
        score, _ = grade_go_terms(["GO:0003700"], ground_truth)
        assert 0.0 < score < 1.0

    def test_hallucinated_ids_score_0(self, ground_truth):
        """GO IDs not in any truth namespace score 0 and do not raise."""
        score, _ = grade_go_terms(["GO:9999999"], ground_truth)
        assert score == 0.0

    def test_invalid_go_format_ignored(self, ground_truth):
        """IDs failing validate_go_id are silently dropped before scoring."""
        score_valid, _ = grade_go_terms(["GO:0003700"], ground_truth)
        score_with_bad, _ = grade_go_terms(["GO:0003700", "NOT_A_GO_ID"], ground_truth)
        assert abs(score_valid - score_with_bad) < 1e-6

    def test_returns_tuple_of_float_and_dict(self, ground_truth):
        """Return type must be (float, dict)."""
        result = grade_go_terms(["GO:0003700"], ground_truth)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_macro_f1_is_mean_of_namespaces(self, ground_truth):
        """macro_f1 == mean of per-namespace F1 values in the returned dict."""
        predicted = ["GO:0003700", "GO:0006915"]
        macro_f1, ns_scores = grade_go_terms(predicted, ground_truth)
        expected_macro = sum(ns_scores.values()) / len(ns_scores)
        assert abs(macro_f1 - expected_macro) < 1e-9

    def test_score_in_valid_range(self, ground_truth):
        """Score is always in [0.0, 1.0]."""
        score, _ = grade_go_terms(["GO:0003700", "GO:0005515"], ground_truth)
        assert 0.0 <= score <= 1.0

    def test_empty_ground_truth_namespace(self):
        """Namespace with empty truth list must not crash; contributes 0 F1."""
        gt = {
            "molecular_function": [],
            "biological_process": ["GO:0006915"],
            "cellular_component": [],
        }
        score, ns = grade_go_terms(["GO:0006915"], gt)
        assert score > 0
        assert ns["molecular_function"] == 0.0
        assert ns["cellular_component"] == 0.0

    def test_per_namespace_dict_has_all_three_keys(self, ground_truth):
        """The namespace dict must contain all three standard namespace keys."""
        _, ns = grade_go_terms([], ground_truth)
        assert "molecular_function" in ns
        assert "biological_process" in ns
        assert "cellular_component" in ns


# ── Disease Grader ─────────────────────────────────────────────────────────────


class TestJaccardSimilarity:
    """Tests for the Jaccard similarity helper used in disease scoring."""

    def test_identical_sets(self):
        """Identical sets → similarity == 1.0."""
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """Completely disjoint sets → similarity == 0.0."""
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        """One shared element in sets of size 2 each → 1/3."""
        result = jaccard_similarity({"a", "b"}, {"b", "c"})
        assert abs(result - 1 / 3) < 1e-9

    def test_both_empty_returns_0(self):
        """Both empty sets → 0.0 (not division-by-zero)."""
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty_returns_0(self):
        """One empty set → 0.0."""
        assert jaccard_similarity({"a"}, set()) == 0.0
        assert jaccard_similarity(set(), {"a"}) == 0.0

    def test_case_insensitive(self):
        """'Li-Fraumeni Syndrome' and 'li-fraumeni syndrome' must match."""
        result = jaccard_similarity(
            {"Li-Fraumeni Syndrome"}, {"li-fraumeni syndrome"}
        )
        assert result == 1.0


class TestGradeDisease:
    """Tests for the hard-tier disease variant association grader."""

    def test_perfect_score_pathogenic(self):
        """Exact Pathogenic + exact disease match → total == 1.0."""
        score, _ = grade_disease(
            Pathogenicity.PATHOGENIC,
            ["Li-Fraumeni syndrome"],
            "Pathogenic",
            ["Li-Fraumeni syndrome"],
        )
        assert abs(score - 1.0) < 1e-6

    def test_perfect_score_benign(self):
        """Exact Benign + exact disease match → total == 1.0."""
        score, _ = grade_disease(
            Pathogenicity.BENIGN,
            ["Familial hypercholesterolemia"],
            "Benign",
            ["Familial hypercholesterolemia"],
        )
        assert abs(score - 1.0) < 1e-6

    def test_flip_penalty_pathogenic_to_benign(self):
        """Predicting Benign when truth is Pathogenic → negative total."""
        score, breakdown = grade_disease(
            Pathogenicity.BENIGN, [], "Pathogenic", ["Li-Fraumeni syndrome"]
        )
        assert score < 0
        assert breakdown["flip_penalty"] == -0.1

    def test_flip_penalty_benign_to_pathogenic(self):
        """Predicting Pathogenic when truth is Benign → negative total."""
        score, breakdown = grade_disease(
            Pathogenicity.PATHOGENIC, [], "Benign", []
        )
        assert score < 0
        assert breakdown["flip_penalty"] == -0.1

    def test_no_flip_penalty_within_tier(self):
        """Pathogenic → Likely pathogenic is same tier — no flip penalty."""
        _, breakdown = grade_disease(
            Pathogenicity.LIKELY_PATHOGENIC, [], "Pathogenic", []
        )
        assert breakdown["flip_penalty"] == 0.0

    def test_vus_no_flip_penalty(self):
        """VUS is tier 2 — predicting VUS when truth is Pathogenic = no flip."""
        _, breakdown = grade_disease(
            Pathogenicity.VUS, [], "Pathogenic", []
        )
        assert breakdown["flip_penalty"] == 0.0

    def test_vus_predicted_benign_truth_no_flip(self):
        """VUS → Benign truth: VUS is uncertain tier, not benign tier — no flip."""
        _, breakdown = grade_disease(
            Pathogenicity.VUS, [], "Benign", []
        )
        assert breakdown["flip_penalty"] == 0.0

    def test_breakdown_keys_present(self):
        """breakdown dict must contain exactly the four expected keys."""
        _, breakdown = grade_disease(
            Pathogenicity.PATHOGENIC, [], "Pathogenic", []
        )
        assert {"pathogenicity_score", "disease_score", "flip_penalty", "total"} == set(
            breakdown.keys()
        )

    def test_disease_score_partial(self):
        """Partial disease overlap → disease_score strictly between 0 and 0.5."""
        _, breakdown = grade_disease(
            Pathogenicity.PATHOGENIC,
            ["Li-Fraumeni syndrome", "Unknown disease"],
            "Pathogenic",
            ["Li-Fraumeni syndrome", "Colorectal cancer"],
        )
        assert 0 < breakdown["disease_score"] < 0.5

    def test_breakdown_total_matches_score(self):
        """The returned total float and breakdown['total'] must be identical."""
        score, breakdown = grade_disease(
            Pathogenicity.PATHOGENIC,
            ["Li-Fraumeni syndrome"],
            "Pathogenic",
            ["Li-Fraumeni syndrome"],
        )
        assert abs(score - breakdown["total"]) < 1e-9

    def test_pathogenicity_same_tier_partial_credit(self):
        """Likely pathogenic vs Pathogenic → pathogenicity_score == 0.25."""
        _, breakdown = grade_disease(
            Pathogenicity.LIKELY_PATHOGENIC, [], "Pathogenic", []
        )
        assert abs(breakdown["pathogenicity_score"] - 0.25) < 1e-9

    @pytest.mark.parametrize(
        "predicted,truth_path",
        [
            (Pathogenicity.PATHOGENIC, "Pathogenic"),
            (Pathogenicity.LIKELY_PATHOGENIC, "Likely pathogenic"),
            (Pathogenicity.BENIGN, "Benign"),
        ],
    )
    def test_exact_pathogenicity_gives_half_point(
        self, predicted: Pathogenicity, truth_path: str
    ):
        """Exact pathogenicity match gives pathogenicity_score == 0.5."""
        _, breakdown = grade_disease(predicted, [], truth_path, [])
        assert abs(breakdown["pathogenicity_score"] - 0.5) < 1e-9
