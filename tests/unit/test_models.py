"""
Unit tests for Pydantic model validation and auto-computed fields.

No network access and no model weights required.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from models import (
    ActionType,
    GONamespace,
    Pathogenicity,
    ProteinAction,
    ProteinObservation,
    ProteinState,
    RewardBreakdown,
    TaskType,
    VariantInfo,
)


# ── VariantInfo ────────────────────────────────────────────────────────────────


class TestVariantInfo:
    """Tests for the VariantInfo model including auto-notation generation."""

    def test_auto_notation_generated(self):
        """variant_notation is built from the three positional fields when omitted."""
        v = VariantInfo(wildtype_aa="A", mutant_aa="V", position=4)
        assert v.variant_notation == "A4V"

    def test_explicit_notation_preserved(self):
        """Explicitly supplied notation is kept as-is."""
        v = VariantInfo(
            wildtype_aa="R",
            mutant_aa="H",
            position=175,
            variant_notation="custom_R175H",
        )
        assert v.variant_notation == "custom_R175H"

    def test_notation_format_correct(self):
        """Standard HGVS-style compact notation: {wt}{pos}{mut}."""
        v = VariantInfo(wildtype_aa="R", mutant_aa="H", position=175)
        assert v.variant_notation == "R175H"

    def test_fields_stored_correctly(self):
        """All four fields are accessible after construction."""
        v = VariantInfo(wildtype_aa="T", mutant_aa="M", position=790)
        assert v.wildtype_aa == "T"
        assert v.mutant_aa == "M"
        assert v.position == 790


# ── ProteinObservation ─────────────────────────────────────────────────────────


class TestProteinObservation:
    """Tests for the ProteinObservation model including auto-computed sequence_length."""

    def _make_obs(self, **overrides) -> ProteinObservation:
        """Helper to build a minimal valid ProteinObservation."""
        defaults = dict(
            protein_id="P01308",
            sequence="MALWMR",
            task_type=TaskType.EASY,
            task_description="Classify the protein.",
            available_tools=["get_esm2_embedding"],
            step_number=0,
            max_steps=10,
        )
        defaults.update(overrides)
        return ProteinObservation(**defaults)

    def test_sequence_length_auto_set(self):
        """sequence_length is computed from the actual sequence string."""
        obs = self._make_obs(sequence="MALWMR")
        assert obs.sequence_length == 6

    def test_variant_info_none_by_default(self):
        """variant_info defaults to None for non-hard tasks."""
        obs = self._make_obs()
        assert obs.variant_info is None

    def test_easy_task_no_variant(self):
        """Easy task observations have no variant_info."""
        obs = self._make_obs(task_type=TaskType.EASY)
        assert obs.variant_info is None

    def test_hard_task_with_variant(self):
        """Hard task observations can carry a VariantInfo."""
        vi = VariantInfo(wildtype_aa="R", mutant_aa="H", position=175)
        obs = self._make_obs(task_type=TaskType.HARD, variant_info=vi)
        assert obs.variant_info is not None
        assert obs.variant_info.variant_notation == "R175H"

    def test_sequence_length_overrides_caller_value(self):
        """Even if caller passes a wrong sequence_length, validator corrects it."""
        obs = self._make_obs(sequence="MALWMR", sequence_length=999)
        assert obs.sequence_length == 6

    @pytest.mark.parametrize("seq,expected_len", [
        ("M", 1),
        ("MALWMR", 6),
        ("MVHLT" * 10, 50),
    ])
    def test_sequence_length_parametrized(self, seq: str, expected_len: int):
        """sequence_length matches len(sequence) for various inputs."""
        obs = self._make_obs(sequence=seq)
        assert obs.sequence_length == expected_len


# ── ProteinAction ──────────────────────────────────────────────────────────────


class TestProteinAction:
    """Tests for ProteinAction validation including action completeness rules."""

    def test_call_tool_without_tool_name_raises(self):
        """CALL_TOOL with no tool_name must raise ValueError."""
        with pytest.raises(Exception):
            ProteinAction(action_type=ActionType.CALL_TOOL)

    def test_submit_without_prediction_raises(self):
        """SUBMIT_PREDICTION with no prediction fields must raise ValueError."""
        with pytest.raises(Exception):
            ProteinAction(action_type=ActionType.SUBMIT_PREDICTION)

    def test_valid_tool_call(self):
        """Valid CALL_TOOL with tool_name passes validation."""
        action = ProteinAction(
            action_type=ActionType.CALL_TOOL,
            tool_name="get_esm2_embedding",
            tool_args={"sequence": "MALWMR"},
        )
        assert action.tool_name == "get_esm2_embedding"

    def test_valid_submit_family(self):
        """SUBMIT_PREDICTION with predicted_family passes validation."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_family="Insulin family",
        )
        assert action.predicted_family == "Insulin family"

    def test_valid_submit_go_terms(self):
        """SUBMIT_PREDICTION with predicted_go_terms passes validation."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_go_terms=["GO:0003700"],
        )
        assert action.predicted_go_terms == ["GO:0003700"]

    def test_valid_submit_pathogenicity(self):
        """SUBMIT_PREDICTION with predicted_pathogenicity passes validation."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_pathogenicity=Pathogenicity.PATHOGENIC,
        )
        assert action.predicted_pathogenicity == Pathogenicity.PATHOGENIC

    def test_reasoning_field_optional(self):
        """reasoning field is optional on any valid action."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_family="Insulin family",
            reasoning="Because of the conserved GIVEQCC motif.",
        )
        assert action.reasoning is not None

    def test_tool_args_optional_on_tool_call(self):
        """tool_args defaults to None when not supplied."""
        action = ProteinAction(
            action_type=ActionType.CALL_TOOL,
            tool_name="get_esm2_embedding",
        )
        assert action.tool_args is None


# ── RewardBreakdown ────────────────────────────────────────────────────────────


class TestRewardBreakdown:
    """Tests for RewardBreakdown auto-total computation."""

    def test_total_auto_computed(self):
        """total == base_score + all penalties."""
        rb = RewardBreakdown(
            base_score=1.0,
            wasted_step_penalty=-0.05,
            redundant_tool_penalty=-0.01,
            flip_penalty=0.0,
        )
        assert abs(rb.total - 0.94) < 1e-9

    def test_all_zeros_total_zero(self):
        """All zero components → total == 0.0."""
        rb = RewardBreakdown(
            base_score=0.0,
            wasted_step_penalty=0.0,
            redundant_tool_penalty=0.0,
            flip_penalty=0.0,
        )
        assert rb.total == 0.0

    def test_negative_total_allowed(self):
        """Total can be negative when penalties exceed base_score."""
        rb = RewardBreakdown(
            base_score=0.0,
            wasted_step_penalty=0.0,
            redundant_tool_penalty=0.0,
            flip_penalty=-0.1,
        )
        assert rb.total == -0.1

    def test_total_recomputed_on_all_fields(self):
        """Verify the sum formula across all four components."""
        rb = RewardBreakdown(
            base_score=0.5,
            wasted_step_penalty=-0.05,
            redundant_tool_penalty=-0.01,
            flip_penalty=-0.1,
        )
        expected = 0.5 + (-0.05) + (-0.01) + (-0.1)
        assert abs(rb.total - expected) < 1e-9


# ── TaskType ───────────────────────────────────────────────────────────────────


class TestTaskType:
    """Tests for the TaskType enum."""

    def test_valid_values(self):
        """All three task types exist."""
        assert TaskType.EASY is not None
        assert TaskType.MEDIUM is not None
        assert TaskType.HARD is not None

    def test_string_values(self):
        """Enum values match the expected lowercase strings."""
        assert TaskType.EASY.value == "easy"
        assert TaskType.MEDIUM.value == "medium"
        assert TaskType.HARD.value == "hard"

    def test_from_value(self):
        """TaskType can be constructed from its string value."""
        assert TaskType("easy") == TaskType.EASY
        assert TaskType("medium") == TaskType.MEDIUM
        assert TaskType("hard") == TaskType.HARD


# ── Pathogenicity ──────────────────────────────────────────────────────────────


class TestPathogenicity:
    """Tests for the Pathogenicity enum."""

    def test_all_five_values_exist(self):
        """There must be exactly five pathogenicity tiers."""
        assert len(Pathogenicity) == 5

    def test_pathogenic_value(self):
        """Pathogenic string value matches ClinVar spelling."""
        assert Pathogenicity.PATHOGENIC.value == "Pathogenic"

    def test_likely_pathogenic_value(self):
        """Likely pathogenic spelling matches ClinVar."""
        assert Pathogenicity.LIKELY_PATHOGENIC.value == "Likely pathogenic"

    def test_vus_value(self):
        """VUS abbreviation expands to full ClinVar label."""
        assert Pathogenicity.VUS.value == "Variant of Uncertain Significance"

    def test_likely_benign_value(self):
        """Likely benign spelling matches ClinVar."""
        assert Pathogenicity.LIKELY_BENIGN.value == "Likely benign"

    def test_benign_value(self):
        """Benign spelling matches ClinVar."""
        assert Pathogenicity.BENIGN.value == "Benign"


# ── GONamespace ────────────────────────────────────────────────────────────────


class TestGONamespace:
    """Tests for the GONamespace enum."""

    def test_three_namespaces(self):
        """Gene Ontology has exactly three top-level namespaces."""
        assert len(GONamespace) == 3

    def test_namespace_values(self):
        """Namespace values match canonical GO strings."""
        assert GONamespace.MOLECULAR_FUNCTION.value == "molecular_function"
        assert GONamespace.BIOLOGICAL_PROCESS.value == "biological_process"
        assert GONamespace.CELLULAR_COMPONENT.value == "cellular_component"
