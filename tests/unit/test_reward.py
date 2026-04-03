"""
Unit tests for reward_calculator.

No model load, no network — all graders are pure and StateManager
uses pre-loaded fixture files from disk.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from core.reward_calculator import calculate_reward, is_redundant_tool_call
from models import ActionType, Pathogenicity, ProteinAction, TaskType

# ── Shared ground truth fixtures ───────────────────────────────────────────────

EASY_GROUND_TRUTH: dict = {
    "protein_id": "P01308",
    "family": "Insulin family",
    "family_choices": [
        "Insulin family", "Globin family", "Kinase family",
        "Protease family", "Receptor family", "Hormone family",
        "Antibody family", "Channel family", "Lipase family",
        "Transferase family",
    ],
    "difficulty": "easy",
}

MEDIUM_GROUND_TRUTH: dict = {
    "protein_id": "P04637",
    "go_terms": {
        "molecular_function": ["GO:0003700", "GO:0005515"],
        "biological_process": ["GO:0006915"],
        "cellular_component": ["GO:0005634"],
    },
    "difficulty": "medium",
}

HARD_GROUND_TRUTH: dict = {
    "protein_id": "P04637",
    "pathogenicity": "Pathogenic",
    "associated_diseases": ["Li-Fraumeni syndrome"],
    "difficulty": "hard",
}


# ── Helper builders ────────────────────────────────────────────────────────────

def _tool_action(seq: str = "MALWMR") -> ProteinAction:
    """Return a CALL_TOOL action for get_esm2_embedding."""
    return ProteinAction(
        action_type=ActionType.CALL_TOOL,
        tool_name="get_esm2_embedding",
        tool_args={"sequence": seq},
    )


def _submit_easy(family: str = "Insulin family") -> ProteinAction:
    """Return a SUBMIT_PREDICTION action for easy tier."""
    return ProteinAction(
        action_type=ActionType.SUBMIT_PREDICTION,
        predicted_family=family,
    )


# ── is_redundant_tool_call ─────────────────────────────────────────────────────


class TestIsRedundantToolCall:
    """Tests for the redundancy-detection helper."""

    def test_not_redundant_first_call(self):
        """Empty history → never redundant."""
        assert is_redundant_tool_call("get_esm2_embedding", {"seq": "M"}, []) is False

    def test_redundant_same_args(self):
        """Identical tool_name + tool_args in history → True."""
        history = [{"tool_name": "get_esm2_embedding", "tool_args": {"sequence": "M"}}]
        assert (
            is_redundant_tool_call("get_esm2_embedding", {"sequence": "M"}, history)
            is True
        )

    def test_not_redundant_different_args(self):
        """Same tool, different args → not redundant."""
        history = [{"tool_name": "get_esm2_embedding", "tool_args": {"sequence": "M"}}]
        assert (
            is_redundant_tool_call(
                "get_esm2_embedding", {"sequence": "MALWMR"}, history
            )
            is False
        )

    def test_not_redundant_different_tool_name(self):
        """Different tool name → not redundant even with same args."""
        history = [{"tool_name": "tool_a", "tool_args": {"x": 1}}]
        assert is_redundant_tool_call("tool_b", {"x": 1}, history) is False

    def test_redundant_dict_arg_order_independent(self):
        """Dict arg order must not affect redundancy detection."""
        history = [
            {"tool_name": "get_esm2_embedding", "tool_args": {"a": 1, "b": 2}}
        ]
        assert (
            is_redundant_tool_call(
                "get_esm2_embedding", {"b": 2, "a": 1}, history
            )
            is True
        )

    def test_not_redundant_when_history_has_different_tool(self):
        """History with different tool name → not redundant."""
        history = [{"tool_name": "other_tool", "tool_args": {"sequence": "M"}}]
        assert (
            is_redundant_tool_call("get_esm2_embedding", {"sequence": "M"}, history)
            is False
        )


# ── calculate_reward — tool calls ─────────────────────────────────────────────


class TestCalculateRewardToolCalls:
    """Tests for reward logic when action_type == CALL_TOOL."""

    def test_tool_call_base_score_zero(self):
        """Non-redundant tool calls always have base_score == 0.0."""
        rb = calculate_reward(
            _tool_action(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.base_score == 0.0

    def test_tool_call_redundant_penalty(self):
        """Redundant tool call incurs a negative redundant_tool_penalty."""
        history = [{"tool_name": "get_esm2_embedding",
                    "tool_args": {"sequence": "MALWMR"}}]
        rb = calculate_reward(
            _tool_action("MALWMR"), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=1, already_submitted=False,
            tool_call_history=history,
        )
        assert rb.redundant_tool_penalty < 0

    def test_tool_call_not_redundant_no_penalty(self):
        """First-time tool call has no redundant_tool_penalty."""
        rb = calculate_reward(
            _tool_action(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.redundant_tool_penalty == 0.0

    def test_tool_call_total_zero_when_no_penalty(self):
        """Non-redundant tool call → total == 0.0."""
        rb = calculate_reward(
            _tool_action(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.total == 0.0


# ── calculate_reward — easy tier ──────────────────────────────────────────────


class TestCalculateRewardEasy:
    """Tests for easy-tier (family classification) reward logic."""

    def test_easy_exact_match_score_1(self):
        """Exact family match → base_score == 1.0."""
        rb = calculate_reward(
            _submit_easy("Insulin family"), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert abs(rb.base_score - 1.0) < 1e-6

    def test_easy_wrong_family_score_0(self):
        """Wrong family → base_score == 0.0."""
        rb = calculate_reward(
            _submit_easy("Globin family"), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.base_score == 0.0

    def test_easy_superfamily_match_score_03(self):
        """Superfamily partial match → base_score == 0.3."""
        rb = calculate_reward(
            _submit_easy("Insulin"), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert abs(rb.base_score - 0.3) < 1e-6

    def test_easy_case_insensitive_match(self):
        """Case-insensitive family match → full credit."""
        rb = calculate_reward(
            _submit_easy("insulin family"), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert abs(rb.base_score - 1.0) < 1e-6


# ── calculate_reward — medium tier ───────────────────────────────────────────


class TestCalculateRewardMedium:
    """Tests for medium-tier (GO term prediction) reward logic."""

    def test_medium_perfect_go_score_1(self):
        """Predicting all true GO IDs → base_score == 1.0."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_go_terms=["GO:0003700", "GO:0005515",
                                 "GO:0006915", "GO:0005634"],
        )
        rb = calculate_reward(
            action, MEDIUM_GROUND_TRUTH, TaskType.MEDIUM,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert abs(rb.base_score - 1.0) < 1e-6

    def test_medium_empty_prediction_score_0(self):
        """Predicting only hallucinated GO IDs that match nothing → base_score == 0.0."""
        # Empty list is rejected by Pydantic; use a valid-format but wrong ID instead.
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_go_terms=["GO:9999999"],
        )
        rb = calculate_reward(
            action, MEDIUM_GROUND_TRUTH, TaskType.MEDIUM,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.base_score == 0.0


    def test_medium_partial_score_between_0_and_1(self):
        """Partial GO prediction → 0.0 < base_score < 1.0."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_go_terms=["GO:0003700"],
        )
        rb = calculate_reward(
            action, MEDIUM_GROUND_TRUTH, TaskType.MEDIUM,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert 0.0 < rb.base_score < 1.0


# ── calculate_reward — hard tier ──────────────────────────────────────────────


class TestCalculateRewardHard:
    """Tests for hard-tier (disease variant) reward logic."""

    def test_hard_perfect_score_1(self):
        """Exact pathogenicity + exact disease match → total == 1.0."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_pathogenicity=Pathogenicity.PATHOGENIC,
            predicted_diseases=["Li-Fraumeni syndrome"],
        )
        rb = calculate_reward(
            action, HARD_GROUND_TRUTH, TaskType.HARD,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert abs(rb.total - 1.0) < 1e-6

    def test_hard_flip_penalty_negative_total(self):
        """Predicting Benign when truth is Pathogenic → negative total."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_pathogenicity=Pathogenicity.BENIGN,
        )
        rb = calculate_reward(
            action, HARD_GROUND_TRUTH, TaskType.HARD,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert rb.total < 0


# ── Wasted step penalty ───────────────────────────────────────────────────────


class TestWastedStepPenalty:
    """Tests for the late-submission penalty."""

    def test_wasted_step_penalty_after_step_3(self):
        """Submitting after step 3 on first submission incurs penalty."""
        early = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        late = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=5, already_submitted=False, tool_call_history=[],
        )
        assert late.wasted_step_penalty < 0
        assert early.wasted_step_penalty == 0.0

    def test_no_penalty_at_step_3(self):
        """Step 3 is the threshold — exactly step 3 gets no penalty."""
        rb = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=3, already_submitted=False, tool_call_history=[],
        )
        assert rb.wasted_step_penalty == 0.0

    def test_penalty_applied_step_4(self):
        """Step 4 (> 3) triggers wasted_step_penalty."""
        rb = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=4, already_submitted=False, tool_call_history=[],
        )
        assert rb.wasted_step_penalty < 0

    def test_no_penalty_when_already_submitted(self):
        """If already_submitted=True, no additional wasted step penalty."""
        rb = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=9, already_submitted=True, tool_call_history=[],
        )
        assert rb.wasted_step_penalty == 0.0


# ── RewardBreakdown contract ───────────────────────────────────────────────────


class TestRewardBreakdownContract:
    """Tests for the RewardBreakdown output guarantees."""

    def test_reward_breakdown_total_is_sum(self):
        """RewardBreakdown.total must equal sum of all component fields."""
        rb = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        expected = (
            rb.base_score
            + rb.wasted_step_penalty
            + rb.redundant_tool_penalty
            + rb.flip_penalty
        )
        assert abs(rb.total - expected) < 1e-9

    def test_reward_deterministic_same_inputs(self):
        """Same inputs always produce an identical RewardBreakdown total."""
        action = ProteinAction(
            action_type=ActionType.SUBMIT_PREDICTION,
            predicted_family="Globin family",
        )
        rb1 = calculate_reward(
            action, EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=2, already_submitted=False, tool_call_history=[],
        )
        rb2 = calculate_reward(
            action, EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=2, already_submitted=False, tool_call_history=[],
        )
        assert rb1.total == rb2.total

    def test_reward_breakdown_has_required_fields(self):
        """RewardBreakdown exposes all four expected fields."""
        rb = calculate_reward(
            _submit_easy(), EASY_GROUND_TRUTH, TaskType.EASY,
            step_number=0, already_submitted=False, tool_call_history=[],
        )
        assert hasattr(rb, "base_score")
        assert hasattr(rb, "wasted_step_penalty")
        assert hasattr(rb, "redundant_tool_penalty")
        assert hasattr(rb, "flip_penalty")
        assert hasattr(rb, "total")
