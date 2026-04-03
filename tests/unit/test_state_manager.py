"""
Unit tests for StateManager episode lifecycle.

Tests cover fixture loading, episode reset, step tracking,
done conditions, and state snapshot correctness.
No model load or network access required.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from constants import MAX_STEPS_PER_EPISODE
from core.state_manager import StateManager
from models import ActionType, ProteinState, TaskType


# ── Initialisation ─────────────────────────────────────────────────────────────


class TestStateManagerInit:
    """Tests for fixture loading at construction time."""

    def test_loads_all_three_fixture_files(self):
        """All three task-type fixture lists must be loaded with 10 entries each."""
        sm = StateManager()
        assert len(sm._fixtures[TaskType.EASY]) == 10
        assert len(sm._fixtures[TaskType.MEDIUM]) == 10
        assert len(sm._fixtures[TaskType.HARD]) == 10

    def test_no_active_episode_on_init(self):
        """step_number is -1 and is_done is False before first reset()."""
        sm = StateManager()
        assert sm.step_number == -1
        assert sm.is_done is False

    def test_tool_call_history_empty_on_init(self):
        """No tool calls before any episode starts."""
        sm = StateManager()
        assert sm.tool_call_history == []

    def test_fixture_entries_have_difficulty_field(self):
        """Every fixture entry must have a 'difficulty' key."""
        sm = StateManager()
        for task_type in TaskType:
            for entry in sm._fixtures[task_type]:
                assert "difficulty" in entry, f"Missing difficulty in {entry}"


# ── Reset ──────────────────────────────────────────────────────────────────────


class TestStateManagerReset:
    """Tests for reset() episode initialisation."""

    def test_reset_easy_returns_easy_fixture(self):
        """Resetting to EASY returns a fixture with difficulty == 'easy'."""
        sm = StateManager()
        fixture, state = sm.reset(TaskType.EASY, seed=0)
        assert fixture["difficulty"] == "easy"

    def test_reset_medium_returns_medium_fixture(self):
        """Resetting to MEDIUM returns a fixture with difficulty == 'medium'."""
        sm = StateManager()
        fixture, _ = sm.reset(TaskType.MEDIUM, seed=0)
        assert fixture["difficulty"] == "medium"

    def test_reset_hard_returns_hard_fixture(self):
        """Resetting to HARD returns a fixture with difficulty == 'hard'."""
        sm = StateManager()
        fixture, _ = sm.reset(TaskType.HARD, seed=0)
        assert fixture["difficulty"] == "hard"

    def test_reset_seed_deterministic(self):
        """Same seed always picks the same protein from the fixture list."""
        sm = StateManager()
        fixture1, _ = sm.reset(TaskType.EASY, seed=42)
        fixture2, _ = sm.reset(TaskType.EASY, seed=42)
        assert fixture1["protein_id"] == fixture2["protein_id"]

    def test_reset_different_seeds_may_differ(self):
        """Different seeds across 10 iterations sample at least 2 distinct proteins."""
        sm = StateManager()
        proteins_seen: set[str] = set()
        for seed in range(10):
            f, _ = sm.reset(TaskType.EASY, seed=seed)
            proteins_seen.add(f["protein_id"])
        assert len(proteins_seen) > 1

    def test_reset_clears_previous_state(self):
        """After a second reset() call, step counter and history are cleared."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type="call_tool",
                tool_name="get_esm2_embedding",
                tool_args={"sequence": "MALWMR"})
        sm.reset(TaskType.MEDIUM, seed=0)
        assert sm.step_number == 0
        assert sm.tool_call_history == []
        assert sm.is_done is False

    def test_state_step_number_zero_after_reset(self):
        """ProteinState returned by reset() has step_number == 0."""
        sm = StateManager()
        _, state = sm.reset(TaskType.EASY, seed=0)
        assert state.step_number == 0

    def test_state_submitted_false_after_reset(self):
        """ProteinState returned by reset() has submitted == False."""
        sm = StateManager()
        _, state = sm.reset(TaskType.EASY, seed=0)
        assert state.submitted is False

    def test_reset_returns_protein_state(self):
        """Second element of reset() tuple is a ProteinState instance."""
        sm = StateManager()
        _, state = sm.reset(TaskType.EASY, seed=0)
        assert isinstance(state, ProteinState)

    def test_custom_episode_id_preserved(self):
        """episode_id override is reflected in the returned state."""
        sm = StateManager()
        _, state = sm.reset(TaskType.EASY, seed=0, episode_id="test-ep-123")
        assert state.episode_id == "test-ep-123"


# ── Step ───────────────────────────────────────────────────────────────────────


class TestStateManagerStep:
    """Tests for step() state advancement."""

    def test_step_increments_counter(self):
        """Each step call increments step_number by 1."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type="call_tool",
                tool_name="get_esm2_embedding",
                tool_args={"sequence": "MALWMR"})
        assert sm.step_number == 1

    def test_multiple_steps_accumulate(self):
        """Three steps → step_number == 3."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        for _ in range(3):
            sm.step(action_type="call_tool",
                    tool_name="get_esm2_embedding",
                    tool_args={"sequence": "M"})
        assert sm.step_number == 3

    def test_tool_call_recorded_in_history(self):
        """CALL_TOOL step records an entry in tool_call_history."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type="call_tool",
                tool_name="get_esm2_embedding",
                tool_args={"sequence": "MALWMR"})
        assert len(sm.tool_call_history) == 1
        assert sm.tool_call_history[0]["tool_name"] == "get_esm2_embedding"

    def test_submit_action_not_recorded_in_tool_history(self):
        """SUBMIT_PREDICTION actions are NOT recorded in tool_call_history."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type=ActionType.SUBMIT_PREDICTION.value)
        assert len(sm.tool_call_history) == 0

    def test_step_before_reset_raises(self):
        """step() before reset() raises RuntimeError."""
        sm = StateManager()
        with pytest.raises(RuntimeError):
            sm.step(action_type="call_tool",
                    tool_name="get_esm2_embedding",
                    tool_args={})

    def test_tool_call_history_is_copy(self):
        """tool_call_history property returns a copy, not the internal list."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type="call_tool",
                tool_name="get_esm2_embedding",
                tool_args={"sequence": "M"})
        history = sm.tool_call_history
        history.clear()
        assert len(sm.tool_call_history) == 1


# ── Done conditions ────────────────────────────────────────────────────────────


class TestStateManagerDone:
    """Tests for is_done flag and episode termination conditions."""

    def test_mark_submitted_sets_done(self):
        """mark_submitted() makes is_done True."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.mark_submitted()
        assert sm.is_done is True

    def test_max_steps_sets_done(self):
        """After MAX_STEPS_PER_EPISODE steps, is_done becomes True."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        for _ in range(MAX_STEPS_PER_EPISODE):
            sm.step(action_type="call_tool",
                    tool_name="get_esm2_embedding",
                    tool_args={"sequence": "M"})
        assert sm.is_done is True

    def test_not_done_before_max_steps(self):
        """One step fewer than MAX_STEPS_PER_EPISODE → is_done False."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        for _ in range(MAX_STEPS_PER_EPISODE - 1):
            sm.step(action_type="call_tool",
                    tool_name="get_esm2_embedding",
                    tool_args={"sequence": "M"})
        assert sm.is_done is False

    def test_not_done_immediately_after_reset(self):
        """Fresh episode is not done."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        assert sm.is_done is False

    def test_not_done_before_any_episode(self):
        """is_done is False when no episode has started."""
        sm = StateManager()
        assert sm.is_done is False


# ── get_state ──────────────────────────────────────────────────────────────────


class TestStateManagerGetState:
    """Tests for the get_state() snapshot method."""

    def test_get_state_returns_protein_state(self):
        """get_state() returns a ProteinState instance."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        state = sm.get_state()
        assert isinstance(state, ProteinState)

    def test_get_state_before_reset_raises(self):
        """get_state() before reset() raises RuntimeError."""
        sm = StateManager()
        with pytest.raises(RuntimeError):
            sm.get_state()

    def test_cumulative_reward_starts_zero(self):
        """Freshly-reset episode has cumulative_reward == 0.0."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        state = sm.get_state()
        assert state.cumulative_reward == 0.0

    def test_state_task_type_matches_reset(self):
        """State task_type matches the task_type passed to reset()."""
        sm = StateManager()
        sm.reset(TaskType.MEDIUM, seed=0)
        assert sm.get_state().task_type == TaskType.MEDIUM

    def test_state_tool_calls_made_increments(self):
        """tool_calls_made in state increments with each CALL_TOOL step."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.step(action_type="call_tool",
                tool_name="get_esm2_embedding",
                tool_args={"sequence": "M"})
        state = sm.get_state()
        assert state.tool_calls_made == 1

    def test_state_submitted_true_after_mark_submitted(self):
        """submitted field in state reflects mark_submitted() call."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        sm.mark_submitted()
        assert sm.get_state().submitted is True


# ── get_current_fixture ────────────────────────────────────────────────────────


class TestGetCurrentFixture:
    """Tests for get_current_fixture()."""

    def test_returns_dict(self):
        """get_current_fixture() returns a dict."""
        sm = StateManager()
        sm.reset(TaskType.EASY, seed=0)
        fixture = sm.get_current_fixture()
        assert isinstance(fixture, dict)

    def test_raises_before_reset(self):
        """get_current_fixture() before reset() raises RuntimeError."""
        sm = StateManager()
        with pytest.raises(RuntimeError):
            sm.get_current_fixture()

    def test_fixture_matches_reset_task_type(self):
        """Fixture difficulty matches the task_type used during reset."""
        sm = StateManager()
        sm.reset(TaskType.HARD, seed=0)
        fixture = sm.get_current_fixture()
        assert fixture["difficulty"] == "hard"
