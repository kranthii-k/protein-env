"""
core/state_manager.py — Episode lifecycle manager for ProteinEnv.

Loads all fixture data at startup and tracks per-episode state: step count,
submission status, tool call history, and the sampled protein/variant.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from pathlib import Path

try:
    from constants import MAX_STEPS_PER_EPISODE
    from models import ActionType, ProteinState, TaskType
except ImportError:
    from protein_env.constants import MAX_STEPS_PER_EPISODE
    from protein_env.models import ActionType, ProteinState, TaskType

logger = logging.getLogger(__name__)


class StateManager:
    """Manage episode state for a single ProteinEnv session.

    Responsible for:
    - Loading fixture data from data/fixtures/ at construction time
    - Sampling proteins/variants for each episode
    - Tracking step count, submission status, and tool call history
    - Producing ProteinState snapshots on demand
    """

    FIXTURE_PATHS: dict[TaskType, str] = {
        TaskType.EASY:   "data/fixtures/easy_proteins.json",
        TaskType.MEDIUM: "data/fixtures/medium_proteins.json",
        TaskType.HARD:   "data/fixtures/hard_variants.json",
    }

    def __init__(self, data_root: Path | None = None) -> None:
        """Initialize StateManager and load all fixtures into memory.

        Args:
            data_root: Root directory for data/fixtures/. Defaults to the
                       directory two levels above this file (project root).

        Returns:
            None.

        Raises:
            FileNotFoundError: If any fixture file is missing.
            ValueError: If any fixture file contains invalid JSON or empty list.
        """
        if data_root is None:
            data_root = Path(__file__).parent.parent

        self._fixtures: dict[TaskType, list[dict]] = {}
        for task_type, rel_path in self.FIXTURE_PATHS.items():
            self._fixtures[task_type] = self._load_fixture(
                data_root / rel_path, task_type
            )

        self._episode_id: str | None = None
        self._task_type: TaskType | None = None
        self._current_fixture: dict | None = None
        self._step_number: int = -1
        self._submitted: bool = False
        self._tool_call_history: list[dict] = []
        self._episode_history: list[dict] = []

    def _load_fixture(self, path: Path, task_type: TaskType) -> list[dict]:
        """Load and validate a single fixture JSON file.

        Args:
            path:      Absolute path to the fixture JSON file.
            task_type: TaskType label (used in error messages only).

        Returns:
            Non-empty list of fixture dicts.

        Raises:
            FileNotFoundError: If the file does not exist at *path*.
            ValueError: If the file is not valid JSON or the list is empty.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Fixture file not found for {task_type.value}: {path}"
            )
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(
                f"Fixture file for {task_type.value} must be a non-empty JSON array."
            )
        logger.info(
            "Loaded %d fixtures for task_type=%s from %s",
            len(data), task_type.value, path,
        )
        return data

    def reset(
        self,
        task_type: TaskType,
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> tuple[dict, ProteinState]:
        """Start a new episode, sampling a protein/variant from fixtures.

        Args:
            task_type:  Which task tier to run (EASY, MEDIUM, HARD).
            seed:       Random seed for deterministic protein sampling.
            episode_id: Override episode UUID. Auto-generated if None.

        Returns:
            Tuple of (fixture_dict, ProteinState) for the new episode.

        Raises:
            Nothing (fixture data already validated at construction).
        """
        rng = random.Random(seed)
        self._current_fixture = rng.choice(self._fixtures[task_type])
        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_type = task_type
        self._step_number = 0
        self._submitted = False
        self._tool_call_history = []
        self._episode_history = []

        protein_id = (
            self._current_fixture.get("protein_id")
            or self._current_fixture.get("variant_id", "unknown")
        )
        logger.info(
            "Episode %s started: task=%s protein=%s",
            self._episode_id, task_type.value, protein_id,
        )
        return self._current_fixture, self.get_state()

    def step(
        self,
        action_type: str | ActionType,
        tool_name: str | None = None,
        tool_args: dict | None = None,
    ) -> None:
        """Advance episode state by one step.

        Args:
            action_type: Type of action taken this step (string or ActionType).
            tool_name:   Tool name if action was CALL_TOOL; None otherwise.
            tool_args:   Tool args if action was CALL_TOOL; None otherwise.

        Returns:
            None.

        Raises:
            RuntimeError: If episode is not active (reset() not called).
        """
        self._assert_active()
        action_str = (
            action_type.value
            if isinstance(action_type, ActionType)
            else str(action_type)
        )
        if action_str == ActionType.CALL_TOOL.value and tool_name:
            self._tool_call_history.append(
                {"tool_name": tool_name, "tool_args": tool_args or {}}
            )
        self._episode_history.append(
            {"step": self._step_number, "action_type": action_str,
             "tool_name": tool_name, "tool_args": tool_args}
        )
        self._step_number += 1

    def mark_submitted(self) -> None:
        """Mark the current episode as having received a prediction submission.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing.
        """
        self._submitted = True

    def get_state(self) -> ProteinState:
        """Return a snapshot of current episode state as a Pydantic model.

        Args:
            None.

        Returns:
            ProteinState with all episode tracking fields populated.

        Raises:
            RuntimeError: If no active episode (reset() not called).
        """
        self._assert_active()
        protein_id = (
            self._current_fixture.get("protein_id")  # type: ignore[union-attr]
            or self._current_fixture.get("variant_id", "unknown")  # type: ignore[union-attr]
        )
        return ProteinState(
            episode_id=self._episode_id,  # type: ignore[arg-type]
            task_type=self._task_type,  # type: ignore[arg-type]
            current_protein_id=protein_id,
            step_number=self._step_number,
            cumulative_reward=0.0,
            submitted=self._submitted,
            tool_calls_made=len(self._tool_call_history),
            episode_history=list(self._episode_history),
        )

    def get_current_fixture(self) -> dict:
        """Return the current episode's raw fixture dict.

        Args:
            None.

        Returns:
            Raw fixture dict (easy_proteins, medium_proteins, or hard_variants entry).

        Raises:
            RuntimeError: If no active episode.
        """
        self._assert_active()
        return self._current_fixture  # type: ignore[return-value]

    def _assert_active(self) -> None:
        """Raise RuntimeError if no episode is currently active.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._episode_id is None:
            raise RuntimeError(
                "No active episode. Call reset() before accessing state."
            )

    @property
    def is_done(self) -> bool:
        """True if the episode has ended (submitted or max steps reached)."""
        if self._step_number < 0:
            return False
        return self._submitted or self._step_number >= MAX_STEPS_PER_EPISODE

    @property
    def step_number(self) -> int:
        """Current 0-indexed step number. -1 if no active episode."""
        return self._step_number

    @property
    def tool_call_history(self) -> list[dict]:
        """List of tool calls made this episode. Each dict has tool_name + tool_args."""
        return list(self._tool_call_history)
