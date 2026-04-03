"""
server/protein_environment.py — Core ProteinEnv environment.

Implements the OpenEnv reset / step / state contract by wiring together
StateManager (episode lifecycle), ESM2Embedder (tool backend), and
reward_calculator (scoring). One instance = one concurrent session.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from ..models import (
        ActionType,
        ProteinAction,
        ProteinObservation,
        ProteinState,
        RewardBreakdown,
        StepInfo,
        StepResult,
        TaskType,
        VariantInfo,
    )
    from ..core.esm2_embedder import ESM2Embedder
    from ..core.state_manager import StateManager
    from ..core.reward_calculator import calculate_reward
    from ..constants import ESM2_MODEL_ID, MAX_STEPS_PER_EPISODE
except ImportError:
    from models import (
        ActionType,
        ProteinAction,
        ProteinObservation,
        ProteinState,
        RewardBreakdown,
        StepInfo,
        StepResult,
        TaskType,
        VariantInfo,
    )
    from core.esm2_embedder import ESM2Embedder
    from core.state_manager import StateManager
    from core.reward_calculator import calculate_reward
    from constants import ESM2_MODEL_ID, MAX_STEPS_PER_EPISODE

logger = logging.getLogger(__name__)

_AVAILABLE_TOOLS: list[str] = ["get_esm2_embedding"]
_KNOWN_TOOLS: frozenset[str] = frozenset(_AVAILABLE_TOOLS)


class ProteinEnvironment:
    """OpenEnv-compliant environment for protein function prediction.

    Wraps StateManager (episode lifecycle), ESM2Embedder (tool backend),
    and reward_calculator (scoring) into the standard reset / step / state API.

    One instance handles one concurrent session. Not thread-safe.
    """

    def __init__(self) -> None:
        """Initialize environment components. Does not load ESM2 weights yet.

        Args:
            None.

        Returns:
            None.

        Raises:
            Nothing.
        """
        self._embedder = ESM2Embedder()
        self._state_manager = StateManager()
        self._done: bool = False
        self._task_type: TaskType | None = None
        self._current_fixture: dict | None = None
        self._cumulative_reward: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_type: str = "easy",
        seed: int | None = None,
        episode_id: str | None = None,
    ) -> ProteinObservation:
        """Start a new episode and return the initial observation.

        Args:
            task_type:  One of "easy", "medium", "hard". Case-insensitive.
            seed:       Optional random seed for reproducible episode sampling.
            episode_id: Optional episode ID override.

        Returns:
            ProteinObservation for the first step.

        Raises:
            ValueError: If task_type is not a valid TaskType value.
        """
        parsed = self._parse_task_type(task_type)
        self._task_type = parsed
        self._done = False
        self._cumulative_reward = 0.0

        fixture, _ = self._state_manager.reset(parsed, seed=seed, episode_id=episode_id)
        self._current_fixture = fixture

        obs = self._build_observation(fixture, parsed)
        logger.info("reset() → task=%s episode=%s", parsed.value, episode_id)
        return obs

    def step(self, action: ProteinAction) -> StepResult:
        """Execute one agent action and return the result.

        Handles two action types:

        1. CALL_TOOL — runs the requested tool (get_esm2_embedding).
           Tool result stored in StepInfo.tool_result. Episode does NOT end.
           Redundant calls incur REDUNDANT_TOOL_PENALTY.

        2. SUBMIT_PREDICTION — scores the prediction via reward_calculator.
           Episode ends (done=True). Final score logged at INFO level.

        Args:
            action: Validated ProteinAction from the agent.

        Returns:
            StepResult with updated observation, reward, done flag, and info.

        Raises:
            RuntimeError: If reset() has not been called.
            RuntimeError: If episode is already done.
            ValueError:   If tool_name is unknown.
        """
        self._assert_ready()

        reward_breakdown = calculate_reward(
            action=action,
            ground_truth=self._current_fixture,  # type: ignore[arg-type]
            task_type=self._task_type,  # type: ignore[arg-type]
            step_number=self._state_manager.step_number,
            already_submitted=self._state_manager.get_state().submitted,
            tool_call_history=self._state_manager.tool_call_history,
        )

        tool_result: dict | None = None
        done_reason: str | None = None

        if action.action_type == ActionType.CALL_TOOL:
            tool_result = self._handle_tool_call(action)
            self._state_manager.step(
                action_type=ActionType.CALL_TOOL,
                tool_name=action.tool_name,
                tool_args=action.tool_args,
            )
        else:
            self._state_manager.step(action_type=ActionType.SUBMIT_PREDICTION)
            self._state_manager.mark_submitted()
            done_reason = "prediction_submitted"
            logger.info(
                "SUBMIT_PREDICTION: task=%s reward=%.4f",
                self._task_type.value if self._task_type else "?",  # type: ignore[union-attr]
                reward_breakdown.total,
            )

        self._cumulative_reward += reward_breakdown.total
        self._done = self._state_manager.is_done

        if self._state_manager.step_number >= MAX_STEPS_PER_EPISODE and not self._done:
            self._done = True
            done_reason = "max_steps_reached"

        obs = self._build_observation(
            self._current_fixture,  # type: ignore[arg-type]
            self._task_type,  # type: ignore[arg-type]
        )
        info = StepInfo(
            reward_breakdown=reward_breakdown,
            tool_result=tool_result,
            done_reason=done_reason,
        )
        return StepResult(
            observation=obs,
            reward=reward_breakdown.total,
            done=self._done,
            info=info,
        )

    def state(self) -> ProteinState:
        """Return the current full episode state snapshot.

        Args:
            None.

        Returns:
            ProteinState with all episode tracking fields populated.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        return self._state_manager.get_state()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        """Guard that ensures reset() has been called and episode is not done.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If reset() not called or episode already done.
        """
        if self._task_type is None or self._current_fixture is None:
            raise RuntimeError("reset() must be called before step().")
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

    def _parse_task_type(self, task_type: str) -> TaskType:
        """Convert a case-insensitive string to a TaskType enum value.

        Args:
            task_type: Raw string such as "easy", "MEDIUM", or "Hard".

        Returns:
            Corresponding TaskType enum member.

        Raises:
            ValueError: If the string does not match any TaskType value.
        """
        normalised = task_type.strip().lower()
        for member in TaskType:
            if member.value == normalised:
                return member
        valid = [m.value for m in TaskType]
        raise ValueError(
            f"Invalid task_type '{task_type}'. Valid values: {valid}"
        )

    def _build_observation(
        self, fixture: dict, task_type: TaskType
    ) -> ProteinObservation:
        """Build a ProteinObservation from a fixture dict and current step info.

        Sequences are taken from the ``sequence`` key for easy/medium fixtures
        and from ``sequence_with_mutation`` for hard variants.

        Args:
            fixture:   Raw fixture dict (easy/medium/hard entry).
            task_type: Current task type.

        Returns:
            ProteinObservation with all fields populated.

        Raises:
            Nothing.
        """
        sequence = fixture.get("sequence") or fixture.get("sequence_with_mutation", "")
        protein_id = fixture.get("protein_id") or fixture.get("variant_id", "unknown")
        variant_info = self._extract_variant_info(fixture, task_type)

        return ProteinObservation(
            protein_id=protein_id,
            sequence=sequence,
            task_type=task_type,
            task_description=self._build_task_description(task_type, fixture),
            available_tools=_AVAILABLE_TOOLS,
            step_number=self._state_manager.step_number,
            max_steps=MAX_STEPS_PER_EPISODE,
            variant_info=variant_info,
        )

    def _extract_variant_info(
        self, fixture: dict, task_type: TaskType
    ) -> VariantInfo | None:
        """Build a VariantInfo from a hard-tier fixture, or return None.

        Args:
            fixture:   Raw fixture dict.
            task_type: Current task type.

        Returns:
            VariantInfo if task_type is HARD and required fields are present;
            None otherwise.

        Raises:
            Nothing.
        """
        if task_type != TaskType.HARD:
            return None
        try:
            return VariantInfo(
                wildtype_aa=fixture["wildtype_aa"],
                mutant_aa=fixture["mutant_aa"],
                position=fixture["position"],
                variant_notation=fixture.get("variant_notation", ""),
            )
        except (KeyError, Exception):
            return None

    def _build_task_description(self, task_type: TaskType, fixture: dict) -> str:
        """Generate a clear plain-English task description for the agent.

        For EASY:   Protein family classification; lists the 10 family_choices.
        For MEDIUM: GO term prediction across 3 namespaces.
        For HARD:   Pathogenicity classification + disease association;
                    includes variant notation and gene name.

        Args:
            task_type: Task type enum.
            fixture:   Fixture dict for current protein/variant.

        Returns:
            Multi-line string task description.

        Raises:
            Nothing.
        """
        if task_type == TaskType.EASY:
            return self._describe_easy(fixture)
        if task_type == TaskType.MEDIUM:
            return self._describe_medium(fixture)
        return self._describe_hard(fixture)

    def _describe_easy(self, fixture: dict) -> str:
        """Build the easy-tier task description.

        Args:
            fixture: Easy protein fixture dict.

        Returns:
            Task description string.

        Raises:
            Nothing.
        """
        choices = fixture.get("family_choices", [])
        choices_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(choices))
        return (
            f"Classify the protein '{fixture.get('name', fixture.get('protein_id'))}' "
            f"into the correct protein family.\n\n"
            f"Choose exactly ONE of the following 10 protein families:\n{choices_str}\n\n"
            f"Submit your answer using SUBMIT_PREDICTION with predicted_family set to "
            f"the exact family name from the list above. You may call get_esm2_embedding "
            f"to obtain a 320-dimensional sequence embedding before deciding."
        )

    def _describe_medium(self, fixture: dict) -> str:
        """Build the medium-tier task description.

        Args:
            fixture: Medium protein fixture dict.

        Returns:
            Task description string.

        Raises:
            Nothing.
        """
        return (
            f"Predict the Gene Ontology (GO) terms for the protein "
            f"'{fixture.get('name', fixture.get('protein_id'))}'.\n\n"
            f"Provide a flat list of GO IDs (format: GO:XXXXXXX) covering all three "
            f"namespaces: molecular_function, biological_process, and cellular_component.\n\n"
            f"Submit your answer using SUBMIT_PREDICTION with predicted_go_terms set to "
            f"a list of GO term IDs (e.g. ['GO:0003677', 'GO:0006915']). "
            f"You may call get_esm2_embedding first to analyse the sequence."
        )

    def _describe_hard(self, fixture: dict) -> str:
        """Build the hard-tier task description.

        Args:
            fixture: Hard variant fixture dict.

        Returns:
            Task description string.

        Raises:
            Nothing.
        """
        gene = fixture.get("gene", "unknown gene")
        wt = fixture.get("wildtype_aa", "?")
        mut = fixture.get("mutant_aa", "?")
        pos = fixture.get("position", "?")
        notation = f"{wt}{pos}{mut}"
        return (
            f"Assess the clinical significance of the missense variant {notation} "
            f"in the {gene} gene.\n\n"
            f"1. Predict pathogenicity using the ClinVar five-tier scale:\n"
            f"   Pathogenic | Likely pathogenic | Variant of Uncertain Significance "
            f"| Likely benign | Benign\n\n"
            f"2. List associated diseases or syndromes.\n\n"
            f"Submit using SUBMIT_PREDICTION with:\n"
            f"  - predicted_pathogenicity: one of the five ClinVar tiers\n"
            f"  - predicted_diseases: list of disease/syndrome names\n\n"
            f"You may call get_esm2_embedding on the mutant sequence before deciding."
        )

    def _handle_tool_call(self, action: ProteinAction) -> dict:
        """Execute a tool call and return a JSON-serialisable result dict.

        Currently supported tools:
          get_esm2_embedding — calls self._embedder.embed_as_list(sequence).
            Returns {"embedding": list[float], "dim": 320, "model": ESM2_MODEL_ID}

        Args:
            action: Action with action_type=CALL_TOOL and tool_name set.

        Returns:
            Tool result dict (JSON-serialisable).

        Raises:
            ValueError: If tool_name is not "get_esm2_embedding".
            ValueError: If required tool_args are missing.
        """
        tool_name = action.tool_name or ""
        if tool_name not in _KNOWN_TOOLS:
            raise ValueError(
                f"Unknown tool '{tool_name}'. Supported tools: {list(_KNOWN_TOOLS)}"
            )
        return self._run_esm2_embedding(action.tool_args or {})

    def _run_esm2_embedding(self, tool_args: dict) -> dict:
        """Run the get_esm2_embedding tool and return the result dict.

        Args:
            tool_args: Dict expected to contain key "sequence".

        Returns:
            Dict with keys "embedding" (list[float]), "dim" (int), "model" (str).

        Raises:
            ValueError: If "sequence" key is missing from tool_args.
        """
        sequence = tool_args.get("sequence")
        if not sequence:
            raise ValueError(
                "get_esm2_embedding requires tool_args['sequence'] to be set."
            )
        embedding = self._embedder.embed_as_list(sequence)
        return {"embedding": embedding, "dim": len(embedding), "model": ESM2_MODEL_ID}
