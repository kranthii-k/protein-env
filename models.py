"""
Pydantic v2 data models for the ProteinEnv OpenEnv environment.

All public types used for inter-component communication (API requests, grader
inputs, step results) are defined here. No raw dicts may cross function
boundaries — callers must use these models explicitly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    """Difficulty tier for an environment episode."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ActionType(str, Enum):
    """Discriminator for agent actions sent to the environment step endpoint."""

    SUBMIT_PREDICTION = "submit_prediction"
    CALL_TOOL = "call_tool"


class Pathogenicity(str, Enum):
    """ClinVar five-tier pathogenicity classification for missense variants."""

    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely pathogenic"
    VUS = "Variant of Uncertain Significance"
    LIKELY_BENIGN = "Likely benign"
    BENIGN = "Benign"


class GONamespace(str, Enum):
    """Gene Ontology top-level namespace identifiers."""

    MOLECULAR_FUNCTION = "molecular_function"
    BIOLOGICAL_PROCESS = "biological_process"
    CELLULAR_COMPONENT = "cellular_component"


# ── Domain sub-models ─────────────────────────────────────────────────────────


class VariantInfo(BaseModel):
    """Structured representation of a missense amino-acid variant.

    Args:
        wildtype_aa: Single-letter code of the reference amino acid (e.g. "R").
        mutant_aa:   Single-letter code of the alternate amino acid (e.g. "H").
        position:    1-indexed position of the substitution in the protein sequence.
        variant_notation: HGVS-style compact notation (e.g. "R175H").
            Auto-populated from the other three fields when left empty.

    Returns:
        A validated VariantInfo instance with variant_notation populated.

    Raises:
        ValidationError: If wildtype_aa, mutant_aa, or position are missing.
    """

    wildtype_aa: str
    mutant_aa: str
    position: int
    variant_notation: str = ""

    @model_validator(mode="after")
    def build_notation(self) -> "VariantInfo":
        """Auto-build HGVS notation when not supplied by the caller.

        Args:
            None (operates on the model instance itself).

        Returns:
            The updated VariantInfo instance.

        Raises:
            Nothing — notation construction is unconditionally safe.
        """
        if not self.variant_notation:
            self.variant_notation = (
                f"{self.wildtype_aa}{self.position}{self.mutant_aa}"
            )
        return self


# ── Observation / action models ───────────────────────────────────────────────


class ProteinObservation(BaseModel):
    """Full environment observation returned to the agent after each step.

    Args:
        protein_id:       UniProt accession (or synthetic identifier).
        sequence:         Amino-acid sequence in single-letter code.
        sequence_length:  Auto-computed from ``sequence`` — callers may omit.
        task_type:        Difficulty tier for this episode.
        task_description: Natural-language description of the task objective.
        available_tools:  List of tool names the agent may invoke.
        step_number:      Zero-indexed current step within the episode.
        max_steps:        Maximum allowed steps before forced termination.
        variant_info:     Populated only for hard-tier (variant) tasks.

    Returns:
        A validated ProteinObservation with sequence_length correctly set.

    Raises:
        ValidationError: If required fields are missing or malformed.
    """

    protein_id: str
    sequence: str
    sequence_length: int = 0
    task_type: TaskType
    task_description: str
    available_tools: list[str]
    step_number: int
    max_steps: int
    variant_info: VariantInfo | None = None

    @model_validator(mode="after")
    def set_sequence_length(self) -> "ProteinObservation":
        """Recompute sequence_length from the actual sequence string.

        Args:
            None (operates on the model instance itself).

        Returns:
            The updated ProteinObservation instance.

        Raises:
            Nothing — len() on a string is always safe.
        """
        self.sequence_length = len(self.sequence)
        return self


class ProteinAction(BaseModel):
    """Agent action submitted to the environment step endpoint.

    Exactly one of two modes must be used:
      - ``SUBMIT_PREDICTION``: Agent commits at least one prediction field.
      - ``CALL_TOOL``: Agent invokes a named environment tool.

    Args:
        action_type:              Discriminator for the action mode.
        predicted_family:         Protein family string (easy tasks).
        predicted_go_terms:       List of GO IDs (medium tasks).
        predicted_pathogenicity:  ClinVar classification (hard tasks).
        predicted_diseases:       List of associated disease names (hard tasks).
        tool_name:                Tool to invoke; required for CALL_TOOL.
        tool_args:                Keyword args forwarded to the tool.
        reasoning:                Optional chain-of-thought from the agent.

    Returns:
        A validated ProteinAction instance.

    Raises:
        ValueError: If CALL_TOOL is missing tool_name, or SUBMIT_PREDICTION
            has no prediction field set.
    """

    action_type: ActionType
    predicted_family: str | None = None
    predicted_go_terms: list[str] | None = None
    predicted_pathogenicity: Pathogenicity | None = None
    predicted_diseases: list[str] | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    reasoning: str | None = None

    @model_validator(mode="after")
    def validate_action_completeness(self) -> "ProteinAction":
        """Enforce that tool calls have a tool_name and predictions have content.

        Args:
            None (operates on the model instance itself).

        Returns:
            The validated ProteinAction instance.

        Raises:
            ValueError: If action invariants are violated.
        """
        if self.action_type == ActionType.CALL_TOOL and not self.tool_name:
            raise ValueError("CALL_TOOL action requires tool_name")

        if self.action_type == ActionType.SUBMIT_PREDICTION:
            has_prediction = any(
                [
                    self.predicted_family,
                    self.predicted_go_terms,
                    self.predicted_pathogenicity,
                ]
            )
            if not has_prediction:
                raise ValueError(
                    "SUBMIT_PREDICTION requires at least one prediction field"
                )
        return self


# ── Reward / result models ────────────────────────────────────────────────────


class RewardBreakdown(BaseModel):
    """Itemised reward signal returned alongside every step result.

    All penalty fields should be passed as non-positive values; the validator
    simply sums them into ``total``.

    Args:
        base_score:            Raw correctness score before penalties.
        wasted_step_penalty:   Applied when step is taken after submission.
        redundant_tool_penalty: Applied when a tool is called with duplicate args.
        flip_penalty:          Applied when a prior prediction is reversed.
        total:                 Auto-computed sum (callers may omit).

    Returns:
        A RewardBreakdown with ``total`` correctly populated.

    Raises:
        Nothing — arithmetic on floats is always safe.
    """

    base_score: float
    wasted_step_penalty: float = 0.0
    redundant_tool_penalty: float = 0.0
    flip_penalty: float = 0.0
    total: float = 0.0

    @model_validator(mode="after")
    def compute_total(self) -> "RewardBreakdown":
        """Recompute total as the sum of all component scores.

        Args:
            None (operates on the model instance itself).

        Returns:
            The updated RewardBreakdown instance.

        Raises:
            Nothing — addition of floats is always safe.
        """
        self.total = (
            self.base_score
            + self.wasted_step_penalty
            + self.redundant_tool_penalty
            + self.flip_penalty
        )
        return self


class StepInfo(BaseModel):
    """Auxiliary metadata attached to every StepResult.

    Args:
        reward_breakdown: Itemised reward components for the step.
        tool_result:      Raw output dict from the invoked tool, if any.
        done_reason:      Human-readable explanation for episode termination.
        grader_details:   Debug string from the grader (e.g. matched GO terms).

    Returns:
        A validated StepInfo instance.

    Raises:
        ValidationError: If reward_breakdown is missing.
    """

    reward_breakdown: RewardBreakdown
    tool_result: dict | None = None
    done_reason: str | None = None
    grader_details: str | None = None


class StepResult(BaseModel):
    """Complete environment response to a single agent step.

    Args:
        observation: Next observation the agent should act on.
        reward:      Scalar reward for this step (mirrors info.reward_breakdown.total).
        done:        Whether the episode has terminated.
        info:        Full itemised breakdown and auxiliary metadata.

    Returns:
        A validated StepResult instance.

    Raises:
        ValidationError: If observation or info are missing.
    """

    observation: ProteinObservation
    reward: float
    done: bool
    info: StepInfo


class ProteinState(BaseModel):
    """Mutable runtime state for a single environment episode.

    Stored server-side and updated after every step. NOT exposed directly to
    the agent (agents receive ProteinObservation, not ProteinState).

    Args:
        episode_id:         UUID string uniquely identifying this episode.
        task_type:          Difficulty tier selected at episode start.
        current_protein_id: UniProt accession of the protein being evaluated.
        step_number:        Zero-indexed step counter (incremented after each step).
        cumulative_reward:  Running sum of per-step rewards.
        submitted:          True once a SUBMIT_PREDICTION action is processed.
        tool_calls_made:    Count of CALL_TOOL actions processed so far.
        episode_history:    Chronological list of raw step dicts (for logging).

    Returns:
        A validated ProteinState instance.

    Raises:
        ValidationError: If required identifiers are missing.
    """

    episode_id: str
    task_type: TaskType
    current_protein_id: str
    step_number: int
    cumulative_reward: float
    submitted: bool
    tool_calls_made: int
    episode_history: list[dict] = []
