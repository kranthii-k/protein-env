"""
core/reward_calculator.py — Central reward computation for ProteinEnv.

Dispatches to the appropriate grader based on TaskType and assembles a
fully-populated RewardBreakdown. All reward logic is deterministic and
stateless — the caller is responsible for persisting ProteinState.
"""

from __future__ import annotations

import json

try:
    from constants import (
        REDUNDANT_TOOL_PENALTY,
        WASTED_STEP_PENALTY,
    )
    from models import ActionType, ProteinAction, RewardBreakdown, TaskType
    from graders.family_grader import grade_family
    from graders.go_grader import grade_go_terms
    from graders.disease_grader import grade_disease
    from models import Pathogenicity
except ImportError:
    from protein_env.constants import (
        REDUNDANT_TOOL_PENALTY,
        WASTED_STEP_PENALTY,
    )
    from protein_env.models import (
        ActionType,
        Pathogenicity,
        ProteinAction,
        RewardBreakdown,
        TaskType,
    )
    from protein_env.graders.family_grader import grade_family
    from protein_env.graders.go_grader import grade_go_terms
    from protein_env.graders.disease_grader import grade_disease

# Step threshold beyond which a submit is considered "late" and penalised.
_LATE_SUBMIT_STEP_THRESHOLD: int = 3


def is_redundant_tool_call(
    tool_name: str,
    tool_args: dict,
    tool_call_history: list[dict],
) -> bool:
    """Return True if an identical tool_name + tool_args call exists in history.

    Comparison is performed by serialising tool_args to a canonically sorted
    JSON string so that dict key ordering does not affect the result.

    Args:
        tool_name:         Name of the tool being called this step.
        tool_args:         Arguments dict for this call.
        tool_call_history: Prior tool calls this episode (dicts with
                           keys ``tool_name`` and ``tool_args``).

    Returns:
        bool — True if the exact same call was made previously.

    Raises:
        Nothing.
    """
    canonical_args = json.dumps(tool_args, sort_keys=True)
    for prior in tool_call_history:
        if prior.get("tool_name") != tool_name:
            continue
        if json.dumps(prior.get("tool_args", {}), sort_keys=True) == canonical_args:
            return True
    return False


def _reward_for_tool_call(
    action: ProteinAction,
    tool_call_history: list[dict],
) -> RewardBreakdown:
    """Build RewardBreakdown for a CALL_TOOL action.

    Base score is always 0.0. Applies REDUNDANT_TOOL_PENALTY if the same
    tool/args combo was already seen in history.

    Args:
        action:            The CALL_TOOL action.
        tool_call_history: Prior tool call records.

    Returns:
        RewardBreakdown with base_score=0.0 and optional redundant_tool_penalty.

    Raises:
        Nothing.
    """
    redundant_penalty = 0.0
    if action.tool_name and action.tool_args is not None:
        if is_redundant_tool_call(
            action.tool_name, action.tool_args, tool_call_history
        ):
            redundant_penalty = REDUNDANT_TOOL_PENALTY

    return RewardBreakdown(
        base_score=0.0,
        redundant_tool_penalty=redundant_penalty,
    )


def _base_score_easy(action: ProteinAction, ground_truth: dict) -> float:
    """Compute base score for an easy-tier SUBMIT_PREDICTION.

    Args:
        action:       SUBMIT_PREDICTION action with predicted_family set.
        ground_truth: Fixture dict containing the ``family`` key.

    Returns:
        float score from grade_family.

    Raises:
        Nothing.
    """
    predicted = action.predicted_family or ""
    return grade_family(predicted, ground_truth.get("family", ""))


def _base_score_medium(action: ProteinAction, ground_truth: dict) -> float:
    """Compute base score for a medium-tier SUBMIT_PREDICTION.

    Args:
        action:       SUBMIT_PREDICTION action with predicted_go_terms set.
        ground_truth: Fixture dict containing the ``go_terms`` key.

    Returns:
        float macro-F1 score from grade_go_terms, in [0.0, 1.0].

    Raises:
        Nothing.
    """
    predicted = action.predicted_go_terms or []
    truth_go = ground_truth.get("go_terms", {})
    macro_f1, _ = grade_go_terms(predicted, truth_go)
    return macro_f1


def _base_score_hard(action: ProteinAction, ground_truth: dict) -> float:
    """Compute base score for a hard-tier SUBMIT_PREDICTION.

    Args:
        action:       SUBMIT_PREDICTION action with predicted_pathogenicity
                      and optionally predicted_diseases set.
        ground_truth: Fixture dict containing ``pathogenicity`` and
                      ``associated_diseases`` keys.

    Returns:
        float total score from grade_disease (may be negative).

    Raises:
        Nothing.
    """
    pathogenicity = action.predicted_pathogenicity or Pathogenicity.VUS
    diseases = action.predicted_diseases or []
    truth_path = ground_truth.get("pathogenicity", "")
    truth_diseases = ground_truth.get("associated_diseases", [])
    total, _ = grade_disease(pathogenicity, diseases, truth_path, truth_diseases)
    return total


def _wasted_step_penalty(step_number: int, already_submitted: bool) -> float:
    """Return WASTED_STEP_PENALTY if the agent submitted late without prior submission.

    Penalty is applied when the agent submits for the first time after step 3,
    signalling they used more steps than necessary before committing.

    Args:
        step_number:       Current 0-indexed step number.
        already_submitted: True if a prior SUBMIT_PREDICTION was processed.

    Returns:
        float — WASTED_STEP_PENALTY (<0) or 0.0.

    Raises:
        Nothing.
    """
    if not already_submitted and step_number > _LATE_SUBMIT_STEP_THRESHOLD:
        return WASTED_STEP_PENALTY
    return 0.0


def _reward_for_submission(
    action: ProteinAction,
    ground_truth: dict,
    task_type: TaskType,
    step_number: int,
    already_submitted: bool,
) -> RewardBreakdown:
    """Build RewardBreakdown for a SUBMIT_PREDICTION action.

    Args:
        action:            The SUBMIT_PREDICTION action.
        ground_truth:      Raw fixture dict for the current protein/variant.
        task_type:         TaskType enum (EASY, MEDIUM, HARD).
        step_number:       Current 0-indexed step number.
        already_submitted: True if a prior prediction was submitted.

    Returns:
        Fully populated RewardBreakdown.

    Raises:
        Nothing.
    """
    if task_type == TaskType.EASY:
        base = _base_score_easy(action, ground_truth)
    elif task_type == TaskType.MEDIUM:
        base = _base_score_medium(action, ground_truth)
    else:
        base = _base_score_hard(action, ground_truth)

    wasted = _wasted_step_penalty(step_number, already_submitted)

    return RewardBreakdown(
        base_score=base,
        wasted_step_penalty=wasted,
    )


def calculate_reward(
    action: ProteinAction,
    ground_truth: dict,
    task_type: TaskType,
    step_number: int,
    already_submitted: bool,
    tool_call_history: list[dict],
) -> RewardBreakdown:
    """Compute reward for a single agent action.

    Dispatch rules:
    - CALL_TOOL:          base_score = 0.0; apply redundant_tool_penalty
                          if the same tool_name + tool_args was used before.
    - SUBMIT_PREDICTION: compute base_score via the task-appropriate grader.
                          Apply wasted_step_penalty if step_number > 3 and
                          this is the agent's first submission.

    RewardBreakdown.total is computed automatically by its model_validator.

    Args:
        action:            The agent's action this step.
        ground_truth:      Raw fixture dict for the current protein/variant.
        task_type:         TaskType enum (EASY, MEDIUM, HARD).
        step_number:       Current 0-indexed step number.
        already_submitted: True if agent already submitted a prediction this episode.
        tool_call_history: List of dicts with keys tool_name, tool_args from prior steps.

    Returns:
        RewardBreakdown with base_score, penalties, and total populated.

    Raises:
        Nothing (graders are pure and do not raise).
    """
    if action.action_type == ActionType.CALL_TOOL:
        return _reward_for_tool_call(action, tool_call_history)

    return _reward_for_submission(
        action, ground_truth, task_type, step_number, already_submitted
    )
