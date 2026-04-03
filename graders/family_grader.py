"""
graders/family_grader.py — Reward grader for easy-tier protein family classification.

All functions in this module are pure: they perform no I/O, hold no state,
and produce no side effects. Deterministic output is guaranteed for identical
inputs.
"""

from __future__ import annotations

try:
    from constants import EASY_EXACT_MATCH_REWARD, EASY_SUPERFAMILY_REWARD
except ImportError:
    from protein_env.constants import EASY_EXACT_MATCH_REWARD, EASY_SUPERFAMILY_REWARD


def grade_family(predicted: str, ground_truth: str) -> float:
    """Score a protein family classification prediction.

    Scoring tiers (evaluated in order, first match wins):
      - 1.0  if predicted == ground_truth (case-insensitive, stripped)
      - 0.3  if ground_truth is contained within predicted OR predicted is
               contained within ground_truth (superfamily partial match —
               e.g. "Insulin" inside "Insulin family")
      - 0.0  otherwise

    Args:
        predicted:    Agent's predicted family name.
        ground_truth: Correct family name from fixture.

    Returns:
        float in {0.0, 0.3, 1.0}

    Raises:
        Nothing.
    """
    pred_norm = predicted.strip().lower()
    truth_norm = ground_truth.strip().lower()

    if not pred_norm:
        return 0.0

    if pred_norm == truth_norm:
        return EASY_EXACT_MATCH_REWARD

    if truth_norm in pred_norm or pred_norm in truth_norm:
        return EASY_SUPERFAMILY_REWARD

    return 0.0


def is_valid_family_choice(predicted: str, valid_choices: list[str]) -> bool:
    """Return True if predicted is one of the valid family choices.

    Comparison is case-insensitive and strips leading/trailing whitespace.
    Used to detect hallucinated family names not in the options presented
    to the agent within the observation.

    Args:
        predicted:     Agent's predicted family name.
        valid_choices: The 10 choices presented in the observation.

    Returns:
        bool — True if predicted (normalised) is in valid_choices (normalised).

    Raises:
        Nothing.
    """
    pred_norm = predicted.strip().lower()
    normalised_choices = {c.strip().lower() for c in valid_choices}
    return pred_norm in normalised_choices
