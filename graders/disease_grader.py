"""
graders/disease_grader.py — Reward grader for hard-tier disease variant association.

Scores variant predictions on two axes:
  1. Pathogenicity classification (ClinVar five-tier match with tier proximity credit)
  2. Disease name overlap (Jaccard similarity, case-insensitive)

A flip penalty is applied when the prediction crosses the clinical significance
boundary (pathogenic ↔ benign).

All functions are pure: no I/O, no state, no side effects.
"""

from __future__ import annotations

try:
    from constants import (
        HARD_DISEASE_WEIGHT,
        HARD_FLIP_PENALTY,
        HARD_PATHOGENICITY_WEIGHT,
    )
    from models import Pathogenicity
except ImportError:
    from protein_env.constants import (
        HARD_DISEASE_WEIGHT,
        HARD_FLIP_PENALTY,
        HARD_PATHOGENICITY_WEIGHT,
    )
    from protein_env.models import Pathogenicity

# ── Severity tiers ────────────────────────────────────────────────────────────
# Pathogenicity values grouped into 3 clinical significance tiers.
_PATHOGENIC_TIER: frozenset[str] = frozenset(
    {Pathogenicity.PATHOGENIC.value, Pathogenicity.LIKELY_PATHOGENIC.value}
)
_UNCERTAIN_TIER: frozenset[str] = frozenset({Pathogenicity.VUS.value})
_BENIGN_TIER: frozenset[str] = frozenset(
    {Pathogenicity.LIKELY_BENIGN.value, Pathogenicity.BENIGN.value}
)

_TIER_MAP: dict[str, int] = {
    **{v: 1 for v in _PATHOGENIC_TIER},
    **{v: 2 for v in _UNCERTAIN_TIER},
    **{v: 3 for v in _BENIGN_TIER},
}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets (case-insensitive).

    jaccard = |A ∩ B| / |A ∪ B|
    Returns 0.0 if both sets are empty.

    Args:
        set_a: First set of strings.
        set_b: Second set of strings.

    Returns:
        float in [0.0, 1.0]

    Raises:
        Nothing.
    """
    norm_a = {s.strip().lower() for s in set_a}
    norm_b = {s.strip().lower() for s in set_b}

    union = norm_a | norm_b
    if not union:
        return 0.0

    intersection = norm_a & norm_b
    return len(intersection) / len(union)


def _score_pathogenicity(
    predicted_value: str,
    ground_truth_value: str,
) -> float:
    """Compute the pathogenicity sub-score (before flip penalty).

    Args:
        predicted_value:    .value of the Pathogenicity enum from the agent.
        ground_truth_value: Raw pathogenicity string from the fixture.

    Returns:
        float — 0.5 for exact match, 0.25 for same tier, 0.0 otherwise.

    Raises:
        Nothing.
    """
    if predicted_value == ground_truth_value:
        return HARD_PATHOGENICITY_WEIGHT

    pred_tier = _TIER_MAP.get(predicted_value)
    truth_tier = _TIER_MAP.get(ground_truth_value)

    if pred_tier is not None and truth_tier is not None and pred_tier == truth_tier:
        return HARD_PATHOGENICITY_WEIGHT / 2.0

    return 0.0


def _compute_flip_penalty(
    predicted_value: str,
    ground_truth_value: str,
) -> float:
    """Return HARD_FLIP_PENALTY if prediction crosses the pathogenic/benign boundary.

    A "flip" occurs when the prediction is in the pathogenic tier and the truth
    is in the benign tier, or vice versa. VUS is never considered a flip.

    Args:
        predicted_value:    .value of the Pathogenicity enum from the agent.
        ground_truth_value: Raw pathogenicity string from the fixture.

    Returns:
        float — HARD_FLIP_PENALTY (<0) if a flip occurred, 0.0 otherwise.

    Raises:
        Nothing.
    """
    pred_in_path = predicted_value in _PATHOGENIC_TIER
    pred_in_benign = predicted_value in _BENIGN_TIER
    truth_in_path = ground_truth_value in _PATHOGENIC_TIER
    truth_in_benign = ground_truth_value in _BENIGN_TIER

    if (pred_in_path and truth_in_benign) or (pred_in_benign and truth_in_path):
        return HARD_FLIP_PENALTY

    return 0.0


def grade_disease(
    predicted_pathogenicity: Pathogenicity,
    predicted_diseases: list[str],
    ground_truth_pathogenicity: str,
    ground_truth_diseases: list[str],
) -> tuple[float, dict]:
    """Score a disease variant prediction.

    Scoring:
      pathogenicity_score:
        - 0.5  exact match (predicted.value == ground_truth_pathogenicity)
        - 0.25 same severity tier
        - 0.0  otherwise

      flip_penalty: HARD_FLIP_PENALTY (-0.1) if prediction crosses the
        pathogenic ↔ benign clinical boundary; 0.0 otherwise.

      disease_score:
        = jaccard_similarity(predicted_diseases, ground_truth_diseases)
          * HARD_DISEASE_WEIGHT

      total = pathogenicity_score + disease_score + flip_penalty
      (NOT clamped — can be negative due to flip penalty.)

    Args:
        predicted_pathogenicity:  Agent's Pathogenicity enum value.
        predicted_diseases:       List of disease names predicted by agent.
        ground_truth_pathogenicity: String pathogenicity from fixture.
        ground_truth_diseases:    List of true disease names from fixture.

    Returns:
        Tuple of (total_score: float, breakdown: dict) where breakdown has keys:
        pathogenicity_score, disease_score, flip_penalty, total.

    Raises:
        Nothing.
    """
    pred_val = predicted_pathogenicity.value

    pathogenicity_score = _score_pathogenicity(pred_val, ground_truth_pathogenicity)
    flip_penalty = _compute_flip_penalty(pred_val, ground_truth_pathogenicity)
    disease_score = (
        jaccard_similarity(set(predicted_diseases), set(ground_truth_diseases))
        * HARD_DISEASE_WEIGHT
    )

    total = pathogenicity_score + disease_score + flip_penalty

    breakdown: dict = {
        "pathogenicity_score": pathogenicity_score,
        "disease_score": disease_score,
        "flip_penalty": flip_penalty,
        "total": total,
    }
    return total, breakdown
