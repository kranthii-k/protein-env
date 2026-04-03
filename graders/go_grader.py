"""
graders/go_grader.py — Reward grader for medium-tier GO term prediction.

Scores are computed as macro-averaged F1 across the three Gene Ontology
namespaces (molecular_function, biological_process, cellular_component).

All functions in this module are pure: no I/O, no state, no side effects.
"""

from __future__ import annotations

import re

# No constants or models needed — namespace keys are plain string literals.

# Canonical GO namespace keys expected in ground-truth dicts
_NAMESPACES: tuple[str, ...] = (
    "molecular_function",
    "biological_process",
    "cellular_component",
)

_GO_ID_PATTERN: re.Pattern[str] = re.compile(r"^GO:\d{7}$")


def validate_go_id(go_id: str) -> bool:
    """Return True if go_id matches the canonical format GO:XXXXXXX (7 digits).

    Args:
        go_id: String to validate.

    Returns:
        bool — True if the string is a well-formed GO term identifier.

    Raises:
        Nothing.
    """
    return bool(_GO_ID_PATTERN.match(go_id))


def _build_namespace_predicted(
    valid_predicted: set[str],
    ground_truth: dict[str, list[str]],
) -> dict[str, set[str]]:
    """Assign each valid predicted GO ID to the namespace(s) where it appears in truth.

    IDs not found in any namespace truth are assigned to all namespaces so
    they still penalise precision globally. When an ID appears in multiple
    namespace truths it is counted in each.

    Args:
        valid_predicted: Set of syntactically valid predicted GO IDs.
        ground_truth:    Dict mapping namespace name to list of true GO IDs.

    Returns:
        Dict mapping namespace name to the per-namespace predicted set.

    Raises:
        Nothing.
    """
    # Build reverse lookup: GO ID → set of namespaces it belongs to
    id_to_namespaces: dict[str, set[str]] = {}
    for ns in _NAMESPACES:
        for go_id in ground_truth.get(ns, []):
            id_to_namespaces.setdefault(go_id, set()).add(ns)

    ns_predicted: dict[str, set[str]] = {ns: set() for ns in _NAMESPACES}
    unassigned: set[str] = set()

    for go_id in valid_predicted:
        namespaces = id_to_namespaces.get(go_id)
        if namespaces:
            for ns in namespaces:
                ns_predicted[ns].add(go_id)
        else:
            unassigned.add(go_id)

    # Unassigned IDs count against precision in every namespace
    for ns in _NAMESPACES:
        ns_predicted[ns] |= unassigned

    return ns_predicted


def _f1_for_namespace(
    ns_predicted: set[str],
    namespace_truth: set[str],
) -> float:
    """Compute namespace-scoped F1 using the per-namespace predicted set.

    Args:
        ns_predicted:    Predicted GO IDs assigned to this namespace.
        namespace_truth: True GO IDs for this namespace.

    Returns:
        float in [0.0, 1.0] — F1 score for this namespace.

    Raises:
        Nothing.
    """
    if not namespace_truth:
        return 0.0

    if not ns_predicted:
        return 0.0

    true_positives = len(ns_predicted & namespace_truth)
    precision = true_positives / len(ns_predicted)
    recall = true_positives / len(namespace_truth)

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


def grade_go_terms(
    predicted: list[str],
    ground_truth: dict[str, list[str]],
) -> tuple[float, dict[str, float]]:
    """Score GO term predictions using macro-averaged F1 across 3 namespaces.

    Ground truth has keys: "molecular_function", "biological_process",
    "cellular_component". Predicted is a flat list of GO IDs (namespace-agnostic
    — the agent doesn't need to specify namespace).

    Scoring per namespace:
      precision = |predicted ∩ true_namespace| / |predicted|  (0 if predicted empty)
      recall    = |predicted ∩ true_namespace| / |true_namespace|  (0 if true empty)
      f1        = 2*p*r / (p+r) if p+r > 0 else 0.0

    Macro F1 = mean of f1 scores across all 3 namespaces.

    Invalid GO IDs (failing validate_go_id) in predicted are silently ignored
    before scoring — they do not inflate the precision denominator.

    Args:
        predicted:    Flat list of GO ID strings from agent.
        ground_truth: Dict mapping namespace name to list of GO IDs.

    Returns:
        Tuple of (macro_f1: float, per_namespace: dict[str, float]).
        macro_f1 is in [0.0, 1.0].
        per_namespace maps each namespace name to its individual F1 score.

    Raises:
        Nothing.
    """
    valid_predicted: set[str] = {g for g in predicted if validate_go_id(g)}

    ns_predicted_map = _build_namespace_predicted(valid_predicted, ground_truth)

    per_namespace: dict[str, float] = {}
    for ns in _NAMESPACES:
        truth_set: set[str] = set(ground_truth.get(ns, []))
        per_namespace[ns] = _f1_for_namespace(ns_predicted_map[ns], truth_set)

    macro_f1 = sum(per_namespace.values()) / len(_NAMESPACES)
    return macro_f1, per_namespace
