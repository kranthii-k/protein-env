"""
Shared pytest fixtures for the ProteinEnv test suite.

All fixtures are function-scoped (the default) unless otherwise noted.
Fixtures that load JSON files do so relative to this file's directory,
making the test suite location-independent.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

try:
    from models import ActionType, Pathogenicity, ProteinAction
except ImportError:
    from protein_env.models import ActionType, Pathogenicity, ProteinAction

# ── Fixture file paths ────────────────────────────────────────────────────────

_FIXTURES_DIR = pathlib.Path(__file__).parent.parent / "data" / "fixtures"
_EASY_JSON = _FIXTURES_DIR / "easy_proteins.json"
_MEDIUM_JSON = _FIXTURES_DIR / "medium_proteins.json"
_HARD_JSON = _FIXTURES_DIR / "hard_variants.json"


# ── JSON fixture loaders ──────────────────────────────────────────────────────


@pytest.fixture
def easy_protein_fixture() -> dict:
    """Return the first entry from easy_proteins.json.

    Args:
        None.

    Returns:
        A dict representing one easy-tier protein fixture record.

    Raises:
        FileNotFoundError: If easy_proteins.json is missing.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with _EASY_JSON.open(encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)
    return records[0]


@pytest.fixture
def medium_protein_fixture() -> dict:
    """Return the first entry from medium_proteins.json.

    Args:
        None.

    Returns:
        A dict representing one medium-tier protein fixture record.

    Raises:
        FileNotFoundError: If medium_proteins.json is missing.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with _MEDIUM_JSON.open(encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)
    return records[0]


@pytest.fixture
def hard_variant_fixture() -> dict:
    """Return the first entry from hard_variants.json.

    Args:
        None.

    Returns:
        A dict representing one hard-tier variant fixture record.

    Raises:
        FileNotFoundError: If hard_variants.json is missing.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with _HARD_JSON.open(encoding="utf-8") as fh:
        records: list[dict] = json.load(fh)
    return records[0]


# ── Synthetic model fixtures ──────────────────────────────────────────────────


@pytest.fixture
def mock_esm2_embedding() -> np.ndarray:
    """Return a deterministic 320-dimensional fake ESM2 embedding.

    The embedding is a constant vector (all elements equal to 0.1) so that
    tests involving embedding arithmetic produce predictable results without
    requiring a GPU or network access.

    Args:
        None.

    Returns:
        A numpy ndarray of shape (320,) with dtype float32.

    Raises:
        Nothing.
    """
    return np.ones(320, dtype=np.float32) * 0.1


@pytest.fixture
def valid_easy_action() -> ProteinAction:
    """Return a valid SUBMIT_PREDICTION action targeting an easy task.

    The predicted family is set to the well-known label "Insulin family"
    (matches the first easy fixture entry) so reward-grader tests can
    exercise the exact-match path without loading fixture data themselves.

    Args:
        None.

    Returns:
        A fully-validated ProteinAction instance with action_type=SUBMIT_PREDICTION.

    Raises:
        ValidationError: Should never raise; values are hard-coded to be valid.
    """
    return ProteinAction(
        action_type=ActionType.SUBMIT_PREDICTION,
        predicted_family="Insulin family",
        reasoning="The sequence contains the conserved GIVEQCC motif of the insulin family.",
    )


@pytest.fixture
def valid_tool_call_action() -> ProteinAction:
    """Return a valid CALL_TOOL action invoking get_esm2_embedding.

    Args:
        None.

    Returns:
        A fully-validated ProteinAction instance with action_type=CALL_TOOL.

    Raises:
        ValidationError: Should never raise; values are hard-coded to be valid.
    """
    return ProteinAction(
        action_type=ActionType.CALL_TOOL,
        tool_name="get_esm2_embedding",
        tool_args={
            "sequence": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"
        },
        reasoning="Requesting ESM2 embedding to understand sequence-level features before predicting family.",
    )
