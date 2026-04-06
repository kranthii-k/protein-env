"""
Module-level constants for the ProteinEnv OpenEnv environment.

All numeric limits, identifiers, reward weights, and penalty coefficients
used across the codebase are defined here to eliminate magic strings/numbers.
"""

# ── Model identifiers ─────────────────────────────────────────────────────────
ESM2_MODEL_ID: str = "facebook/esm2_t6_8M_UR50D"

# ── Sequence / model capacity limits ─────────────────────────────────────────
MAX_SEQUENCE_LENGTH: int = 1022          # ESM2 hard limit (tokens, excl. [CLS]/[EOS])
EMBEDDING_DIM: int = 320                 # Output dim for esm2_t6_8M

# ── Episode configuration ─────────────────────────────────────────────────────
MAX_STEPS_PER_EPISODE: int = 10

# ── Environment metadata ──────────────────────────────────────────────────────
ENV_NAME: str = "protein-env"
ENV_VERSION: str = "0.1.0"
SPEC_VERSION: int = 1
PORT: int = 7860

# ── Reward weights — Easy (family classification) ─────────────────────────────
EASY_EXACT_MATCH_REWARD: float = 1.0
EASY_SUPERFAMILY_REWARD: float = 0.3

# ── Reward weights — Medium (GO term prediction) ──────────────────────────────
MEDIUM_MAX_REWARD: float = 1.0

# ── Reward weights — Hard (disease variant association) ───────────────────────
HARD_PATHOGENICITY_WEIGHT: float = 0.5
HARD_DISEASE_WEIGHT: float = 0.5

# ── Penalty coefficients (applied as negative values to total reward) ─────────
HARD_FLIP_PENALTY: float = -0.1          # Changing a prior prediction
WASTED_STEP_PENALTY: float = -0.05      # Step taken after prediction was submitted
REDUNDANT_TOOL_PENALTY: float = -0.01   # Calling the same tool with identical args
