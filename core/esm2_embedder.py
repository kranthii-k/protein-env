"""
core/esm2_embedder.py — ESM2 protein sequence embedder.

Wraps facebook/esm2_t6_8M_UR50D via HuggingFace Transformers.
Model weights are lazy-loaded on the first call to embed() to avoid
paying the download cost at server startup in environments that may
never call the tool.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from transformers import EsmModel, EsmTokenizer

try:
    from constants import EMBEDDING_DIM, ESM2_MODEL_ID, MAX_SEQUENCE_LENGTH
except ImportError:
    from protein_env.constants import EMBEDDING_DIM, ESM2_MODEL_ID, MAX_SEQUENCE_LENGTH

logger = logging.getLogger(__name__)

# Standard 20 AAs + accepted ambiguous codes
_VALID_AA_PATTERN: re.Pattern[str] = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBJOUXZ]+$")


class ESM2Embedder:
    """Wraps facebook/esm2_t6_8M_UR50D for protein sequence embedding.

    Lazy-loads model on first call to embed(). Thread-unsafe — one instance
    per environment session. CPU-only by design (hackathon infra constraint).
    """

    def __init__(self) -> None:
        """Initialize embedder without loading model weights."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded: bool = False

    def _load(self) -> None:
        """Load ESM2 tokenizer and model weights from HuggingFace hub.

        Called lazily on the first embed() call.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If model weights cannot be downloaded or loaded.
        """
        try:
            from transformers import EsmModel, EsmTokenizer  # noqa: PLC0415
            logger.info("Loading ESM2 tokenizer: %s", ESM2_MODEL_ID)
            self._tokenizer = EsmTokenizer.from_pretrained(ESM2_MODEL_ID)
            logger.info("Loading ESM2 model weights: %s", ESM2_MODEL_ID)
            self._model = EsmModel.from_pretrained(ESM2_MODEL_ID)
            self._model.eval()
        except ImportError as exc:
            raise RuntimeError(
                "transformers package is required to load ESM2. "
                "Install with: pip install transformers torch"
            ) from exc

        param_count = sum(p.numel() for p in self._model.parameters()) // 1_000_000
        logger.info(
            "ESM2 model loaded: %s (%dM parameters)", ESM2_MODEL_ID, param_count
        )
        self._loaded = True

    def _validate_sequence(self, sequence: str) -> str:
        """Strip whitespace and validate amino-acid alphabet.

        Args:
            sequence: Raw sequence string from caller.

        Returns:
            Cleaned, upper-cased sequence string.

        Raises:
            ValueError: If the sequence is empty or contains non-AA characters.
        """
        cleaned = sequence.strip().upper()
        if not cleaned:
            raise ValueError("Sequence must not be empty.")
        if not _VALID_AA_PATTERN.match(cleaned):
            invalid = set(cleaned) - set("ACDEFGHIKLMNPQRSTVWYBJOUXZ")
            raise ValueError(
                f"Sequence contains invalid characters: {sorted(invalid)}"
            )
        return cleaned

    def _truncate_if_needed(self, sequence: str) -> str:
        """Truncate sequence to MAX_SEQUENCE_LENGTH with a WARNING log.

        Args:
            sequence: Cleaned amino-acid sequence.

        Returns:
            Sequence of length <= MAX_SEQUENCE_LENGTH.

        Raises:
            Nothing.
        """
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            logger.warning(
                "Sequence length %d exceeds MAX_SEQUENCE_LENGTH %d — truncating.",
                len(sequence),
                MAX_SEQUENCE_LENGTH,
            )
            return sequence[:MAX_SEQUENCE_LENGTH]
        return sequence

    def _compute_embedding(self, sequence: str) -> np.ndarray:
        """Run the ESM2 forward pass and mean-pool residue embeddings.

        Strips the [CLS] (position 0) and [EOS] (last position) tokens
        before averaging so only real residue representations are pooled.

        Args:
            sequence: Validated, truncated amino-acid sequence.

        Returns:
            np.ndarray of shape (EMBEDDING_DIM,), dtype float32.

        Raises:
            Nothing (after successful model load).
        """
        import numpy as np  # noqa: PLC0415
        import torch       # noqa: PLC0415
        inputs = self._tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)

        # hidden_states: (1, seq_len+2, hidden_dim)
        hidden = outputs.last_hidden_state[0]  # (seq_len+2, hidden_dim)
        residue_hidden = hidden[1:-1]           # strip [CLS] and [EOS]
        pooled = residue_hidden.mean(dim=0)     # (hidden_dim,)
        return pooled.cpu().numpy().astype(np.float32)

    def embed(self, sequence: str) -> np.ndarray:
        """Compute mean-pooled ESM2 embedding for a protein sequence.

        Truncates sequences longer than MAX_SEQUENCE_LENGTH (1022 AA).
        Strips whitespace from sequence before tokenizing.
        Removes <cls> and <eos> tokens before mean-pooling.

        Args:
            sequence: Amino acid sequence using single-letter codes.

        Returns:
            np.ndarray of shape (EMBEDDING_DIM,) = (320,), dtype float32.

        Raises:
            RuntimeError: If model loading fails.
            ValueError: If sequence is empty or contains non-AA characters.
        """
        if not self._loaded:
            self._load()

        cleaned = self._validate_sequence(sequence)
        cleaned = self._truncate_if_needed(cleaned)
        embedding = self._compute_embedding(cleaned)

        assert embedding.shape == (EMBEDDING_DIM,), (
            f"Embedding shape mismatch: expected ({EMBEDDING_DIM},), got {embedding.shape}"
        )
        return embedding

    def embed_as_list(self, sequence: str) -> list[float]:
        """Convenience wrapper returning embed() result as a Python list.

        Used for JSON-serializable tool responses.

        Args:
            sequence: Amino acid sequence.

        Returns:
            List of 320 floats.

        Raises:
            RuntimeError: If model loading fails.
            ValueError: If sequence is invalid.
        """
        return self.embed(sequence).tolist()

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded into memory."""
        return self._loaded
