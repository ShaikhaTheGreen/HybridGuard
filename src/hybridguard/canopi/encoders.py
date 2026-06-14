"""
canopi.encoders
===============
The FROZEN encoder E in the CANOPI pipeline:  x -> c(x) [L0] -> E(.) -> P(.).

E is never trained (that is the whole point — all the learning lives in the
projection head P). This module gives two interchangeable frozen encoders behind
one interface so the same training/eval code runs on a GPU box and on a laptop:

  * SentenceTransformerEncoder  — the real semantic encoder used for paper
    numbers (e.g. a multilingual MiniLM / LaBSE). Needs torch + sentence-
    transformers ([full] extra). Embeddings are L2-normalized and cached.
  * HashingEncoder — a deterministic, dependency-free char-n-gram hashing
    encoder. NOT a semantic encoder: it exists so the canonicalize->encode->
    project path and all unit tests run with numpy only. It must never be used
    for reported results (guarded by `is_semantic = False`).

Every encoder applies the installed L0 canonicalizer first (canonicalize_first=
True) so E always sees the canonical form, exactly as at inference.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from hybridguard.canonicalize import canonicalize

__all__ = [
    "BaseEncoder",
    "HashingEncoder",
    "SentenceTransformerEncoder",
    "get_encoder",
]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class BaseEncoder:
    """Common interface. Subclasses implement `_encode_raw`."""

    name: str = "base"
    dim: int = 0
    is_semantic: bool = False

    def __init__(self, canonicalize_first: bool = True, cache: bool = True):
        self.canonicalize_first = canonicalize_first
        self._cache: dict[str, np.ndarray] = {} if cache else None

    def _prep(self, text: str) -> str:
        return canonicalize(text).canonical if self.canonicalize_first else text

    def _encode_raw(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalized embeddings [n, dim] for the (canonicalized) texts."""
        texts = list(texts)
        prepped = [self._prep(t) for t in texts]
        if self._cache is None:
            out = self._encode_in_batches(prepped, batch_size)
            return _l2_normalize(out)
        # Cache by canonical text.
        missing = [t for t in prepped if t not in self._cache]
        uniq_missing = list(dict.fromkeys(missing))
        if uniq_missing:
            vecs = _l2_normalize(self._encode_in_batches(uniq_missing, batch_size))
            for t, v in zip(uniq_missing, vecs):
                self._cache[t] = v
        return np.vstack([self._cache[t] for t in prepped])

    def _encode_in_batches(self, prepped: List[str], batch_size: int) -> np.ndarray:
        chunks = []
        for i in range(0, len(prepped), batch_size):
            chunks.append(self._encode_raw(prepped[i : i + batch_size]))
        return np.vstack(chunks) if chunks else np.zeros((0, self.dim))


class HashingEncoder(BaseEncoder):
    """Deterministic char-n-gram hashing encoder. numpy-only, frozen, NOT semantic.

    Stable across processes (uses blake2b, not Python's salted hash). Useful as a
    test double and as a trivial lower-bound front-end; never report its numbers.
    """

    is_semantic = False

    def __init__(self, dim: int = 256, ngram_min: int = 3, ngram_max: int = 5, **kw):
        super().__init__(**kw)
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.name = f"hashing-{dim}"

    def _ngrams(self, text: str):
        t = f" {text.lower()} "
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(t) - n + 1):
                yield t[i : i + n]

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for g in self._ngrams(text):
                h = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] & 1) else -1.0
                out[r, idx] += sign
        return out


class SentenceTransformerEncoder(BaseEncoder):
    """Frozen sentence-transformers encoder (real semantic E for paper numbers).

    Lazy-imports sentence_transformers so the module imports without torch. The
    model is put in eval() and never receives gradients.
    """

    is_semantic = True

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        **kw,
    ):
        super().__init__(**kw)
        self.model_name = model_name
        self.name = model_name.split("/")[-1]
        self._device = device
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._model.eval()
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def _encode_raw(self, texts: List[str]) -> np.ndarray:
        self._ensure()
        import torch  # lazy

        with torch.no_grad():
            emb = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we L2-normalize in encode()
                show_progress_bar=False,
            )
        return np.asarray(emb, dtype=np.float32)


def get_encoder(spec: Optional[dict] = None, **overrides) -> BaseEncoder:
    """Factory. spec={'backend': 'sentence-transformers'|'hashing'|'auto', ...}.

    'auto' uses sentence-transformers if importable, else falls back to hashing
    (so smoke tests never fail on a torch-less box). Reported runs must set
    backend explicitly to a semantic encoder.
    """
    spec = dict(spec or {})
    spec.update(overrides)
    backend = spec.pop("backend", "auto")
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "sentence-transformers"
        except Exception:
            backend = "hashing"
    if backend == "hashing":
        return HashingEncoder(**{k: spec[k] for k in ("dim", "ngram_min", "ngram_max", "canonicalize_first", "cache") if k in spec})
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEncoder(**{k: spec[k] for k in ("model_name", "device", "canonicalize_first", "cache") if k in spec})
    raise ValueError(f"unknown encoder backend: {backend!r}")
