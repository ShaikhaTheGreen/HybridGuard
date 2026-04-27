"""HybridGuard: canonicalization-aware, contrastive, adversarially trained
prompt-injection detection.

Public API
----------
Minimal install (``pip install -e .``)::

    from hybridguard import canonicalize, CanonicalResult

Full install (``pip install -e .[full]``) also exposes the training and
evaluation pipeline (Phase 2 of the library restructure, coming soon).

Command-line::

    hybridguard --help
    hybridguard canonicalize "text to clean"
    hybridguard reproduce --seeds 42,2025,7,1337,314   # Phase 2
"""

from .canonicalize import canonicalize, CanonicalResult, MAX_DECODE_DEPTH

__version__ = "0.2.0"
__all__ = [
    "canonicalize",
    "CanonicalResult",
    "MAX_DECODE_DEPTH",
    "__version__",
]
