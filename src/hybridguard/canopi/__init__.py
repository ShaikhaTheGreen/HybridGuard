"""
CANOPI — Canonicalization + projection-head representation for prompt-injection
detection. Trains a projection head P on top of the FROZEN HybridGuard L0
canonicalizer + a frozen encoder E with a JOINT objective (multi-view intent-
invariance + partial-AUC low-FPR term). No generative model at inference.

Inference:  x -> c(x) [hybridguard.canonicalize] -> E(x) [frozen] -> P(x) [unit-
norm] -> detector head -> s(x) -> 1[s(x) >= tau].

This package is added ALONGSIDE the existing hybridguard code; it does not modify
the L0 canonicalizer or CLI. See docs_NPL/CANOPI_INTEGRATION_PLAN.md.
"""
from __future__ import annotations

__canopi_version__ = "0.1.0"

# Light imports only (numpy). Torch-dependent symbols are imported lazily inside
# their modules so `import hybridguard.canopi` works on a torch-less box.
from . import metrics  # noqa: F401

__all__ = ["metrics", "__canopi_version__"]
