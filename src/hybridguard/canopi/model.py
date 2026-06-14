"""
canopi.model
============
Inference pipeline (NO generative model at inference):

    x -> c(x) [L0 canonicalize] -> E(.) [FROZEN encoder] -> P(.) [trainable head,
    unit-norm] -> detector head -> score s(x) -> decision 1[s(x) >= tau]

`CanopiModel` (torch) is the trainable object used by train.py. `CanopiInference`
(numpy) runs a trained model anywhere — including a torch-less laptop — by reading
the projection/detector weights as numpy arrays, so the end-to-end path is
testable and deployable without the training stack. A randomly-initialized
`NumpyProjectionHead` lets the canonicalize->encode->project geometry be unit
tested before any training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["ProjectionHead", "DetectorHead", "CanopiModel", "NumpyProjectionHead", "CanopiInference"]


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


# ---------------------------------------------------------------------------
# torch trainable model
# ---------------------------------------------------------------------------

def _build_torch_model(in_dim, hidden, out_dim, depth, dropout):
    import torch
    from torch import nn

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            layers, d = [], in_dim
            for _ in range(max(depth - 1, 0)):
                layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            layers += [nn.Linear(d, out_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, e):
            z = self.net(e)
            return nn.functional.normalize(z, dim=-1)

    class DetectorHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(out_dim, 1)

        def forward(self, z):
            return self.fc(z).squeeze(-1)

    class _Canopi(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ProjectionHead()
            self.det = DetectorHead()

        def project(self, e):
            return self.proj(e)

        def logits(self, e):
            return self.det(self.proj(e))

        def forward(self, e):
            return self.logits(e)

    return _Canopi()


class ProjectionHead:  # facade for import symmetry; real nn.Module built lazily
    pass


class DetectorHead:
    pass


@dataclass
class CanopiModel:
    """Trainable wrapper around the frozen encoder + torch projection/detector.

    The encoder is FROZEN (numpy embeddings precomputed once); only proj+det train.
    """

    encoder: object
    in_dim: int
    hidden: int = 256
    out_dim: int = 128
    depth: int = 2
    dropout: float = 0.1
    device: Optional[str] = None

    def __post_init__(self):
        import torch

        self._torch = torch
        self.net = _build_torch_model(self.in_dim, self.hidden, self.out_dim, self.depth, self.dropout)
        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self._device)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Frozen-encoder embeddings of canonicalized texts (no grad)."""
        return self.encoder.encode(list(texts))

    def project_tensor(self, e):
        return self.net.project(e)

    def logits_tensor(self, e):
        return self.net.logits(e)

    def score(self, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """End-to-end probability s(x) in [0,1]."""
        torch = self._torch
        e = self.embed(texts)
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(e), batch_size):
                t = torch.as_tensor(e[i : i + batch_size], dtype=torch.float32, device=self._device)
                out.append(torch.sigmoid(self.net.logits(t)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def to_inference(self) -> "CanopiInference":
        """Export numpy weights for torch-free inference."""
        sd = {k: v.detach().cpu().numpy() for k, v in self.net.state_dict().items()}
        return CanopiInference(self.encoder, sd, self.depth)


# ---------------------------------------------------------------------------
# numpy inference / test heads
# ---------------------------------------------------------------------------

class NumpyProjectionHead:
    """Deterministic random projection head (numpy). For geometry unit tests and
    as an inference head when torch weights are loaded via `set_weights`."""

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128, depth: int = 2, seed: int = 1337):
        rng = np.random.default_rng(seed)
        self.W, d = [], in_dim
        dims = [in_dim] + [hidden] * max(depth - 1, 0) + [out_dim]
        for a, b in zip(dims[:-1], dims[1:]):
            self.W.append((rng.standard_normal((a, b)) / np.sqrt(a), np.zeros(b)))

    def set_weights(self, weights):
        self.W = weights

    def forward(self, e: np.ndarray) -> np.ndarray:
        h = np.asarray(e, dtype=float)
        for i, (w, b) in enumerate(self.W):
            h = h @ w + b
            if i < len(self.W) - 1:
                h = np.where(h > 0, h, 0.5 * (np.exp(np.minimum(h, 0)) - 1))  # GELU-ish
        return _l2(h)


class CanopiInference:
    """Torch-free inference: encoder -> numpy projection -> numpy linear detector."""

    def __init__(self, encoder, state_dict: Optional[dict] = None, depth: int = 2):
        self.encoder = encoder
        self.depth = depth
        self._proj_W = None
        self._det = None
        if state_dict is not None:
            self._load(state_dict)

    def _load(self, sd: dict):
        # Reconstruct ordered Linear weights from torch state_dict naming.
        proj_layers = []
        lin_idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("proj.net.") and k.endswith(".weight")})
        for li in lin_idx:
            w = sd[f"proj.net.{li}.weight"].T  # torch stores [out,in]
            b = sd[f"proj.net.{li}.bias"]
            proj_layers.append((np.asarray(w), np.asarray(b)))
        self._proj = NumpyProjectionHead(proj_layers[0][0].shape[0], depth=len(proj_layers))
        self._proj.set_weights(proj_layers)
        self._det = (np.asarray(sd["det.fc.weight"]).ravel(), float(np.asarray(sd["det.fc.bias"]).ravel()[0]))

    def project(self, texts: Sequence[str]) -> np.ndarray:
        return self._proj.forward(self.encoder.encode(list(texts)))

    def score(self, texts: Sequence[str]) -> np.ndarray:
        z = self.project(texts)
        w, b = self._det
        return 1.0 / (1.0 + np.exp(-(z @ w + b)))
