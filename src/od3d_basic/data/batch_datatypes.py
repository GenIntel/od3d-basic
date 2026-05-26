from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class OD3D_ModelData:
    """Container for neural-network output tensors."""
    feat:          Optional[Tensor] = None  # (B, F)
    feats:         Optional[Tensor] = None  # (B, N, F)
    featmap:       Optional[Tensor] = None  # (B, F, H, W)
    latent:        Optional[Tensor] = None
    latent_mu:     Optional[Tensor] = None
    latent_logvar: Optional[Tensor] = None
