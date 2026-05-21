import logging

logger = logging.getLogger(__name__)
import torch
from dataclasses import dataclass
from typing import List, Union


@dataclass
class OD3D_ModelData:
    latent: torch.Tensor = None  # BxF
    latent_mu: torch.Tensor = None  # BxF
    latent_logvar: torch.Tensor = None  # BxF
    feat: torch.Tensor = None  # BxF
    feat_mu: torch.Tensor = None  # BxF
    feat_logvar: torch.Tensor = None  # BxF
    feats: torch.Tensor = None  # BxNxF
    pts3d: torch.Tensor = None  # BxNx3
    featmap: torch.Tensor = None  # BxCxHxW
    featmap_mu: torch.Tensor = None  # BxCxHxW
    featmap_logvar: torch.Tensor = None  # BxCxHxW
    featmaps: Union[
        List[torch.Tensor],
        torch.Tensor,
    ] = None  # List(BxCxHixWi) or BxNxCxHxW
    mask: torch.Tensor = None  # Bx1xHxW
    masks: Union[List[torch.Tensor], torch.Tensor] = None  # List(Bx1xHixWi) or BxNxHxW
    masks_scores: torch.Tensor = None  # BxN
