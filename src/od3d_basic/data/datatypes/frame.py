from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor


def _stack_field(values: list) -> Optional[Tensor]:
    if any(v is None for v in values):
        return None
    return torch.stack(values, dim=0)


def _pca_to_rgb(featmap: Tensor) -> Tensor:
    """Project (H, W, F) feature map to (H, W, 3) via PCA, output in [0, 1]."""
    H, W, F = featmap.shape
    flat = featmap.reshape(-1, F).float()
    flat = flat - flat.mean(0)
    _, _, V = torch.pca_lowrank(flat, q=3, niter=4)
    pca = (flat @ V).reshape(H, W, 3)
    for c in range(3):
        ch = pca[..., c]
        pca[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    return pca.clamp(0, 1)


@dataclass(kw_only=True)
class Frame:
    frame_id:     str
    cam_intr4x4:  Optional[Tensor]       = None  # (4, 4)
    rgb:          Optional[Tensor]       = None  # (3, H, W)
    depth:        Optional[Tensor]       = None  # (H, W)
    depth_mask:   Optional[Tensor]       = None  # (H, W)  bool
    mask:         Optional[Tensor]       = None  # (H, W)  bool
    feat:         Optional[Tensor]       = None  # (F,)
    featmap:      Optional[Tensor]       = None  # (F, H, W)
    featmap_lvls: Optional[List[Tensor]] = None  # L x (F, H_l, W_l)

    def viz(self, show: bool = True) -> Optional[Tensor]:
        """Interactive modality viewer with CheckButtons to toggle panels.

        When show=True opens a matplotlib window; checkboxes select/deselect
        rgb, mask, depth, depth_mask, and featmap PCA.
        Returns the current (3, H, W_total) composed image in [0, 1].
        """
        import torch.nn.functional as F

        all_panels: dict[str, Tensor] = {}

        if self.rgb is not None:
            rgb = self.rgb.float()
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            all_panels["rgb"] = rgb.clamp(0, 1)

        if self.mask is not None:
            all_panels["mask"] = self.mask.float().unsqueeze(-1).expand(-1, -1, 3).clone()

        if self.depth is not None:
            d = self.depth.float()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            all_panels["depth"] = d.unsqueeze(-1).expand(-1, -1, 3).clone()

        if self.depth_mask is not None:
            all_panels["depth_mask"] = self.depth_mask.float().unsqueeze(-1).expand(-1, -1, 3).clone()

        if self.featmap is not None:
            all_panels["featmap pca"] = _pca_to_rgb(self.featmap)

        if not all_panels:
            return None

        labels = list(all_panels.keys())
        active = {l: True for l in labels}

        def _compose() -> Tensor:
            visible = [all_panels[l] for l in labels if active[l]]
            if not visible:
                return torch.zeros(1, 1, 3)
            H_out = max(p.shape[0] for p in visible)
            resized = []
            for p in visible:
                h, w = p.shape[:2]
                if h != H_out:
                    p = F.interpolate(
                        p.permute(2, 0, 1).unsqueeze(0),
                        size=(H_out, max(1, round(w * H_out / h))),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                resized.append(p)
            return torch.cat(resized, dim=1)

        if show:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import CheckButtons

            fig = plt.figure(figsize=(max(4, 3 * len(labels)), 4))
            ax_img   = fig.add_axes([0.0, 0.18, 1.0, 0.82])
            ax_check = fig.add_axes([0.05, 0.02, 0.9, 0.12])

            im = ax_img.imshow(_compose().cpu().numpy())
            ax_img.set_title(self.frame_id, fontsize=9)
            ax_img.axis("off")

            check = CheckButtons(ax_check, labels, actives=[True] * len(labels))

            def on_toggle(label: str) -> None:
                active[label] = not active[label]
                img = _compose()
                im.set_data(img.cpu().numpy())
                im.set_extent([0, img.shape[1], img.shape[0], 0])
                ax_img.set_xlim(0, img.shape[1])
                ax_img.set_ylim(img.shape[0], 0)
                fig.canvas.draw_idle()

            check.on_clicked(on_toggle)
            plt.show()

        return _compose().permute(2, 0, 1)  # (3, H, W_total)


@dataclass
class FrameBatch:
    """Stacked across B Frame samples."""
    cam_intr4x4:  Optional[Tensor]       = None  # (B, 4, 4)
    rgb:          Optional[Tensor]       = None  # (B, 3, H, W)
    depth:        Optional[Tensor]       = None  # (B, H, W)
    depth_mask:   Optional[Tensor]       = None  # (B, H, W)
    mask:         Optional[Tensor]       = None  # (B, H, W)  bool
    feat:         Optional[Tensor]       = None  # (B, F)
    featmap:      Optional[Tensor]       = None  # (B, F, H, W)
    featmap_lvls: Optional[List[Tensor]] = None  # L x (B, F, H_l, W_l)


def collate_frames(
    samples: list[Frame],
    include: Optional[set[str]] = None,
) -> FrameBatch:
    def _get(attr):
        vals = [getattr(s, attr) for s in samples]
        if include and attr not in include:
            return None
        return _stack_field(vals)

    def _get_lvls(attr):
        if include and attr not in include:
            return None
        per_sample = [getattr(s, attr) for s in samples]
        if any(v is None for v in per_sample):
            return None
        return [
            torch.stack([per_sample[b][l] for b in range(len(per_sample))])
            for l in range(len(per_sample[0]))
        ]

    return FrameBatch(
        cam_intr4x4  = _get("cam_intr4x4"),
        rgb          = _get("rgb"),
        depth        = _get("depth"),
        depth_mask   = _get("depth_mask"),
        mask         = _get("mask"),
        feat         = _get("feat"),
        featmap      = _get("featmap"),
        featmap_lvls = _get_lvls("featmap_lvls"),
    )
