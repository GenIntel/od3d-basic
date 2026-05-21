import logging

logger = logging.getLogger(__name__)
from od3d_basic.cv.optimization.ransac import ransac
from od3d_basic.cv.optimization.ransac import ransac
from od3d_basic.cv.geometry.fit.tform4x4 import fit_tform4x4, score_tform4x4_fit
from functools import partial

from od3d_basic.cv.cluster.embed import pca
from od3d_basic.cv.optimization.gradient_descent import (
    gradient_descent_se3,
)


def se3_align_mesh(
    pts_ref,
    pts_src,
    seq_ref_feats,
    seq_src_feats,
    dist_app_weight=0.2,
    geo_cyclic_weight_temp=0.9,
    app_cyclic_weight_temp=0.9,
    ransac_samples=1000,
    ransac_score_perc=1.0,
    device=None,
    embed_type=None,
    embed_dim=128,
    reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
    refine_optimization_steps=20,
    refine_lr=2e-2,
    refine_pts_weight=0.0,
    refine_arap_weight=1.0,
    refine_reg_weight=0.0,
    refine_arap_geo_std=0.05,
):
    # seq_src_feats: List[torch.Tensor] or torch.Tensor, N x K x F
    # seq_ref_feats: List[torch.Tensor] or torch.Tensor, N x K x F
    # embed_type: str, e.g "pca"
    # embed_dim: int

    dist_ref_src = calc_feats_dist_ref_src(
        seq_ref_feats=seq_ref_feats,
        seq_src_feats=seq_src_feats,
        device=device,
        embed_type=embed_type,
        embed_dim=embed_dim,
        reduce_type=reduce_type,
    )

    src_tform4x4_ref = ransac(
        pts=pts_ref,
        fit_func=partial(
            fit_tform4x4,
            pts_ref=pts_src,
            dist_ref=dist_ref_src,
        ),
        score_func=partial(
            score_tform4x4_fit,
            pts_ref=pts_src,
            dist_app_ref=dist_ref_src,
            dist_app_weight=dist_app_weight,
            geo_cyclic_weight_temp=geo_cyclic_weight_temp,
            app_cyclic_weight_temp=app_cyclic_weight_temp,
            score_perc=ransac_score_perc,
        ),
        fits_count=ransac_samples,
        fit_pts_count=4,
    )

    # _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(
    #     pts=pts_ref,
    #     tform4x4=src_tform4x4_ref[None,],
    #     pts_ref=pts_src,
    #     dist_app_ref=dist_ref_src,
    #     return_dists=True,
    #     dist_app_weight=dist_app_weight,
    #     geo_cyclic_weight_temp=geo_cyclic_weight_temp,
    #     app_cyclic_weight_temp=app_cyclic_weight_temp,
    #     score_perc=ransac_score_perc,
    # )

    if refine_optimization_steps > 0:
        (
            src_tform4x4_ref,
            ref_pts_offset,
        ) = gradient_descent_se3(
            pts=pts_ref,
            models=src_tform4x4_ref,
            score_func=partial(
                score_tform4x4_fit,
                pts_ref=pts_src,
                dist_app_ref=dist_ref_src,
                dist_app_weight=dist_app_weight,
                geo_cyclic_weight_temp=geo_cyclic_weight_temp,
                app_cyclic_weight_temp=app_cyclic_weight_temp,
                score_perc=ransac_score_perc,
            ),
            steps=refine_optimization_steps,
            lr=refine_lr,
            pts_weight=refine_pts_weight,
            arap_weight=refine_arap_weight,
            arap_geo_std=refine_arap_geo_std,
            reg_weight=refine_reg_weight,
            return_pts_offset=True,
        )

    _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(
        pts=pts_ref,
        tform4x4=src_tform4x4_ref[None,],
        pts_ref=pts_src,
        dist_app_ref=dist_ref_src,
        return_dists=True,
        dist_app_weight=dist_app_weight,
        geo_cyclic_weight_temp=geo_cyclic_weight_temp,
        app_cyclic_weight_temp=app_cyclic_weight_temp,
        score_perc=ransac_score_perc,
    )

    #
    #     all_pred_ref_pts_offset[category][s][
    #         ref_vertices_mask
    #     ] = ref_pts_offset
    #
    #
    # _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(
    #     pts=pts_ref,
    #     tform4x4=src_tform4x4_ref[None,],
    #     pts_ref=pts_src,
    #     dist_app_ref=dist_ref_src,
    #     return_dists=True,
    #     dist_app_weight=self.config.dist_appear_weight,
    #     geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
    #     app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
    #     score_perc=self.config.ransac.score_perc,
    # )

    return src_tform4x4_ref


def calc_feats_dist_ref_src(
    seq_ref_feats,
    seq_src_feats,
    device=None,
    embed_type=None,
    embed_dim=128,
    reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
):
    # seq_src_feats: List[torch.Tensor] or torch.Tensor, N x K x F
    # seq_ref_feats: List[torch.Tensor] or torch.Tensor, N x K x F
    # embed_type: str, e.g "pca"
    # embed_dim: int

    import torch
    from od3d_basic.cv.io import get_default_device

    if device is None:
        device = get_default_device()

    # perform pca on seq1_feats and seq2_feats and visualize
    if isinstance(seq_ref_feats, list):
        seq1_verts_count = len(seq_ref_feats)
        seq2_verts_count = len(seq_src_feats)
        dist_verts_seq1_seq2 = (
            torch.ones(size=(seq1_verts_count, seq2_verts_count)).to(
                device=device,
            )
            * torch.inf
        )

        # Vertices1+2 x Viewpoints x F
        seq12_feats_padded = torch.nn.utils.rnn.pad_sequence(
            seq_ref_feats + seq_src_feats,
            batch_first=True,
            padding_value=torch.nan,
        ).to(device=device)
        F = seq12_feats_padded.shape[-1]
        V = seq12_feats_padded.shape[-2]
        seq12_feats_padded_mask = ~seq12_feats_padded.isnan().all(dim=-1)

        if embed_type is None:
            pass
        elif embed_type == "pca":
            seq12_feats_padded_embed = torch.zeros(
                size=(seq12_feats_padded.shape[:-1] + (embed_dim,)),
            ).to(
                device=device,
                dtype=seq12_feats_padded.dtype,
            )
            seq12_feats_padded_embed[:] = torch.nan
            seq12_feats_padded_embed[seq12_feats_padded_mask] = pca(
                seq12_feats_padded[seq12_feats_padded_mask],
                C=embed_dim,
            )
            seq12_feats_padded = seq12_feats_padded_embed
            F = embed_dim
        else:
            logger.warning(f"unknown embed type {embed_type}")

        P = seq1_verts_count  # ensures that 11 GB are enough
        logger.info(
            f"seq1 verts {seq1_verts_count}, seq2 verts {seq2_verts_count}, seq1 partial {(seq1_verts_count // P)}, viewpoints max {V}",
        )

        for p in range(P):
            if p < P - 1:
                seq1_verts_partial = torch.arange(seq1_verts_count)[
                    (seq1_verts_count // P) * p : (seq1_verts_count // P) * (p + 1)
                ].to(device=device)
            else:
                seq1_verts_partial = torch.arange(seq1_verts_count)[
                    (seq1_verts_count // P) * p :
                ].to(
                    device=device,
                )
            seq1_verts_partial_count = len(seq1_verts_partial)
            # logger.info(seq1_verts_partial)
            # Vertices1 x Viewpoints x F
            seq1_feats_padded = seq12_feats_padded[seq1_verts_partial].clone()
            seq2_feats_padded = seq12_feats_padded[seq1_verts_count:].clone()

            seq1_feats_padded_mask = seq12_feats_padded_mask[seq1_verts_partial].clone()
            seq2_feats_padded_mask = seq12_feats_padded_mask[seq1_verts_count:].clone()

            # Vertices1 x Viewpoints x Vertices2 x Viewpoints
            if reduce_type.startswith("negdot"):
                dists_verts_feats_seq1_seq2 = -torch.einsum(
                    "bnf,bkf->bnk",
                    seq1_feats_padded.reshape(-1, F)[None,],
                    seq2_feats_padded.reshape(-1, F)[None,],
                ).reshape(
                    seq1_verts_partial_count,
                    V,
                    seq2_verts_count,
                    V,
                )
            else:
                dists_verts_feats_seq1_seq2 = torch.cdist(
                    seq1_feats_padded.reshape(-1, F)[None,],
                    seq2_feats_padded.reshape(-1, F)[None,],
                ).reshape(
                    seq1_verts_partial_count,
                    V,
                    seq2_verts_count,
                    V,
                )
            dists_verts_feats_seq1_seq2_mask = (
                seq1_feats_padded_mask[:, :, None, None]
                * seq2_feats_padded_mask[None, None, :, :]
            )
            dist_verts_seq1_seq2_inf_mask = (
                dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)
                .flatten(
                    2,
                )
                .sum(
                    dim=-1,
                )
                == 0.0
            )

            if (
                reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN
                or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN
            ):
                # replace nan values with inf
                dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(
                    torch.inf,
                )
                dist_verts_seq1_seq2[seq1_verts_partial] = (
                    dists_verts_feats_seq1_seq2.permute(
                        0,
                        2,
                        1,
                        3,
                    )
                    .flatten(
                        2,
                    )
                    .min(dim=-1)
                    .values
                )
            elif (
                reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.AVG
                or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_AVG
            ):
                dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(
                    0.0,
                )
                dists_verts_feats_seq1_seq2_mask = (
                    dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                )

                dist_verts_seq1_seq2_partial = (
                    dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).flatten(
                        2,
                    )
                    * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(
                        2,
                    )
                ).sum(dim=-1) / (
                    dists_verts_feats_seq1_seq2_mask.permute(
                        0,
                        2,
                        1,
                        3,
                    )
                    .flatten(2)
                    .sum(
                        dim=-1,
                    )
                    + 1e-10
                )
                dist_verts_seq1_seq2_partial[dist_verts_seq1_seq2_inf_mask] = torch.inf
                dist_verts_seq1_seq2[seq1_verts_partial] = dist_verts_seq1_seq2_partial
                del dist_verts_seq1_seq2_partial
            elif (
                reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG
                or reduce_type == OD3D_MESH_FEATS_DIST_REDUCE_TYPES.NEGDOT_MIN_AVG
            ):
                dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(
                    torch.inf,
                )
                dists_verts_feats_seq1_seq2_mask = (
                    dists_verts_feats_seq1_seq2_mask.nan_to_num(0.0)
                )
                dist_verts_seq1_seq2_partial = (
                    (
                        dists_verts_feats_seq1_seq2.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .min(
                            dim=-1,
                        )
                        .values.nan_to_num(
                            posinf=0.0,
                        )
                        * dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )[:, :, :, 0]
                    ).sum(dim=-1)
                    + (
                        dists_verts_feats_seq1_seq2.permute(
                            0,
                            2,
                            1,
                            3,
                        )
                        .min(
                            dim=-2,
                        )
                        .values.nan_to_num(
                            posinf=0.0,
                        )
                        * dists_verts_feats_seq1_seq2_mask.permute(
                            0,
                            2,
                            1,
                            3,
                        )[:, :, 0, :]
                    ).sum(dim=-1)
                ) / (
                    dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[
                        :,
                        :,
                        0,
                        :,
                    ].sum(
                        dim=-1,
                    )
                    + dists_verts_feats_seq1_seq2_mask.permute(
                        0,
                        2,
                        1,
                        3,
                    )[
                        :,
                        :,
                        :,
                        0,
                    ].sum(dim=-1)
                    + 1e-10
                )
                dist_verts_seq1_seq2_partial[dist_verts_seq1_seq2_inf_mask] = torch.inf
                dist_verts_seq1_seq2[seq1_verts_partial] = dist_verts_seq1_seq2_partial
                del dist_verts_seq1_seq2_partial
            else:
                logger.warning(f"Unknown reduce type {reduce_type}.")

            del dists_verts_feats_seq1_seq2
            del dists_verts_feats_seq1_seq2_mask
            del dist_verts_seq1_seq2_inf_mask
            del seq1_verts_partial
        del seq12_feats_padded
        del seq12_feats_padded_mask
        seq_ref_feats.clear()
        seq_src_feats.clear()
    else:
        if reduce_type.startswith("negdot"):
            dist_verts_seq1_seq2 = -torch.einsum(
                "nf,kf->nk",
                seq_ref_feats.to(device=device),
                seq_src_feats.to(device=device),
            )
        else:
            dist_verts_seq1_seq2 = torch.cdist(
                seq_ref_feats.to(device=device),
                seq_src_feats.to(device=device),
            )

    # if not fpath_dist_verts_mesh_feats.parent.exists():
    #    fpath_dist_verts_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
    # torch.save(dist_verts_seq1_seq2.detach().cpu(), fpath_dist_verts_mesh_feats)
    # logger.info(f"save mesh feats dist at {fpath_dist_verts_mesh_feats}")
    # del dist_verts_seq1_seq2

    del seq_ref_feats
    del seq_src_feats
    torch.cuda.empty_cache()

    return dist_verts_seq1_seq2


"""
    refine_optimization_steps: 0 # 20
    refine_lr: 2e-2 # 2e-2
    refine_pts_weight: 0. # 0.5 # 0.1 too small, 1 too large
    refine_arap_weight: 1. # 1.
    refine_reg_weight: 0. # 0.1
    refine_arap_geo_std: 0.05 # 0.05 - 0.1

    ransac:
      samples: 1000
      score_perc: 1.
    use_only_first_reference: False
    geo_cyclic_weight_temp: 0.9  # 0.9
    app_cyclic_weight_temp: 0.9  # 0.9
    gt_cam_tform_obj_source: labeled_cuboid # zsp_labeled, labeled, labeled_cuboid or '' for only align
    dist_appear_weight: 0.2
    global_optimization_steps: 1
"""
