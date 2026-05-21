import logging

logger = logging.getLogger(__name__)
from od3d_basic.od3d_datasets.co3d import frame
from od3d_basic.od3d_datasets.frame import OD3D_Frame
from od3d_basic.cv.transforms.transform import OD3D_Transform
from od3d_basic.cv.visual.crop import crop, crop_with_bbox
from od3d_basic.od3d_datasets.frame import OD3D_FRAME_MODALITIES
import torch


class BBoxsCrops(OD3D_Transform):
    def __init__(self, H, W, bboxs_from_masks=False):
        super().__init__()
        self.H = H
        self.W = W
        self.bboxs_from_masks = bboxs_from_masks

        self.mode_bilinear = "bilinear"  # "bilinear" "nearest_v2"
        self.mode_nearest = "nearest_v2"  # "bilinear" "nearest_v2"
        #self.mode_mask = "nearest_v2"  # "bilinear" "nearest_v2""
        #self.mode_pxl_cat_id = "nearest_v2"

    def __call__(self, frame: OD3D_Frame):
        # logger.info(f"Frame name {self.name}")

        apply_rgb = OD3D_FRAME_MODALITIES.RGB in frame.modalities
        apply_rgb_mask = OD3D_FRAME_MODALITIES.RGB_MASK in frame.modalities
        apply_rgb_orig = OD3D_FRAME_MODALITIES.RGB_ORIG in frame.modalities

        apply_mask = OD3D_FRAME_MODALITIES.MASK in frame.modalities
        apply_obj_syms = OD3D_FRAME_MODALITIES.OBJ_SYMS in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_SYMS in frame.modalities
        apply_obj_kpts3d = OD3D_FRAME_MODALITIES.OBJ_KPTS3D in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_KPTS3D in frame.modalities
        apply_obj_kpts3d_mask = OD3D_FRAME_MODALITIES.OBJ_KPTS3D_MASK in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_KPTS3D_MASK in frame.modalities
        apply_obj_kpts2d_mask = OD3D_FRAME_MODALITIES.OBJ_KPTS2D_MASK in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_KPTS2D_MASK in frame.modalities

        apply_obj_size3d = OD3D_FRAME_MODALITIES.OBJ_SIZE3D in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_SIZE3D in frame.modalities
        apply_obj_size1d = OD3D_FRAME_MODALITIES.OBJ_SIZE1D in frame.modalities or OD3D_FRAME_MODALITIES.OBJS_SIZE1D in frame.modalities

        #apply_mask_dt = False #  OD3D_FRAME_MODALITIES.MASK_DT in frame.modalities
        #apply_mask_inv_dt = False #  OD3D_FRAME_MODALITIES.MASK_INV_DT in frame.modalities
        apply_depth = OD3D_FRAME_MODALITIES.DEPTH in frame.modalities
        apply_depth_mask = OD3D_FRAME_MODALITIES.DEPTH_MASK in frame.modalities
        apply_pxl_cat_id = OD3D_FRAME_MODALITIES.PXL_CAT_ID in frame.modalities

        apply_cat_id = OD3D_FRAME_MODALITIES.CATEGORIES_IDS in frame.modalities or OD3D_FRAME_MODALITIES.CATEGORY_ID in frame.modalities
        apply_cat = OD3D_FRAME_MODALITIES.CATEGORIES in frame.modalities or OD3D_FRAME_MODALITIES.CATEGORY in frame.modalities

        #apply_cat_id = OD3D_FRAME_MODALITIES.CATEGORY_ID in frame.modalities
        #apply_cat = OD3D_FRAME_MODALITIES.CATEGORY in frame.modalities

        apply_mesh = OD3D_FRAME_MODALITIES.MESH in frame.modalities
        pxl_cat_id_ids = frame.get_pxl_cat_id().unique()
        if 0 not in pxl_cat_id_ids: 
            pxl_cat_id_ids = torch.cat([torch.LongTensor([0]), pxl_cat_id_ids])

        if self.bboxs_from_masks:
            # mask = (batch.pxl_cat_id == batch.pxl_cat_id.flatten(1).unique(dim=-1)[..., None, None])[..., 1:, :, :]
            # pxl_cat_id = mask * (batch.pxl_cat_id.flatten(1).unique(dim=-1)[..., 1:, None, None])
            # while mask.shape[1] > mask_amodal.shape[0]:
            #     id_filter = mask.flatten(2).sum(dim=-1).min(dim=-1).indices
            #     mask = torch.cat([mask[:, :id_filter], mask[:, id_filter+1:]], dim=1)
            #     pxl_cat_id = torch.cat([pxl_cat_id[:, :id_filter], pxl_cat_id[:, id_filter+1:]], dim=1)
            # #from od3d_basic.cv.visual.show import show_imgs
            # #show_imgs([a, mask.cpu()])
            # # from od3d_basic.cv.visual.show import show_img, show_imgs
            # # show_imgs(mask[0], height=512, width=512)
            # # mask: [1, 16, 512, 512]
            # # from od3d_basic.cv.visual.show import show_img, show_imgs
            # # show_imgs([scene[:, 0, 0], mask[0]], height=512, width=512)
            # from od3d_basic.cv.visual.draw import get_bboxs_from_masks
            # # preprocess_bbox(self, bboxs, bbox_id, frame, batch, H_out, W_out)
            # bboxs_mask_modal = get_bboxs_from_masks(mask)[0]


            from od3d_basic.cv.visual.draw import get_bboxs_from_masks
            # preprocess_bbox(self, bboxs, bbox_id, frame, batch, H_out, W_out)

            pxl_cat_id = frame.get_pxl_cat_id()
            mask = (pxl_cat_id == pxl_cat_id.flatten(1).unique(dim=-1)[..., None, None])[..., 1:, :, :]
            # indices of largest masks
            masks_ids_sorted = mask.flatten(2).sum(dim=-1).sort(descending=True).indices[0]
            if hasattr(frame.meta, 'l_objs_valid') and frame.meta.l_objs_valid is not None and len(frame.meta.l_objs_valid) < len(masks_ids_sorted):
                logger.info(f"Filtering bboxs from masks for frame {frame.name} with {mask.shape[1]} objects as number of valid objects is {len(frame.meta.l_objs_valid)}.")
                # filter for largest indices, but do not change the order of masks
                masks_ids_sorted = masks_ids_sorted[:len(frame.meta.l_objs_valid)]
                # make masks ids sorted in ascending order
                masks_ids_sorted = masks_ids_sorted.sort(descending=False).values
                mask = mask[:, masks_ids_sorted]
                
                pxl_cat_id_ids = pxl_cat_id_ids[torch.cat([torch.LongTensor([0]), masks_ids_sorted + 1])]

            bboxs = get_bboxs_from_masks(mask)[0]
            #if hasattr(frame.meta, 'l_objs_valid') and frame.meta.l_objs_valid is not None and len(frame.meta.l_objs_valid) != bboxs.shape[0]:
            #    assert False, f"Number of bboxs {bboxs.shape[0]} does not match number of objects valid {len(frame.meta.l_objs_valid)} for frame {frame.name}"

            frame.bboxs = bboxs

        if OD3D_FRAME_MODALITIES.BBOXS in frame.modalities:
            if frame.bboxs is not None or (hasattr(frame, 'fpath_bboxs') and frame.fpath_bboxs is not None and frame.fpath_bboxs.exists()):
                bboxs = frame.get_bboxs()
                if bboxs is None:
                    frame.bboxs = []
                    bboxs = []
                    return None
            else:
                frame.bboxs = []
                bboxs = []
                return None

            frame.modalities_stacked = []
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.BBOXS)

            bboxs_sel = []
            bboxs_ids = []
            cam_intr4x4s = []
            rgbs = []
            rgbs_mask = []
            rgbs_orig = []
            masks = []
            masks_dt = []
            masks_inv_dt = []
            depths = []
            depths_masks = []
            pxl_cat_ids = []
            sizes = []
            bboxs_tformed = []
            categories = []
            obj_syms = []
            obj_size3d = []
            obj_size1d = []
            obj_kpts3d = []
            obj_kpts3d_mask = []
            obj_kpts2d_mask = []
            mesh = None

            if apply_cat_id:
                _categories_ids = torch.LongTensor([frame.all_categories.index(cat)
                                                    if cat in frame.all_categories else 0 for cat in frame.categories])

            categories_ids = []
            for r in range(len(bboxs)):

                feats_resize_nearest = []
                feats_resize_nearest_dim = []
                feats_resize_bilinear = []
                feats_resize_bilinear_dim = []

                if apply_cat and frame.categories[r] not in frame.all_categories:
                    logger.info(f"Skipping frame {frame.name} {r} as category {frame.categories[r]} is not in all_categories.")
                    continue
                    
                if not frame.meta.l_objs_valid[r]:
                    logger.info(f"Skipping frame {frame.name} {r} as object is not valid.")
                    continue
                    
                if apply_obj_kpts3d:
                    if frame.fpaths_objs_kpts3d is None:
                        logger.info(f"Skipping frame {frame.name} as fpath obj_kpts3d for all objects is None.")
                        continue
                    if frame.fpaths_objs_kpts3d[r] is None:
                        logger.info(f"Skipping frame {frame.name} as fpath obj_kpts3d for object {r} is None.")
                        continue
                    if not frame.fpaths_objs_kpts3d[r].exists():
                        logger.info(f"Skipping frame {frame.name} as fpath obj_kpts3d {frame.fpaths_objs_kpts3d[r]} does not exist.")
                        continue
                if apply_obj_kpts2d_mask:
                    if frame.fpaths_objs_kpts2d_mask is None:
                        logger.info(f"Skipping frame {frame.name} as fpath obj_kpts2d_mask for all objects is None.")
                        continue
                    if frame.fpaths_objs_kpts2d_mask[r] is None:
                        logger.info(f"Skipping frame {frame.name} as fpath obj_kpts2d_mask for object {r} is None.")
                        continue
                    if not frame.fpaths_objs_kpts2d_mask[r].exists():   
                        logger.info(f"Skipping frame {frame.name} as obj_kpts2d_mask for object {r} does not exist")
                        continue

                mask = frame.get_pxl_cat_id() == pxl_cat_id_ids[r+1]
                if mask.sum() == 0:
                    logger.info(f"Skipping frame {frame.name} {r} as mask is empty.")
                    continue
                
                #if len(bboxs_ids) > 0:
                #   break

                bboxs_ids.append(r)

                # x_min, y_min, x_max, y_max
                bbox = bboxs[r]
                # rgb, cam_crop_tform_cam = crop(

                if apply_obj_syms:
                    obj_syms.append(frame.get_objs_syms()[r])

                if apply_obj_size3d:
                    obj_size3d.append(frame.get_objs_size3d()[r])
                
                if apply_obj_size1d:
                    obj_size1d.append(frame.get_objs_size1d()[r])

                if apply_obj_kpts3d:
                    obj_kpts3d.append(frame.get_objs_kpts3d()[r])
                if apply_obj_kpts3d_mask:
                    obj_kpts3d_mask.append(frame.get_objs_kpts3d_mask()[r])
                if apply_obj_kpts2d_mask:
                    obj_kpts2d_mask.append(frame.get_objs_kpts2d_mask()[r])

                if apply_cat:
                    categories.append(frame.categories[r])

                if apply_cat_id:
                    categories_ids.append(_categories_ids[r])

                if apply_rgb:
                    feats_resize_bilinear.append(frame.get_rgb())
                    feats_resize_bilinear_dim.append(feats_resize_bilinear[-1].shape[0])

                if apply_rgb_orig:
                    feats_resize_bilinear.append(frame.get_rgb_orig())
                    feats_resize_bilinear_dim.append(feats_resize_bilinear[-1].shape[0])

                if len(feats_resize_bilinear) > 0:
                    feats_resize_bilinear = torch.cat(feats_resize_bilinear, dim=0)
                    logger.info(f"Cropping rgb with bbox {bbox} for frame {frame.name} with shape {feats_resize_bilinear.shape}")
                    feats_resize_bilinear, cam_crop_tform_cam = crop_with_bbox(img=feats_resize_bilinear, bbox=bbox,
                                                                              H_out=self.H, W_out=self.W,
                                                                              mode=self.mode_bilinear)

                    if apply_rgb:
                        rgbs.append(feats_resize_bilinear[:feats_resize_bilinear_dim[0]].clone().to(dtype=frame.get_rgb().dtype))
                        feats_resize_bilinear = feats_resize_bilinear[feats_resize_bilinear_dim[0]:].clone()
                        feats_resize_bilinear_dim = feats_resize_bilinear_dim[1:]

                    if apply_rgb_orig:
                        rgbs_orig.append(feats_resize_bilinear[:feats_resize_bilinear_dim[0]].clone().to(dtype=frame.get_rgb_orig().dtype))
                        feats_resize_bilinear = feats_resize_bilinear[feats_resize_bilinear_dim[0]:].clone()
                        feats_resize_bilinear_dim = feats_resize_bilinear_dim[1:]

                if apply_depth:
                    feats_resize_nearest.append(frame.get_depth())
                    feats_resize_nearest_dim.append(feats_resize_nearest[-1].shape[0])

                if apply_mask:
                    mask = frame.get_pxl_cat_id() == pxl_cat_id_ids[r+1]
                    frame.mask = mask

                    feats_resize_nearest.append(frame.get_mask())
                    feats_resize_nearest_dim.append(feats_resize_nearest[-1].shape[0])

                if apply_rgb_mask:
                    feats_resize_nearest.append(frame.get_rgb_mask())
                    feats_resize_nearest_dim.append(feats_resize_nearest[-1].shape[0])

                if apply_depth_mask:
                    feats_resize_nearest.append(frame.get_depth_mask())
                    feats_resize_nearest_dim.append(feats_resize_nearest[-1].shape[0])

                if apply_pxl_cat_id:
                    feats_resize_nearest.append(frame.get_pxl_cat_id())
                    feats_resize_nearest_dim.append(feats_resize_nearest[-1].shape[0])

                if len(feats_resize_nearest) > 0:
                    feats_resize_nearest = torch.cat(feats_resize_nearest, dim=0)

                    logger.info(f"Cropping features with bbox {bbox} for frame {frame.name} with shape {feats_resize_nearest.shape}")
                    feats_resize_nearest, cam_crop_tform_cam = crop_with_bbox(img=feats_resize_nearest, bbox=bbox,
                                                                              H_out=self.H, W_out=self.W,
                                                                              mode=self.mode_nearest)

                    if apply_depth:
                        depths.append(feats_resize_nearest[:feats_resize_nearest_dim[0]].clone().to(dtype=frame.get_depth().dtype))
                        feats_resize_nearest = feats_resize_nearest[feats_resize_nearest_dim[0]:].clone()
                        feats_resize_nearest_dim = feats_resize_nearest_dim[1:]

                    if apply_mask:
                        masks.append( feats_resize_nearest[:feats_resize_nearest_dim[0]].clone().to(dtype=frame.get_mask().dtype))
                        feats_resize_nearest = feats_resize_nearest[feats_resize_nearest_dim[0]:].clone()
                        feats_resize_nearest_dim = feats_resize_nearest_dim[1:]

                    if apply_rgb_mask:
                        rgbs_mask.append(feats_resize_nearest[:feats_resize_nearest_dim[0]].clone().to(dtype=frame.get_rgb_mask().dtype))
                        feats_resize_nearest = feats_resize_nearest[feats_resize_nearest_dim[0]:].clone()
                        feats_resize_nearest_dim = feats_resize_nearest_dim[1:]

                    if apply_depth_mask:
                        depths_masks.append(feats_resize_nearest[:feats_resize_nearest_dim[0]].clone().to(dtype=frame.get_depth_mask().dtype))
                        feats_resize_nearest = feats_resize_nearest[feats_resize_nearest_dim[0]:].clone()
                        feats_resize_nearest_dim = feats_resize_nearest_dim[1:]

                    if apply_pxl_cat_id:
                        pxl_cat_ids.append(feats_resize_nearest[:feats_resize_nearest_dim[0]].clone().to(dtype=frame.get_pxl_cat_id().dtype))
                        feats_resize_nearest = feats_resize_nearest[feats_resize_nearest_dim[0]:].clone()
                        feats_resize_nearest_dim = feats_resize_nearest_dim[1:]

                from od3d_basic.cv.geometry.transform import transf2d_bbox
                bbox_tformed = transf2d_bbox(bbox=bbox.clone(), transf3x3=cam_crop_tform_cam)
                bboxs_tformed.append(bbox_tformed)

                #sizes.append([self.H, self.W])
                cam_intr4x4 = torch.bmm(
                    cam_crop_tform_cam[None,].clone(),
                    frame.get_cam_intr4x4()[None,].clone(),
                )[0]
                cam_intr4x4s.append(cam_intr4x4)

            if len(bboxs_ids) == 0:
                return None
            
            if apply_mesh:
                frame.mesh = frame.get_mesh()
                frame.mesh = [ frame.mesh.get_meshes_with_ids([bbox_id], clone=True) for bbox_id in bboxs_ids]
                #frame.mesh = [frame.get_mesh(objs_ids=[bbox_id]) for bbox_id in bboxs_ids] # mesh.get_meshes_with_ids(meshes_ids=bboxs_ids)
                # this is buggy (line above)
            frame.size = torch.LongTensor([self.H, self.W])

            frame.bboxs = frame.bboxs[bboxs_ids][:, None,].clone()
            bboxs_tformed = torch.stack(bboxs_tformed, dim=0).clone()
            frame.bboxs = bboxs_tformed[:, None]
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.BBOXS)
            frame.bbox = bboxs_tformed
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.BBOX)
            # works
            #cam_tform4x4_obj = frame.get_cam_tform4x4_objs()[bboxs_ids]
            #frame.obj_tform4x4_objs = frame.get_obj_tform4x4_objs(objs_ids=bboxs_ids)

            # works (in both cases the other variable is not stored, as get only stores the exact variable

            frame.set_cam_tform4x4_obj_scene(frame.get_cam_tform4x4_obj().clone())
            cam_tform4x4_obj = frame.get_cam_tform4x4_objs()[bboxs_ids].clone()
            
            frame.obj_tform4x4_objs = frame.get_obj_tform4x4_objs(objs_ids=bboxs_ids).clone()
            frame.obj_tform4x4_objs = torch.eye(4)[None,].repeat(frame.obj_tform4x4_objs.shape[0], 1, 1) 
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_TFORM4X4_OBJS)

            # note: projection does not change as we scale the depth z to the object as well
            scale = 1. / (
                cam_tform4x4_obj[:, :3, :3]
                .norm(dim=-1, keepdim=True)
                .mean(dim=-2, keepdim=True)
            )
            
            cam_tform4x4_obj[:, :3] = cam_tform4x4_obj[:, :3] * scale
            
            frame.set_cam_scale1d(scale.squeeze(-1).squeeze(-1))
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CAM_SCALE1D)

            frame.cam_tform4x4_obj = cam_tform4x4_obj
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ)

            # note here one does not remove the scale
            frame.set_cam_intr4x4_scene(frame.get_cam_intr4x4().clone())
            cam_intr4x4s = torch.stack(cam_intr4x4s, dim=0)
            frame.cam_intr4x4 = cam_intr4x4s
            frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CAM_INTR4X4)
            
            if apply_obj_syms:
                obj_syms = torch.stack(obj_syms, dim=0)
                frame.obj_syms = obj_syms
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_SYMS)
                frame.objs_syms = frame.obj_syms
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_SYMS)

            if apply_obj_size1d:
                obj_size1d = torch.stack(obj_size1d, dim=0)
                frame.obj_size1d = obj_size1d
                frame.obj_size1d = ((frame.obj_size1d / frame.obj_size1d) * 2. )  / frame.cam_scale1d
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_SIZE1D)
                frame.objs_size1d = frame.obj_size1d
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_SIZE1D)

            if apply_obj_size3d:
                obj_size3d = torch.stack(obj_size3d, dim=0)
                frame.obj_size3d = obj_size3d.clone()
                frame.obj_size3d = ((frame.obj_size3d / frame.obj_size3d.max(dim=0).values) * 2. )  / frame.cam_scale1d[:, None]
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_SIZE3D)
                frame.objs_size3d = frame.obj_size3d
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_SIZE3D)

            if apply_obj_kpts3d:
                obj_kpts3d = torch.stack(obj_kpts3d, dim=0)
                frame.obj_kpts3d = obj_kpts3d
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_KPTS3D)
                frame.objs_kpts3d = obj_kpts3d
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_KPTS3D)

            if apply_obj_kpts3d_mask:
                obj_kpts3d_mask = torch.stack(obj_kpts3d_mask, dim=0)
                frame.obj_kpts3d_mask = obj_kpts3d_mask
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_KPTS3D_MASK)
                frame.objs_kpts3d_mask = obj_kpts3d_mask
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_KPTS3D_MASK)

            if apply_obj_kpts2d_mask:
                obj_kpts2d_mask = torch.stack(obj_kpts2d_mask, dim=0)
                frame.obj_kpts2d_mask = obj_kpts2d_mask
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJ_KPTS2D_MASK)
                frame.objs_kpts2d_mask = obj_kpts2d_mask
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.OBJS_KPTS2D_MASK)

            if apply_cat:
                frame.category = categories
                frame.categories = categories
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CATEGORY)
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CATEGORIES)
            if apply_cat_id:
                frame.category_id = torch.stack(categories_ids, dim=0)
                frame.categories_ids = categories_ids
                del categories_ids
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CATEGORY_ID)
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.CATEGORIES_IDS)
            if apply_rgb:
                frame.set_rgb_scene(frame.get_rgb().clone())
                rgbs = torch.stack(rgbs, dim=0)
                frame.rgb = rgbs
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.RGB)
            if apply_rgb_mask:
                rgbs_mask = torch.stack(rgbs_mask, dim=0)
                frame.rgb_mask = rgbs_mask
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.RGB_MASK)
            if apply_depth_mask:
                depths_masks = torch.stack(depths_masks, dim=0)
                frame.depth_mask = depths_masks
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.DEPTH_MASK)
            if apply_rgb_orig:
                rgbs_orig = torch.stack(rgbs_orig, dim=0)
                frame.rgb_orig = rgbs_orig
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.RGB_ORIG)
            if apply_mask:
                masks = torch.stack(masks, dim=0)
                frame.mask = masks
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.MASK)
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.MASK_DT)
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.MASK_INV_DT)
            if apply_depth:
                depths = torch.stack(depths, dim=0)
                frame.depth = depths
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.DEPTH)
            if apply_pxl_cat_id:
                pxl_cat_ids = torch.stack(pxl_cat_ids, dim=0)
                frame.pxl_cat_id = pxl_cat_ids
                frame.modalities_stacked.append(OD3D_FRAME_MODALITIES.PXL_CAT_ID)

        # this is more complicated
        #if hasattr(frame, 'pair') and frame.pair is not None:
        #    frame.pair = self(frame.pair)
        return frame
