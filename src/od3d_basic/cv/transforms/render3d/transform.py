import torch
from od3d_basic.cv.transforms.transform import OD3D_Transform

# from omegaconf import DictConfig
# import genesis as gs
# pip install genesis-world


class Render3D(OD3D_Transform):
    def __init__(
        self,
        H,
        W,
        # apply_txtr=False,
        # config: DictConfig = None,
        # scale_min=None,
        # scale_max=None,
        # center_rel_shift_xy_min=[0.0, 0.0],
        # center_rel_shift_xy_max=[0.0, 0.0],
        # scale_with_mask=None,
        # scale_with_dist=None,
        # scale_with_pad=True,
        # center_use_mask=False,
        # scale_selection="shorter",
    ):
        super().__init__()
        self.H = H
        self.W = W

    def __call__(self, frame):
        device = "cuda"
        # device = 'cpu'

        meshes = frame.get_mesh(device=device)

        cam_tform4x4_obj = frame.get_cam_tform4x4_obj().clone().to(device)
        cam_intr4x4 = frame.get_cam_intr4x4().clone().to(device)
        imgs_sizes = frame.size.clone()
        with torch.no_grad():
            mods = meshes.render(
                imgs_sizes=imgs_sizes,
                cams_tform4x4_obj=cam_tform4x4_obj[None,],
                cams_intr4x4=cam_intr4x4[None,],
                modalities=["rgb", "mask"],
            )

            rgb = frame.get_rgb()
            mask = frame.get_mask()
            frame.rgb = (mods["rgb"][0] * 255).to(dtype=rgb.dtype, device=rgb.device)
            frame.mask = (mods["mask"][0]).to(dtype=mask.dtype, device=mask.device)

            # mask = (mods["mask"][0]).to(dtype=mask.dtype, device=mask.device)
            # import numpy as np
            # import cv2
            # mask_np = np.uint8((mask > 0.5).numpy()[0] * 255.)
            # mask_dt = torch.FloatTensor(cv2.distanceTransform(mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE))[None,]

            # from od3d_basic.cv.visual.show import show_imgs
            # show_imgs([frame.mask, (mods["mask"][0]).to(dtype=mask.dtype, device=mask.device)])
            del meshes
            del cam_intr4x4
            del cam_tform4x4_obj

        return frame
        # frame.rgb =
        # ########################## init ##########################
        # gs.init(seed=0, precision="32", logging_level="debug")
        #
        # ########################## create a scene ##########################
        # scene = gs.Scene(
        #     sim_options=gs.options.SimOptions(),
        #     viewer_options=gs.options.ViewerOptions(
        #         res=(self.W, self.H),
        #         camera_pos=(8.5, 0.0, 4.5),
        #         camera_lookat=(3.0, 0.0, 0.5),
        #         camera_fov=50,
        #     ),
        #     rigid_options=gs.options.RigidOptions(enable_collision=False, gravity=(0, 0, 0)),
        #     # renderer=gs.renderers.RayTracer(  # type: ignore
        #     #     env_surface=gs.surfaces.Emission(
        #     #         emissive_texture=gs.textures.ImageTexture(
        #     #             image_path="textures/indoor_bright.png",
        #     #         ),
        #     #     ),
        #     #     env_radius=15.0,
        #     #     env_euler=(0, 0, 180),
        #     #     lights=[
        #     #         {"pos": (0.0, 0.0, 10.0), "radius": 3.0, "color": (15.0, 15.0, 15.0)},
        #     #     ],
        #     # ),
        # )
        #
        # ########################## materials ##########################
        #
        # ########################## entities ##########################
        # # floor
        # scene.add_entity(
        #     morph=gs.morphs.Plane(
        #         pos=(0.0, 0.0, -0.5),
        #     ),
        #     surface=gs.surfaces.Aluminium(
        #         ior=10.0,
        #     ),
        # )
        #
        # # user specified external color texture
        # scene.add_entity(
        #     morph=gs.morphs.Mesh(
        #         file=fpath_mesh,
        #         scale=0.5,
        #         pos=(0.0, -3, 0.0),
        #     ),
        #     #surface=gs.surfaces.Rough(
        #     #    diffuse_texture=gs.textures.ColorTexture(
        #     #        color=(1.0, 0.5, 0.5),
        #     #    ),
        #     #),
        # )
        #
        # ########################## cameras ##########################
        # cam_0 = scene.add_camera(
        #     res=(1600, 900),
        #     pos=(8.5, 0.0, 1.5),
        #     lookat=(3.0, 0.0, 0.7),
        #     fov=60,
        #     GUI=True,
        #     spp=512,
        # )
        # scene.build()
        #
        # ########################## forward + backward twice ##########################
        # horizon = 1
        #
        # for i in range(horizon):
        #     scene.step()
        #     cam_0.render()
        #
        # scene.reset()
