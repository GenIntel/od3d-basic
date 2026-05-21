import cv2
import numpy as np
import torch
from od3d_basic.cv.visual.blend import rgb_to_range01

# Initialize variables to store the coordinates
clicked_position = None


def pick_pixel(img):  #
    # img: 3xHxW
    img_cv = tensor_to_cv_img(img)

    import cv2

    def get_pixel_location(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            global clicked_position
            clicked_position = torch.LongTensor([x, y])  # Store the coordinates
            pixel_value = img_cv[y, x]
            print(f"Clicked pixel at {clicked_position}: Value = {pixel_value}")

    # Load the image

    # Create a window and bind the mouse callback function
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", get_pixel_location)

    # Display the image and wait for a key press to exit
    while True:
        cv2.imshow("Image", img_cv)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    global clicked_position
    # Check if a pixel was clicked and print the coordinates
    if clicked_position is not None:
        clicked_position = clicked_position.to(device=img.device)
        print("Last clicked position:", clicked_position)

    cv2.destroyAllWindows()
    return clicked_position


def random_colors_as_img(K: int, res=100, device="cpu"):
    colors = get_colors(K * 3, device=device)
    colors = colors[torch.randperm(K * 3).to(device=device)]
    img = torch.zeros(size=(3, res, res * K)).to(device=device)
    for k in range(K):
        img[:, :, res * k : res * (k + 1)] = colors[k][:, None, None].repeat(
            1,
            res,
            res,
        )
    return img


def tensor_to_cv_img(x_in):
    # x_in : CxHxW float32
    # x_out : HxWxC uint8
    if x_in.dtype == torch.uint8:
        x_in = x_in * 1.0
    else:
        x_in = rgb_to_range01(x_in) * 255.0
        # x_in = x_in * 255.0
    x_in = torch.clamp(x_in, min=0.0, max=255.0)
    x_out = (x_in.permute(1, 2, 0).cpu().detach().numpy()).astype(np.uint8)
    x_out = x_out[:, :, ::-1]
    return x_out

def cv_img_to_tensor(x_in):
    # x_in : HxWxC uint8
    # x_out : CxHxW float32

    x_out = x_in[:, :, ::-1].copy()
    x_out = torch.from_numpy(x_out).permute(2, 0, 1)
    x_out = x_out.to(dtype=torch.float)
    return x_out


from typing import List

def add_boolean_table(img: torch.Tensor, table: torch.Tensor, text: List = None):
    """
    Args:
        img (torch.Tensor): 3xHxW
        table (torch.Tensor): RxC
        W: int
    Return:
        img (torch.Tensor): 3xHxW
    """
    dtype = img.dtype
    device = img.device

    W = img.shape[-1]
    R, C = table.shape
    K = R * C
    column_offset = W // 4
    margin = 5
    entry_width = (W) // C
    font_scale = entry_width / 60.0
    margin_top = int(40 * font_scale)
    H = R * entry_width
    letters_count_max = 0
    if text is not None:
        for t, _text in enumerate(text):
            if len(_text) > letters_count_max:
                letters_count_max = len(_text)
    W_text = int(font_scale * letters_count_max * 20.0)

    img_text = torch.ones(size=(3, H, W_text)).to(dtype=dtype, device=device)
    img_table = torch.ones(size=(3, H, W)).to(dtype=dtype, device=device)

    if text is not None:
        for t, _text in enumerate(text):
            img_text = draw_text_in_rgb(
                img=img_text,
                text=_text,
                leftTopCornerOfText=(margin, margin + margin_top + entry_width * t),
                fontColor=(0, 0, 0),
                fontScale=font_scale,
            )

    pxls = torch.stack(
        torch.meshgrid(torch.arange(R), torch.arange(C), indexing="ij"),
        dim=-1,
    )
    pxls = pxls.flip(dims=[-1])  # x, y
    pxls = ((pxls + 0.5) * entry_width).to(dtype=int, device=device)
    # pxls[:, :, 0] += column_offset
    pxls = pxls.reshape(-1, 2)
    K = pxls.shape[0]
    blue = torch.Tensor([0, 0, 1.0]).to(dtype=dtype, device=device)
    green = torch.Tensor([0, 1.0, 0]).to(dtype=dtype, device=device)
    colors = (
        table[..., None].to(dtype=dtype, device=device) * green[None, None]
        + (~table[..., None]).to(dtype=dtype, device=device) * blue[None, None]
    )
    colors = colors.reshape(-1, 3)
    img_table = (
        draw_pixels(
            img_table,
            pxls=pxls,
            colors=colors,
            radius_in=0,
            radius_out=entry_width // 5,
        )
        / 255.0
    )

    img_table = torch.cat([img_text, img_table], dim=-1)

    img = torch.cat(
        [
            torch.ones(img.shape[0], img.shape[1], W_text).to(
                dtype=dtype,
                device=device,
            ),
            img,
        ],
        dim=-1,
    )
    img = torch.cat([img, img_table], dim=1)

    return img


def draw_bboxs(img, bboxs, color=(255, 255, 255), line_width=2):
    # img: 3xHxW, bboxs: Nx[x0, y0, x1, y1]
    if isinstance(bboxs, torch.Tensor) and bboxs.dim() == 1:
        img = draw_bbox(img=img, bbox=bboxs, color=color, line_width=line_width)
    else:
        for bbox in bboxs:
            img = draw_bbox(img=img, bbox=bbox, color=color, line_width=line_width)
    return img


def draw_bbox(img, bbox, color=(255, 255, 255), line_width=2):
    # img: 3xHxW, bbox: [x0, y0, x1, y1]

    device = img.device

    bbox = bbox.detach().cpu().numpy()
    bbox = np.round(bbox).astype(np.int32)
    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, line_width)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device=device, dtype=torch.uint8)
    return img


def get_bboxs_from_masks(masks):
    # masks: ...xHxW
    # bboxs: ...x4
    H, W = masks.shape[-2:]
    masks_batches = masks.shape[:-2]
    masks = masks.reshape(-1, H, W)
    from torchvision.ops import masks_to_boxes

    try:
        boxes = masks_to_boxes(masks > 0.5).long()
    except Exception:
        boxes = torch.ones((masks.shape[0], 4)).long()
        boxes[:, 0] = 0
        boxes[:, 1] = 0
        boxes[:, 2] = W - 1
        boxes[:, 3] = H - 1
    boxes = boxes.reshape(*masks_batches, 4)
    return boxes



def draw_gradient_line(img, p0, p1, c0, c1, steps=10, thickness=2):
    p0 = np.array(p0)
    p1 = np.array(p1)
    c0 = np.array(c0)
    c1 = np.array(c1)

    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps

        pt_start = (1 - t0) * p0 + t0 * p1
        pt_end   = (1 - t1) * p0 + t1 * p1

        color = (1 - t0) * c0 + t0 * c1

        cv2.line(
            img,
            pt1=tuple(pt_start.astype(int)),
            pt2=tuple(pt_end.astype(int)),
            color=tuple(color.astype(int).tolist()),
            thickness=thickness,
        )
    return img



def batch_draw_bbox3d(imgs, objs_size3d, cams_intr4x4, cams_tform4x4_obj, cams_scale1d=None, down_sample_rate=None, thickness=2):
    B = len(imgs)
    for b in range(B):
        imgs[b] = draw_bbox3d(img=imgs[b], obj_size3d=objs_size3d[b], 
                              cam_intr4x4=cams_intr4x4[b], cam_tform4x4_obj=cams_tform4x4_obj[b], 
                              cam_scale1d=cams_scale1d[b] if cams_scale1d is not None else None, 
                              down_sample_rate=down_sample_rate, thickness=thickness)
    return imgs

def draw_bbox3d(img, obj_size3d, cam_intr4x4, cam_tform4x4_obj, cam_scale1d=None, down_sample_rate=None, thickness=2):
    from od3d_basic.cv.visual.blend import rgb_to_range01
    from od3d_basic.cv.geometry.transform import proj3d2d_broadcast, tform4x4_broadcast, cam_intr4x4_downsample
    from od3d_basic.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast, inv_tform4x4, get_a_tform4x4_b_scale1d_transl_only, get_a_tform4x4_b_scale1d_rot_and_transl
    from od3d_basic.cv.geometry.objects3d.meshes import Meshes

    bbox3d = Meshes.from_objs_size3d(obj_size3d=obj_size3d.reshape(-1, 3))
    if cam_scale1d is not None:
        cam_tform4x4_obj = get_a_tform4x4_b_scale1d_transl_only(a_tform4x4_b=cam_tform4x4_obj, scale1d=1. / cam_scale1d, clone=True)
    else:
        cam_tform4x4_obj = cam_tform4x4_obj.clone()
    
    if down_sample_rate is not None:
        cam_intr4x4, _ = cam_intr4x4_downsample(cams_intr4x4=cam_intr4x4.clone(), down_sample_rate=down_sample_rate)
    cam_proj4x4_obj = tform4x4_broadcast(cam_intr4x4, cam_tform4x4_obj)

    bbox3d_verts_proj2d = proj3d2d_broadcast(pts3d=bbox3d.verts, proj4x4=cam_proj4x4_obj)
    matrix_edges = (((bbox3d.verts[None, :, :] - bbox3d.verts[:, None, :]).abs() == 0.).sum(dim=-1) == 2)
    # get only upper diagonal edges (to avoid duplicates)
    matrix_edges = torch.triu(matrix_edges, diagonal=1)
    edge_indices = matrix_edges.nonzero(as_tuple=False)
    lines2d = bbox3d_verts_proj2d[edge_indices] # K x 2 x 2 (K, start/end, x/y)

    colors = (bbox3d.verts[edge_indices] > 0.) * 1. # K x 2 x 3 (K, start/end, r/g/b)
    img = draw_lines(img=img, lines=lines2d, colors=colors, thickness=thickness)

    return img

def draw_lines(img, lines, colors=None, thickness=2, steps=10):
    # lines: K x 2 x 2
    K = lines.shape[0]
    H = img.shape[-2]
    W = img.shape[-1]

    lines[..., 0] = lines[..., 0].clamp(-thickness, W + thickness)
    lines[..., 1] = lines[..., 1].clamp(-thickness, H + thickness)

    device = img.device
    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    if colors is None:
        colors = get_colors(K) * 1.0
    elif isinstance(colors, tuple) or isinstance(colors, list):
        colors = torch.from_numpy(np.tile(np.array(colors), reps=(K, 1)) / 255)
    colors = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
    for k in range(K):
        if colors.ndim == 3:
            img = draw_gradient_line(img, 
                                     p0=(int(lines[k, 0, 0].item()), int(lines[k, 0, 1].item())), 
                                     p1=(int(lines[k, 1, 0].item()), int(lines[k, 1, 1].item())),
                                     c0=colors[k, 0, :],
                                     c1=colors[k, 1, :],
                                     steps=steps,
                                     thickness=thickness)
        else:
            cv2.line(
                img,
                pt1=(int(lines[k, 0, 0].item()), int(lines[k, 0, 1].item())),
                pt2=(int(lines[k, 1, 0].item()), int(lines[k, 1, 1].item())),
                color=(
                    colors[k, 0].item(),
                    colors[k, 1].item(),
                    colors[k, 2].item(),
                ),
                thickness=thickness,
            )
    # img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device=device, dtype=torch.uint8)
    return img

    # cv2.line(image, start_point, end_point, color, thickness)


def draw_pixels(img, pxls, colors=None, radius_in=3, radius_out=5):
    if pxls.dim() > 2:
        K = pxls.shape[0]
        pxls = pxls.clone().reshape(-1, 2)
    else:
        # pxls: K x 2
        K, _ = pxls.shape

    N = pxls.shape[0]
    device = img.device
    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if colors is None:
        colors = get_colors(K) * 1.0
    elif isinstance(colors, tuple) or isinstance(colors, list):
        colors = torch.from_numpy(np.tile(np.array(colors), reps=(K, 1)) / 255)
    colors = (colors.detach().cpu().numpy() * 255).astype(np.uint8)

    # img = cv2.circle(img.copy(), (240, 240), 5, (255, 255, 255), -1)
    H, W = img.shape[:2]
    HW = max(H, W)
    pxls = pxls.clone().nan_to_num(-3 * HW).clamp(-3 * HW, 3 * HW)

    for k in range(N):
        cv2.circle(
            img,
            (int(pxls[k, 0].item()), int(pxls[k, 1].item())), # x, y
            radius_out,
            (
                colors[k // (N//K), 0].item(),
                colors[k // (N//K), 1].item(),
                colors[k // (N//K), 2].item(),
            ),
            -1,
        )
        if radius_in > 0:
            cv2.circle(
                img,
                (int(pxls[k, 0].item()), int(pxls[k, 1].item())),
                radius_in,
                (255, 255, 255),
                -1,
            )

    # img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device=device, dtype=torch.uint8)
    return img


def floats2colors(colors_floats, color_map=cv2.COLORMAP_VIRIDIS):
    # colors_floats: K,
    torch_colors = (
        torch.from_numpy(
            cv2.applyColorMap(
                tensor_to_cv_img(colors_floats.repeat(1, 1, 1)),
                color_map,
            ),
        )
        / 255.0
    ).flip(dims=(-1,))
    return torch_colors[0]


def get_colors(
    K,
    device=None,
    last_white=False,
    last_white_grey=False,
    first_white=False,
    K_rel=None,
    color_map=cv2.COLORMAP_HSV,  # jet, winter, cool ,magma, viridis, twilight, twilight shifted, cividis
    randperm=False,
):
    if last_white_grey:
        K = K - 2
        color_grey = 0.7 * torch.ones(size=(1, 3))
        color_white = torch.ones(size=(1, 3))
    elif last_white or first_white:
        K = K - 1
        color_white = torch.ones(size=(1, 3))

    if K > 0:
        if K_rel is None:
            colors_floats = (
                torch.arange(K).repeat(1, 1, 1).type(torch.float32) + 1.0
            ) / (K + 1)
        else:
            alpha_rel = 0.8
            colors_rel_floats = (
                torch.arange(K_rel).repeat(1, 1, 1).type(torch.float32) + 1.0
            ) / (K_rel + 1)
            colors_not_rel_floats = (
                torch.arange(K - K_rel).repeat(1, 1, 1).type(torch.float32) + 1.0
            ) / (K - K_rel + 1)
            colors_floats = torch.cat(
                (
                    colors_rel_floats * alpha_rel,
                    colors_rel_floats[:, :, -1:] * alpha_rel
                    + (1.0 - alpha_rel) * colors_not_rel_floats,
                ),
                dim=2,
            )
        torch_colors = (
            torch.from_numpy(
                cv2.applyColorMap(
                    tensor_to_cv_img(colors_floats),
                    color_map,
                ),
            )
            / 255.0
        )

        torch_colors = torch_colors[0]
    else:
        torch_colors = torch.ones((0, 3))

    if last_white_grey:
        torch_colors = torch.cat((torch_colors, color_grey, color_white), dim=0)
    elif last_white:
        torch_colors = torch.cat((torch_colors, color_white), dim=0)
    elif first_white:
        torch_colors = torch.cat((color_white, torch_colors), dim=0)

    if device is not None:
        torch_colors = torch_colors.to(device)

    if randperm:
        torch_colors = torch_colors[torch.randperm(len(torch_colors))]
    # K x 3
    return torch_colors


def draw_text_as_img(H: int, W: int, text: str, fontScale=1.0, lineThickness: int = 2):
    img = torch.zeros(size=(3, H, W))
    return draw_text_in_rgb(
        img,
        text=text,
        fontScale=fontScale,
        lineThickness=lineThickness,
    )


def draw_text_in_rgb(
    img,
    text="title0",
    fontScale=None,
    lineThickness: int = None,
    fontColor=(125, 125, 125),
    leftTopCornerOfText=None,
):

    # 3xHxW
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    if fontScale is None:
        fontScale = 1. * (W * 1. / 800.)

    if lineThickness is None:
        lineThickness = max(1, int( 2 * (W * 1. / 512.) ))

    if leftTopCornerOfText is None:
        left = int(10 * (W * 1. / 512.))
        top = int(20 * (W * 1. / 512.))
        leftTopCornerOfText = (left, top)

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, line in enumerate(text.split("\n")):
        gap = cv2.getTextSize(line, font, fontScale, lineThickness)[0][1] + 5
        topLeftCornerOfLine = (leftTopCornerOfText[0], leftTopCornerOfText[1] + gap * i)
        if (
            isinstance(fontColor[0], list)
            or isinstance(fontColor[0], tuple)
            or isinstance(fontColor[0], torch.Tensor)
        ):
            fontColorLine = fontColor[i]
            if isinstance(fontColorLine, torch.Tensor):
                fontColorLine = fontColorLine.tolist()
        else:
            fontColorLine = fontColor
        cv2.putText(
            img,
            line,
            topLeftCornerOfLine,
            font,
            fontScale,
            fontColorLine,
            lineThickness,
        )

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    if dtype == torch.uint8:
        img = img * 255.0
    img = img.to(device=device, dtype=dtype)

    return img




def draw_corresp_batch(rgbs, kpts2d_annots_gt, kpts2d_annots_pred, kpts2d_acc_pck01=None):
    from od3d_basic.cv.visual.show import imgs_to_img
    # logger.info(batch.name_unique)
    # kpts2d_acc_pck01 = results_batch['kpts2d_acc/pck01']
    # kpts2d_annots_gt = batch.kpts2d_annots
    # kpts2d_annots_pred = results_batch[f"kpts2d_annots"]
    # rgbs = batch.rgbs
    imgs = []
    B = len(rgbs)
    for b in range(B):
        rgb_src = rgb_to_range01(rgbs[b][0].clone())
        rgb_trg = rgb_to_range01(rgbs[b][1].clone())

        kpts2d_annots_gt_src = kpts2d_annots_gt[b][0].clone()
        kpts2d_annots_gt_trg = kpts2d_annots_gt[b][1].clone()
        kpts2d_annots_pred_trg = kpts2d_annots_pred[b].clone()

        if kpts2d_acc_pck01 is not None:
            kpts2d_acc_pck01_b = kpts2d_acc_pck01[b]
        else:
            kpts2d_acc_pck01_b = None

        imgs.append(draw_corresp(rgb_src, rgb_trg, kpts2d_annots_gt_src, kpts2d_annots_pred_trg,
                                 kpts2d_acc_pck01=kpts2d_acc_pck01_b))

    img = imgs_to_img(imgs)
    return img

def draw_corresp(rgb_src, rgb_trg, kpts2d_annots_gt_src, kpts2d_annots_pred_trg, kpts2d_acc_pck01=None,
                 kpts2d_modal_mask=None, kpts2d_annots_gt_trg=None, kpts2d_annots_gt_trg_syms=None, axes_lines2d_trg=None,
                 line_thickness = 3, radius_in = 3, radius_out = 7):
    from od3d_basic.cv.visual.show import imgs_to_img
    device = rgb_src.device

    kpts2d_annots_gt_src = kpts2d_annots_gt_src.to(device=device)
    kpts2d_annots_pred_trg = kpts2d_annots_pred_trg.to(device=device)

    if kpts2d_annots_gt_trg is not None:
        kpts2d_annots_gt_trg = kpts2d_annots_gt_trg.to(device=device)
    if kpts2d_annots_gt_trg_syms is not None:
        kpts2d_annots_gt_trg_syms = kpts2d_annots_gt_trg_syms.to(device=device)
    if kpts2d_acc_pck01 is not None:
        kpts2d_acc_pck01 = kpts2d_acc_pck01.to(device=device)
    if kpts2d_modal_mask is not None:
        kpts2d_modal_mask = kpts2d_modal_mask.to(device=device)
    if axes_lines2d_trg is not None:
        axes_lines2d_trg = axes_lines2d_trg.to(device=device)

    rgb_src = draw_pixels(rgb_src, pxls=kpts2d_annots_gt_src, radius_in=radius_in, radius_out=radius_out)
    rgb_trg = draw_pixels(rgb_trg, pxls=kpts2d_annots_pred_trg, radius_in=0, radius_out=radius_out)
    if kpts2d_annots_gt_trg is not None:
        rgb_trg = draw_pixels(rgb_trg, pxls=kpts2d_annots_gt_trg, radius_in=radius_in, radius_out=radius_out)
    if kpts2d_annots_gt_trg_syms is not None:
        rgb_trg = draw_pixels(rgb_trg, pxls=kpts2d_annots_gt_trg_syms, radius_in=radius_in, radius_out=radius_out)
    if axes_lines2d_trg is not None:
        rgb_trg = draw_lines(img=rgb_trg, lines=axes_lines2d_trg)
    # rgb.append(rgb_orig)

    H, W = rgb_src.shape[-2:]
    img = imgs_to_img(torch.stack([rgb_src[None], rgb_trg[None]], dim=0), pad=0)

    kpts2d_annots_pred_trg[..., 1] += H
    #kpts2d_annots_gt_src[..., 1] += H
    #kpts2d_annots_gt_src[..., 0] +=  W
    #kpts2d_annots_pred_trg[..., 0] += b * W

    green = torch.FloatTensor([0, 1., 0]).to(device)
    blue = torch.FloatTensor([0, 0, 1.]).to(device)
    red_green_blue = torch.FloatTensor([1., 1., 1.]).to(device)
    if kpts2d_acc_pck01 is not None:
        colors_ours = green[None,] * kpts2d_acc_pck01[:, None] + blue[None,] * (1. - kpts2d_acc_pck01[:, None])
    else:
        colors_ours = green[None,] * torch.ones_like(kpts2d_annots_gt_src[..., 0][:, None]).to(device)

    if kpts2d_modal_mask is not None:
        colors_ours = (colors_ours * 0.8 + 0.5 * red_green_blue[None,] * (1. - 1. * kpts2d_modal_mask[:, None])).clamp(0., 1.)

    img = draw_lines(img, lines=torch.stack([kpts2d_annots_gt_src.to(device), kpts2d_annots_pred_trg.to(device)], dim=1), colors=colors_ours,
                     thickness=line_thickness)

    return img



def draw_coordinate_frames(cam_intr4x4, cam_tform4x4_obj, rgb, down_sample_rate=None, size=None, obj_syms=None, thickness=2):
    return batch_draw_coordinate_frames(cam_intr4x4[None], cam_tform4x4_obj[None], rgb[None], 
                                        obj_syms=obj_syms[None,] if obj_syms is not None else None,
                                        down_sample_rate=down_sample_rate, size=size, thickness=thickness)[0]


def batch_draw_coordinate_frames(cam_intr4x4, cam_tform4x4_obj, rgb, down_sample_rate=None, obj_syms=None, size=None, thickness=2):
    device = cam_tform4x4_obj.device
    B = rgb.shape[0]

    from od3d_basic.cv.geometry.transform import cam_intr4x4_downsample, proj3d2d_broadcast, tform4x4_broadcast

    if down_sample_rate is not None and size is not None:
        cam_intr4x4_down, _ = cam_intr4x4_downsample(cams_intr4x4=cam_intr4x4, imgs_sizes=size,
                                                        down_sample_rate=down_sample_rate)
    else:
        cam_intr4x4_down = cam_intr4x4.clone()
    
    if obj_syms is None:
        lines3d = torch.stack([torch.eye(3) * 0, torch.eye(3)], dim=1).to(device=device)
    else:
        lines3d = torch.stack([-1 * torch.eye(3) , torch.eye(3)], dim=1).to(device=device)

    cam_proj4x4 = tform4x4_broadcast(cam_intr4x4_down, cam_tform4x4_obj)
    lines3d_first_dims = lines3d.shape[:-1]
    cam_proj4x4_first_dims = cam_proj4x4.shape[:-2]
    lines3d = lines3d.expand(*cam_proj4x4_first_dims, *lines3d_first_dims, 3)
    lines3d = lines3d.clone()

    if obj_syms is not None:
        # start 0ing if not axis 180degree symmetric
        lines3d[..., 0, :] *= (obj_syms[:, :, None].clone() == 2)
        
        lines3d = lines3d.clone()
        # start end 0ing if continous symmetric
        lines3d[..., 0, :] *= (obj_syms[:, :, None].clone() != -1)
        lines3d[..., 1, :] *= (obj_syms[:, :, None].clone() != -1)

    cam_proj4x4 = cam_proj4x4.expand(*lines3d_first_dims, *cam_proj4x4_first_dims, 4, 4).permute(*((list(
        range(len(lines3d_first_dims), len(lines3d_first_dims) + len(cam_proj4x4_first_dims)))) + list(
        range(len(lines3d_first_dims))) + [-2, -1]))
    lines_rgb = lines3d[..., 1, :]
    lines2d = proj3d2d_broadcast(lines3d, cam_proj4x4)
    lines2d = torch.nan_to_num(lines2d, nan=0., posinf=0., neginf=0.)

    from od3d_basic.cv.visual.draw import draw_lines
    for b in range(B):
        rgb[b] = draw_lines(rgb[b], lines=lines2d[b], colors=lines_rgb[b], thickness=thickness)
        rgb[b] = draw_lines(rgb[b], lines=lines2d[b] * 0., colors=lines_rgb[0], thickness=thickness)

    return rgb
