import logging

logger = logging.getLogger(__name__)
import urllib.request
import time
import sys
from pathlib import Path
import tarfile
import zipfile
import shutil
import os
import gdown
from hydra import compose, initialize, initialize_config_dir
import inspect
from omegaconf import DictConfig, OmegaConf
import json
from typing import Dict
import subprocess
import re
import importlib
import datetime

import os
import shutil
import concurrent.futures
import re


def now_to_str():
    return dt_to_str(datetime.datetime.now())

def now_to_str_dir():
    return dt_to_str_dir(datetime.datetime.now())

def dt_to_str_dir(dt):
    return dt.strftime("%Y_%m_%d__%H_%M_%S")

def dt_from_str_dir(dt_str):
    return datetime.datetime.strptime(dt_str, "%Y_%m_%d__%H_%M_%S")

def dt_to_str(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def dt_from_str(dt_str):
    return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def td_to_hours(td):
    return td.seconds // 3600

def cfg_path_to_path(_path):
    path = (str(_path).replace(" ", "").replace("'", "").replace(".", "").replace(",", "").replace("[", "").replace("]", ""))
    path = Path(path)
    return path 

def td_to_mins(td):
    return td.seconds // 60


def is_fpath_video(fpath: Path):
    if fpath is None:
        return False
    fpath = Path(fpath)
    return fpath.suffix in [".mp4", ".avi", ".mov", ".mkv", ".webm"]


def is_fpath_image(fpath: Path):
    return fpath.suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def get_valid_fpath(fpath: Path):
    return Path(re.sub(r'''[\[\]\s,'""]''', "", str(fpath)))

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration),
    )
    sys.stdout.flush()


def download(url: str, fpath: Path):
    if fpath.exists():
        logging.warning(f"File {fpath} already exists. Skip download {url}.")
    else:
        if not fpath.parent.exists():
            fpath.parent.mkdir(parents=True)

        if "google.com" in url:
            gdown.download(url=url, output=str(fpath), fuzzy=True)
        else:
            urllib.request.urlretrieve(url, fpath, reporthook)


def unzip(fpath: Path, dst: Path, remove_zip=True):
    with zipfile.ZipFile(fpath, "r") as zip_ref:
        zip_ref.extractall(dst)
    if remove_zip:
        os.remove(fpath)


def untar(fpath: Path, dst: Path):
    file = tarfile.open(fpath)
    file.extractall(dst)
    file.close()
    os.remove(fpath)


def move_dir(src: Path, dst: Path):
    for _fpath in src.iterdir():
        shutil.move(_fpath, dst)
    shutil.rmtree(src)




def parallel_rmtree(path, workers=8):
    if not os.path.exists(path):
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for root, dirs, files in os.walk(path, topdown=False):
            # Delete files in parallel
            for f in files:
                file_path = os.path.join(root, f)
                futures.append(executor.submit(os.unlink, file_path))
            # Delete dirs in parallel
            for d in dirs:
                dir_path = os.path.join(root, d)
                futures.append(executor.submit(os.rmdir, dir_path))

        # Wait for all deletions to finish
        for f in futures:
            f.result()

    os.rmdir(path)

def rm_dir(path: Path, fast=False):
    if not fast:
        try:
            logger.info(f"removing directory {path}")
            #path.unlink() # only works for empty directory
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(e)
    else:
        removed_dir = False
        while not removed_dir:
            try:
                logger.info(f"removing directory fast {path}")
                parallel_rmtree(path)
                removed_dir = True
                # shutil.rmtree(path)

                # couldnt measure improvements

                #path.unlink() # only works for empty directory
                #run_cmd(cmd = f"rm -r {path}", logger=logger)

                # path_empty = path.parent.joinpath("empty")
                # path_empty.mkdir(exist_ok=True, parents=True)
                # run_cmd(cmd=f"rsync -av --delete {path_empty}/ {path}/", logger=logger)
                # shutil.rmtree(path_empty)
                # shutil.rmtree(path)

                # rsync -av --delete --files-from=/dev/null / /path/to/target/dir/
            except Exception as e:
                logger.warning(e)

from tqdm import tqdm
from omegaconf import open_dict
import multiprocessing


def load_single_hierarchical_config(
    procnum,
    return_dict,
    config_dir_rel,
    benchmark,
    platform,
    ablations,
    overrides,
):
    """worker function"""
    logger.info(f"start process {procnum}")
    return_dict[procnum] = procnum
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        overrides = (
            [
                f"+ablations/{Path(ablation).parent}={Path(ablation).stem}"
                for ablation in ablations
            ]
            + [
                "platform=" + platform,
            ]
            + overrides
        )
        cfg = compose(config_name=benchmark, overrides=overrides)

        with open_dict(cfg):
            cfg.ablation_name = "_".join(
                [Path(ablation).stem for ablation in ablations],
            )

        logger.info(cfg.ablation_name)
        return_dict[procnum] = cfg


def load_multiple_hierarchical_configs(
    benchmark="defaults",
    platform="local",
    multiple_ablations=[],
    mp=True,
):
    config_dir_rel = "../../config"
    cfgs = []
    overrides = []
    if mp:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        for a, ablations in tqdm(enumerate(multiple_ablations)):
            p = multiprocessing.Process(
                target=load_single_hierarchical_config,
                args=(
                    a,
                    return_dict,
                    config_dir_rel,
                    benchmark,
                    platform,
                    ablations,
                    overrides,
                ),
            )
            jobs.append(p)
            p.start()

        for a, proc in enumerate(jobs):
            proc.join()
            cfgs.append(return_dict[a])
    else:
        return_dict = {}
        for a, ablations in tqdm(enumerate(multiple_ablations)):
            load_single_hierarchical_config(a, return_dict, config_dir_rel, benchmark, platform, ablations,
                                            overrides)

            cfgs.append(return_dict[a])
    return cfgs


def load_hierarchical_config(
    benchmark="defaults",
    platform="local",
    ablations=[],
    overrides=[],
):
    config_dir_rel = "../../config"
    with initialize(version_base=None, config_path=config_dir_rel, job_name="test_app"):
        overrides = (
            [
                f"+ablations/{Path(ablation).parent}={Path(ablation).stem}"
                for ablation in ablations
            ]
            + ["platform=" + platform]
            + overrides
        )
        cfg = compose(config_name=benchmark, overrides=overrides)

        # if ablations is None:
        #    cfg = compose(config_name=benchmark, overrides=["platform=" + platform] + overrides)
        # else:
        #    cfg = compose(config_name=benchmark, overrides=["+ablations=" + ablation, "platform=" + platform] + overrides)
    return cfg


def read_config_intern(
    rfpath: Path = None,
    benchmark="defaults",
    platform="local",
    overrides=[],
):
    config_dir_rel = "../../config"

    try:
        with initialize(
            version_base=None,
            config_path=config_dir_rel,
            job_name="test_app",
        ):
            if rfpath is not None:
                cfg = compose(
                    config_name=benchmark,
                    overrides=[
                        f"+{rfpath.parent}=" + str(rfpath.stem),
                        "platform=" + platform,
                    ]
                    + overrides,
                )
            else:
                cfg = compose(
                    config_name=benchmark,
                    overrides=[
                        "platform=" + platform,
                    ]
                    + overrides,
                )

        cfg = cfg
        if rfpath is not None:
            for key in str(rfpath.parent).split("/"):
                cfg = cfg.get(key)
    except hydra.errors.ConfigCompositionException as e:
        fpath = Path("config").joinpath(rfpath)
        logger.warning(e)
        logger.warning("trying to read with OmegaConf")
        cfg = OmegaConf.load(fpath)

    return cfg


import hydra.errors


def read_config_extern(fpath: Path):
    try:
        with initialize_config_dir(
            config_dir=str(fpath.parent.absolute()),
            job_name="test_app_extern",
        ):
            cfg = compose(config_name=fpath.stem)
    except hydra.errors.ConfigCompositionException as e:
        logger.warning(e)
        logger.warning("trying to read with OmegaConf")
        cfg = OmegaConf.load(fpath)

    return cfg


def write_config_to_json_file(config: DictConfig, fpath: Path):
    write_json(config=dict(config), fpath=fpath)


def write_json(config: Dict, fpath: Path):
    fpath.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(fpath.expanduser(), "w") as outfile:
        json.dump(config, outfile)


def ensure_path(obj):
    return obj if isinstance(obj, Path) else Path(obj)


def read_json(fpath: Path):
    fpath = ensure_path(fpath)
    with open(fpath.expanduser()) as openfile:
        config = json.load(openfile)
    return config


def read_yaml(fpath: Path, resolve=True):
    cfg = OmegaConf.load(fpath)
    cfg = OmegaConf.to_container(cfg, resolve=resolve)
    return cfg


def run_cmd(cmd, logger, live=False, background=False):
    if logger is not None:
        logger.info(f"Run command {cmd}")
    if live:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
        )

        for line in process.stdout:
            # Print or process the live output as needed
            logger.info(line)
            # print(line, end='')

        # Wait for the subprocess to complete
        process.wait()

        # Retrieve the return code of the subprocess
        return_code = process.returncode
        logger.info(f"Return Code {return_code}")

    else:
        if not background:
            res = subprocess.run(cmd, capture_output=True, shell=True)
            if logger is not None:
                logger.info(res.stdout.decode("utf-8"))
                logger.info(res.stderr.decode("utf-8"))
            return res.stdout.decode("utf-8")
        else:
            # child = RunCmdBackgroundProcess(cmd, os.getpid())
            from multiprocessing import Process

            pid = os.getpid()
            child_proc = Process(target=run_child, args=(cmd, pid))
            child_proc.daemon = True
            child_proc.start()


def run_child(cmd, parent_pid):
    """
    Start a child process by running self._cmd.
    Wait until the parent process (self._parent) has died, then kill the
    child.
    """
    import psutil
    from time import sleep

    _parent = psutil.Process(pid=parent_pid)
    _child = subprocess.Popen(cmd, shell=True)
    try:
        # with open("log.txt", "a") as myfile:
        #    myfile.write(_parent.status())
        while (
            _parent.status() == psutil.STATUS_RUNNING
            or _parent.status() == psutil.STATUS_SLEEPING
        ):
            sleep(1)
    except psutil.NoSuchProcess:
        pass
    finally:
        _child.terminate()


def read_str_from_file(fpath: Path):
    with open(fpath) as file:
        data = file.read().rstrip()
    return data


def write_str_to_file(fpath: Path, text: str):
    with open(fpath, "w") as file:
        file.write(text)


from typing import List
from copy import deepcopy
from enum import Enum


def write_dict_as_yaml(fpath: Path, _dict: Dict, save_enum_as_str=False):
    if save_enum_as_str:
        _dict = {
            key: str(value) if isinstance(value, Enum) else value
            for key, value in deepcopy(_dict).items()
        }

    conf = OmegaConf.create(_dict)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as fp:  #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)


def read_dict_from_yaml(fpath: Path):
    #logger.info("start read...")
    with open(fpath) as fp:
        loaded = OmegaConf.load(fp.name)
    # fpath = "/data/lmbraid19/sommerl/datasets/Omni6DPose_Preprocess/subset/omni6dpose_test_bbox_mask_amodal_mangosteen.yaml"
    #logger.info("end read...")
    return loaded


# import pyarrow as pa
# import pyarrow.parquet as pq
# def save_dict_as_pandas_df(fpath: Path, _dict: Dict):
#
#     # Convert PyTorch tensors to NumPy arrays
#     data_np = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in _dict.items()}
#
#     # Convert NumPy arrays to PyArrow arrays
#     arrays = {key: pa.array(value) if isinstance(value, np.ndarray) else value for key, value in data_np.items()}
#
#     # Create a PyArrow Table from the arrays
#     table = pa.Table.from_pydict(arrays)
#
#     # Write the table to a Parquet file
#     pq.write_table(table, fpath)
#
# def load_dict_from_parquet(fpath):
#     # Read the Parquet file into a PyArrow Table
#     table = pq.read_table(fpath)
#     # Access the schema of the table
#     schema = table.schema
#
#     # Convert PyArrow arrays to NumPy arrays
#     arrays = {column.name: column.to_numpy() if schema.field(column.name).type == pa.Array else column for column in table.columns}
#
#     # Convert NumPy arrays to PyTorch tensors
#     data = {key: torch.tensor(value) if isinstance(value, np.array) else value for key, value in arrays.items()}
#
#     return data


def multiply_metric_with_unit(metric_with_unit, factor):

    # s = "40gb"

    match = re.fullmatch(r"(\d+)([a-zA-Z]+)", metric_with_unit)
    number = int(match.group(1))
    unit = match.group(2)

    # multiply
    new_value = number * factor

    result = f"{new_value}{unit}"

    return result

def write_list_as_yaml(fpath: Path, _list: List[str]):
    conf = OmegaConf.create(_list)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as fp:  #  tempfile.NamedTemporaryFile()
        OmegaConf.save(config=conf, f=fp.name)


def read_list_from_yaml(fpath: Path):
    with open(fpath) as fp:
        loaded = OmegaConf.load(fp.name)
    return loaded


def get_obj_from_config(*args, config: DictConfig, **kwargs):
    class_name_split = config.class_name.split(".")
    module_name = ".".join(class_name_split[:-1])
    class_name = class_name_split[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    config_kwargs = config.get("kwargs", {})
    config_kwargs = {key: config_kwargs.get(key) for key in inspect.getfullargspec(class_.__init__)[0][1:] if config_kwargs.get(key) is not None}
    return class_(*args, **{**kwargs, **config_kwargs})


def dict_depth(d):
    if isinstance(d, dict) or isinstance(d, DictConfig):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


_MESH_EXTS = {".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"}


def _mesh_to_trimesh(m):
    import trimesh
    import numpy as np
    vertices = m.verts.numpy()
    faces = m.faces.numpy()
    visual = None
    if m.vert_colors is not None:
        vc = (m.vert_colors.numpy() * 255).clip(0, 255).astype(np.uint8)
        visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
    elif m.texture is not None and m.verts_uvs is not None:
        from PIL import Image
        tex_np = (m.texture.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(tex_np)
        uv = m.verts_uvs.numpy().copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # flip y back to trimesh UV convention
        visual = trimesh.visual.TextureVisuals(uv=uv, image=pil_img)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def _pil_to_tensor(image):
    import torch
    import numpy as np
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr[:, :, :3]).permute(2, 0, 1)


def _load_trimesh_texture(m):
    mat = getattr(m.visual, "material", None)
    if mat is None:
        return None
    if hasattr(mat, "image") and mat.image is not None:
        return _pil_to_tensor(mat.image)
    if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
        return _pil_to_tensor(mat.baseColorTexture)
    return _flat_color_texture(m)


def _flat_color_texture(m):
    import torch
    import numpy as np
    mat = getattr(getattr(m, "visual", None), "material", None)
    if mat is None:
        return None
    if hasattr(mat, "main_color") and mat.main_color is not None:
        c = torch.from_numpy(mat.main_color[:3] / 255.0).float()
        return (c[:, None, None] * torch.ones(3, 500, 500))
    return None


def _load_mesh(entry: Path):
    from od3d_basic.data.modalities import Mesh
    mesh_file = None
    if entry.is_file() and entry.suffix.lower() in _MESH_EXTS:
        mesh_file = entry
    elif entry.is_dir():
        for ext in (".obj", ".ply", ".glb", ".gltf", ".stl", ".fbx"):
            candidates = sorted(entry.glob(f"*{ext}"))
            if candidates:
                mesh_file = candidates[0]
                break
    if mesh_file is None:
        return None, None
    try:
        import torch
        import trimesh
        import numpy as np

        # Load without force="mesh" to preserve UV/texture visuals; handle Scene
        loaded = trimesh.load(mesh_file)
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            m = geoms[0] if geoms else None
        else:
            m = loaded
        if m is None or not isinstance(m, trimesh.Trimesh):
            return None, None

        verts = torch.tensor(m.vertices, dtype=torch.float32)  # (V, 3)
        faces = torch.tensor(m.faces, dtype=torch.int64)       # (F, 3)

        # center and normalize to unit bounding box
        v_min = verts.min(dim=0).values
        v_max = verts.max(dim=0).values
        center = (v_min + v_max) * 0.5
        scale  = (v_max - v_min).max().clamp(min=1e-8).item()
        verts  = (verts - center) / scale

        # tform4x4: maps normalized → original  (original = scale * normalized + center)
        tform = torch.eye(4, dtype=torch.float32)
        tform[:3, :3] = torch.eye(3) * scale
        tform[:3,  3] = center

        vert_colors = None
        verts_uvs   = None
        faces_uvs   = None
        texture     = None

        # Check UV/texture first — TextureVisuals also exposes vertex_colors (baked,
        # often black), so the UV branch must take priority.
        if isinstance(m.visual, trimesh.visual.TextureVisuals):
            uv = m.visual.uv
            if uv is not None:
                verts_uvs = torch.from_numpy(np.array(uv, dtype=np.float32))
                verts_uvs[:, 1] = 1.0 - verts_uvs[:, 1]  # flip y to image convention
                faces_uvs = faces.clone()
                texture = _load_trimesh_texture(m)
            else:
                texture = _flat_color_texture(m)
                if texture is not None:
                    verts_uvs = 0.5 * torch.ones((verts.shape[0], 2), dtype=torch.float32)
                    faces_uvs = faces.clone()
        elif isinstance(m.visual, trimesh.visual.ColorVisuals):
            vc = m.visual.vertex_colors
            if vc is not None:
                vert_colors = torch.from_numpy(
                    np.array(vc, dtype=np.float32)[:, :3] / 255.0
                )

        return (
            Mesh(verts=verts, faces=faces,
                 vert_colors=vert_colors,
                 verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture=texture),
            tform,
        )
    except Exception:
        return None, None
