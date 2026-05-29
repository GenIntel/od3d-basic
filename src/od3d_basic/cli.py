"""
o3x — od3d_basic command-line interface.

Usage:
  o3x dataset fetch     --config <yaml> [--url URL]
  o3x dataset index     --config <yaml> [--db FILE]
  o3x dataset visualize --config <yaml> [--db FILE] [--limit N] [--object-id ID]
                                         [--filter-has-kpts] [--render]
                                         [--render-frames N] [--renderer BACKEND]
  o3x platform setup    -p <platform>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── dataset sub-parser ────────────────────────────────────────────────────────

def _build_dataset_parser(sub):
    p = sub.add_parser("dataset", help="Dataset commands (fetch, index, visualize)")
    ds_sub = p.add_subparsers(dest="dataset_command", required=True)

    def _add_config(q):
        q.add_argument(
            "--config", required=True, type=Path, metavar="YAML",
            help="Path to a DatasetConfig YAML (must contain class_name)",
        )

    p_fetch = ds_sub.add_parser("fetch", help="Download / prepare the dataset")
    _add_config(p_fetch)
    p_fetch.add_argument("--url", default=None, metavar="URL")

    p_index = ds_sub.add_parser("index", help="Build SQLite index from on-disk data")
    _add_config(p_index)
    p_index.add_argument("--db", type=Path, default=None, metavar="FILE")

    p_vis = ds_sub.add_parser("visualize", help="Summarize and optionally render dataset objects")
    _add_config(p_vis)
    p_vis.add_argument("--db", type=Path, default=None, metavar="FILE")
    p_vis.add_argument("--limit", type=int, default=20, metavar="N")
    p_vis.add_argument("--object-id", default=None, metavar="ID")
    p_vis.add_argument("--filter-has-kpts", action="store_true")
    p_vis.add_argument("--render", action="store_true")
    p_vis.add_argument("--render-frames", type=int, default=4, metavar="N")
    p_vis.add_argument("--renderer", choices=["pyrender", "nvdiffrast"], default="pyrender")


def _run_dataset(args):
    from od3d_basic.dataset.cli import _load_class_from_config

    cls, cfg = _load_class_from_config(args.config)

    if args.dataset_command == "fetch":
        cls.fetch(cfg, url=args.url)
    elif args.dataset_command == "index":
        cls.index(cfg, db=args.db)
    elif args.dataset_command == "visualize":
        if args.filter_has_kpts:
            cfg.filter_has_kpts = True
        cls.visualize(
            cfg,
            db=args.db,
            limit=args.limit,
            object_id=args.object_id,
            render=args.render,
            render_frames=args.render_frames,
            renderer=args.renderer,
        )


# ── platform sub-parser ───────────────────────────────────────────────────────

def _build_platform_parser(sub):
    p = sub.add_parser("platform", help="Platform management commands")
    plat_sub = p.add_subparsers(dest="platform_command", required=True)

    p_setup = plat_sub.add_parser(
        "setup",
        help="Copy and run the repository setup script on a remote platform",
    )
    p_setup.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )

    p_overview = plat_sub.add_parser(
        "overview",
        help="Show job queue status on a remote platform",
    )
    p_overview.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )
    p_overview.add_argument(
        "--configs", action="store_true",
        help="Also print the resolved platform config",
    )

    p_runi = plat_sub.add_parser(
        "runi",
        help="Open an interactive shell on a compute node via srun --pty bash",
    )
    p_runi.add_argument(
        "-p", "--platform", default="slurm", metavar="PLATFORM",
        help="Platform name matching a config in configs/platform/ (default: slurm)",
    )


def _load_platform_config(platform: str):
    """Load a platform config using Hydra, with configs/platform/ as the config root."""
    from hydra import initialize_config_dir, compose

    configs_dir = (Path(__file__).parent.parent / "configs" / "platform").resolve()
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Platform config directory not found: {configs_dir}")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(configs_dir),
        job_name="platform_setup",
    ):
        cfg = compose(config_name=platform)

    return cfg, configs_dir



def _multiply_metric_with_unit(metric_with_unit: str, factor: int) -> str:
    """Multiply a value-with-unit string (e.g. '5gb') by an integer factor."""
    import re
    m = re.match(r"^(\d+(?:\.\d+)?)(\D+)$", str(metric_with_unit).strip())
    if m:
        return f"{int(float(m.group(1)) * factor)}{m.group(2)}"
    return str(metric_with_unit)


def _make_sbatch_script(cfg, job_name: str, env_vars: dict, remote_setup_script: str) -> str:
    """Return a complete sbatch script string built from platform config values."""

    node_count      = cfg.get("node_count", 1)
    gpu_count       = cfg.get("gpu_count_per_node", 1)
    cpu_count       = cfg.get("cpu_count_per_gpu", 8)
    ram_per_cpu     = cfg.get("ram_per_cpu", "5gb")
    walltime        = cfg.get("walltime", "24:00:00")
    partition       = cfg.get("partition", None)
    nodes_exclude   = cfg.get("nodes_exclude", None)
    restart         = cfg.get("restart_upon_fail", False)
    # path_home is defined in the custom overlay; fall back to path_ws
    path_home       = cfg.get("path_home", cfg.get("path_ws", "/tmp"))

    total_mem = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    optional = {
        "requeue":        "#SBATCH --requeue"                    if restart       else "",
        "partition":      f"#SBATCH --partition {partition}"     if partition     else "",
        "nodes_exclude":  f"#SBATCH --exclude {nodes_exclude}"   if nodes_exclude else "",
    }

    env_block = "\n".join(f"export {k}={v!r}" for k, v in env_vars.items())

    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH --nodes {node_count}",
        "#SBATCH --ntasks-per-node 1",
        f"#SBATCH --time {walltime}",
        f"#SBATCH --gres gpu:{gpu_count}",
        f"#SBATCH --cpus-per-task {cpu_count}",
        f"#SBATCH --mem {total_mem}",
        "#SBATCH --open-mode=append",
        f"#SBATCH -o {path_home}/slurm_jobs/%x_%j.o",
        "#SBATCH --mail-type=FAIL",
        "#SBATCH --signal=B:SIGUSR1@60",
    ]
    for v in optional.values():
        if v:
            lines.append(v)

    lines += [
        "",
        "set -euo pipefail",
        "",
        env_block,
        "",
        f"bash {remote_setup_script}",
    ]
    return "\n".join(lines) + "\n"


def _run_platform_setup(args):
    import re
    import subprocess
    from omegaconf import OmegaConf, open_dict

    platform = args.platform

    print(f"Loading platform config '{platform}'…")
    cfg, configs_dir = _load_platform_config(platform)

    with open_dict(cfg):
        cfg.setup = True

    print(OmegaConf.to_yaml(cfg))

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    path_ws        = cfg.get("path_ws", "")
    path_cuda      = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version = str(cfg.get("python_version", "3.10"))
    branch         = cfg.get("branch", "main")
    pull           = cfg.get("pull", True)
    pull_submodules = cfg.get("pull_submodules", True)
    username       = cfg.get("username", "")
    path_home      = cfg.get("path_home", path_ws)

    # Walk up from __file__ to find the outermost git repo (the repo that
    # contains od3d-basic as a submodule) via --show-superproject-working-tree.
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            text=True,
            cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        repo_name = Path(local_repo_root).name
    except subprocess.CalledProcessError:
        repo_name = "housecorr3d"
        local_repo_root = str(Path.cwd())

    # Inject the GitHub token into the HTTPS remote URL so git clone on the
    # cluster can authenticate through the proxy without SSH keys.
    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True, cwd=local_repo_root,
        ).strip()
        # Convert SSH → HTTPS if needed: git@github.com:Org/Repo → https://github.com/Org/Repo
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        # Strip any existing token then prepend the new one
        plain = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain.replace("https://", f"https://{token}@") if token else plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = Path(local_repo_root).name

    # Find the setup script in the parent repo
    setup_script_local = Path(local_repo_root) / "setup" / "setup_slurm.sh"
    if not setup_script_local.is_file():
        raise FileNotFoundError(f"Setup script not found: {setup_script_local}")

    remote_setup  = f"{path_ws}/setup_slurm.sh"
    remote_sbatch = f"{path_ws}/setup_slurm_job.sh"

    def _scp(local, remote):
        target = f"{ssh_host}:{remote}"
        if username:
            target = f"{username}@{ssh_host}:{remote}"
        print(f"Copying {local} → {target}")
        subprocess.run(["scp", str(local), target], check=True)

    # Build sbatch wrapper with #SBATCH headers from the platform config
    _proxy = "http://tfproxy.informatik.intra.uni-freiburg.de:8080"
    env_vars = {
        "PATH_WS":         path_ws,
        "PATH_CUDA":       path_cuda,
        "PYTHON_VERSION":  python_version,
        "REPO_URL":        repo_url,   # housecorr3d HTTPS URL with token
        "REPO_NAME":       repo_name,  # derived from remote URL, e.g. HouseCorr3Dv2
        "BRANCH":          branch,
        "PULL":            "true" if pull else "false",
        "PULL_SUBMODULES": "true" if pull_submodules else "false",
        "HTTP_PROXY":      _proxy,
        "HTTPS_PROXY":     _proxy,
        "http_proxy":      _proxy,
        "https_proxy":     _proxy,
    }
    sbatch_script = _make_sbatch_script(
        cfg,
        job_name=f"setup_{repo_name}",
        env_vars=env_vars,
        remote_setup_script=remote_setup,
    )

    # Write sbatch script to a temp file and SCP both scripts
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
        tmp.write(sbatch_script)
        tmp_path = tmp.name

    try:
        _scp(setup_script_local, remote_setup)
        _scp(tmp_path, remote_sbatch)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Ensure output log directory exists, then submit via sbatch
    remote_cmd = (
        f"mkdir -p {path_home}/slurm_jobs && "
        f"chmod +x {remote_setup} {remote_sbatch} && "
        f"sbatch {remote_sbatch}"
    )
    print(f"Submitting setup job on {ssh_host}…")
    subprocess.run(["ssh", ssh_host, remote_cmd], check=True)


def _fetch_jobs(ssh_host: str, username: str) -> list:
    """Return job list from sacct (last 24 h) as a list of dicts."""
    import subprocess
    fields      = ["JobID", "JobName", "State", "ExitCode", "Elapsed", "Start", "End", "Partition", "NodeList"]
    start_expr  = "$(date -d '24 hours ago' +'%Y-%m-%dT%H:%M:%S')"
    cmd = (
        f"sacct --starttime={start_expr}"
        f" --format={','.join(fields)!r}"
        f" --parsable2 --allocations"
    )
    if username:
        cmd += f" -u {username}"
    result = subprocess.run(["ssh", ssh_host, cmd], capture_output=True, text=True, check=True)
    lines  = [l for l in result.stdout.splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    return [dict(zip(fields, row.split("|"))) for row in lines[1:]]


def _open_log(ssh_host: str, path_home: str, job: dict) -> None:
    """Open the job log in less via ssh -t. Prints an error if the file is missing."""
    import subprocess
    log_path = f"{path_home}/slurm_jobs/{job['JobName']}_{job['JobID']}.o"
    check = subprocess.run(
        ["ssh", ssh_host, f"test -f {log_path!r} && echo yes || echo no"],
        capture_output=True, text=True,
    )
    if check.stdout.strip() != "yes":
        print(f"\nLog not found: {log_path}")
        print("(Jobs not submitted via `o3x platform setup` may write logs elsewhere.)")
        input("Press Enter to return…")
        return
    subprocess.run(["ssh", "-t", ssh_host, f"less +G {log_path!r}"])


def _kill_job(ssh_host: str, job: dict) -> None:
    """Cancel a SLURM job via scancel after confirmation."""
    import subprocess
    job_id = job["JobID"]
    print(f"\nscancel {job_id} ({job['JobName']})  [y/N] ", end="", flush=True)
    if input().strip().lower() != "y":
        return
    result = subprocess.run(["ssh", ssh_host, f"scancel {job_id}"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"scancel failed: {result.stderr.strip()}")
    else:
        print(f"Job {job_id} cancelled.")
    input("Press Enter to continue…")


def _overview_tui(stdscr, jobs: list, ssh_host: str, title: str):
    """
    Curses TUI for job selection.
    Returns: ('view', job_dict) | ('refresh',) | ('quit',)
    """
    import curses

    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_BLACK,  curses.COLOR_CYAN)   # selected row
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_WHITE)  # column header
    curses.init_pair(3, curses.COLOR_RED,    -1)                  # FAILED / ERROR
    curses.init_pair(4, curses.COLOR_GREEN,  -1)                  # COMPLETED
    curses.init_pair(5, curses.COLOR_YELLOW, -1)                  # RUNNING / other
    curses.init_pair(6, curses.COLOR_WHITE,  curses.COLOR_BLUE)   # title / status bar

    COLS = [
        ("JobID",     10, ">"),
        ("JobName",   26, "<"),
        ("State",     14, "<"),
        ("ExitCode",   8, ">"),
        ("Elapsed",   10, "<"),
        ("Start",     19, "<"),
        ("Partition", 12, "<"),
    ]

    def fmt_row(job):
        parts = []
        for field, w, align in COLS:
            val = str(job.get(field, ""))
            val = val[:w] if align == "<" else val
            parts.append(f"{val:{align}{w}}")
        return "  ".join(parts)

    def state_attr(state):
        if any(s in state for s in ("FAIL", "ERROR", "TIMEOUT", "OUT_OF")):
            return curses.color_pair(3)
        if "COMPLET" in state:
            return curses.color_pair(4)
        return curses.color_pair(5)

    current = 0
    offset  = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # ── title bar ──────────────────────────────────────────────
        bar = f" {title}  [R] refresh  [K] kill  [Q] quit "
        stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        stdscr.addstr(0, 0, bar[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)

        # ── column header ──────────────────────────────────────────
        hdr_job  = {f: f for f, *_ in COLS}
        stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
        stdscr.addstr(1, 0, fmt_row(hdr_job)[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)

        # ── job rows ───────────────────────────────────────────────
        list_h  = h - 3          # title + header + status bar
        visible = jobs[offset: offset + list_h]
        for i, job in enumerate(visible):
            y   = i + 2
            idx = i + offset
            row = fmt_row(job)[:w - 1]
            if idx == current:
                stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(y, 0, row.ljust(w - 1))
                stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                attr = state_attr(job.get("State", ""))
                stdscr.attron(attr)
                stdscr.addstr(y, 0, row)
                stdscr.attroff(attr)

        # ── status bar ─────────────────────────────────────────────
        count = f"[{current + 1}/{len(jobs)}]" if jobs else "[0/0]"
        status = f" {count}  ↑↓ navigate   Enter / L : show logs   K kill   R refresh   Q quit"
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(h - 1, 0, status[:w - 1].ljust(w - 1))
        stdscr.attroff(curses.color_pair(6))

        stdscr.refresh()
        key = stdscr.getch()

        if key in (ord("q"), ord("Q"), 27):
            return ("quit",)
        elif key in (ord("r"), ord("R")):
            return ("refresh",)
        elif key == curses.KEY_UP:
            if current > 0:
                current -= 1
                if current < offset:
                    offset -= 1
        elif key == curses.KEY_DOWN:
            if current < len(jobs) - 1:
                current += 1
                if current >= offset + list_h:
                    offset += 1
        elif key in (ord("\n"), ord("l"), ord("L"), curses.KEY_ENTER) and jobs:
            return ("view", jobs[current])
        elif key in (ord("k"), ord("K")) and jobs:
            return ("kill", jobs[current])


def _run_platform_overview(args):
    import curses
    import subprocess
    from omegaconf import OmegaConf

    platform = args.platform
    cfg, _   = _load_platform_config(platform)

    ssh_host  = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    username  = cfg.get("username", "")
    path_home = cfg.get("path_home", cfg.get("path_ws", ""))

    if args.configs:
        print("=" * 60)
        print(f"Platform config: {platform}")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        input("Press Enter to open the job overview…")

    # ── active jobs: show as plain text before entering the TUI ─────
    squeue_fmt = "%.10i %.12P %.30j %.10u %.10T %.12M %.12l %.5D %R"
    squeue_cmd = f"squeue --format={squeue_fmt!r}"
    if username:
        squeue_cmd += f" -u {username}"
    print("=" * 60)
    print(f"Active jobs on {ssh_host}" + (f" (user: {username})" if username else ""))
    print("=" * 60)
    subprocess.run(["ssh", ssh_host, squeue_cmd], check=True)
    print()

    # ── TUI loop ────────────────────────────────────────────────────
    tui_title = f"SLURM overview · {ssh_host}" + (f" · {username}" if username else "") + " · last 24 h"
    jobs = _fetch_jobs(ssh_host, username)

    while True:
        action = curses.wrapper(lambda scr: _overview_tui(scr, jobs, ssh_host, tui_title))

        if action[0] == "quit":
            break
        elif action[0] == "refresh":
            jobs = _fetch_jobs(ssh_host, username)
        elif action[0] == "view":
            _open_log(ssh_host, path_home, action[1])
        elif action[0] == "kill":
            _kill_job(ssh_host, action[1])
            jobs = _fetch_jobs(ssh_host, username)


def _run_platform_runi(args):
    import subprocess

    cfg, _ = _load_platform_config(args.platform)

    ssh_host = cfg.get("ssh")
    if not ssh_host or ssh_host is False:
        raise ValueError(
            f"Platform '{args.platform}' has no ssh host configured (ssh: {ssh_host!r})"
        )

    partition     = cfg.get("partition", None)
    node_count    = cfg.get("node_count", 1)
    gpu_count     = cfg.get("gpu_count_per_node", 1)
    cpu_count     = cfg.get("cpu_count_per_gpu", 8)
    ram_per_cpu   = cfg.get("ram_per_cpu", "5gb")
    walltime      = cfg.get("walltime", "24:00:00")
    nodes_exclude = cfg.get("nodes_exclude", None)
    total_mem     = _multiply_metric_with_unit(ram_per_cpu, cpu_count)

    srun = (
        f"srun"
        f" --nodes {node_count}"
        f" --ntasks-per-node 1"
        f" --gres gpu:{gpu_count}"
        f" --cpus-per-task {cpu_count}"
        f" --mem {total_mem}"
        f" --time {walltime}"
    )
    if partition:
        srun += f" --partition {partition}"
    if nodes_exclude:
        srun += f" --exclude {nodes_exclude}"
    import os, re, shlex
    from omegaconf import OmegaConf
    from pathlib import Path

    path_ws    = cfg.get("path_ws", "")
    path_cuda       = cfg.get("path_cuda", "/usr/local/cuda-12.4")
    python_version  = str(cfg.get("python_version", "3.10"))
    branch          = str(cfg.get("branch", "main"))
    pull            = str(cfg.get("pull", True)).lower()
    pull_subs       = str(cfg.get("pull_submodules", True)).lower()

    cuda_tag   = "cu" + os.path.basename(path_cuda).replace("cuda-", "").replace(".", "")
    py_tag     = "py" + python_version.replace(".", "")
    venv_path  = f"{path_ws}/venv_{py_tag}_{cuda_tag}" if path_ws else ""

    # Derive REPO_URL (with token) and REPO_NAME from the local git remote —
    # same logic as _run_platform_setup.
    token = OmegaConf.select(cfg, "credentials.github.token", default="") or ""
    try:
        submodule_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, cwd=Path(__file__).parent,
        ).strip()
        superproject = subprocess.check_output(
            ["git", "rev-parse", "--show-superproject-working-tree"], text=True, cwd=submodule_root,
        ).strip()
        local_repo_root = superproject if superproject else submodule_root
        raw_remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, cwd=local_repo_root,
        ).strip()
        if raw_remote.startswith("git@"):
            raw_remote = re.sub(r"git@github\.com:", "https://github.com/", raw_remote)
        plain    = re.sub(r"https://[^@]+@", "https://", raw_remote)
        repo_url  = plain.replace("https://", f"https://{token}@") if token else plain
        repo_name = Path(re.sub(r"\.git$", "", plain.split("/")[-1])).name
    except subprocess.CalledProcessError:
        repo_url  = ""
        repo_name = ""

    if path_ws:
        srun += f" --chdir {path_ws}"
    _proxy = "http://tfproxy.informatik.intra.uni-freiburg.de:8080"
    srun += (
        f" --export=ALL"
        f",HTTP_PROXY={_proxy}"
        f",HTTPS_PROXY={_proxy}"
        f",http_proxy={_proxy}"
        f",https_proxy={_proxy}"
        f",PATH_WS={path_ws}"
        f",PATH_CUDA={path_cuda}"
        f",PYTHON_VERSION={python_version}"
        f",REPO_URL={repo_url}"
        f",REPO_NAME={repo_name}"
        f",BRANCH={branch}"
        f",PULL={pull}"
        f",PULL_SUBMODULES={pull_subs}"
        f",CUDA_HOME={path_cuda}"
        f",CUDACXX={path_cuda}/bin/nvcc"
    )

    # Write a small activation script to the remote so bash --init-file can
    # source it without wrapping srun in a bash -c subshell (which breaks the PTY).
    init_lines = [
        "[ -f ~/.bashrc ] && . ~/.bashrc",
        f"export PATH={path_cuda}/bin:$PATH",
        f"export LD_LIBRARY_PATH={path_cuda}/lib64:${{LD_LIBRARY_PATH:-}}",
        f"export CPATH=${{CPATH:-}}:{path_cuda}/targets/x86_64-linux/include",
        f"export LIBRARY_PATH=${{LIBRARY_PATH:-}}:{path_cuda}/targets/x86_64-linux/lib",
    ]
    if venv_path:
        init_lines.append(f"[ -d {venv_path} ] && source {venv_path}/bin/activate")
    remote_init = f"{path_ws}/.od3d_init" if path_ws else "~/.od3d_init"
    init_script = "\n".join(init_lines)
    subprocess.run(["ssh", ssh_host, f"cat > {remote_init}"], input=init_script, text=True, check=True)

    srun += f" --pty bash --init-file {remote_init}"

    print(f"Opening interactive session on {ssh_host} in {path_ws or '~'}…")
    subprocess.run(["ssh", "-t", ssh_host, srun])


def _run_platform(args):
    if args.platform_command == "setup":
        _run_platform_setup(args)
    elif args.platform_command == "overview":
        _run_platform_overview(args)
    elif args.platform_command == "runi":
        _run_platform_runi(args)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="o3x",
        description="od3d_basic CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _build_dataset_parser(sub)
    _build_platform_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "dataset":
        _run_dataset(args)
    elif args.command == "platform":
        _run_platform(args)


if __name__ == "__main__":
    main()
