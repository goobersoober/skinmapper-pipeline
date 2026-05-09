"""
Train 2D Gaussian Splatting on the prepared dataset.

We invoke the upstream 2DGS train.py as a subprocess so we get all of its
optimisations (densification, pruning, normal regularisation) without having
to fork the codebase.

Tuned defaults for skin/limb capture (vs. the stock 2DGS scene defaults):
  - Lower iterations (10k vs 30k) — skin doesn't need ultra-fine detail
  - Stronger densification floor — smooth limbs don't need aggressive splits
  - Higher normal-consistency loss weight — favours coherent skin surfaces
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


def train_2dgs(workdir: Path, iterations: int = 10_000,
               normal_loss_weight: float = 0.10,
               depth_dir: Path | None = None) -> Path:
    """
    Run 2DGS training. Returns path to trained ply (point_cloud_<iter>.ply).
    """
    twod_gs_root = Path("/workspace/2dgs")
    out_dir = workdir / "output"
    out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(twod_gs_root / "train.py"),
        "-s", str(workdir),
        "-m", str(out_dir),
        "--iterations", str(iterations),
        "--lambda_normal", str(normal_loss_weight),
        "--save_iterations", str(iterations),
        "--quiet",
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{twod_gs_root}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(cmd, cwd=str(twod_gs_root), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"2DGS training failed (exit {proc.returncode})")

    # Find resulting ply
    ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply.exists():
        # Fall back to any iteration we can find
        candidates = sorted((out_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
        if not candidates:
            raise RuntimeError(f"No trained ply found in {out_dir}")
        ply = candidates[-1]
    return ply
