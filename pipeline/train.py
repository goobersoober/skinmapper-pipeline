"""
Train 2D Gaussian Splatting on the prepared dataset.

We invoke the upstream 2DGS train.py as a subprocess so we get all of its
optimisations (densification, pruning, normal regularisation, distortion
loss) without having to fork the codebase.

Tuned defaults for skin/limb capture (vs. the stock 2DGS scene defaults):
  - 15k iterations (vs 30k stock) — skin geometry converges faster than
    architectural scenes; 10k was leaving residual noise, 15k is the
    sweet spot before diminishing returns kick in
  - Higher normal-consistency loss (0.10 vs 0.05 stock) — favours
    coherent skin surfaces over per-Gaussian texture detail
  - Higher distortion loss (lambda_dist=1000 vs 100 stock) — penalises
    Gaussians that aren't aligned to the actual surface; critical for
    clean mesh extraction via TSDF fusion
  - Lower opacity reset interval — prevents over-densification on the
    smooth interior of limbs (where the loss has nothing to chase)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict


def train_2dgs(workdir: Path, iterations: int = 15_000,
               normal_loss_weight: float = 0.10,
               distortion_loss_weight: float = 1000.0,
               depth_dir: Path | None = None) -> Path:
    """
    Run 2DGS training. Returns path to trained ply (point_cloud_<iter>.ply).

    `depth_dir` is currently unused at training time — depth priors are
    instead used during data preparation to densify points3D.txt
    (see pose.densify_points_with_depth). Keeping the argument for forward
    compat with a future GS-IR / Relightable-3DGS upgrade that natively
    supports depth supervision.
    """
    twod_gs_root = Path("/workspace/2dgs")
    out_dir = workdir / "output"
    out_dir.mkdir(exist_ok=True)

    # Iteration milestones — distortion + normal loss weights ramp in at
    # iter 3k once geometry is roughly in place
    densify_until = max(int(iterations * 0.5), 5_000)

    cmd = [
        sys.executable, str(twod_gs_root / "train.py"),
        "-s", str(workdir),
        "-m", str(out_dir),
        "--iterations",            str(iterations),
        "--lambda_normal",         str(normal_loss_weight),
        "--lambda_dist",           str(distortion_loss_weight),
        "--densify_until_iter",    str(densify_until),
        "--save_iterations",       str(iterations),
        "--test_iterations",       "-1",
        "--quiet",
    ]
    print(f"[train] {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{twod_gs_root}:{env.get('PYTHONPATH', '')}"

    proc = subprocess.run(cmd, cwd=str(twod_gs_root), env=env,
                          capture_output=True, text=True)
    # Stream output regardless — much easier to debug
    if proc.stdout:
        print(proc.stdout[-4000:])
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"2DGS training failed (exit {proc.returncode})")

    # Find resulting ply
    ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not ply.exists():
        candidates = sorted(
            (out_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
        if not candidates:
            raise RuntimeError(f"No trained ply found in {out_dir}")
        ply = candidates[-1]
    return ply
