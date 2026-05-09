"""
MASt3R pose estimation. Replaces COLMAP for sparse-view, low-texture inputs.

Output is the directory layout 2DGS / 3DGS expect:
  workdir/
    images/   <- the input JPEGs (already there)
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Add MASt3R to path (Dockerfile sets PYTHONPATH but be explicit)
sys.path.insert(0, "/workspace/mast3r")
sys.path.insert(0, "/workspace/mast3r/dust3r")


def estimate_poses(image_paths: List[Path], workdir: Path,
                   device: str = "cuda") -> Dict[str, object]:
    """
    Run MASt3R global alignment over all images. Write COLMAP-format sparse
    reconstruction into workdir/sparse/0/.

    Returns: { 'n_registered': int, 'n_points': int }
    """
    from mast3r.model import AsymmetricMASt3R
    from mast3r.utils.coarse_to_fine import coarse_matching
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    ckpt = "/workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()

    images = load_images([str(p) for p in image_paths], size=512)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None,
                      symmetrize=True)

    output = inference(pairs, model, device, batch_size=1, verbose=False)

    scene = global_aligner(output, device=device,
                           mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine",
                                   lr=0.01)

    poses = scene.get_im_poses().detach().cpu().numpy()      # (N, 4, 4)
    intrinsics = scene.get_intrinsics().detach().cpu().numpy()  # (N, 3, 3)
    pts3d = scene.get_pts3d()                                 # list of (h, w, 3) per view
    confs = scene.get_conf()                                  # list of (h, w)

    sparse_dir = workdir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = workdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Make sure images are in workdir/images
    for p in image_paths:
        dst = images_dir / p.name
        if not dst.exists():
            dst.write_bytes(p.read_bytes())

    # ----- write cameras.txt (one PINHOLE per image) -----
    with (sparse_dir / "cameras.txt").open("w") as f:
        for i, K in enumerate(intrinsics, start=1):
            img = Image.open(image_paths[i - 1])
            w, h = img.size
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            # Scale intrinsics from MASt3R's working size (512) to native res
            mast3r_size = 512
            sx = w / mast3r_size
            sy = h / mast3r_size
            f.write(f"{i} PINHOLE {w} {h} "
                    f"{fx*sx:.4f} {fy*sy:.4f} {cx*sx:.4f} {cy*sy:.4f}\n")

    # ----- write images.txt (extrinsic per image) -----
    def mat_to_quat(R: np.ndarray) -> np.ndarray:
        # COLMAP convention: qw, qx, qy, qz
        m = R
        t = np.trace(m)
        if t > 0:
            s = 0.5 / np.sqrt(t + 1.0)
            qw = 0.25 / s
            qx = (m[2, 1] - m[1, 2]) * s
            qy = (m[0, 2] - m[2, 0]) * s
            qz = (m[1, 0] - m[0, 1]) * s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                qw = (m[2, 1] - m[1, 2]) / s
                qx = 0.25 * s
                qy = (m[0, 1] + m[1, 0]) / s
                qz = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                qw = (m[0, 2] - m[2, 0]) / s
                qx = (m[0, 1] + m[1, 0]) / s
                qy = 0.25 * s
                qz = (m[1, 2] + m[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                qw = (m[1, 0] - m[0, 1]) / s
                qx = (m[0, 2] + m[2, 0]) / s
                qy = (m[1, 2] + m[2, 1]) / s
                qz = 0.25 * s
        return np.array([qw, qx, qy, qz])

    with (sparse_dir / "images.txt").open("w") as f:
        for i, T in enumerate(poses, start=1):
            # MASt3R returns world->cam? actually cam->world. Invert for COLMAP
            T_inv = np.linalg.inv(T)
            R = T_inv[:3, :3]
            t = T_inv[:3, 3]
            q = mat_to_quat(R)
            name = image_paths[i - 1].name
            f.write(f"{i} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {i} {name}\n")
            f.write("\n")  # empty 2D points line

    # ----- write points3D.txt — sample a sparse subset of the dense output -----
    n_pts = 0
    with (sparse_dir / "points3D.txt").open("w") as f:
        idx = 1
        for view_i, (pts, conf) in enumerate(zip(pts3d, confs)):
            pts_np = pts.detach().cpu().numpy()    # (h, w, 3)
            conf_np = conf.detach().cpu().numpy()  # (h, w)
            mask = conf_np > np.percentile(conf_np, 50)
            xyz = pts_np[mask]
            # Subsample to ~5k points per view
            if len(xyz) > 5000:
                sel = np.random.choice(len(xyz), 5000, replace=False)
                xyz = xyz[sel]
            for p in xyz:
                f.write(f"{idx} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 200 200 200 0.1\n")
                idx += 1
                n_pts += 1

    return {"n_registered": len(poses), "n_points": n_pts}
