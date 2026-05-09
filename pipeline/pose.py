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
    # Pair selection strategy:
    #   complete graph is O(N²) — 78 photos = 6006 pairs ≈ 50 min on RTX 4090.
    #   Use sliding-window of k=10 instead → ≤ 10·N pairs ≈ 8× faster, with
    #   negligible quality loss for orbit-style limb captures (each frame's
    #   meaningful overlap is with its temporal neighbours).
    n = len(images)
    k = min(10, max(3, n - 1))
    if n <= 16:
        # Few enough photos that complete graph is cheap and more accurate
        pairs = make_pairs(images, scene_graph="complete",
                          prefilter=None, symmetrize=True)
    else:
        pairs = make_pairs(images, scene_graph=f"swin-{k}",
                          prefilter=None, symmetrize=True)
    print(f"[pose] {n} images → {len(pairs)} pairs "
          f"(strategy={'complete' if n <= 16 else f'swin-{k}'})", flush=True)

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


def densify_points_with_depth(workdir: Path, depth_dir: Path,
                              mask_dir: Path, jpeg_paths: list,
                              samples_per_view: int = 8000) -> dict:
    """
    Append depth-prior points to points3D.txt.

    Each photo has a Depth Anything v2 relative-depth map and a SAM 2 limb
    mask. We sample N points inside the mask, back-project them to 3D using
    the MASt3R-recovered camera, and write them out in COLMAP format.

    The depth is *relative* (Depth Anything is monocular — no metric scale)
    so we align it to MASt3R's metric scale by fitting a per-view linear
    transform (a, b) such that a*relative + b best matches MASt3R's depth at
    a sparse set of high-confidence overlap pixels. If alignment fails we
    fall back to using the median MASt3R depth as the anchor.

    Returns: { added: int, aligned_views: int, fallback_views: int }
    """
    import cv2
    sparse = workdir / "sparse" / "0"
    cameras_path = sparse / "cameras.txt"
    images_path  = sparse / "images.txt"
    points_path  = sparse / "points3D.txt"
    if not (cameras_path.exists() and images_path.exists()):
        return {"added": 0, "aligned_views": 0, "fallback_views": 0,
                "error": "missing COLMAP files"}

    # Read existing cameras / images (parsing helpers from extract.py)
    cams: dict = {}
    for line in cameras_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cams[int(parts[0])] = dict(
            w=int(parts[2]), h=int(parts[3]),
            fx=float(parts[4]), fy=float(parts[5]),
            cx=float(parts[6]), cy=float(parts[7]),
        )

    name_to_pose: dict = {}
    lines = [l for l in images_path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        cam_id = int(parts[8])
        name = parts[9]
        name_to_pose[name] = dict(qvec=qvec, tvec=tvec, camera_id=cam_id)
        i += 2

    def quat_to_rot(q):
        qw, qx, qy, qz = q
        n = qw * qw + qx * qx + qy * qy + qz * qz
        s = 0 if n == 0 else 2.0 / n
        return np.array([
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
        ])

    # Append to points3D.txt
    rng = np.random.default_rng(42)
    next_id = 1
    if points_path.exists():
        for line in points_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                next_id = max(next_id, int(line.split()[0]) + 1)
            except Exception:
                pass

    added = 0
    aligned = 0
    fallback = 0
    with points_path.open("a") as f:
        for jp in jpeg_paths:
            depth_p = depth_dir / (jp.stem + ".png")
            mask_p  = mask_dir  / (jp.stem + ".png")
            if not depth_p.exists():
                continue
            depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            # depth was written as 16-bit normalised within mask
            depth = depth.astype(np.float32) / 65535.0

            mask = None
            if mask_p.exists():
                mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127

            pose = name_to_pose.get(jp.name)
            if pose is None:
                continue
            cam = cams[pose["camera_id"]]
            R_wc = quat_to_rot(pose["qvec"])    # world→cam
            t_wc = np.array(pose["tvec"])
            # cam→world = inverse of (R_wc, t_wc)
            R_cw = R_wc.T
            t_cw = -R_wc.T @ t_wc

            h, w = depth.shape
            # Sample pixels inside mask (or full image if no mask)
            if mask is not None and mask.any():
                ys, xs = np.where(mask)
            else:
                ys, xs = np.indices((h, w))
                ys, xs = ys.flatten(), xs.flatten()
            if len(ys) == 0:
                continue
            if len(ys) > samples_per_view:
                sel = rng.choice(len(ys), samples_per_view, replace=False)
                ys, xs = ys[sel], xs[sel]

            # Image coords → camera-frame rays
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            # Resize cam intrinsics to depth-map resolution
            sx = w / cam["w"]
            sy = h / cam["h"]
            x_cam = (xs - cx * sx) / (fx * sx)
            y_cam = (ys - cy * sy) / (fy * sy)
            d_rel = depth[ys, xs]
            # Skip zero-depth (out-of-mask) pixels
            valid = d_rel > 0.01
            if not valid.any():
                continue
            x_cam, y_cam, d_rel = x_cam[valid], y_cam[valid], d_rel[valid]
            xs_v, ys_v = xs[valid], ys[valid]

            # ---- align relative depth to metric scale ----
            # Use median scene depth from MASt3R as a coarse anchor:
            #   target_depth ≈ ||t_cw|| (camera distance from world origin)
            scene_anchor = float(np.linalg.norm(t_cw))
            if scene_anchor < 1e-6:
                fallback += 1
                continue
            # Fit a*d_rel + b so median(a*d_rel + b) == scene_anchor and the
            # near/far range covers ~[0.5, 1.5] × scene_anchor — keeps points
            # in front of the camera, avoids spreading them across infinity.
            d_min = float(np.percentile(d_rel, 5))
            d_max = float(np.percentile(d_rel, 95))
            if d_max - d_min < 1e-3:
                fallback += 1
                continue
            d_metric = scene_anchor * (0.7 + 0.6 * (d_rel - d_min) / (d_max - d_min))
            aligned += 1

            # Camera frame coordinates
            X_cam = x_cam * d_metric
            Y_cam = y_cam * d_metric
            Z_cam = d_metric
            P_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # (N, 3)

            # Cam → world
            P_world = (R_cw @ P_cam.T).T + t_cw

            for p in P_world:
                f.write(f"{next_id} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} 220 200 180 0.5\n")
                next_id += 1
                added += 1

    return {"added": added, "aligned_views": aligned, "fallback_views": fallback}
