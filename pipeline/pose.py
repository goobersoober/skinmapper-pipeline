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
    from dust3r.image_pairs import make_pairs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    from pipeline.pair_select import (compute_dinov2_embeddings,
                                       top_k_pairs, make_retrieval_pairs)

    images = load_images([str(p) for p in image_paths], size=512)
    n = len(images)

    # Pair selection — same approach Polycam/KIRI use:
    #   1. Compute DINOv2 global feature per image
    #   2. For each image, take its top-k most similar images by cosine sim
    # This catches close-up + wide pairs of the same area regardless of
    # capture order, unlike a temporal sliding window.
    if n <= 16:
        pairs = make_pairs(images, scene_graph="complete",
                          prefilter=None, symmetrize=True)
        strategy = "complete"
    else:
        print(f"[pose] computing DINOv2 embeddings for {n} images...",
              flush=True)
        embeddings = compute_dinov2_embeddings(image_paths, device=device)
        # k=8 = solid pair density (still far better than temporal swin-k).
        # k=12 OOMs host RAM at ~pair 900 because each pair's view/pred
        # dicts hold ~80MB of feature maps and we accumulate all of them
        # before global_aligner runs. To raise this back to 12 we'd need
        # to stream pair outputs to disk between chunks (TODO).
        k = 8
        pair_idx = top_k_pairs(embeddings, k=k)
        pairs = make_retrieval_pairs(images, pair_idx, symmetrize=True)
        strategy = f"dinov2-top{k}"

    print(f"[pose] {n} images → {len(pairs)} pairs (strategy={strategy})",
          flush=True)

    # Load MASt3R AFTER DINOv2 has been freed (lower peak VRAM)
    ckpt = "/workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()

    # Helper: move every GPU tensor inside a (possibly nested) dict/list to CPU.
    # CRITICAL: without this, each chunk's outputs accumulate in GPU memory
    # and the loop OOMs around ~900 pairs even on a 24GB card.
    def _to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {key: _to_cpu(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [_to_cpu(val) for val in obj]
        if isinstance(obj, tuple):
            return tuple(_to_cpu(val) for val in obj)
        return obj

    # Chunked inference with BF16 mixed precision + graceful failure handling.
    # BF16 cuts VRAM use ~2× and is ~2× faster on Ampere/Hopper, with no
    # measurable quality loss for MASt3R's transformer architecture.
    #
    # MASt3R's inference() returns view1/view2/pred1/pred2 as DICTS where
    # values are stacked tensors of shape (chunk_size, ...) plus list-of-int
    # fields like `idx`. We collect the per-chunk dicts and then concatenate
    # them along dim 0 (tensors) / extend (lists) so global_aligner sees a
    # single combined dict, not a list of dicts.
    CHUNK = 32
    chunk_view1: list = []
    chunk_view2: list = []
    chunk_pred1: list = []
    chunk_pred2: list = []
    failed = 0
    for i in range(0, len(pairs), CHUNK):
        chunk = pairs[i:i + CHUNK]
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = inference(chunk, model, device,
                                batch_size=4, verbose=False)
            # Convert to CPU immediately — frees VRAM between chunks.
            chunk_view1.append(_to_cpu(out["view1"]))
            chunk_view2.append(_to_cpu(out["view2"]))
            chunk_pred1.append(_to_cpu(out["pred1"]))
            chunk_pred2.append(_to_cpu(out["pred2"]))
            del out
            print(f"[pose] inference {min(i+CHUNK, len(pairs))}/{len(pairs)} pairs",
                  flush=True)
        except Exception as e:
            failed += len(chunk)
            print(f"[pose] chunk {i}-{i+CHUNK} failed ({type(e).__name__}): {e}",
                  flush=True)
        torch.cuda.empty_cache()

    if failed > 0:
        ok = len(pairs) - failed
        print(f"[pose] {failed}/{len(pairs)} pairs failed — continuing with {ok}",
              flush=True)
        if ok < len(pairs) * 0.5:
            raise RuntimeError(
                f"Too many pair failures ({failed}/{len(pairs)}) — "
                "registration would be unreliable."
            )

    def _merge_chunk_dicts(chunks: list) -> dict:
        """Combine a list of per-chunk dicts into one concatenated dict."""
        if not chunks:
            return {}
        keys = list(chunks[0].keys())
        merged: dict = {}
        for k in keys:
            vals = [c[k] for c in chunks if k in c]
            if not vals:
                continue
            first = vals[0]
            if isinstance(first, torch.Tensor):
                merged[k] = torch.cat(vals, dim=0)
            elif isinstance(first, list):
                merged[k] = [item for sub in vals for item in sub]
            elif isinstance(first, tuple):
                # Flatten tuples too (rare)
                merged[k] = tuple(item for sub in vals for item in sub)
            else:
                # Scalar / non-collectable: just keep the first
                merged[k] = first
        return merged

    output = {
        "view1": _merge_chunk_dicts(chunk_view1),
        "view2": _merge_chunk_dicts(chunk_view2),
        "pred1": _merge_chunk_dicts(chunk_pred1),
        "pred2": _merge_chunk_dicts(chunk_pred2),
    }
    del chunk_view1, chunk_view2, chunk_pred1, chunk_pred2

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
    # MASt3R's `load_images(..., size=512)` resizes the *long edge* to 512
    # while preserving aspect ratio. So for a 1600×1200 input the working
    # frame is 512×384, and the intrinsics it returns are in that frame.
    # We therefore scale by a SINGLE long-edge ratio, not separate sx/sy.
    mast3r_long = 512
    with (sparse_dir / "cameras.txt").open("w") as f:
        for i, K in enumerate(intrinsics, start=1):
            img = Image.open(image_paths[i - 1])
            w, h = img.size
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            scale = max(w, h) / mast3r_long
            f.write(f"{i} PINHOLE {w} {h} "
                    f"{fx*scale:.4f} {fy*scale:.4f} "
                    f"{cx*scale:.4f} {cy*scale:.4f}\n")

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

    For each photo:
      1. Read the Depth Anything v2 relative-depth map + SAM 2 limb mask.
      2. Project the existing MASt3R world points into this camera. Their
         z-coords give us metric depth at known image pixels.
      3. Sample the relative depth at those same pixels.
      4. Least-squares fit  a*relative + b = metric  using RANSAC for
         robustness against outliers (depth-net failures, occlusions).
      5. Apply (a, b) to the full depth map → metric depth per pixel.
      6. Sample N points inside the mask, back-project to 3D world coords
         using the metric depth, append to points3D.txt.

    Why bother: 2DGS bootstraps from points3D.txt. The denser + more
    accurate the init, the faster + smoother the convergence — especially
    on textureless skin where photometric loss has nothing to chase.

    Returns: { added, aligned_views, fallback_views, ransac_views }
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

    # Read existing world points (from MASt3R) — used as alignment ground truth
    world_points = []
    if points_path.exists():
        for line in points_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                world_points.append((int(parts[0]),
                                     float(parts[1]), float(parts[2]),
                                     float(parts[3])))
            except ValueError:
                continue
    P_world_existing = np.array([[x, y, z] for _, x, y, z in world_points],
                                dtype=np.float64) if world_points else np.zeros((0, 3))
    next_id = (max((pid for pid, *_ in world_points), default=0) + 1
               if world_points else 1)

    rng = np.random.default_rng(42)
    added = 0
    aligned = 0
    ransac_views = 0
    fallback = 0

    def fit_ransac(rel: np.ndarray, met: np.ndarray,
                   iters: int = 200, inlier_thr: float = 0.05
                   ) -> tuple[float, float, int] | None:
        """RANSAC fit a*rel + b = met. Returns (a, b, n_inliers) or None."""
        n = len(rel)
        if n < 8:
            return None
        best = None
        for _ in range(iters):
            i, j = rng.choice(n, 2, replace=False)
            if abs(rel[i] - rel[j]) < 1e-3:
                continue
            a = (met[i] - met[j]) / (rel[i] - rel[j])
            b = met[i] - a * rel[i]
            if a < 1e-3:  # depth must increase with relative depth
                continue
            err = np.abs(a * rel + b - met)
            inliers = err < (inlier_thr * np.median(np.abs(met)) + 1e-3)
            n_in = int(inliers.sum())
            if best is None or n_in > best[2]:
                best = (a, b, n_in, inliers)
        if best is None or best[2] < 8:
            return None
        # Refit on inliers via least squares
        a, b, n_in, inliers = best
        A = np.stack([rel[inliers], np.ones(n_in)], axis=1)
        sol, *_ = np.linalg.lstsq(A, met[inliers], rcond=None)
        return float(sol[0]), float(sol[1]), n_in

    with points_path.open("a") as f:
        for jp in jpeg_paths:
            depth_p = depth_dir / (jp.stem + ".png")
            mask_p  = mask_dir  / (jp.stem + ".png") if mask_dir else None
            if not depth_p.exists():
                continue
            depth = cv2.imread(str(depth_p), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            depth = depth.astype(np.float32) / 65535.0

            mask = None
            if mask_p is not None and mask_p.exists():
                mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127

            pose = name_to_pose.get(jp.name)
            if pose is None:
                continue
            cam = cams[pose["camera_id"]]
            R_wc = quat_to_rot(pose["qvec"])      # world→cam
            t_wc = np.array(pose["tvec"])
            R_cw = R_wc.T
            t_cw = -R_wc.T @ t_wc

            h, w = depth.shape
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            scale_dx = w / cam["w"]
            scale_dy = h / cam["h"]

            # ---- Step A: project existing MASt3R points into this view ----
            # → produces (rel, metric) pairs for affine fitting
            (a_fit, b_fit) = (None, None)
            if len(P_world_existing) > 0:
                P_cam = (R_wc @ P_world_existing.T).T + t_wc
                z = P_cam[:, 2]
                in_front = z > 1e-3
                if in_front.any():
                    u_px = (P_cam[in_front, 0] / z[in_front] * fx + cx) * scale_dx
                    v_px = (P_cam[in_front, 1] / z[in_front] * fy + cy) * scale_dy
                    z_in = z[in_front]
                    inside = (u_px >= 0) & (u_px < w) & (v_px >= 0) & (v_px < h)
                    if inside.any():
                        u_i = u_px[inside].astype(np.int32)
                        v_i = v_px[inside].astype(np.int32)
                        z_i = z_in[inside]
                        rel_at_pts = depth[v_i, u_i]
                        # Drop pixels where depth is 0 (mask bg)
                        valid = rel_at_pts > 0.01
                        if valid.sum() >= 12:
                            res = fit_ransac(rel_at_pts[valid], z_i[valid])
                            if res is not None:
                                a_fit, b_fit, n_in = res
                                ransac_views += 1
                                aligned += 1

            # ---- Step B: fallback alignment if RANSAC failed ----
            if a_fit is None:
                scene_anchor = float(np.linalg.norm(t_cw))
                if scene_anchor < 1e-6:
                    fallback += 1
                    continue
                # Map relative percentile range to ±30% of scene_anchor
                d_min_pct = float(np.percentile(depth[depth > 0.01], 5))
                d_max_pct = float(np.percentile(depth[depth > 0.01], 95))
                if d_max_pct - d_min_pct < 1e-3:
                    fallback += 1
                    continue
                a_fit = (0.6 * scene_anchor) / (d_max_pct - d_min_pct)
                b_fit = scene_anchor * 0.7 - a_fit * d_min_pct
                fallback += 1

            # ---- Step C: sample mask interior, back-project ----
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

            d_rel = depth[ys, xs]
            valid = d_rel > 0.01
            if not valid.any():
                continue
            ys, xs, d_rel = ys[valid], xs[valid], d_rel[valid]

            d_metric = a_fit * d_rel + b_fit
            # Reject points behind the camera or absurdly far
            ok = (d_metric > 0.05) & (d_metric < 10.0)
            if not ok.any():
                continue
            ys, xs, d_metric = ys[ok], xs[ok], d_metric[ok]

            # Image coords → camera-frame rays (in *image* pixel units)
            x_cam = (xs - cx * scale_dx) / (fx * scale_dx) * d_metric
            y_cam = (ys - cy * scale_dy) / (fy * scale_dy) * d_metric
            P_cam = np.stack([x_cam, y_cam, d_metric], axis=1)
            P_world = (R_cw @ P_cam.T).T + t_cw

            for p in P_world:
                f.write(f"{next_id} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} "
                        f"220 200 180 0.5\n")
                next_id += 1
                added += 1

    return {"added": added, "aligned_views": aligned,
            "ransac_views": ransac_views, "fallback_views": fallback}
