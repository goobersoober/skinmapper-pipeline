"""
RunPod serverless entry point.

Input shape (job["input"]):
    photos_url:   str   — presigned URL or http(s) URL to a zip of photos
    scan_type:    str   — "design" | "content" | "post_tattoo"  (default "design")
    body_part:    str   — "leg" | "forearm" | "calf" | "thigh" | ...
    iterations:   int   — optional, default 10000
    upload_url:   str   — presigned URL to PUT the result zip back

Output shape:
    {
      ok: bool,
      result_url?: str,        # echoes upload_url on success
      stats: { ... },
      error?: str,
    }

If `upload_url` is not given, the handler returns the OBJ + textures as
base64-encoded bytes inside the output (small zips only — prefer upload_url).
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import runpod  # type: ignore

sys.path.insert(0, "/workspace")

from pipeline.utils    import (list_photos, normalise_to_jpeg,
                               reject_blurry, make_workdir, cleanup)
from pipeline.segment  import segment_folder
from pipeline.depth    import estimate_depth_folder
from pipeline.pose     import estimate_poses
from pipeline.train    import train_2dgs
from pipeline.extract  import extract_mesh, bake_textures


# -------- helpers --------

def _download(url: str, dst: Path) -> None:
    print(f"[io] downloading {url} → {dst}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)


def _upload(url: str, src: Path) -> None:
    print(f"[io] uploading {src} → {url}", flush=True)
    data = src.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": "application/zip"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status >= 300:
            raise RuntimeError(f"upload failed: HTTP {r.status}")


def _zip_dir(folder: Path, zip_path: Path,
             include: list[str] | None = None) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(folder)
            if include and not any(p.name == n or p.name.endswith(n) for n in include):
                continue
            z.write(p, arcname=str(rel))


def _unzip(zip_path: Path, dst: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst)


# -------- core pipeline --------

def _step(label: str, t_prev: list[float]):
    """Log elapsed time since the previous step and reset the marker."""
    now = time.time()
    elapsed = now - t_prev[0]
    print(f"[step] {label}  (+{elapsed:.1f}s)", flush=True)
    t_prev[0] = now


def run_pipeline(job_input: dict) -> dict:
    t0 = time.time()
    t_step = [t0]
    stats: dict = {}

    scan_type = job_input.get("scan_type", "design")
    body_part = job_input.get("body_part", "leg")
    iterations = int(job_input.get("iterations", 15_000))

    work = make_workdir()
    print(f"[pipe] workdir = {work}", flush=True)
    print(f"[pipe] scan_type={scan_type} body_part={body_part} "
          f"iterations={iterations}", flush=True)

    try:
        # 1. Get photos
        zip_path = work / "photos.zip"
        photos_url = job_input.get("photos_url")
        if photos_url:
            _download(photos_url, zip_path)
        elif "photos_b64" in job_input:
            zip_path.write_bytes(base64.b64decode(job_input["photos_b64"]))
        else:
            raise ValueError("no photos_url or photos_b64 in input")

        raw_dir = work / "raw"
        raw_dir.mkdir()
        _unzip(zip_path, raw_dir)

        # Some users zip a single folder; flatten one level if needed
        photo_root = raw_dir
        sub = [p for p in raw_dir.iterdir() if p.is_dir()]
        if len(sub) == 1 and not list_photos(raw_dir):
            photo_root = sub[0]

        photos = list_photos(photo_root)
        if len(photos) < 15:
            raise ValueError(f"need at least 15 photos, got {len(photos)}")
        stats["photos_input"] = len(photos)
        _step(f"unzip — {len(photos)} photos", t_step)

        # 2. Blur rejection
        kept, rejected = reject_blurry(photos, threshold=60.0)
        stats["photos_blur_rejected"] = len(rejected)
        if len(kept) < 15:
            print(f"[pre] blur threshold too aggressive — keeping all", flush=True)
            kept = photos

        # 3. Normalise to JPEGs (HEIC→JPEG, downsize to ≤1600px)
        jpeg_dir = work / "jpegs"
        kept_jpegs = normalise_to_jpeg(kept, jpeg_dir, max_dim=1600)
        stats["photos_used"] = len(kept_jpegs)
        _step(f"preprocess — {len(kept_jpegs)} JPEGs ready "
              f"({len(rejected)} blur-rejected)", t_step)

        # 4. SAM 2 segmentation — produces binary masks (and a masked preview).
        # CRITICAL: masks are NOT applied before pose estimation. MASt3R needs
        # full visual context (including background features) to triangulate
        # camera positions reliably on low-texture skin. Masks are used later
        # to constrain 2DGS training and to bound the depth-prior point cloud.
        mask_dir   = work / "masks"
        masked_dir = work / "masked"   # preview only; not used for matching
        seg_stats = segment_folder(jpeg_dir, mask_dir, masked_dir,
                                   body_part=body_part)
        stats["segmentation"] = seg_stats
        _step(f"SAM 2 — {seg_stats}", t_step)

        # GUARD: catastrophic segmentation failure
        seg_total = sum(seg_stats.values())
        seg_real  = seg_stats.get("success", 0) + seg_stats.get("fallback", 0)
        if seg_total > 0 and seg_real / seg_total < 0.5:
            print(f"[seg] WARNING: only {seg_real}/{seg_total} photos "
                  f"segmented — masks will be unreliable. Disabling "
                  f"mask-based pixel rejection in texture bake.", flush=True)
            mask_dir = None  # downstream falls back to using all pixels

        # 5. Depth Anything v2 priors — masked to limb only
        depth_dir = work / "depths"
        n_depth = estimate_depth_folder(jpeg_dir, depth_dir, mask_dir=mask_dir)
        stats["depth_maps"] = n_depth
        _step(f"Depth Anything v2 — {n_depth} maps", t_step)

        # 6. MASt3R poses on the ORIGINAL JPEGs (backgrounds intact)
        jpeg_paths = sorted(p for p in jpeg_dir.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg"})
        pose_stats = estimate_poses(jpeg_paths, work)
        stats["pose"] = pose_stats
        _step(f"MASt3R poses — {pose_stats}", t_step)

        # GUARD: pose registration must include enough photos to be useful
        n_reg = pose_stats.get("n_registered", 0)
        if n_reg < max(10, int(0.7 * len(jpeg_paths))):
            raise RuntimeError(
                f"Pose estimation registered only {n_reg}/{len(jpeg_paths)} "
                f"photos — too few for reliable reconstruction. Likely cause: "
                f"insufficient overlap between consecutive shots, or motion blur"
            )

        # 6b. Densify points3D.txt with depth-prior projection. Cap total
        # added points to avoid OOM during 2DGS init.
        MAX_DENSIFY = 200_000
        per_view = max(1000, MAX_DENSIFY // max(1, len(jpeg_paths)))
        from pipeline.pose import densify_points_with_depth
        densify_stats = densify_points_with_depth(
            workdir=work, depth_dir=depth_dir, mask_dir=mask_dir,
            jpeg_paths=jpeg_paths, samples_per_view=per_view,
        )
        stats["depth_densify"] = densify_stats
        _step(f"Depth-init densify — {densify_stats}", t_step)

        # 7. 2DGS training
        ply = train_2dgs(work, iterations=iterations,
                         normal_loss_weight=0.10,
                         distortion_loss_weight=1000.0,
                         depth_dir=depth_dir)
        stats["trained_ply"] = str(ply)
        _step(f"2DGS training — {iterations} iters", t_step)

        # 8. Mesh extraction
        model_dir = work / "output"
        mesh_ply = extract_mesh(work, model_dir)
        _step(f"mesh extraction — {mesh_ply.name}", t_step)

        # GUARD: validate mesh has actual content
        try:
            import trimesh
            _check = trimesh.load(str(mesh_ply), process=False)
            n_v, n_f = len(_check.vertices), len(_check.faces)
            stats["mesh_raw"] = {"verts": n_v, "faces": n_f}
            print(f"[mesh] raw mesh: {n_v} verts, {n_f} faces", flush=True)
            if n_f < 1000:
                raise RuntimeError(
                    f"Extracted mesh has only {n_f} faces — TSDF fusion "
                    f"likely failed. Check 2DGS training quality."
                )
        except RuntimeError:
            raise
        except Exception as _mc_e:
            print(f"[mesh] mesh validation skipped: {_mc_e}", flush=True)

        # 9. Texture baking — masks gate which photo pixels contribute
        out_dir = work / "out"
        artefacts = bake_textures(mesh_ply, work, out_dir,
                                  tex_size=4096, mask_dir=mask_dir)
        _step(f"texture baking — {len(artefacts)} artefacts", t_step)

        # 10. Pick texture(s) by scan_type
        keep_original = scan_type in ("content", "post_tattoo", "both")
        keep_albedo   = scan_type in ("design", "both", "default") or scan_type not in (
            "content", "post_tattoo"
        )
        if keep_original and not keep_albedo:
            artefacts["albedo"].unlink(missing_ok=True)
            mtl = (
                "newmtl skin\nillum 0\n"
                "Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
                f"map_Kd {artefacts['original'].name}\n"
                f"map_Ka {artefacts['original'].name}\n"
            )
        elif keep_albedo and not keep_original:
            artefacts["original"].unlink(missing_ok=True)
            mtl = (
                "newmtl skin\nillum 0\n"
                "Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
                f"map_Kd {artefacts['albedo'].name}\n"
                f"map_Ka {artefacts['albedo'].name}\n"
            )
        else:
            mtl = (
                "newmtl skin\nillum 0\n"
                "Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\n"
                f"map_Kd {artefacts['original'].name}\n"
                f"map_Ka {artefacts['original'].name}\n"
            )
        artefacts["mtl"].write_text(mtl)

        # 11. Zip result
        result_zip = work / "result.zip"
        _zip_dir(out_dir, result_zip)

        # 12. Upload or return inline
        upload_url = job_input.get("upload_url")
        if upload_url:
            _upload(upload_url, result_zip)
            output = {"ok": True, "result_url": upload_url, "stats": stats}
        else:
            output = {
                "ok": True,
                "result_b64": base64.b64encode(result_zip.read_bytes()).decode(),
                "stats": stats,
            }

        stats["seconds"] = round(time.time() - t0, 1)
        return output

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "stats": stats,
                "seconds": round(time.time() - t0, 1)}
    finally:
        if os.environ.get("KEEP_WORKDIR") != "1":
            cleanup(work)


# -------- RunPod entry --------

def handler(job):
    return run_pipeline(job["input"])


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
