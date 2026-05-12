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

# Heartbeat prints so RunPod logs always show which import is taking time.
# When the worker looks "hung" early on, it's usually one of these blocking.
def _hb(label: str) -> None:
    print(f"[boot] {time.strftime('%H:%M:%S')} {label}", flush=True)

_hb("handler.py top — runpod next")
import runpod  # type: ignore
_hb("runpod imported — adding sys.path /workspace")

sys.path.insert(0, "/workspace")

_hb("importing pipeline.utils")
from pipeline.utils    import (list_photos, normalise_to_jpeg,
                               reject_blurry, make_workdir, cleanup)
_hb("importing pipeline.segment (SAM2 + GroundingDINO)")
from pipeline.segment  import segment_folder
_hb("importing pipeline.depth (Depth Anything v2)")
from pipeline.depth    import estimate_depth_folder
_hb("importing pipeline.pose (MASt3R + DINOv2)")
from pipeline.pose     import estimate_poses
_hb("importing pipeline.train (2DGS)")
from pipeline.train    import train_2dgs
_hb("importing pipeline.extract (mesh + textures)")
from pipeline.extract  import extract_mesh, bake_textures
_hb("all pipeline modules imported — ready to receive jobs")


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

        # Sanity cap at 200 photos — enough for the largest realistic capture
        # (full backpiece with mixed close-up + wide shots). Subsample evenly
        # only if user uploads more, to guard against abuse / runaway captures.
        MAX_PHOTOS = 200
        if len(photos) > MAX_PHOTOS:
            step = len(photos) / MAX_PHOTOS
            photos = [photos[int(i * step)] for i in range(MAX_PHOTOS)]
            print(f"[pre] subsampled to {len(photos)} photos (was {stats['photos_input']})",
                  flush=True)

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

        # 6c. Apply masks to training images so 2DGS only reconstructs the
        # limb, not the surrounding scene. Save the originals first so
        # texture baking can restore them afterwards (project_views_to_texture
        # needs original photo colors, not blacked-out training versions).
        images_dir = work / "images"
        images_orig_dir = work / "images_orig"
        if mask_dir is not None and mask_dir.exists():
            import cv2
            import numpy as np
            images_orig_dir.mkdir(exist_ok=True)
            n_masked = 0
            for img_p in sorted(images_dir.iterdir()):
                if img_p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                # Save original so we can restore for texture baking
                shutil.copy2(img_p, images_orig_dir / img_p.name)

                mask_p = mask_dir / (img_p.stem + ".png")
                if not mask_p.exists():
                    continue
                img = cv2.imread(str(img_p))
                mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
                if img is None or mask is None:
                    continue
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                # Soft edge: erode mask 3px then gaussian-blur so the limb
                # boundary doesn't become a hard cliff in the reconstruction
                mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
                mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
                mask_f = (mask.astype(np.float32) / 255.0)[..., None]
                masked = (img.astype(np.float32) * mask_f).astype(np.uint8)
                cv2.imwrite(str(img_p), masked,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                n_masked += 1
            print(f"[pipe] masked {n_masked} training images "
                  f"(background → black, originals saved for bake)",
                  flush=True)
            stats["images_masked"] = n_masked
        _step("apply masks to training images", t_step)

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

        # 8b. Restore original (unmasked) images for texture baking. We
        # masked them to black for 2DGS training so geometry stayed inside
        # the limb; but project_views_to_texture needs the real photo
        # colors. mask_dir still gates which pixels are sampled, so the
        # baked texture stays clean either way.
        if images_orig_dir.exists():
            for orig in sorted(images_orig_dir.iterdir()):
                shutil.copy2(orig, images_dir / orig.name)
            print(f"[pipe] restored {sum(1 for _ in images_orig_dir.iterdir())} "
                  "original images for texture bake", flush=True)

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

        # 10b. Save a few mask previews + photo samples for debugging.
        # When the mesh comes back fragmented or off-shape, the first thing
        # to check is whether the SAM2 masks are tight to the limb.
        try:
            import cv2 as _cv2
            debug_dir = out_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            if mask_dir and mask_dir.exists():
                masks = sorted(mask_dir.glob("*.png"))
                # Take 4 evenly-spaced samples
                if masks:
                    step = max(1, len(masks) // 4)
                    samples = masks[::step][:4]
                    for m in samples:
                        photo = work / "images_orig" / (m.stem + ".jpg")
                        if not photo.exists():
                            photo = work / "jpegs" / (m.stem + ".jpg")
                        if photo.exists():
                            img = _cv2.imread(str(photo))
                            mask = _cv2.imread(str(m), _cv2.IMREAD_GRAYSCALE)
                            if img is not None and mask is not None:
                                mask3 = _cv2.cvtColor(mask, _cv2.COLOR_GRAY2BGR)
                                if mask3.shape != img.shape:
                                    mask3 = _cv2.resize(mask3,
                                        (img.shape[1], img.shape[0]))
                                # Side-by-side: original | mask | masked
                                masked = _cv2.bitwise_and(img, mask3)
                                combined = _cv2.hconcat([img, mask3, masked])
                                _cv2.imwrite(
                                    str(debug_dir / f"mask_preview_{m.stem}.jpg"),
                                    combined,
                                    [int(_cv2.IMWRITE_JPEG_QUALITY), 75])
                    print(f"[pipe] wrote {len(samples)} mask previews to debug/",
                          flush=True)
        except Exception as _e:
            print(f"[pipe] mask preview generation skipped: {_e}", flush=True)

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
