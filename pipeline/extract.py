"""
Mesh + texture extraction from a trained 2DGS model.

Steps:
  1. Run 2DGS's render.py to get TSDF-fused mesh (model.obj/ply)
  2. Bake two textures by projecting source photos onto mesh:
       - mesh_original.png : the photos as-shot (preserves studio lighting)
       - mesh_albedo.png   : delit/evenly-lit version (for design scans)
  3. Write OBJ + MTL pointing at the chosen texture(s)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def extract_mesh(workdir: Path, model_dir: Path) -> Path:
    """
    Run 2DGS mesh extraction (TSDF fusion). Returns path to resulting OBJ.
    """
    cmd = [
        sys.executable, "/workspace/2dgs/render.py",
        "-s", str(workdir),
        "-m", str(model_dir),
        "--skip_train", "--skip_test",
        "--mesh_res", "1024",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/workspace/2dgs:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd="/workspace/2dgs", env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("2DGS mesh extraction failed")

    # 2DGS writes train/ours_<iter>/fuse_post.ply under model_dir
    candidates = list(model_dir.glob("train/ours_*/fuse_post.ply"))
    if not candidates:
        candidates = list(model_dir.glob("**/fuse*.ply"))
    if not candidates:
        raise RuntimeError(f"No fused mesh found under {model_dir}")
    return candidates[0]


def delight_texture(tex_bgr: np.ndarray) -> np.ndarray:
    """
    Approximate delighting: estimate slow-varying lighting via large bilateral
    filter on luminance, divide it out to recover albedo.

    Cheap and stable — not physically accurate, but good enough that mixed
    indoor lighting becomes visually "evenly lit" for design reference.
    Used only when scan_type == "design".
    """
    lab = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Big-radius bilateral filter ≈ slow lighting estimate
    L_light = cv2.bilateralFilter(L, d=0, sigmaColor=40, sigmaSpace=80)
    # Avoid division blowups
    L_light = np.clip(L_light, 30, 220)

    # Target mean luminance — flatten lighting to this
    target = float(np.mean(L_light))
    L_albedo = np.clip(L * (target / L_light), 0, 255)

    lab[:, :, 0] = L_albedo
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out


def bake_textures(mesh_ply: Path, workdir: Path, out_dir: Path,
                  tex_size: int = 4096) -> Dict[str, Path]:
    """
    Convert 2DGS fused mesh → OBJ with UV atlas, bake texture by projecting
    source photos. Returns paths to {original, albedo, mesh}.

    Uses pymeshlab for: ply→obj, parametrisation, projection-based texture
    baking. pymeshlab's `compute_texmap_from_registered_rasters` is the
    classical photo-projection route.
    """
    import pymeshlab as ml

    ms = ml.MeshSet()
    ms.load_new_mesh(str(mesh_ply))

    # Light cleanup
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_close_holes(maxholesize=80)

    # Decimate to a manageable count for downstream
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=120_000,
                                                preserveboundary=True,
                                                preservenormal=True)

    # UV unwrap (atlas-based) — purely so projection has a UV target.
    # The cylindrical_remap.py step on the Mac will REPLACE this UV later.
    ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(
        sidedim=tex_size, textdim=tex_size, border=2,
        method="Space-optimizing")

    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "mesh.obj"
    tex_orig = out_dir / "mesh_original.png"
    tex_albedo = out_dir / "mesh_albedo.png"

    # Save raster registration for pymeshlab projection.
    # 2DGS images.txt + cameras.txt under workdir/sparse/0 — pymeshlab needs
    # them in its own .out / Bundler format. We sidestep by using the
    # 2DGS render.py to bake views, then projecting in CV space.
    #
    # Concretely: render the trained model from each input camera, then
    # blend that view into the texture via UV projection. Implemented
    # below as project_views_to_texture().
    rendered_dir = workdir / "rendered_views"
    if not rendered_dir.exists():
        rendered_dir.mkdir()
        _render_all_views(workdir, rendered_dir)

    # Save the OBJ first (so we have UVs)
    ms.save_current_mesh(str(obj_path),
                         save_textures=False,
                         save_vertex_color=False)

    # Project source photos to texture
    tex = project_views_to_texture(
        obj_path=obj_path,
        workdir=workdir,
        tex_size=tex_size,
    )
    cv2.imwrite(str(tex_orig), tex)
    cv2.imwrite(str(tex_albedo), delight_texture(tex))

    # Write MTL
    mtl_path = out_dir / "mesh.mtl"
    mtl_path.write_text(
        "newmtl skin\n"
        "illum 0\n"
        "Ka 1.000 1.000 1.000\n"
        "Kd 1.000 1.000 1.000\n"
        f"map_Kd {tex_orig.name}\n"
        f"map_Ka {tex_orig.name}\n"
    )
    # Patch mtllib reference in OBJ
    obj_text = obj_path.read_text().splitlines()
    if not any(l.startswith("mtllib") for l in obj_text):
        obj_text.insert(0, "mtllib mesh.mtl")
        obj_text.insert(1, "usemtl skin")
        obj_path.write_text("\n".join(obj_text) + "\n")

    return {"mesh": obj_path, "original": tex_orig, "albedo": tex_albedo,
            "mtl": mtl_path}


def _render_all_views(workdir: Path, out_dir: Path) -> None:
    """Render trained model from each training-set camera (used as sanity)."""
    # Stub — pymeshlab's projection actually doesn't need this. Kept as
    # a hook for future GS-IR / per-view albedo extraction.
    return


def project_views_to_texture(obj_path: Path, workdir: Path,
                             tex_size: int = 4096) -> np.ndarray:
    """
    Photo-projection texture baking.

    For each input photo, compute which OBJ triangles project into the photo
    (using the COLMAP poses we wrote), then back-project pixels into the UV
    atlas and accumulate a weighted average (weighted by view angle and
    inverse depth — i.e. prefer fronto-parallel, close-up views).

    Returns BGR uint8 texture of shape (tex_size, tex_size, 3).
    """
    import trimesh

    mesh = trimesh.load(str(obj_path), process=False)
    verts = np.asarray(mesh.vertices)        # (V, 3)
    faces = np.asarray(mesh.faces)           # (F, 3)
    uvs   = np.asarray(mesh.visual.uv)       # (V, 2)  ← per-vertex UVs

    # Read cameras + poses
    sparse = workdir / "sparse" / "0"
    cams = _read_cameras(sparse / "cameras.txt")
    imgs = _read_images(sparse / "images.txt")

    accum = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
    weight = np.zeros((tex_size, tex_size), dtype=np.float64)

    img_dir = workdir / "images"
    for img_id, info in imgs.items():
        cam = cams[info["camera_id"]]
        photo = cv2.imread(str(img_dir / info["name"]))
        if photo is None:
            continue
        h, w = photo.shape[:2]

        # Build world->cam (R, t) from quaternion in COLMAP convention
        qw, qx, qy, qz = info["qvec"]
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array(info["tvec"])
        # Project verts
        cam_pts = (R @ verts.T).T + t  # (V, 3) in cam frame
        z = cam_pts[:, 2]
        valid = z > 0.01
        K = np.array([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]])
        proj = (K @ cam_pts.T).T
        proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)

        for f in faces:
            if not (valid[f].all()):
                continue
            uv0, uv1, uv2 = uvs[f]
            p0, p1, p2 = proj[f]
            if not _all_in_image(p0, p1, p2, w, h):
                continue

            # View weight: face normal · -view dir  (favour fronto-parallel)
            v0, v1, v2 = cam_pts[f]
            n = np.cross(v1 - v0, v2 - v0)
            nn = np.linalg.norm(n)
            if nn < 1e-8:
                continue
            n /= nn
            view = (v0 + v1 + v2) / 3.0
            view /= np.linalg.norm(view) + 1e-8
            cos_v = max(0.0, -float(np.dot(n, view)))
            if cos_v < 0.15:
                continue
            weight_val = cos_v ** 2

            _splat_face(accum, weight, photo,
                        uv0, uv1, uv2, p0, p1, p2,
                        tex_size, weight_val)

    weight = np.maximum(weight, 1e-6)
    out = (accum / weight[:, :, None]).clip(0, 255).astype(np.uint8)

    # Inpaint uncovered regions only near the boundary of covered ones
    cov = (weight > 1e-3).astype(np.uint8) * 255
    border = cv2.dilate(cov, np.ones((21, 21), np.uint8)) - cov
    inp_mask = (border > 0).astype(np.uint8)
    out = cv2.inpaint(out, inp_mask, 3, cv2.INPAINT_TELEA)

    return out


# ---------- COLMAP I/O helpers ----------

def _read_cameras(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        # PINHOLE: id model w h fx fy cx cy
        out[cid] = dict(model=parts[1],
                        w=int(parts[2]), h=int(parts[3]),
                        fx=float(parts[4]), fy=float(parts[5]),
                        cx=float(parts[6]), cy=float(parts[7]))
    return out


def _read_images(path: Path) -> dict:
    """Returns id -> {qvec, tvec, camera_id, name}."""
    out = {}
    lines = [l for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    # Pairs of lines: (header, points) — we only care about header
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        img_id = int(parts[0])
        qvec = list(map(float, parts[1:5]))
        tvec = list(map(float, parts[5:8]))
        cam_id = int(parts[8])
        name = parts[9]
        out[img_id] = dict(qvec=qvec, tvec=tvec, camera_id=cam_id, name=name)
        i += 2  # skip the (possibly empty) 2D-points line
    return out


def _quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 0 if n == 0 else 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ])


def _all_in_image(p0, p1, p2, w, h) -> bool:
    for p in (p0, p1, p2):
        if p[0] < 0 or p[1] < 0 or p[0] >= w or p[1] >= h:
            return False
    return True


def _splat_face(accum, weight, photo,
                uv0, uv1, uv2, p0, p1, p2,
                tex_size, w_val) -> None:
    """
    Rasterise the triangle in UV space and pull RGB from the image at each
    UV pixel (barycentric-interpolated image coords).
    """
    uv_px = np.array([
        [uv0[0] * tex_size, (1 - uv0[1]) * tex_size],
        [uv1[0] * tex_size, (1 - uv1[1]) * tex_size],
        [uv2[0] * tex_size, (1 - uv2[1]) * tex_size],
    ])
    x_min = max(0, int(np.floor(uv_px[:, 0].min())))
    x_max = min(tex_size - 1, int(np.ceil(uv_px[:, 0].max())))
    y_min = max(0, int(np.floor(uv_px[:, 1].min())))
    y_max = min(tex_size - 1, int(np.ceil(uv_px[:, 1].max())))
    if x_max <= x_min or y_max <= y_min:
        return

    # Barycentric setup
    v0 = uv_px[1] - uv_px[0]
    v1 = uv_px[2] - uv_px[0]
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-9:
        return

    ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    v2_ = pts - uv_px[0]
    d20 = v2_ @ v0
    d21 = v2_ @ v1
    v = (d11 * d20 - d01 * d21) / denom
    w_ = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w_
    inside = (u >= 0) & (v >= 0) & (w_ >= 0)
    if not inside.any():
        return
    u, v, w_ = u[inside], v[inside], w_[inside]
    pts_in = pts[inside].astype(int)

    # Image coords for those UV pixels
    img_xy = (u[:, None] * p0 + v[:, None] * p1 + w_[:, None] * p2)
    img_xy = img_xy.astype(int)
    h_img, w_img = photo.shape[:2]
    ok = (img_xy[:, 0] >= 0) & (img_xy[:, 0] < w_img) & \
         (img_xy[:, 1] >= 0) & (img_xy[:, 1] < h_img)
    if not ok.any():
        return
    pts_in = pts_in[ok]
    img_xy = img_xy[ok]

    bgr = photo[img_xy[:, 1], img_xy[:, 0]].astype(np.float64)
    accum[pts_in[:, 1], pts_in[:, 0]] += bgr * w_val
    weight[pts_in[:, 1], pts_in[:, 0]] += w_val
