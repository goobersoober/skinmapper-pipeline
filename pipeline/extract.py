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


# 2DGS's render.py and mesh_utils.py have undeclared deps:
#   - mediapy (in render_utils.py)
#   - open3d  (in mesh_utils.py, used for TSDF mesh fusion)
# Neither is in our requirements.txt. Auto-install on first use so we
# don't need a Docker rebuild whenever 2DGS adds another import. Once
# requirements.txt covers them, this becomes a no-op.
_2DGS_DEPS = [
    ("mediapy", "mediapy"),
    ("open3d",  "open3d"),
    ("xatlas",  "xatlas"),    # robust UV atlas packer for texture bake
]


def _ensure_2dgs_deps() -> None:
    missing = []
    for mod, pkg in _2DGS_DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"[extract] installing missing 2DGS deps: {missing}", flush=True)
    # --ignore-installed skips trying to uninstall system distutils packages
    # like `blinker 1.4` that open3d pulls in as a transitive dep. Without
    # this, the install fails on Ubuntu Python with:
    #   "Cannot uninstall blinker 1.4 — distutils installed project"
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--ignore-installed", "blinker", *missing]
    )
    print(f"[extract] installed {missing}", flush=True)


def _ensure_pymeshlab_system_libs() -> None:
    """pymeshlab's IO plugins (libio_base.so etc) dlopen libOpenGL.so.0,
    libEGL.so.1, etc. The base image doesn't ship them. apt-install at
    runtime as root since RunPod containers run as root anyway. Permanent
    fix is in Dockerfile (libopengl0 libegl1).
    """
    lib_path = Path("/usr/lib/x86_64-linux-gnu/libOpenGL.so.0")
    if lib_path.exists():
        return
    print("[extract] installing pymeshlab system libs (libopengl0 libegl1)",
          flush=True)
    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    try:
        subprocess.check_call(
            ["apt-get", "update", "-qq"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["apt-get", "install", "-y", "--no-install-recommends",
             "libopengl0", "libegl1"],
            env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("[extract] system libs installed", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[extract] apt install failed ({e}) — pymeshlab may not "
              "be able to read PLY files. Permanent fix is in Dockerfile.",
              flush=True)


def extract_mesh(workdir: Path, model_dir: Path) -> Path:
    """
    Run 2DGS mesh extraction (TSDF fusion). Returns path to resulting OBJ.
    """
    _ensure_2dgs_deps()
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
                  tex_size: int = 4096,
                  mask_dir: Path | None = None) -> Dict[str, Path]:
    """
    Convert 2DGS fused mesh → OBJ with UV atlas, bake texture by projecting
    source photos. Returns paths to {original, albedo, mesh}.

    Uses pymeshlab for: ply→obj, parametrisation, projection-based texture
    baking. pymeshlab's `compute_texmap_from_registered_rasters` is the
    classical photo-projection route.
    """
    # Make sure runtime deps are installed. The 2DGS-era pipeline called
    # _ensure_2dgs_deps() inside extract_mesh, but the depth-fusion path
    # skips extract_mesh entirely, so bake_textures is now the first
    # call site. Idempotent — no-op once installed.
    _ensure_2dgs_deps()
    _ensure_pymeshlab_system_libs()
    # pymeshlab is no longer used for decimation OR UV — both are broken
    # in v2025. Open3D handles decimation, xatlas handles UV, and we
    # write the OBJ manually. pymeshlab stays a runtime dep only because
    # later parts of bake (texture projection) still use it.

    # --- Open3D mesh decimation ---
    import open3d as o3d
    o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    n_in = len(o3d_mesh.triangles)
    print(f"[extract] loaded mesh: {len(o3d_mesh.vertices)} verts, "
          f"{n_in} faces", flush=True)
    TARGET_FACES = 60_000
    if n_in > TARGET_FACES:
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=TARGET_FACES,
        )
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_unreferenced_vertices()
    n_out = len(o3d_mesh.triangles)
    print(f"[extract] decimated to {n_out} faces", flush=True)

    # --- xatlas UV parametrisation ---
    # xatlas is the same library Blender and Substance use under the hood.
    # It produces clean non-overlapping UVs even for arbitrary topology.
    import xatlas
    vertices = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.uint32)
    print(f"[extract] xatlas parametrise: {len(vertices)} verts, "
          f"{len(faces)} faces", flush=True)
    vmapping, new_faces, uvs = xatlas.parametrize(vertices, faces)
    print(f"[extract] xatlas done: {len(vmapping)} unique uv-verts, "
          f"{len(new_faces)} faces", flush=True)
    # vmapping[i] = index into original `vertices` for new vertex i.
    # `new_faces` indexes into the new vertex array.
    new_vertices = vertices[vmapping]

    # --- Write OBJ with UVs manually (much faster + simpler than pymeshlab) ---
    # Textures saved as JPEG, not PNG. At 4096x4096 the PNG is ~15MB;
    # base64-wrapped in RunPod's JSON response that's ~20MB, which RunPod
    # rejects with 400 Bad Request on the job-done endpoint. JPEG q92
    # of a skin-texture atlas is ~1-3MB — fits comfortably.
    # (Once we wire up presigned S3 upload_url, this becomes moot and
    # we can go back to PNG for lossless texture.)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "mesh.obj"
    tex_orig = out_dir / "mesh_original.jpg"
    tex_albedo = out_dir / "mesh_albedo.jpg"
    with obj_path.open("w") as f:
        f.write("mtllib mesh.mtl\nusemtl skin\n")
        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        # OBJ is 1-indexed; pair each vertex with the matching UV index.
        for face in new_faces:
            a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
    print(f"[extract] wrote {obj_path}", flush=True)

    del o3d_mesh

    # Project source photos to texture (use SAM 2 masks to discount
    # background pixels — only limb pixels contribute to the bake)
    tex = project_views_to_texture(
        obj_path=obj_path,
        workdir=workdir,
        tex_size=tex_size,
        mask_dir=mask_dir,
    )
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    cv2.imwrite(str(tex_orig), tex, jpeg_params)
    cv2.imwrite(str(tex_albedo), delight_texture(tex), jpeg_params)

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


def project_views_to_texture(obj_path: Path, workdir: Path,
                             tex_size: int = 4096,
                             mask_dir: Path | None = None) -> np.ndarray:
    """
    Photo-projection texture baking — vectorised, GPU-friendly.

    Strategy: render the UV atlas to find which texel maps to which 3D point,
    then for each input photo, project all those 3D points into the photo
    in one matrix multiply. No per-face Python loop.

    Steps:
      1. Build a UV→3D lookup grid:
         For each texel (u, v), find which face it falls inside, compute
         the barycentric coords, and write the world-space 3D point and
         normal at that texel. (This is "rasterise the atlas".)
      2. For each photo:
         - Project the entire 3D point grid into the photo (one mat-mul).
         - Compute view angle and visibility.
         - Sample photo pixels at projected coords (cv2.remap is bilinear).
         - Accumulate into the texture with view-angle weighting.

    Returns BGR uint8 (tex_size, tex_size, 3).
    """
    import trimesh

    mesh = trimesh.load(str(obj_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)        # (V, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64)             # (F, 3)
    if getattr(mesh.visual, "uv", None) is None:
        raise RuntimeError(
            f"OBJ at {obj_path} has no UV coordinates — pymeshlab "
            "parametrisation step must have failed silently"
        )
    uvs = np.asarray(mesh.visual.uv, dtype=np.float64)         # (V, 2)
    if len(uvs) != len(verts):
        raise RuntimeError(
            f"UV/vertex count mismatch: {len(uvs)} UVs vs {len(verts)} verts"
        )

    # Step 1: rasterise UV atlas → per-texel (3D point, face_index)
    pos_map, face_map = _rasterise_uv_atlas(verts, faces, uvs, tex_size)
    # pos_map: (S, S, 3) world coords, NaN where uncovered
    # face_map: (S, S) int64 face indices, -1 where uncovered

    # Per-face normal (lazy; only used for view-angle weighting)
    f_v0 = verts[faces[:, 0]]
    f_v1 = verts[faces[:, 1]]
    f_v2 = verts[faces[:, 2]]
    f_n  = np.cross(f_v1 - f_v0, f_v2 - f_v0)
    f_n /= (np.linalg.norm(f_n, axis=1, keepdims=True) + 1e-8)
    # Lookup normal per texel
    face_idx = face_map.copy()
    face_idx[face_idx < 0] = 0
    nrm_map = f_n[face_idx]
    nrm_map[face_map < 0] = 0  # mask out

    # Read cameras + poses
    sparse = workdir / "sparse" / "0"
    cams = _read_cameras(sparse / "cameras.txt")
    imgs = _read_images(sparse / "images.txt")

    # BEST-VIEW selection instead of weighted averaging.
    # We accumulated colour over all views before, which mathematically
    # dulls saturation: bright red + several greys → washed-out pink.
    # Instead: for each texel, keep only the colour from the single
    # most-face-on photo (highest cos(view_angle)). Sharper, fully
    # saturated, true to the source photo. Same approach Polycam uses.
    best_color  = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    best_weight = np.zeros((tex_size, tex_size),    dtype=np.float32)

    valid_mask = (face_map >= 0)
    pos_flat = pos_map.reshape(-1, 3)
    nrm_flat = nrm_map.reshape(-1, 3)
    H_W = tex_size * tex_size

    img_dir = workdir / "images"
    for img_id, info in imgs.items():
        cam = cams[info["camera_id"]]
        photo = cv2.imread(str(img_dir / info["name"]))
        if photo is None:
            continue
        h, w = photo.shape[:2]

        # Optional: mask the photo so masked regions contribute zero
        photo_mask = None
        if mask_dir is not None:
            mp = mask_dir / (Path(info["name"]).stem + ".png")
            if mp.exists():
                photo_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if photo_mask is not None and photo_mask.shape != photo.shape[:2]:
                    photo_mask = cv2.resize(photo_mask, (w, h),
                                            interpolation=cv2.INTER_NEAREST)

        qw, qx, qy, qz = info["qvec"]
        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array(info["tvec"], dtype=np.float64)

        # Project all texel points: (N, 3) @ R.T + t  →  cam frame
        # Vectorised: cam_pts = pos_flat @ R.T + t
        cam_pts = pos_flat @ R.T + t  # (N, 3)
        z = cam_pts[:, 2]

        K = np.array([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]], dtype=np.float64)
        # Project
        proj = cam_pts @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u_px = proj[:, 0] / proj[:, 2]
            v_px = proj[:, 1] / proj[:, 2]

        # In-bounds + in-front mask
        inb = (z > 0.01) & (u_px >= 0) & (u_px < w - 1) & \
              (v_px >= 0) & (v_px < h - 1) & valid_mask.reshape(-1)

        if not inb.any():
            continue

        # View-angle weight: angle between face normal and view direction
        view = cam_pts / (np.linalg.norm(cam_pts, axis=1, keepdims=True) + 1e-8)
        # face normals are in WORLD frame; rotate to cam frame: n_cam = R @ n_world
        n_cam = nrm_flat @ R.T
        cos_v = -(n_cam * view).sum(axis=1)  # +ve when facing camera
        cos_v = np.clip(cos_v, 0, 1)
        # Lower threshold: 0.15 ≈ 81° from normal (very grazing) → reject
        inb &= (cos_v > 0.15)
        if not inb.any():
            continue

        # Sample photo at projected coords (bilinear via cv2.remap)
        map_x = np.full((tex_size, tex_size), -1, dtype=np.float32)
        map_y = np.full((tex_size, tex_size), -1, dtype=np.float32)
        map_x.flat[inb] = u_px[inb].astype(np.float32)
        map_y.flat[inb] = v_px[inb].astype(np.float32)

        sampled = cv2.remap(photo, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))

        w_tex = np.zeros(tex_size * tex_size, dtype=np.float32)
        w_tex[inb] = (cos_v[inb] ** 2).astype(np.float32)

        if photo_mask is not None:
            # Zero-out weight where the photo's own mask says background
            # (uses sampled mask via the same remap)
            mask_remap = cv2.remap(
                photo_mask, map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            w_tex *= (mask_remap.flatten() > 127).astype(np.float32)

        w_tex2d = w_tex.reshape(tex_size, tex_size)
        # Best-view update: where this photo's weight beats the current
        # best, replace colour AND weight. Everywhere else, leave as-is.
        better = w_tex2d > best_weight
        if better.any():
            best_color[better] = sampled[better].astype(np.float32)
            best_weight[better] = w_tex2d[better]

    out = best_color.clip(0, 255).astype(np.uint8)

    # Inpaint near the boundary of covered regions only — never invent
    # interior detail
    cov = (best_weight > 1e-3).astype(np.uint8) * 255
    border = cv2.dilate(cov, np.ones((21, 21), np.uint8)) - cov
    inp_mask = (border > 0).astype(np.uint8)
    out = cv2.inpaint(out, inp_mask, 3, cv2.INPAINT_TELEA)

    return out


# ---------- UV atlas rasterisation (helper) ----------

def _rasterise_uv_atlas(verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
                        tex_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    For each texel, find which 3D world-space point it represents.

    Returns:
        pos_map  : (tex_size, tex_size, 3) — world-space point at each texel
                   (zeros where no face covers that texel)
        face_map : (tex_size, tex_size)    — face index (-1 if uncovered)

    Uses cv2.fillConvexPoly with sub-pixel barycentric interpolation.
    """
    pos_map = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    face_map = -np.ones((tex_size, tex_size), dtype=np.int64)

    for fi, f in enumerate(faces):
        v0, v1, v2 = verts[f]
        uv0, uv1, uv2 = uvs[f]
        # UV px (note: V flipped — UVs are bottom-origin, image top-origin)
        p = np.array([
            [uv0[0] * tex_size, (1 - uv0[1]) * tex_size],
            [uv1[0] * tex_size, (1 - uv1[1]) * tex_size],
            [uv2[0] * tex_size, (1 - uv2[1]) * tex_size],
        ])
        x_min = max(0, int(np.floor(p[:, 0].min())))
        x_max = min(tex_size - 1, int(np.ceil(p[:, 0].max())))
        y_min = max(0, int(np.floor(p[:, 1].min())))
        y_max = min(tex_size - 1, int(np.ceil(p[:, 1].max())))
        if x_max <= x_min or y_max <= y_min:
            continue

        # Barycentric basis
        v0_uv = p[1] - p[0]
        v1_uv = p[2] - p[0]
        d00 = v0_uv @ v0_uv
        d01 = v0_uv @ v1_uv
        d11 = v1_uv @ v1_uv
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-9:
            continue

        ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
        pts = np.stack([xs, ys], axis=-1).astype(np.float32) - p[0]
        d20 = pts @ v0_uv
        d21 = pts @ v1_uv
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        if not inside.any():
            continue

        # Interpolate world position
        u_in = u[inside][..., None]
        v_in = v[inside][..., None]
        w_in = w[inside][..., None]
        pos = u_in * v0 + v_in * v1 + w_in * v2

        ys_in = ys[inside]
        xs_in = xs[inside]
        # Only write if not already written (atlas should be disjoint)
        empty = face_map[ys_in, xs_in] < 0
        ys_e, xs_e = ys_in[empty], xs_in[empty]
        pos_map[ys_e, xs_e] = pos[empty].astype(np.float32)
        face_map[ys_e, xs_e] = fi

    return pos_map, face_map


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


# (legacy per-face splat helpers removed — replaced by vectorised
# project_views_to_texture above)
