"""
Depth-map fusion + Poisson surface reconstruction.

Why this exists: 2DGS reconstruction requires real parallax across the
subject. For single-side vertical-pan tattoo scans (typical user
capture), the camera never goes around the limb — there's parallax up
and down but very little across the limb width. 2DGS in that case
produces fragmented "flat sheet" gaussians that TSDF can't fuse into
a coherent surface.

This module bypasses 2DGS entirely and instead:
  1. Reads the dense world-space point cloud written by
     pose.densify_points_with_depth (each photo's masked pixels
     back-projected using Depth Anything v2 metric-aligned depth).
  2. Builds an Open3D point cloud, estimates per-point normals.
  3. Runs Poisson surface reconstruction → a single smooth closed-ish
     mesh that wraps the photographed surface (a half-cylinder for
     single-side scans, a closed shape for orbital scans).
  4. Trims low-density regions (ghost geometry behind the camera path).

The output is a single PLY ready for the same xatlas + texture-bake
stage as before. Speed: ~30 sec on CPU vs ~6 min for 2DGS training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _read_points3d(path: Path) -> np.ndarray:
    """Load points3D.txt → (N, 3) float64 world-space coords."""
    pts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                pts.append((float(parts[1]),
                            float(parts[2]),
                            float(parts[3])))
            except ValueError:
                continue
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def fuse_depths_to_mesh(workdir: Path,
                        out_ply: Optional[Path] = None,
                        poisson_depth: int = 9,
                        density_quantile: float = 0.05) -> Path:
    """Fuse the dense depth-back-projected point cloud into a mesh.

    Parameters
    ----------
    workdir
        Pipeline workdir. Reads `sparse/0/points3D.txt`.
    out_ply
        Where to write the fused mesh. Defaults to
        `workdir/output/fuse_post.ply` so it mirrors the 2DGS path
        the rest of the pipeline expects.
    poisson_depth
        Octree depth for Poisson reconstruction. 9 = ~512^3 effective
        resolution. Bump to 10 for sharper detail at 2× memory + time.
    density_quantile
        Trim vertices whose Poisson density is below this quantile of
        the global density distribution. Removes ghost geometry that
        Poisson produces in empty regions far from any input points.
    """
    import open3d as o3d

    points_path = workdir / "sparse" / "0" / "points3D.txt"
    if not points_path.exists():
        raise FileNotFoundError(f"points3D.txt missing at {points_path}")

    xyz = _read_points3d(points_path)
    if len(xyz) < 5000:
        raise RuntimeError(
            f"too few densified points ({len(xyz)}) for Poisson fusion. "
            "Bump samples_per_view in densify_points_with_depth or "
            "check the depth/mask outputs."
        )
    print(f"[fuse] loaded {len(xyz):,} world points", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Outlier removal — drop isolated points that don't have ≥8 neighbours
    # within their natural neighbourhood. Cleans up RANSAC misalignments
    # and depth-map glitches at limb edges.
    pcd, _idx = pcd.remove_statistical_outlier(nb_neighbors=20,
                                               std_ratio=2.0)
    print(f"[fuse] after outlier removal: {len(pcd.points):,} points",
          flush=True)

    # Estimate normals — required for Poisson. Use a search radius based
    # on the point cloud's spatial extent so it auto-scales.
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = float(np.linalg.norm(bbox.get_extent()))
    radius_n = extent * 0.01     # 1% of scene diagonal — typical good choice
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_n, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)
    print(f"[fuse] normals estimated (radius {radius_n:.4f})", flush=True)

    # Poisson surface reconstruction
    mesh, densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False)
    )
    print(f"[fuse] Poisson done: {len(mesh.vertices):,} verts, "
          f"{len(mesh.triangles):,} faces", flush=True)

    # Trim ghost geometry: Poisson always produces a watertight surface,
    # but verts in regions where the input point cloud was sparse have
    # very low density and are usually nonsense. Drop them.
    densities = np.asarray(densities)
    if density_quantile > 0 and len(densities):
        thr = float(np.quantile(densities, density_quantile))
        keep = densities >= thr
        mesh.remove_vertices_by_mask(~keep)
        print(f"[fuse] trimmed density<{thr:.2f}: kept "
              f"{int(keep.sum()):,}/{len(densities):,} verts",
              flush=True)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Keep only the largest connected component. After Poisson + density
    # trimming we typically still have small floating triangle clusters
    # from outlier points (floor leaks, mask edges). The limb is by far
    # the largest connected blob; everything else is debris.
    try:
        cluster_ids, cluster_n_tri, _cluster_area = (
            mesh.cluster_connected_triangles())
        cluster_ids = np.asarray(cluster_ids)
        cluster_n_tri = np.asarray(cluster_n_tri)
        if len(cluster_n_tri) > 1:
            largest = int(np.argmax(cluster_n_tri))
            drop_mask = cluster_ids != largest
            n_dropped = int(drop_mask.sum())
            mesh.remove_triangles_by_mask(drop_mask)
            mesh.remove_unreferenced_vertices()
            kept = len(cluster_n_tri) - 1
            print(f"[fuse] kept largest of {len(cluster_n_tri)} clusters; "
                  f"dropped {kept} smaller clusters ({n_dropped:,} tris)",
                  flush=True)
        else:
            print(f"[fuse] mesh is one connected component ({int(cluster_n_tri[0]):,} tris)",
                  flush=True)
    except Exception as _e:
        print(f"[fuse] connected-component cleanup skipped: {_e}",
              flush=True)

    if out_ply is None:
        # Mirror the 2DGS output path so the rest of handler.py + extract
        # work unchanged
        out_dir = workdir / "output" / "train" / "ours_15000"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_ply = out_dir / "fuse_post.ply"

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    print(f"[fuse] wrote {out_ply}  ({out_ply.stat().st_size/1e6:.1f} MB)",
          flush=True)
    return out_ply
