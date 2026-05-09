"""
Utility functions: HEIC decoding, photo loading, blur detection,
working-directory helpers.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image

# Register HEIC opener so PIL.Image.open(...) handles .heic / .HEIC
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def list_photos(folder: Path) -> List[Path]:
    """Return sorted list of photo paths in folder (by filename)."""
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    return sorted(files)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 ndarray. Handles HEIC + EXIF rotation."""
    img = Image.open(path)
    # Apply EXIF orientation if present
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    img = img.convert("RGB")
    return np.asarray(img)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def laplacian_blur_score(img_rgb: np.ndarray) -> float:
    """Higher = sharper. <100 is typically blurry. <50 is unusable."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def reject_blurry(paths: List[Path], threshold: float = 60.0) -> tuple[List[Path], List[Path]]:
    """Return (kept, rejected) paths. Rejected fall below sharpness threshold."""
    kept, rejected = [], []
    for p in paths:
        try:
            img = load_rgb(p)
            # Downsample for speed - sharpness scale-invariant enough
            h, w = img.shape[:2]
            scale = 1024 / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            score = laplacian_blur_score(img)
            if score >= threshold:
                kept.append(p)
            else:
                rejected.append(p)
        except Exception as e:
            print(f"  ! could not read {p.name}: {e}")
            rejected.append(p)
    return kept, rejected


def normalise_to_jpeg(src_paths: List[Path], dst_dir: Path,
                      max_dim: int = 1600) -> List[Path]:
    """
    Convert all photos to JPEG of bounded resolution in dst_dir.
    Most reconstruction libs are happiest with JPEG ≤ 1600px.
    Returns the list of new JPEG paths in input order.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for i, p in enumerate(src_paths):
        img = load_rgb(p)
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        out_path = dst_dir / f"img_{i:04d}.jpg"
        Image.fromarray(img).save(out_path, quality=95, subsampling=0)
        out.append(out_path)
    return out


def make_workdir(prefix: str = "skinmap_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def cleanup(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
