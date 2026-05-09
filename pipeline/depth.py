"""
Depth Anything v2 — per-image relative depth maps used as priors during
2DGS training. Especially helps on textureless skin where pure photometric
loss has nothing to lock onto.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image


class DepthAnythingV2:
    """Wraps the HuggingFace pipeline for Depth Anything v2 large."""

    def __init__(self, device: str = "cuda",
                 model_name: str = "depth-anything/Depth-Anything-V2-Large-hf"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device).eval()

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return float32 relative depth, same H×W as input. Higher = nearer."""
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        if next(self.model.parameters()).dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # (1, h, w)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=image_rgb.shape[:2],
            mode="bicubic", align_corners=False,
        )[0, 0]
        return depth.float().cpu().numpy()


def estimate_depth_folder(image_dir: Path, depth_dir: Path,
                          mask_dir: Path | None = None,
                          device: str = "cuda") -> int:
    """
    Run Depth Anything v2 on every image. Save depth as 16-bit PNG
    (depth normalised to [0, 65535] across the masked region only — out-of-mask
    set to 0). Returns number of depth maps written.
    """
    depth_dir.mkdir(parents=True, exist_ok=True)
    da = DepthAnythingV2(device=device)

    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n = 0
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        depth = da.predict(img)

        # Optional masking: only keep depth inside limb mask
        mask = None
        if mask_dir is not None:
            mp = mask_dir / (p.stem + ".png")
            if mp.exists():
                mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127)

        if mask is not None and mask.any():
            # Normalise within mask only — much more useful for training
            valid = depth[mask]
            lo, hi = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
        else:
            lo, hi = float(np.percentile(depth, 1)), float(np.percentile(depth, 99))

        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((depth - lo) / (hi - lo), 0, 1)
        if mask is not None:
            norm[~mask] = 0
        depth_u16 = (norm * 65535.0).astype(np.uint16)

        out = depth_dir / (p.stem + ".png")
        cv2.imwrite(str(out), depth_u16)
        n += 1
    return n
