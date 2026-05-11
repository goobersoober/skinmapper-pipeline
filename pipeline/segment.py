"""
SAM 2 + GroundingDINO limb segmentation.

For each photo:
  1. GroundingDINO finds the bounding box for the body part (text-prompted)
  2. SAM 2 generates a precise mask from that box
  3. Mask is morphologically cleaned and saved alongside the image

Background pixels (mask == 0) are set to a neutral grey so 3DGS treats
them as low-confidence regions during training.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# ---- Compatibility shim for GroundingDINO + modern transformers ----
# GroundingDINO uses a BertModelWarper that calls get_head_mask on the
# wrapped BertModel. In transformers >= 4.36 the method was hoisted to
# ModuleUtilsMixin and some Bert variants no longer expose it through the
# warper's attribute proxy. Patch it directly onto BertModel + the warper.
try:
    from transformers.models.bert.modeling_bert import BertModel as _BertModel
    from transformers.modeling_utils import ModuleUtilsMixin as _MUM

    def _get_head_mask(self, head_mask, num_hidden_layers,
                       is_attention_chunked=False):
        if head_mask is not None:
            head_mask = _MUM.get_head_mask(
                self, head_mask, num_hidden_layers, is_attention_chunked)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    _BertModel.get_head_mask = _get_head_mask
    print("[segment] applied BertModel.get_head_mask patch", flush=True)

    # Also patch GroundingDINO's BertModelWarper if importable
    try:
        from groundingdino.models.GroundingDINO import bertwarper as _bw
        if hasattr(_bw, "BertModelWarper"):
            _bw.BertModelWarper.get_head_mask = _get_head_mask
            print("[segment] applied BertModelWarper.get_head_mask patch",
                  flush=True)
    except Exception as _e_bw:
        print(f"[segment] BertModelWarper patch skipped: {_e_bw}", flush=True)
except Exception as _e:
    print(f"[segment] BertModel patch skipped: {_e}", flush=True)

# Map iOS body_part values → GroundingDINO text prompts
PROMPT_BY_BODY_PART: Dict[str, str] = {
    "forearm":   "human forearm . arm . hand .",
    "upper_arm": "human upper arm . shoulder .",
    "calf":      "human calf . lower leg . ankle . foot .",
    "thigh":     "human thigh . upper leg .",
    "leg":       "human leg .",
    "arm":       "human arm . hand .",
    "hand":      "human hand . fingers .",
    "foot":      "human foot . toes . ankle .",
    "back":      "human back .",
    "chest":     "human chest . torso .",
    "neck":      "human neck .",
    "torso":     "human torso .",
}


def _prompt_for(body_part: str) -> str:
    return PROMPT_BY_BODY_PART.get(body_part, "human limb . skin .")


class LimbSegmenter:
    """Lazy-loaded GroundingDINO + SAM 2 segmenter."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._sam = None
        self._gdino = None

    # -------- model loading (deferred until first call) --------

    def _load_sam(self):
        if self._sam is not None:
            return self._sam
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        ckpt = "/workspace/sam2/checkpoints/sam2.1_hiera_large.pt"
        cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(cfg, ckpt, device=self.device)
        self._sam = SAM2ImagePredictor(sam2_model)
        return self._sam

    def _load_gdino(self):
        if self._gdino is not None:
            return self._gdino
        from groundingdino.util.inference import load_model
        import groundingdino

        # GroundingDINO config can live at slightly different paths depending
        # on whether it was installed via `groundingdino-py` (PyPI) or built
        # from source. Try both.
        gdino_root = Path(groundingdino.__file__).parent
        candidates = [
            gdino_root / "config" / "GroundingDINO_SwinT_OGC.py",
            gdino_root.parent / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
            Path("/tmp/gdino/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
        ]
        cfg_path = next((c for c in candidates if c.exists()), None)
        if cfg_path is None:
            raise FileNotFoundError(
                f"GroundingDINO config not found. Searched: {candidates}"
            )
        ckpt = "/workspace/checkpoints/groundingdino_swint_ogc.pth"
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint missing: {ckpt}")
        self._gdino = load_model(str(cfg_path), ckpt)
        self._gdino.to(self.device)
        return self._gdino

    # -------- per-image inference --------

    def detect_box(self, image_rgb: np.ndarray, prompt: str,
                   box_thresh: float = 0.30,
                   text_thresh: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
        """Return (x0, y0, x1, y1) of best body-part match, or None."""
        from groundingdino.util.inference import predict
        import groundingdino.datasets.transforms as T

        gdino = self._load_gdino()

        # GroundingDINO expects normalised, transformed tensor
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        h, w = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        image_t, _ = transform(pil, None)

        boxes, logits, phrases = predict(
            model=gdino, image=image_t, caption=prompt,
            box_threshold=box_thresh, text_threshold=text_thresh,
            device=self.device,
        )
        if len(boxes) == 0:
            return None

        # boxes are cxcywh normalised — pick the one with highest score
        idx = int(torch.argmax(logits).item())
        cx, cy, bw, bh = boxes[idx].cpu().numpy()
        x0 = int(max(0, (cx - bw / 2) * w))
        y0 = int(max(0, (cy - bh / 2) * h))
        x1 = int(min(w, (cx + bw / 2) * w))
        y1 = int(min(h, (cy + bh / 2) * h))
        return (x0, y0, x1, y1)

    def mask_from_box(self, image_rgb: np.ndarray,
                      box: Tuple[int, int, int, int]) -> np.ndarray:
        """Run SAM 2 with box prompt. Returns uint8 mask (H, W) ∈ {0, 255}."""
        sam = self._load_sam()
        sam.set_image(image_rgb)
        masks, scores, _ = sam.predict(
            box=np.array(box)[None, :],
            multimask_output=False,
        )
        mask = (masks[0] > 0).astype(np.uint8) * 255
        return mask

    def fallback_centre_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """If detection fails, fall back to SAM 2 with a centre point prompt."""
        sam = self._load_sam()
        sam.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
        point = np.array([[w // 2, h // 2]])
        labels = np.array([1])
        masks, scores, _ = sam.predict(
            point_coords=point, point_labels=labels,
            multimask_output=True,
        )
        # Pick the highest-scoring mask
        best = int(np.argmax(scores))
        mask = (masks[best] > 0).astype(np.uint8) * 255
        return mask

    @staticmethod
    def clean_mask(mask: np.ndarray) -> np.ndarray:
        """Morphological cleanup: close small holes, keep largest component."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

        # Keep largest connected component
        n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
        if n <= 1:
            return m
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return ((labels == largest).astype(np.uint8) * 255)


def segment_folder(image_dir: Path, mask_dir: Path, masked_dir: Path,
                   body_part: str = "leg",
                   device: str = "cuda") -> Dict[str, int]:
    """
    For every image in image_dir:
      - run GroundingDINO + SAM 2
      - save binary mask to mask_dir
      - save image with background→neutral-grey to masked_dir
    Returns counts: {success, fallback, failed}.
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    seg = LimbSegmenter(device=device)
    prompt = _prompt_for(body_part)

    counts = {"success": 0, "fallback": 0, "failed": 0}

    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        try:
            box = seg.detect_box(img, prompt)
            if box is not None:
                mask = seg.mask_from_box(img, box)
                tag = "success"
            else:
                mask = seg.fallback_centre_mask(img)
                tag = "fallback"
            mask = seg.clean_mask(mask)
            counts[tag] += 1
        except Exception as e:
            print(f"  ! segmentation failed on {p.name}: {e}")
            counts["failed"] += 1
            # keep a fully-white mask so the photo isn't wasted
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255

        # Save mask
        cv2.imwrite(str(mask_dir / (p.stem + ".png")), mask)

        # Save image with background masked to neutral grey (128, 128, 128)
        out = img.copy()
        bg = mask == 0
        out[bg] = 128
        Image.fromarray(out).save(masked_dir / p.name, quality=95)

    return counts
