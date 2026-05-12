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

    # GroundingDINO never actually prunes attention heads, so head_mask is
    # always None or unused. Short-circuit to a no-op that returns the
    # expected list-of-None shape. This dodges all dtype/device issues in
    # the original `_convert_head_mask_to_5d` codepath.
    def _get_head_mask_noop(self, head_mask, num_hidden_layers,
                            is_attention_chunked=False):
        return [None] * num_hidden_layers

    _BertModel.get_head_mask = _get_head_mask_noop
    print("[segment] applied BertModel.get_head_mask no-op patch",
          flush=True)
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
        """SAM2 with a centre-region multi-point prompt.

        Tattoo-scan photos always centre the limb, but the limb often
        fills most of the frame and contains many strong visual features
        (tattoo designs, dark hair, skin folds). A single centre point
        causes SAM2 to lock onto one tattoo's silhouette — yielding a
        tiny mask that's a fraction of the actual limb. Two fixes:

        1. Use a 3x3 grid of positive prompts spread across the central
           60% of the frame. Multiple anchors force SAM2 to interpret
           "the foreground object" as the whole limb, not one feature.

        2. Pick the LARGEST mask from multimask_output, not the highest
           confidence. The 3 masks SAM2 returns roughly correspond to
           sub-part / part / whole-object scales; we want the largest.
           We also gate on a minimum coverage so we never accept a
           pathologically tiny mask.
        """
        sam = self._load_sam()
        sam.set_image(image_rgb)
        h, w = image_rgb.shape[:2]

        # 3x3 grid in the central 60% of the frame — points at
        # 0.2, 0.5, 0.8 across both axes
        gx = np.array([0.2, 0.5, 0.8]) * w
        gy = np.array([0.2, 0.5, 0.8]) * h
        xx, yy = np.meshgrid(gx, gy)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)
        labels = np.ones(len(points), dtype=np.int32)

        masks, scores, _ = sam.predict(
            point_coords=points, point_labels=labels,
            multimask_output=True,
        )
        # Score each candidate by area + a tiny preference for high score
        # to break ties. We MUST cover at least 8% of the frame, else
        # something's gone wrong with all candidates.
        frame_px = h * w
        best_idx = -1
        best_score = -1.0
        for i, (m, s) in enumerate(zip(masks, scores)):
            area_frac = float((m > 0).sum()) / frame_px
            if area_frac < 0.08:
                continue
            # Composite score: heavily weight area, mild weight on
            # SAM2 confidence
            composite = area_frac + 0.05 * float(s)
            if composite > best_score:
                best_score = composite
                best_idx = i
        if best_idx < 0:
            # All masks were tiny — return the largest one regardless
            areas = [(m > 0).sum() for m in masks]
            best_idx = int(np.argmax(areas))

        mask = (masks[best_idx] > 0).astype(np.uint8) * 255
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
    gdino_warning_logged = False
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        mask = None
        tag = None

        # Try GroundingDINO + SAM2 box-prompted segmentation first
        try:
            box = seg.detect_box(img, prompt)
            if box is not None:
                mask = seg.mask_from_box(img, box)
                tag = "success"
        except Exception as e:
            if not gdino_warning_logged:
                # Log the first failure only; the rest will fall through quietly
                print(f"  ! GroundingDINO failed on {p.name} ({e}); "
                      f"using SAM2 centre-point fallback for all photos",
                      flush=True)
                gdino_warning_logged = True

        # Fall back to SAM2 with a centre-point prompt. This bypasses
        # GroundingDINO entirely (no Bert/transformers issues) and works
        # fine for our usecase because the subject is always centred in
        # tattoo-scan captures.
        if mask is None:
            try:
                mask = seg.fallback_centre_mask(img)
                tag = "fallback"
            except Exception as e:
                print(f"  ! SAM2 centre-point also failed on {p.name}: {e}",
                      flush=True)
                counts["failed"] += 1
                # All-white mask only as a true last resort
                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                tag = None

        if tag is not None:
            mask = seg.clean_mask(mask)
            counts[tag] += 1

        # Save mask
        cv2.imwrite(str(mask_dir / (p.stem + ".png")), mask)

        # Save image with background masked to neutral grey (128, 128, 128)
        out = img.copy()
        bg = mask == 0
        out[bg] = 128
        Image.fromarray(out).save(masked_dir / p.name, quality=95)

    return counts
