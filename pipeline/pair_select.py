"""
Retrieval-based pair selection using DINOv2.

Why this exists: temporal sliding windows (swin-k) only match each photo with
its k nearest temporal neighbours. This breaks if the user:
  - Takes close-up shots, then resets and takes wide shots
  - Backtracks during capture
  - Shoots from multiple distances

Retrieval-based pairing computes a global feature embedding for each photo
(DINOv2) and matches each photo with its top-k most semantically similar
photos. This catches close-up+wide pairs of the same area regardless of
temporal order — the same approach Polycam and KIRI use.

DINOv2-small is ~22M params, ~50ms per image on RTX 4090. For 100 photos,
embedding extraction takes ~5 sec total — negligible compared to MASt3R.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image


@torch.no_grad()
def compute_dinov2_embeddings(image_paths: List[Path],
                              device: str = "cuda") -> torch.Tensor:
    """
    Compute L2-normalised DINOv2 embeddings for each image.
    Returns: (N, D) tensor on CPU.
    """
    from transformers import AutoImageProcessor, AutoModel

    name = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    embeddings = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # CLS token = global image descriptor
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        embeddings.append(emb.cpu())

    # Free DINOv2 weights before MASt3R loads
    del model, processor
    torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0)


def top_k_pairs(embeddings: torch.Tensor, k: int = 12) -> List[Tuple[int, int]]:
    """
    For each image, find indices of its top-k most similar images.
    Returns deduplicated, undirected pair list.
    """
    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T          # (N, N) cosine sim (normalised)
    sim.fill_diagonal_(-1.0)                 # exclude self-match

    seen = set()
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        top = torch.topk(sim[i], k=min(k, n - 1)).indices.tolist()
        for j in top:
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((i, j))
    return pairs


def make_retrieval_pairs(images: list, pair_indices: List[Tuple[int, int]],
                         symmetrize: bool = True) -> list:
    """
    Construct dust3r-format pair list from arbitrary index pairs.
    `images` is the list of image dicts produced by dust3r's load_images().
    """
    pairs = []
    for i, j in pair_indices:
        if i == j:
            continue
        pairs.append((images[i], images[j]))
        if symmetrize:
            pairs.append((images[j], images[i]))
    return pairs
