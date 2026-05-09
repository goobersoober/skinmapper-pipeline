# SkinMapper neural reconstruction pipeline
# Base: RunPod official PyTorch image (handles CUDA + PyTorch versioning)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache

# System deps: HEIC support, OpenCV deps, build tools, ffmpeg, git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip \
    libheif-dev libheif1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps (pinned). Keep this layer small so rebuilds are quick.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# --- 2D Gaussian Splatting ---
# https://github.com/hbb1/2d-gaussian-splatting
RUN git clone --depth 1 https://github.com/hbb1/2d-gaussian-splatting.git /workspace/2dgs && \
    cd /workspace/2dgs && \
    git submodule update --init --recursive && \
    pip install --no-cache-dir submodules/diff-surfel-rasterization && \
    pip install --no-cache-dir submodules/simple-knn

# --- MASt3R (pose estimation) ---
# https://github.com/naver/mast3r
RUN git clone --depth 1 --recursive https://github.com/naver/mast3r.git /workspace/mast3r && \
    cd /workspace/mast3r && \
    pip install --no-cache-dir -r requirements.txt 2>/dev/null || true && \
    cd dust3r && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# --- SAM 2 (segmentation) ---
# https://github.com/facebookresearch/sam2
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /workspace/sam2 && \
    cd /workspace/sam2 && \
    pip install --no-cache-dir -e . && \
    mkdir -p checkpoints && \
    wget -q -O checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- GroundingDINO (text-prompted segmentation, pairs with SAM 2) ---
RUN pip install --no-cache-dir groundingdino-py && \
    mkdir -p /workspace/checkpoints && \
    wget -q -O /workspace/checkpoints/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# --- Depth Anything v2 (depth priors) ---
# Loaded at runtime via HuggingFace transformers — no build step needed.

# --- MASt3R checkpoint ---
RUN mkdir -p /workspace/mast3r/checkpoints && \
    wget -q -O /workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# --- Pipeline code ---
COPY pipeline /workspace/pipeline
COPY handler.py /workspace/handler.py

# Python path so pipeline modules + 3rd-party libs all resolve
ENV PYTHONPATH="/workspace:/workspace/2dgs:/workspace/mast3r:/workspace/mast3r/dust3r:/workspace/sam2"

CMD ["python", "-u", "/workspace/handler.py"]
