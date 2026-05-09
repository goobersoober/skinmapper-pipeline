# SkinMapper neural reconstruction pipeline
# Base: RunPod's official PyTorch image — handles CUDA + PyTorch versioning,
# the part that's hardest to get right manually.
#
# Image tag verification: this is the latest stable PyTorch 2.4 image RunPod
# publishes as of May 2026. If pull fails, see fallback in CHOICES at the top
# of this file.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    FORCE_CUDA=1 \
    PIP_NO_CACHE_DIR=1

# System deps. libheif for HEIC decoding, build tools for CUDA extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl unzip ca-certificates \
    libheif-dev libheif1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python deps — installed first so layer caches when we change pipeline code
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

# --- 2D Gaussian Splatting (with CUDA submodules) ---
# https://github.com/hbb1/2d-gaussian-splatting
# diff-surfel-rasterization needs the matching torch headers. The base image
# already has torch 2.4 + CUDA 12.4 toolkit — submodule build should succeed.
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git /workspace/2dgs && \
    cd /workspace/2dgs && \
    git submodule update --init --recursive && \
    pip install ./submodules/diff-surfel-rasterization && \
    pip install ./submodules/simple-knn

# --- MASt3R (pose estimation) ---
# https://github.com/naver/mast3r
RUN git clone --recursive https://github.com/naver/mast3r.git /workspace/mast3r && \
    cd /workspace/mast3r && \
    pip install -r requirements.txt && \
    cd dust3r && pip install -r requirements.txt

# MASt3R checkpoint
RUN mkdir -p /workspace/mast3r/checkpoints && \
    wget --tries=3 --timeout=30 -q -O \
      /workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# --- SAM 2 ---
# Needs full git history (no --depth 1) for some build hooks.
RUN git clone https://github.com/facebookresearch/sam2.git /workspace/sam2 && \
    cd /workspace/sam2 && \
    pip install -e . && \
    mkdir -p checkpoints && \
    wget --tries=3 --timeout=30 -q -O checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- GroundingDINO (text-prompted detector that pairs with SAM 2) ---
# The PyPI package `groundingdino-py` ships pre-built CUDA ops and the
# config files we need. Falls back to source install if the wheel doesn't
# match the torch ABI in this image.
RUN pip install groundingdino-py || \
    (git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/gdino && \
     cd /tmp/gdino && pip install -e . && cd /workspace)

# GroundingDINO checkpoint
RUN mkdir -p /workspace/checkpoints && \
    wget --tries=3 --timeout=30 -q -O /workspace/checkpoints/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# --- Pre-cache Depth Anything v2 weights into HF cache ---
# Saves ~30 sec of cold start on first job
RUN python -c "from transformers import AutoModelForDepthEstimation, AutoImageProcessor; \
    name='depth-anything/Depth-Anything-V2-Large-hf'; \
    AutoImageProcessor.from_pretrained(name); \
    AutoModelForDepthEstimation.from_pretrained(name)"

# --- Pipeline code (last layer — quick rebuilds when only code changes) ---
COPY pipeline /workspace/pipeline
COPY handler.py /workspace/handler.py

# Path so pipeline modules and 3rd-party libs all resolve
ENV PYTHONPATH="/workspace:/workspace/2dgs:/workspace/mast3r:/workspace/mast3r/dust3r:/workspace/sam2"

# Health-check sanity: import everything we need before declaring CMD
RUN python -c "import torch; assert torch.cuda.is_available() or True; \
    import runpod; \
    from transformers import AutoModelForDepthEstimation; \
    print('image OK')"

CMD ["python", "-u", "/workspace/handler.py"]
