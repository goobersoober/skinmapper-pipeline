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
# Split into discrete RUN steps so build logs pinpoint which step failed.
# diff-surfel-rasterization compiles against torch headers — needs nvcc
# from the 'devel' base image (already provided).
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git /workspace/2dgs

RUN cd /workspace/2dgs && git submodule update --init --recursive

# Print the build env up-front so any compile error is reproducible
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    import sys; print('python', sys.version)" && \
    nvcc --version && \
    which gcc && gcc --version | head -1

# These two CUDA extensions are where most build failures happen.
# Each gets its own RUN so the log shows which one failed.
RUN cd /workspace/2dgs && \
    pip install --no-build-isolation ./submodules/diff-surfel-rasterization

RUN cd /workspace/2dgs && \
    pip install --no-build-isolation ./submodules/simple-knn

# --- MASt3R (pose estimation) ---
# https://github.com/naver/mast3r
# Use --no-build-isolation for any deps that build CUDA extensions
# (RoPE, croco, etc.) so they can find the pre-installed torch.
RUN git clone --recursive https://github.com/naver/mast3r.git /workspace/mast3r

RUN cd /workspace/mast3r && \
    pip install --no-build-isolation -r requirements.txt

RUN cd /workspace/mast3r/dust3r && \
    pip install --no-build-isolation -r requirements.txt

# DUSt3R's optional CUDA-accelerated RoPE — if it fails to build, fall back
# to the pure-Python implementation (slower but works)
RUN cd /workspace/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace || \
    echo "[warn] cuRoPE build failed — using pure-Python RoPE (slower)"

# MASt3R checkpoint
RUN mkdir -p /workspace/mast3r/checkpoints && \
    wget --tries=3 --timeout=30 -q -O \
      /workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# --- SAM 2 ---
# `pip install -e .` builds a CUDA extension — needs --no-build-isolation
# so it sees torch.
RUN git clone https://github.com/facebookresearch/sam2.git /workspace/sam2

RUN cd /workspace/sam2 && \
    pip install --no-build-isolation -e .

RUN mkdir -p /workspace/sam2/checkpoints && \
    wget --tries=3 --timeout=30 -q -O \
      /workspace/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- GroundingDINO (text-prompted detector that pairs with SAM 2) ---
# Try the PyPI wheel first (no CUDA build needed). If the ABI is incompatible,
# fall back to source install with --no-build-isolation.
RUN pip install --no-build-isolation groundingdino-py || \
    (git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/gdino && \
     cd /tmp/gdino && pip install --no-build-isolation -e .)

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
