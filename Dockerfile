# SkinMapper neural reconstruction pipeline — build 2026-05-11
# Base: CUDA 11.8 devel — NVCC 11.8 compiles all CUDA extensions.
# Torch is explicitly pinned to 2.5.1+cu118 before any package install
# so SAM 2's "torch>=2.5.1" requirement doesn't pull the latest torch
# (which now bundles CUDA 13.x and requires a driver most RunPod workers
# don't have yet).
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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
    libopengl0 libegl1 \
    ffmpeg \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# --- CRITICAL: pin torch BEFORE any other package install ---
# SAM 2 requires torch>=2.5.1. Without this pin, pip installs the latest
# torch which currently pulls CUDA 13.x — incompatible with CUDA 12.x drivers.
# We pin to 2.5.1+cu118 so it matches the NVCC version in this base image.
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
      "torch==2.5.1+cu118" \
      "torchvision==0.20.1+cu118" \
      "torchaudio==2.5.1+cu118" \
      --index-url https://download.pytorch.org/whl/cu118

# Python deps — installed after torch is pinned
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt

# --- 2D Gaussian Splatting (with CUDA submodules) ---
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git /workspace/2dgs

RUN cd /workspace/2dgs && git submodule update --init --recursive

# Confirm torch + CUDA env before compiling extensions
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); \
    import sys; print('python', sys.version)" && \
    nvcc --version && \
    which gcc && gcc --version | head -1

RUN cd /workspace/2dgs && \
    pip install --no-build-isolation ./submodules/diff-surfel-rasterization

RUN cd /workspace/2dgs && \
    pip install --no-build-isolation ./submodules/simple-knn

# --- MASt3R (pose estimation) ---
RUN git clone --recursive https://github.com/naver/mast3r.git /workspace/mast3r

RUN cd /workspace/mast3r && \
    pip install --no-build-isolation -r requirements.txt

RUN cd /workspace/mast3r/dust3r && \
    pip install --no-build-isolation -r requirements.txt

RUN cd /workspace/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace || \
    echo "[warn] cuRoPE build failed — using pure-Python RoPE (slower)"

# MASt3R checkpoint
RUN mkdir -p /workspace/mast3r/checkpoints && \
    wget --tries=3 --timeout=30 -q -O \
      /workspace/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
      https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# --- SAM 2 ---
# Install with --no-deps so pip does NOT try to satisfy torch>=2.5.1
# by pulling a newer torch. Our pinned 2.5.1+cu118 already satisfies it.
# Then install SAM 2's non-torch deps explicitly.
RUN git clone https://github.com/facebookresearch/sam2.git /workspace/sam2

RUN cd /workspace/sam2 && \
    pip install --no-build-isolation --no-deps -e . && \
    pip install "hydra-core>=1.3.2" "iopath>=0.1.10" portalocker

RUN mkdir -p /workspace/sam2/checkpoints && \
    wget --tries=3 --timeout=30 -q -O \
      /workspace/sam2/checkpoints/sam2.1_hiera_large.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# --- GroundingDINO ---
RUN pip install --no-build-isolation groundingdino-py || \
    (git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/gdino && \
     cd /tmp/gdino && pip install --no-build-isolation -e .)

RUN mkdir -p /workspace/checkpoints && \
    wget --tries=3 --timeout=30 -q -O /workspace/checkpoints/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# --- Pre-cache Depth Anything v2 weights ---
RUN python -c "from transformers import AutoModelForDepthEstimation, AutoImageProcessor; \
    name='depth-anything/Depth-Anything-V2-Large-hf'; \
    AutoImageProcessor.from_pretrained(name); \
    AutoModelForDepthEstimation.from_pretrained(name)"

# --- Pipeline code: cloned once at build, pulled fresh at each job start ---
# This means code changes (handler.py, pipeline/*.py) deploy instantly —
# just push to GitHub. No Docker rebuild or rollout needed.
RUN git clone https://github.com/goobersoober/skinmapper-pipeline.git \
      /workspace/skinmapper-pipeline

ENV PYTHONPATH="/workspace/skinmapper-pipeline:/workspace/2dgs:/workspace/mast3r:/workspace/mast3r/dust3r:/workspace/sam2"

RUN python -c "import torch; assert torch.cuda.is_available() or True; \
    import runpod; \
    from transformers import AutoModelForDepthEstimation; \
    print('image OK — torch', torch.__version__, 'cuda', torch.version.cuda)"

# start.sh: pulls latest code then launches handler — runs on every job start
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

CMD ["/workspace/start.sh"]
