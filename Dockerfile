# One image for both pods: rendering (RTX 4090) and generation (RTX A6000 / A100).
# Holds dependencies only; the code lives on the network volume (/workspace/cvpr_vail, git pull to update).
# Base image keeps RunPod's SSH / Jupyter start script. Needs a host driver that supports CUDA 12.8
# (in RunPod's deploy filter, choose CUDA 12.8 or newer).
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    BPY_PYTHON=/opt/bpy/bin/python

# ffmpeg for video I/O; X/GL libraries the headless bpy wheel links against
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libxrender1 libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libsm6 libice6 libgl1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/cvpr_vail/requirements.txt
RUN pip install --no-cache-dir -r /opt/cvpr_vail/requirements.txt

# Blender as a Python module: bpy 5.2 needs Python 3.13, so it gets its own venv
RUN pip install --no-cache-dir uv \
    && uv venv --python 3.13 /opt/bpy \
    && uv pip install --python /opt/bpy --no-cache bpy==5.2.2 numpy pyyaml \
    && /opt/bpy/bin/python -c "import bpy; print('bpy', bpy.app.version_string)"

RUN python -c "import torch, diffusers, transformers; print('torch', torch.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
