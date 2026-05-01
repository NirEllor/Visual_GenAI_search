FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/workspace/.cache/torch

# Mount these at runtime: -v /host/latents:/workspace/latents etc.
VOLUME ["/workspace/latents", "/workspace/checkpoints", "/workspace/results"]

