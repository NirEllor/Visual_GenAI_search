#!/bin/bash
# Shared configuration — sourced by every run_*.sh script.

EMAIL="ellorwaizner.nir@mail.huji.ac.il"

PROJECT="/cs/labs/raananf/ellorw.nir/distillation/Distillation_Research"
IMG="/cs/labs/raananf/ellorw.nir/images/distill.sif"
TORCH_CACHE="/cs/labs/raananf/ellorw.nir/torch_cache"

RUN="source /cs/labs/raananf/ellorw.nir/venv/bin/activate && cd $PROJECT &&"

DIMS=(64 128 256 384 512 1024)
SIZES=(250000 500000 1000000 2000000)