#!/usr/bin/env bash
# Single-GPU CosmoPredict2GR00T training for Piper fold (pi05-aligned settings).
#
# Usage:
#   cd /path/to/starVLA-axibo
#   export PYTHONPATH=.
#   bash examples/Robotwin/train_files/run_piper_cosmo_pi05_aligned.sh
#
# For CosmoPredict2PI (slower, layer-wise head), override:
#   FRAMEWORK=CosmoPredict2PI bash examples/Robotwin/train_files/run_piper_cosmo_pi05_aligned.sh
#
# W&B (override via env or pass through to train_starvla.py):
#   WANDB_ENTITY=ryanrahman WANDB_PROJECT=starvla-piper bash ...

set -euo pipefail

FRAMEWORK="${FRAMEWORK:-CosmoPredict2GR00T}"
CONFIG=examples/Robotwin/train_files/starvla_train_piper_cosmo_pi05_aligned.yaml
RUN_ID="${RUN_ID:-piperx_fold_cosmo_pi05_align_$(date +%m%d)}"
RUN_ROOT="${RUN_ROOT:-playground/Checkpoints}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-ryanrahman}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla-piper}"

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "Framework: ${FRAMEWORK}"
echo "Run ID:    ${RUN_ID}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES}"

ACCEL_CONFIG=examples/Robotwin/train_files/deepspeed/accelerate_zero2_single_gpu.yaml

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.name "${FRAMEWORK}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  --trainer.max_train_steps 15000 \
  --trainer.save_interval 5000 \
  --trainer.gradient_accumulation_steps 8 \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.data_mix arx_x5_pi05 \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  "$@"
