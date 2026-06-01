#!/usr/bin/env bash
# Single-GPU CosmoPredict2GR00T training for Piper fold (pi05-aligned settings).
#
# Usage:
#   cd /path/to/starVLA-axibo
#   export PYTHONPATH=.
#   bash examples/Robotwin/train_files/run_piper_cosmo_pi05_aligned.sh
#
# OOM at per_device_batch_size 16 on CosmoPredict2PI — use grad accum for effective batch 16:
#   per_device_batch_size 1 × gradient_accumulation_steps 16  (default, fits VRAM)
#   per_device_batch_size 2 × gradient_accumulation_steps 8   (try if you have headroom)
#   per_device_batch_size 4 × gradient_accumulation_steps 4   (may still OOM on Cosmo PI)
#
# For CosmoPredict2PI (layer-wise head):
#   FRAMEWORK=CosmoPredict2PI bash examples/Robotwin/train_files/run_piper_cosmo_pi05_aligned.sh
#
# W&B: entity defaults to your logged-in W&B default (omit username unless it is your entity).
#   WANDB_ENTITY=sohaib03-mcmaster-university WANDB_PROJECT=starvla-piper bash ...
#   WANDB_MODE=disabled bash ...   # skip logging

set -euo pipefail

FRAMEWORK="${FRAMEWORK:-CosmoPredict2GR00T}"
CONFIG=examples/Robotwin/train_files/starvla_train_piper_cosmo_pi05_aligned.yaml
RUN_ID="${RUN_ID:-piperx_fold_cosmo_pi05_align_50k}"
RUN_ROOT="${RUN_ROOT:-playground/Checkpoints}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla-piper}"

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "Framework: ${FRAMEWORK}"
echo "Run ID:    ${RUN_ID}"
echo "GPU:       ${CUDA_VISIBLE_DEVICES}"

WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

ACCEL_CONFIG=examples/Robotwin/train_files/deepspeed/accelerate_zero2_single_gpu.yaml

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --framework.name "${FRAMEWORK}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 5000 \
  --trainer.num_warmup_steps 2500 \
  --trainer.gradient_accumulation_steps 16 \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.data_mix arx_x5_pi05 \
  "${WANDB_ARGS[@]}" \
  "$@"
