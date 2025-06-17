#!/bin/bash

# SYSU-MM01 Training Script for MLKD
# Modify the paths below according to your setup
# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH="YOUR_DATA_PATH"                    # Path to SYSU-MM01 dataset
TEACHER_PATH="YOUR_TEACHER_MODEL_PATH"        # Path to teacher model (.t file)
MODEL_PATH="YOUR_MODEL_PATH"                  # Directory to save trained models
LOG_PATH="YOUR_LOG_PATH"                      # Directory to save training logs
VIS_LOG_PATH="YOUR_VIS_LOG_PATH"             # Directory to save tensorboard logs

# DATA_PATH="/home/lunet/coyz2/mlkd_reid/passion/dataset/sysu/"                    # Path to SYSU-MM01 dataset
# TEACHER_PATH="/home/lunet/coyz2/mlkd_reid/DAKD/save_model/sysu/teacher/debug_sysu_mem_p4_n8_ch3_lr_0.00035_adam_best.t"        # Path to teacher model (.t file)
# MODEL_PATH="/home/lunet/coyz2/mlkd_reid/DAKD/save_model/sysu/2025_6_17/ablation3/"                  # Directory to save trained models
# LOG_PATH="/home/lunet/coyz2/mlkd_reid/DAKD/log/sysu/2025_6_17/ablation3/"                      # Directory to save training logs
# VIS_LOG_PATH="/home/lunet/coyz2/mlkd_reid/DAKD/vis_log/sysu/2025_6_17/ablation3/"             # Directory to save tensorboard logs

# =============================================================================
# TRAINING COMMAND
# =============================================================================

python train.py \
    --dataset sysu \
    --data_path "$DATA_PATH" \
    --teacher_path "$TEACHER_PATH" \
    --model_path "$MODEL_PATH" \
    --log_path "$LOG_PATH" \
    --vis_log_path "$VIS_LOG_PATH" \
    --ablation full_mlkd \
    --lr 0.00035 \
    --batch-size 8 \
    --gpu 0,1,2,3 \

echo "Training completed!"