#!/bin/bash

# SYSU-MM01 Testing Script for DAKD
# Modify the paths below according to your setup

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH="YOUR_DATA_PATH"                    # Path to dataset directory
RESUME_PATH="YOUR_MODEL_PATH"                  # Path to trained model (.pth or .t file)

# DATA_PATH="/home/lunet/coyz2/mlkd_reid/passion/dataset/sysu/"                    # Path to SYSU-MM01 dataset
# RESUME_PATH="/home/lunet/coyz2/mlkd_reid/DAKD/save_model/student/best_student_model_epoch_105.pth"                  # Directory to save trained models

# =============================================================================
# TESTING COMMAND
# =============================================================================

python test.py \
    --dataset sysu \
    --data_path "$DATA_PATH" \
    --resume "$RESUME_PATH" \
    --mode all \
    --gpu 0 \
    --rerank

echo "Testing completed!"