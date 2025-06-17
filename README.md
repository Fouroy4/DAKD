# A Dual-Aligned Knowledge Self-Distillation Framework for Visible-Infrared Cross-Modal Person Re-Identification

This repository contains PyTorch implementation of our proposed dual-aligned knowledge self-distillation framework for visible-infrared cross-modal person re-identification on SYSU-MM01 [1] and RegDB [2] datasets.

## Environment Setup

```bash
# Create conda environment and install dependencies
conda env create -f environment.yml
conda activate reid
```

**Experimental Environment:**
- System: Ubuntu 22.04.5 LTS
- Python: 3.9.18
- PyTorch: 2.2.0 with CUDA 12.1
- GPU: NVIDIA A100-SXM4-80GB (single GPU sufficient)

## Dataset Preparation

### Download Datasets
Download the preprocessed datasets from Google Drive:
- [SYSU-MM01 & RegDB Datasets](https://drive.google.com/drive/folders/1jJFLXwbVgvgHY0c5D8a6LWXqRcEXbr_h?usp=drive_link)

### Setup Datasets
1. Download both datasets from the link above
2. Extract the files to your dataset directory
3. Update the `--data_path` parameter in training script to point to your dataset location

## Pre-trained Models

### Download Pre-trained Checkpoints
- [Teacher Model (Baseline)](https://drive.google.com/file/d/1zA49HITsPtKBxggBh98-EproHcq6NdDW/view?usp=drive_link) - Pre-trained teacher model for knowledge distillation
- [Student Model (DAKD)](https://drive.google.com/file/d/16YW1uKIK2zhM2LHSPt-Y25XwSvlKEyxq/view?usp=sharing) - Trained student model with our framework

### Model Usage
1. Download the teacher model for training new student models
2. Download the student model for direct testing/evaluation
3. Update the `--teacher_path` and `--model_path` parameters accordingly

## Training

### Quick Start
Use the provided training script for SYSU-MM01:
```bash
chmod +x train.sh
./train.sh
```

### Testing
Use the provided testing script:
```bash
chmod +x test.sh
./test.sh
```

### Ablation Configurations
- `only_kd`: Only traditional knowledge distillation
- `mlkd_no_instance`: Without instance-level distillation
- `mlkd_no_class`: Without class-level distillation
- `full_mlkd`: Complete framework