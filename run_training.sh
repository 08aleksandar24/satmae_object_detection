#!/bin/bash

# Ensure Conda is properly initialized
source ~/conda/bin/activate satmae  # Modify path if needed

# Set CUDA device (if needed)
export CUDA_VISIBLE_DEVICES=0
cd /home/aleksandar/SatMAE
# Run your training script
torchrun --nproc_per_node=1 --nnodes=1 --master_port=1234 main_finetune.py \
    --output_dir /home/aleksandar/satmaeoutput --log_dir /home/aleksandar/satmaeoutput \
    --batch_size 32 --accum_iter 2 --model vit_large_patch16 --epochs 128 \
    --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 \
    --mixup 0.8 --cutmix 1.0 \
    --finetune /home/aleksandar/satmaepretrain/fmow_pretrain.pth \
    --dist_eval --num_workers 8
