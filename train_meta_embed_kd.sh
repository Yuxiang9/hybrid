#!/bin/bash

# MetaEmbed-KD Training Script
# Uses the three-component loss from DeeperImpact paper:
# L = λ_DI * L_DeepImpact + λ_LI * L_late-int + λ_KD * L_KD
#
# Key features:
# - Reuses existing DistillationScores dataset (1 pos + ~50 hard negatives)
# - Margin loss on sparse scores
# - InfoNCE on dense scores  
# - Knowledge distillation from combined to individual heads

torchrun --nproc_per_node=2 \
    -m src.deep_impact.train \
    --meta_embed_kd \
    --dataset_path required_files/distillation_scores/rank_zephyr_alpha_l6_distill_scores_train.pkl.gz \
    --queries_path required_files/queries/msmarco-passage/queries.train.tsv \
    --collection_path required_files/collections/msmarco-passage/collection.tsv \
    --checkpoint_dir checkpoints_meta_embed_kd \
    --qrels_path required_files/collections/msmarco-passage/qrels.train.tsv \
    --max_length 256 \
    --seed 42 \
    --batch_size 8 \
    --lr 1e-5 \
    --save_every 5000 \
    --eval_every 500 \
    --save_best \
    --gradient_accumulation_steps 2 \
    --dense_dim 128 \
    --num_expansion_tokens 64 \
    --lambda_di 1.0 \
    --lambda_li 1.0 \
    --lambda_kd 1.0 \
    --margin 0.2 \
    --kd_temperature 1.0

# Loss Component Weights:
# --lambda_di    : Weight for DeepImpact margin loss on sparse scores
#                  Ensures positive doc scores higher than negatives
#
# --lambda_li    : Weight for InfoNCE contrastive loss on dense scores
#                  Late interaction with MaxSim
#
# --lambda_kd    : Weight for knowledge distillation
#                  Combined score teaches individual heads
#
# Hyperparameters:
# --margin       : Margin for DeepImpact loss (0.2 = soft, 1.0 = hard)
# --kd_temperature : Temperature for InfoNCE and KD (lower = sharper)
#
# Model Architecture:
# --dense_dim            : Dense embedding dimension (128 recommended)
# --num_expansion_tokens : Number of expansion tokens (64 or 128)
#
# Optional: Fixed weights instead of learned
# Add --fixed_weights to use 0.5/0.5 combination weights (not learned)
#
# Dataset:
# Uses DistillationScores which provides:
# - 1 positive (highest scored by teacher)
# - ~50 hard negatives (lowest scored by teacher)
# - Teacher scores for KD loss

