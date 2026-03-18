#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=1

# =========================
# 实验名（必须和训练一致）
# =========================
EXP_NAME="vae_synth_u"

# =========================
# 路径配置
# =========================

# real data（通常用 test 或 valid）
REAL_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
# fake data（你的生成结果）
FAKE_PATH="/playpen/haochenz/VerbalTS_reimplement/verbalts_orig_save/synth_u_vae_embed/text2ts_msmdiffmv/0/real_text_samples.pt"
# VAE checkpoint
CKPT_PATH="./fid_vae_ckpts/${EXP_NAME}/best.pt"
# 输出（其实现在只是打印，可以以后扩展）
SAVE_PATH="./fid_results/${EXP_NAME}.txt"
mkdir -p ./fid_results

# =========================
# 模型参数（必须和训练一致）
# =========================

BATCH_SIZE=128

HIDDEN_SIZE=128
NUM_LAYERS=2
NUM_HEADS=8
LATENT_DIM=64

# =========================
# 运行
# =========================

echo "=========================="
echo "Running FID Evaluation"
echo "Experiment: ${EXP_NAME}"
echo "=========================="

python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path ${CKPT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --latent_dim ${LATENT_DIM} \
    --save_path ${SAVE_PATH}

echo "=========================="
echo "FID Done!"
echo "=========================="