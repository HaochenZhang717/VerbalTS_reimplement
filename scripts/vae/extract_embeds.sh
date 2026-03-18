#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=3

# =========================
# 路径配置
# =========================

EXP_NAME="vae_synth_u"

DATA_PATH="/playpen/haochenz/LitsDatasets/128_len_ts/synthetic_u/train_ts.npy"
CKPT_PATH="./vae_ckpts/${EXP_NAME}/best.pt"

SAVE_PATH="./vae_embeddings/synth_u/train_${EXP_NAME}.npy"

mkdir -p ./vae_embeddings

# =========================
# 模型参数（必须和训练一致）
# =========================

BATCH_SIZE=128

HIDDEN_SIZE=128
NUM_LAYERS=2
NUM_HEADS=8
LATENT_DIM=64

# embedding 类型
MODE="mean"   # mean / full / flatten

# =========================
# 运行
# =========================

echo "Extracting embeddings..."
echo "Mode: ${MODE}"

python extract_embedding.py \
    --data_path ${DATA_PATH} \
    --ckpt_path ${CKPT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --num_layers ${NUM_LAYERS} \
    --num_heads ${NUM_HEADS} \
    --latent_dim ${LATENT_DIM} \
    --mode ${MODE} \
    --save_path ${SAVE_PATH}

echo "Done!"