#!/bin/bash
# 参数（在项目根目录运行）
INPUT_PTH="rwkv-init.pth"
DATA_FILE="../train.jsonl"
OUT_DIR="out/run-01"
DEVICES="0,1,2,3"
CTX_LEN=256
M_BSZ=2
GRAD_ACCUM=2
GRAD_CP=1
HEAD_CHUNK=64
LR_INIT=6e-4
LR_FINAL=6e-4
WARMUP_STEPS=0
TRAIN_STEPS=10
SAVE_EVERY=10

# 训练
./build/rwkv7_train "$INPUT_PTH" \
    --dataset "$DATA_FILE" --out "$OUT_DIR" --devices "$DEVICES" \
    --ctx "$CTX_LEN" --batch "$M_BSZ" --accum "$GRAD_ACCUM" \
    --activation-checkpoint "$GRAD_CP" --head-chunk "$HEAD_CHUNK" \
    --lr "$LR_INIT" --lr-final "$LR_FINAL" --warmup "$WARMUP_STEPS" \
    --steps "$TRAIN_STEPS" --save-every "$SAVE_EVERY"
