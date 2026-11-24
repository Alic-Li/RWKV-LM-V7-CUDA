#!/bin/bash
# 参数（在项目根目录运行）
INPUT_PTH="rwkv-init.pth"
DATA_FILE="../train.jsonl"
OUT_DIR="out/run-01"
DEVICES="0,1,2,3"
CTX_LEN=512
M_BSZ=8
GRAD_ACCUM=2
GRAD_CP=1
HEAD_CHUNK=64
LR_INIT=6e-4
LR_FINAL=6e-4
WARMUP_STEPS=10
NCCL_BUCKET_MB=25                    # 每个 FP32 梯度通信桶的大小（MiB）
TRAIN_STEPS=""                       # 留空遍历全部窗口一次

# 训练
./build/rwkv7_train "$INPUT_PTH" \
    --dataset "$DATA_FILE" --out "$OUT_DIR" --devices "$DEVICES" \
    --ctx "$CTX_LEN" --batch "$M_BSZ" --accum "$GRAD_ACCUM" \
    --activation-checkpoint "$GRAD_CP" --head-chunk "$HEAD_CHUNK" \
    --lr "$LR_INIT" --lr-final "$LR_FINAL" --warmup "$WARMUP_STEPS" \
    --nccl-bucket-mb "$NCCL_BUCKET_MB" --steps "$TRAIN_STEPS"
