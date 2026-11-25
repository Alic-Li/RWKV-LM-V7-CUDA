#!/usr/bin/env bash

# 参数（在项目根目录运行）
BUILD_DIR="${BUILD_DIR:-./build}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
INPUT_PTH="rwkv-init.pth"
DATA_FILE="../train.jsonl"
OUT_DIR="out/run-01"
CTX_LEN=512
M_BSZ=16
GRAD_ACCUM=1
GRAD_CP=1
HEAD_CHUNK=65536
LR_INIT=6e-4
LR_FINAL=6e-4
WARMUP_STEPS=10
NCCL_BUCKET_MB=25
EPOCH_SAVE=1
TRAIN_STEPS=""  # 留空表示遍历全部窗口一次

ARGS=("$INPUT_PTH" --dataset "$DATA_FILE" --out "$OUT_DIR"
      --ctx "$CTX_LEN" --batch "$M_BSZ" --accum "$GRAD_ACCUM"
      --activation-checkpoint "$GRAD_CP" --head-chunk "$HEAD_CHUNK"
      --lr "$LR_INIT" --lr-final "$LR_FINAL" --warmup "$WARMUP_STEPS"
      --nccl-bucket-mb "$NCCL_BUCKET_MB" --epoch-save "$EPOCH_SAVE" --steps "$TRAIN_STEPS")

# 训练（传入命令行参数时使用传入的参数）
set -euo pipefail
if (( $# )); then ARGS=("$@"); fi
[[ $WORLD_SIZE =~ ^[1-9][0-9]*$ ]] || { echo "WORLD_SIZE must be a positive integer" >&2; exit 1; }
if (( WORLD_SIZE == 1 )); then
    export LOCAL_RANK=0
    exec "$BUILD_DIR/rwkv7_train" "${ARGS[@]}"
fi

export RWKV_RENDEZVOUS
RWKV_RENDEZVOUS=$(mktemp -d)
pids=()
cleanup() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        sleep 1
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
    wait || true
    rm -rf -- "$RWKV_RENDEZVOUS"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

for ((rank=0; rank<WORLD_SIZE; rank++)); do
    LOCAL_RANK=$rank "$BUILD_DIR/rwkv7_train" "${ARGS[@]}" &
    pids+=("$!")
done
for ((rank=0; rank<WORLD_SIZE; rank++)); do wait -n; done
pids=()
