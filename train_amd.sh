#!/usr/bin/env bash

# 参数（在项目根目录运行）
BUILD_DIR="${BUILD_DIR:-./build-rocm}"
export WORLD_SIZE=8
INPUT_PTH="rwkv-init.pth"
DATA_FILE="../train.jsonl"
OUT_DIR="out/run-01"
CTX_LEN=4096
M_BSZ=42
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
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HSA_XNACK=0 HSA_NO_SCRATCH_RECLAIM=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2
topology=$(rocm-smi --showtoponuma)

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
    node=$(awk -v gpu="GPU[$rank]" '$1 == gpu && /Numa Node:/ {print $NF}' <<<"$topology")
    [[ $node =~ ^[0-9]+$ ]] || { echo "Missing NUMA node for GCD $rank" >&2; exit 1; }
    LOCAL_RANK=$rank numactl --cpunodebind="$node" --membind="$node" \
        "$BUILD_DIR/rwkv7_train" "${ARGS[@]}" &
    pids+=("$!")
done
for ((rank=0; rank<WORLD_SIZE; rank++)); do wait -n; done
pids=()
