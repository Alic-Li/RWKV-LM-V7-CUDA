#!/usr/bin/env bash

# mi250-1：GCD 1，PCI 0000:34:00.0。进程内使用设备 0。
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export ROCR_VISIBLE_DEVICES=1
export RWKV_INFER_NUMA_NODE=0

# source 时仅准备环境；直接执行时为推理命令绑定本地 CPU 和内存。
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then return 0; fi
if (( $# == 0 )); then
    echo "用法：$0 推理命令 [参数...]" >&2
    exit 2
fi
exec numactl --cpunodebind="$RWKV_INFER_NUMA_NODE" --membind="$RWKV_INFER_NUMA_NODE" "$@"
