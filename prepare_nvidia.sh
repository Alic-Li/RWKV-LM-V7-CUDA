#!/usr/bin/env bash

# 参数（在项目根目录运行）
BUILD_DIR="${BUILD_DIR:-./build}"
INIT_PTH="rwkv-init.pth"
N_LAYER=12
N_EMBD=768
VOCAB_SIZE=65536
SEED=1234
DEVICE=0

ARGS=("$INIT_PTH" --layers "$N_LAYER" --channels "$N_EMBD" --vocab "$VOCAB_SIZE"
      --seed "$SEED" --device "$DEVICE")

# 初始化权重（传入命令行参数时使用传入的参数）
exec "$BUILD_DIR/rwkv7_init" "${@-"${ARGS[@]}"}"
