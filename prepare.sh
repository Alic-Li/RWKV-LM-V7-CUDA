#!/bin/bash
# 参数（在项目根目录运行）
N_LAYER="12"
N_EMBD="768"
VOCAB_SIZE=65536
SEED=1234
DEVICE=0
INIT_PTH="rwkv-init.pth"

# 初始化权重
"${BUILD_DIR:-./build}/rwkv7_init" "$INIT_PTH" \
    --layers "$N_LAYER" --channels "$N_EMBD" --vocab "$VOCAB_SIZE" \
    --seed "$SEED" --device "$DEVICE"
