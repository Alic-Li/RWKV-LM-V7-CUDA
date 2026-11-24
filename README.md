# RWKV-7 CUDA/C++ 预训练

`rwkv7_init` 独立生成初始权重。`rwkv7_train` 从一个 x070 PyTorch `.pth` state dict 推断模型维度，训练所有模型参数，并按步输出 BF16 state dict。两个程序都可直接运行，无需 Python 环境或 LibTorch。训练使用本仓库的 WKV、TimeMix、ChannelMix 和分块 Head/L2Wrap 融合算子，线性层和反向矩阵乘法使用 cuBLAS。

## 编译

需要 Linux、CMake 3.24+、支持 C++17 的 CUDA 主机编译器、CUDA Toolkit 12+ 和 libarchive 开发包。GPU 需要支持 BF16，即计算能力 8.0 以上。多卡额外需要 NCCL 头文件和库。CMake 默认构建本机 GPU 架构。

本机的 RTX PRO 6000 Blackwell 使用以下命令。CUDA Toolkit 或主机编译器安装路径不同时，对应替换参数即可。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
```

## 初始化与训练

```bash
mkdir -p out
./build/rwkv7_init out/rwkv-init.pth --layers 12 --channels 768 --seed 1234

./build/rwkv7_train out/rwkv-init.pth \
  --dataset corpus.jsonl --out out/run \
  --ctx 1024 --batch 2 --accum 4 --steps 1000 \
  --lr 6e-4 --lr-final 1e-5 --warmup 10 --devices 0
```

已有 x070 权重时可以直接运行训练程序。初始化沿用原项目 x070 模型的参数形状、层相关常量和正交初始化增益。随机数序列由 C++ 的种子控制。Head size 固定为 64，默认 FFN 宽度为 embedding 宽度的 4 倍，默认词表大小为 65536。训练支持 PyTorch ZIP 格式中的 BF16、FP16 和 FP32 输入权重，计算使用 BF16 激活以及 FP32 参数梯度累积、主权重和 AdamW 状态。

`--batch` 是每卡的 microbatch 大小，`--accum` 是每次更新的 microbatch 数。`--ctx` 是窗口上限，须为 16 的倍数。默认重新计算各层激活，可用 `--activation-checkpoint 0` 保留激活，用 `--head-chunk` 控制单次计算的有效目标行数，用 `--wkv-v3 1` 切换 WKV v3。每个 microbatch 的有效长度按目标位置对齐到 16。

`--devices 0,1` 使用单机 NCCL 数据并行，每卡持有完整模型和优化器状态。交叉熵与 L2Wrap 梯度统一除以全卡和梯度累积期间的有效目标数，再通过 NCCL 对梯度求和。只由首卡写出 `rwkv-<step>.pth`。脚本参数 `EPOCH_SAVE` 对应 `--epoch-save`，默认为 1。保存周期沿用原项目从 0 开始的本次训练轮号。例如设为 2 时，进度条上第 1、3、5 轮结束时保存。设为 0 时仅在训练结束时保存。最后不足一轮也保存。已存在的输出目录会为新训练创建独立子目录。继续训练时可用 `--start-step` 延续步号与 warmup 进度，用上一条日志的 `next_window` 设置 `--start-window` 以续接取样位置。权重文件只保存模型参数，重新加载会重新创建 AdamW 状态。

## 数据掩码

JSONL 与 `rwkv_lightning_cuda` 的文本约定一致。每行是独立文档，可以使用 `text` 对该行的所有可预测 token 计算损失，也可使用 `segments` 指定掩码。

```json
{"text":"普通预训练样本。"}
{"segments":[{"text":"问：2+2 等于几？","train":false},{"text":"答：4。","train":true,"end_target":0},{"text":"\n","train":false}]}
```

每个 segment 独立分词。`train:false` 的 token 仍进入前向并为后续目标提供上下文。掩码看的是下一个 token，即预测目标所在位置。`end_target:0` 将训练段后第一个上下文专用 token 位置的预测目标设为 EOS，原输入 token 保持原样。训练首次读取 JSONL 时会在源文件旁写入 `.rwkvidx`，保存字节位置和 token 数，后续按索引读取并分词。源文件和词表的时间戳及文件大小用于判断是否重建索引。自定义词表路径用 `--tokenizer`。

对于 binidx，传入数据前缀、`.idx` 或 `.bin` 路径即可。程序 mmap 读取 MMIDIDX v1 索引、token 文件和必需的掩码。默认掩码路径为 `<prefix>.mask`，可用 `--mask PATH` 指定。掩码每 token 一个原始字节，与 `.bin` 中的 token 逐一对应。

| 字节值 | 预测此位置的标签 |
| --- | --- |
| `0` | 忽略，token 仍可作为上下文 |
| `1` | 使用原 token |
| `2` | 使用 EOS 0，保留原输入 token |

自动遍历时，每份长度为 L 的文档贡献 ceil((L−1)/ctx) 个窗口（L 至少为 2）。每 40,320 个窗口划分一轮，轮内步数是该轮窗口数除以全局 batch 后向上取整。

训练使用 binidx 中的文档边界。超过上下文长度的文档会拆成相邻窗口，窗口间复用一个重叠 token 保持目标对齐，递归状态按窗口重置。文档首 token 和填充位置不贡献损失。

`pretrain/compat` 中的 PTH 读取、JSONL 解析和 tokenizer 实现来自本地的 `rwkv_lightning_cuda` 参考项目，构建自身无需访问该项目。

## HIP 算子库

`hip/` 提供与 CUDA 训练路径对应的 9 个 HIP 源文件。原生 C++ 接口位于 `hip/rwkv7_training_ops.hpp`，由调用方传入 HIP stream 和设备缓冲区。安装 ROCm 后，可从项目根目录按目标 GPU 架构单独构建静态库，例如 MI300 使用以下命令。

```bash
cmake -S hip -B build-hip -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx942
cmake --build build-hip
```

根目录的初始化与训练程序使用 NVIDIA CUDA 后端。HIP 构建目标为算子库 `rwkv7_hip_ops`。


每个完整 mini epoch 读取 40320 个上下文窗口，与 RWKV-LM-V7 的样本数一致。全局批大小为每卡 batch × GPU 数 × 梯度累积次数。在 `train.sh` 中将 `TRAIN_STEPS` 留空，训练器会根据数据索引和全局批大小计算步数，读到本次数据遍历结束。mini epoch 边界和数据尾部均可使用不足一批的窗口。全掩码批计入数据进度，跳过 AdamW 更新。填写正整数可指定批次数，允许跨数据遍历循环取样。`NCCL_BUCKET_MB` 以 MiB 设置每次 NCCL FP32 梯度求和的最大通信量。
