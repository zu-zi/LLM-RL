#!/usr/bin/env bash
set -e

export LLMRL_ROOT="/mnt/afs/wanzunian/wenwen/LLM-RL"

# HF 缓存全部进项目内（不往 ~/.cache 乱写）
export HF_HOME="$LLMRL_ROOT/.cache/hf"
export TRANSFORMERS_CACHE="$LLMRL_ROOT/.cache/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_OFFLINE=0

# 训练产出（wandb/metrics/ckpt）全部进项目内
export WANDB_MODE=offline
export WANDB_DIR="$LLMRL_ROOT/Results/wandb"

# CUDA + torch 行为
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# H200 属于 compute capability 9.0（H100/H200/GH200 都是 9.0）:contentReference[oaicite:0]{index=0}
export TORCH_CUDA_ARCH_LIST="9.0"

# 临时目录也锁进项目
export TMPDIR="$LLMRL_ROOT/.tmp"

# CUDA_HOME 自动探测（优先 nvcc）
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(cd "$(dirname "$(which nvcc)")/.." && pwd)"
else
  export CUDA_HOME="/usr/local/cuda"
fi
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 给训练脚本用的路径钩子（下面第6步会配合代码改造）
export LLMRL_WORKDIR="$LLMRL_ROOT"