# bash setup_sglang.sh                # 默认：env=sgl-cu124，python=3.10，使用清华镜像加速，安装cuda-toolkit
# bash setup_sglang.sh --env-name myenv --py 3.11 --no-mirror --no-cuda-toolkit

#!/usr/bin/env bash
# setup_sglang.sh
# 一键搭建 SGLang 环境（PyTorch 2.5.1 + CUDA 12.4）并自测

set -euo pipefail

########## 可调参数（也可用命令行覆盖） ##########
ENV_NAME="sgl-cu124"
PY_VER="3.10"
USE_MIRROR=1           # 1=使用清华镜像加速（仅针对非 torch/flashinfer 包）, 0=不用
INSTALL_CUDA_TOOLKIT=1 # 1=安装cuda-toolkit=12.4（提供nvcc），0=不装

# 版本固定（稳定组合）
TORCH_VER="2.5.1"
TV_VER="0.20.1"
TA_VER="2.5.1"

########## 解析命令行 ##########
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --py) PY_VER="$2"; shift 2;;
    --no-mirror) USE_MIRROR=0; shift 1;;
    --no-cuda-toolkit) INSTALL_CUDA_TOOLKIT=0; shift 1;;
    -h|--help)
      echo "Usage: bash setup_sglang.sh [--env-name NAME] [--py 3.10|3.11] [--no-mirror] [--no-cuda-toolkit]"
      exit 0;;
    *) echo "Unknown arg: $1" && exit 1;;
  esac
done

########## 函数 ##########
die() { echo "ERROR: $*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"
}

########## 预检查 ##########
need_cmd bash
need_cmd python || true
need_cmd conda

# 让 conda 可被非交互脚本激活
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=== 配置参数 ==="
echo "ENV_NAME           = $ENV_NAME"
echo "PY_VER             = $PY_VER"
echo "USE_MIRROR         = $USE_MIRROR"
echo "INSTALL_CUDA_TOOLKIT = $INSTALL_CUDA_TOOLKIT"
echo "TORCH/TV/TA        = $TORCH_VER / $TV_VER / $TA_VER"
echo

########## 创建/激活环境 ##########
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[1/7] 已存在环境 $ENV_NAME"
else
  echo "[1/7] 创建 conda 环境：$ENV_NAME (python=$PY_VER)"
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"

########## 可选：镜像配置 ##########
if [[ $USE_MIRROR -eq 1 ]]; then
  export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "[2/7] 已启用清华 PyPI 镜像加速（不影响 torch/flashinfer 的专用索引）"
else
  unset PIP_INDEX_URL || true
  echo "[2/7] 未启用镜像（全部走官方源）"
fi

python - <<'PY'
import sys, platform
print("[INFO] Python:", sys.version.split()[0], "| Platform:", platform.platform())
PY

########## 安装 PyTorch 2.5.1 + cu124 ##########
echo "[3/7] 安装 PyTorch/cu124 及相关组件（官方 cu124 源）..."
python -m pip install -U pip
pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("[CHECK] torch:", torch.__version__, "| cuda runtime from torch:", torch.version.cuda)
print("[CHECK] CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[CHECK] Device 0:", torch.cuda.get_device_name(0))
PY

########## （可选）安装 cuda-toolkit 以提供 nvcc ##########
if [[ $INSTALL_CUDA_TOOLKIT -eq 1 ]]; then
  echo "[4/7] 安装 cuda-toolkit=12.4（nvcc，用于编译CUDA扩展更稳）..."
  conda install -y -c nvidia cuda-toolkit=12.4
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  if command -v nvcc >/dev/null 2>&1; then
    echo "[CHECK] nvcc 存在：$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1)"
  else
    echo "[WARN] 未找到 nvcc（通常也可不装，但编译扩展时可能需要）"
  fi
else
  echo "[4/7] 跳过安装 cuda-toolkit（如后续需编译扩展，可自行安装）"
fi

########## 安装 sglang、sgl-kernel、flashinfer、NLP 依赖 ##########
echo "[5/7] 安装 sglang（镜像或官方）..."
pip install "sglang[all]"

echo "[5/7] 安装 sgl-kernel..."
pip install -U sgl-kernel

echo "[5/7] 安装 flashinfer（torch2.5 + cu124 专用轮子）..."
pip install flashinfer \
  --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/

echo "[5/7] 安装 Transformers/Accelerate/Tiktoken..."
pip install transformers accelerate tiktoken

########## 版本打印 ##########
python - <<'PY'
import sglang, torch
print("[CHECK] sglang:", sglang.__version__)
print("[CHECK] torch:", torch.__version__, "| torch.cuda:", torch.version.cuda, "| cuda_available:", torch.cuda.is_available())
PY

########## 生成并运行 SGLang 离线引擎测试 ##########
echo "[6/7] 生成测试脚本 test_sglang_offline.py ..."
cat > test_sglang_offline.py <<'PY'
import torch
import sglang as sgl
from transformers import AutoTokenizer

def test_basic():
    print("="*60)
    print("基本功能自检")
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("SGLang:", sgl.__version__)

def test_engine():
    print("="*60)
    print("初始化 SGLang 离线引擎（Transformers 后端）")
    llm = sgl.Engine(
        model_path="gpt2",          # 可换成 "Qwen/Qwen2.5-1.5B-Instruct" 等更大的模型
        impl="transformers",        # 直接走 HF Transformers 后端
        tokenizer="gpt2",
        disable_cuda_graph=True     # 小模型/单卡更稳
    )
    prompt = "Hello, how are you?"
    sampling = {
        "max_new_tokens": 20,
        "temperature": 0.7,
        "stop_token_ids": [AutoTokenizer.from_pretrained("gpt2").eos_token_id],
    }
    print("输入：", prompt)
    out = llm.generate([prompt], sampling)
    print("输出：", out[0])
    llm.shutdown()

if __name__ == "__main__":
    test_basic()
    test_engine()
    print("\n🎉 OK! SGLang 离线引擎跑通。")
PY

echo "[7/7] 运行测试脚本 ..."
python test_sglang_offline.py

echo
echo "============================== DONE =============================="
echo "环境：$ENV_NAME  已配置完毕并通过测试。"
echo "如需再次使用：  conda activate $ENV_NAME"
echo "测试脚本：       $(pwd)/test_sglang_offline.py"
