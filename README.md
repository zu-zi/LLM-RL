# LLM & RL(中文初版)
+ 目标：基于 LLM 代码库[nanoGPT](https://github.com/karpathy/nanoGPT)，使用原生 PyTorch 实现 LLM + RL 的一系列算法变体，掌握最新 RLVR 方向的研究趋势
+ 实现的 LLM + RL 训练算法如下:
  + [PPO](https://arxiv.org/pdf/1707.06347)
  + [GRPO(deepseek)](https://arxiv.org/pdf/2402.03300)
  + [DAPO](https://arxiv.org/abs/2503.14476)
  + [token entropy相关](https://arxiv.org/abs/2506.01939)
+ 基础模型：使用 nanoGPT 训练出来的 gpt2-large
+ 显卡：32GB-vGPU * 1
+ 数据集："openbmb/UltraFeedback"
+ 奖励模型："OpenAssistant/reward-model-deberta-v3-large-v2"
<!-- + 评测基准：RewardBench v1/v2 -->
+ rollout：[sglang](https://github.com/sgl-project/sglang)引擎加速
+ wandb结果: 默认sglang rollout (后面跑一个曲线更好看的出来？)
    + [PPO](https://wandb.ai/zu_zi666/hlhf-ppo)
    + [GRPO](https://wandb.ai/zu_zi666/hlhf-grpo):sglang * no sglang
    + [DAPO](https://wandb.ai/zu_zi666/hlhf-dapo):token entropy * no token entropy

# build environment(autodl * 32GB)
```
apt-get update
apt-get install -y git
git clone https://github.com/zu-zi/LLM-RL.git
cd LLM-RL

# 0 基础环境
python3 -m pip install -U pip

# PyTorch 2.8 + cu128
python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

python3 -m pip install -U numpy transformers datasets tiktoken bitsandbytes accelerate

# 1 环境变量
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"    # 兼容 20/30/40 系
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Hugging Face 缓存放到大盘
mkdir -p /root/autodl-tmp/hf
export HF_HOME=/root/autodl-tmp/hf
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf

# CUDA Path
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# 2 SGLang
python3 -m pip install -U "sglang[all]"
python3 -m pip install -U sgl-kernel

# 3 FlashInfer

python3 -m pip install -U flashinfer-python flashinfer-cubin

python3 -m pip install -U --index-url https://flashinfer.ai/whl/cu128 flashinfer-jit-cache

# 4 版本校验
python3 - <<'PY'
import torch, importlib
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
try:
    import sglang
    print("sglang:", sglang.__version__)
except Exception as e:
    print("sglang import error:", e)
try:
    fi = importlib.import_module("flashinfer")
    print("flashinfer imported OK")
except Exception as e:
    print("flashinfer import error:", e)
PY

# python3 test_sglang.py

# data
python3 data/RL_dataset/prepare.py

# train
python3 train_PPO.py
# python3 train_GRPO.py
# python3 train_DAPO.py
```
h200新版本环境：
```
conda activate /mnt/afs/wanzunian/wenwen/envs/env-efftoken
python -m pip install -U pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install -U numpy transformers datasets tiktoken bitsandbytes accelerate wandb
python -m pip install -U torch-c-dlpack-ext

python -m pip install -U --no-cache-dir --timeout 300 --retries 30 \
  -i https://pypi.org/simple \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  "sglang[all]"

python -m pip install -U --no-cache-dir --timeout 300 --retries 30 \
  -i https://pypi.org/simple \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  sgl-kernel

python -m pip install -U flashinfer-python flashinfer-cubin
python -m pip install -U --index-url https://flashinfer.ai/whl/cu128 flashinfer-jit-cache

cd /mnt/afs/wanzunian/wenwen/LLM-RL

mkdir -p .cache/hf
mkdir -p .cache/sglang/sgl_pool
mkdir -p .cache/sglang/actor_exports
mkdir -p Results
mkdir -p .tmp

# 每次开新 shell
conda activate /mnt/afs/wanzunian/wenwen/envs/env-efftoken
cd /mnt/afs/wanzunian/wenwen/LLM-RL
source env_h200.sh
```
# 运行时：
+ 每次开机重新执行：export HF_ENDPOINT=https://hf-mirror.com

+ 离线上传wandb
```
# 登录
wandb login   # 输入你的 API key

# 同步某个 run
wandb sync /root/autodl-tmp/Results/wandb/run-2025xxxx_xxxxxx-*

# 或一次性同步整个目录下的所有离线 runs
wandb sync /root/autodl-tmp/Results/wandb
```

+ 每次开新轮，需要清空旧池数据
```
rm -f  /root/autodl-tmp/sgl_pool/roll_*.jsonl
rm -f  /root/autodl-tmp/actor_exports/current
rm -rf /root/autodl-tmp/actor_exports/ts_*
mkdir -p /root/autodl-tmp/sgl_pool /root/autodl-tmp/actor_exports

# 可选：新一轮想要全新日志
rm -f /root/autodl-tmp/Results/metrics.csv
rm -f /root/autodl-tmp/Results/rollout_logs/*.log
```

+ 清理数据盘
sudo rm -rf /root/autodl-tmp/.Trash-0/*

# 项目结构
```
LLM-RL
├── data
│   └── RL_dataset
│       └── prepare.py
├── Results/
├── RL
│   ├── DAPO.py
│   ├── GRPO.py
│   └── PPO.py
├── utils
│   └── rollout_pool.py
├── .gitignore
├── model.py
├── QA.md
├── README_nanoGPT.md
├── README.md
├── rollout_worker.py
├── sample.py
├── test_reward.py
├── test_sglang.py
├── train_DAPO.py
├── train_GRPO.py
├── train_PPO.py
└── train.py
```
+ data/RL_dataset/prepare.py：RL 训练数据的清洗与标准化
+ Results/：训练产出目录（权重、日志、评测结果等）,默认放到/root/autodl-tmp/,可改
+ RL/DAPO.py：DAPO 算法实现
+ RL/GRPO.py：GRPO 算法实现
+ RL/PPO.py：PPO 算法实现
+ utils/rollout_pool.py：离线/并发安全的经验池；负责样本入队、TTL、去重、容量管理等
+ model.py：原nanoGPT模型装配与封装（actor、reward 的统一加载与前向）
+ QA.md：学习记录，待删
+ README_nanoGPT.md：nanoGPT 上游文档
+ README.md：本项目说明（运行指引、结构与结果）
+ rollout_worker.py：rollout 采样进程，调用 actor 生成样本并写入 pool
+ sample.py：原nanoGPT文件，从已训模型采样生成文本，用于定性测试
+ test_reward.py：校验/可视化 reward model 打分流程是否正常
+ test_sglang.py：sglang环境准备检测，连通性与接口自检（sglang 离线池/导出是否可用）
+ train_DAPO.py：DAPO 训练入口脚本
+ train_GRPO.py：GRPO 训练入口脚本
+ train_PPO.py：PPO 训练入口脚本
+ train.py：原nanoGPT训练脚本，RL扩展基础

***
## run
```
source /mnt/afs/wanzunian/wenwen/bin/env_common.sh
export CONDARC=$W/.condarc
conda activate $W/envs/env-efftoken
source env_h200.sh
python3 data/RL_dataset/prepare.py

TOKSEL_MODE=full python -u train_DAPO.py

TOKSEL_MODE=entropy_ratio TOKSEL_RHO=0.2 python -u train_DAPO.py
TOKSEL_MODE=entropy_ratio TOKSEL_RHO=0.8 python -u train_DAPO.py

TOKSEL_MODE=entropy_budget TOKSEL_K=256 python -u train_DAPO.py
TOKSEL_MODE=entropy_budget TOKSEL_K=512 python -u train_DAPO.py

TOKSEL_MODE=random_budget TOKSEL_K=256 python -u train_DAPO.py
TOKSEL_MODE=random_budget TOKSEL_K=512 python -u train_DAPO.py

# 画图
python tools/plot_metrics.py --csv Results/<你的run目录>/metrics.csv
python tools/plot_metrics.py \
  --csv  Results/<entropy_budget_run>/metrics.csv \
  --csv2 Results/<random_budget_run>/metrics.csv \
  --label2 randomK256


# baseline
TOKSEL_MODE=full python -u train_DAPO.py

# “现象复现”：比例太小可能不稳
TOKSEL_MODE=entropy_ratio TOKSEL_RHO=0.2 python -u train_DAPO.py

# 固定预算：量化 Neff
TOKSEL_MODE=entropy_budget TOKSEL_K=256 python -u train_DAPO.py

# 同预算对照：验证质量是否有贡献
TOKSEL_MODE=random_budget TOKSEL_K=256 python -u train_DAPO.py
```