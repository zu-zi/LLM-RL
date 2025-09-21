# configs/GRPO_config.py
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# —— 算法开关 —— #
use_ppo  = False
use_grpo = True

# —— 训练基本面 —— #
wandb_log  = True
from datetime import datetime
wandb_project  = "LLM-RL-GRPO"
wandb_run_name = f"grpo_gpt2l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

out_dir        = "/root/autodl-tmp/Results_GRPO"
eval_interval  = 10
max_iters      = 1500
seed_base      = 1337
compile        = False
backend        = "nccl"
device         = "cuda"

# —— 模型 / 上下文 —— #
init_from   = "gpt2-large"
block_size  = 256
bias        = False
dropout     = 0.0

# —— 批次 / 优化 —— #
batch_size                  = 4
gradient_accumulation_steps = 1
RL_learning_rate            = 1.7e-6    # GRPO 无 critic，lr 稍敢一点；与 PPO 同级更稳
weight_decay                = 5e-3
beta1, beta2                = 0.9, 0.95
max_grad_norm               = 0.5

# —— PPO 专属占位（GRPO 不用 critic，不影响）—— #
vf_clip       = None
ppo_clip      = 0.2
entropy_coef  = 0.003       # 比 PPO 再小一点，避免过分发散
kl_ctl_init   = 0.9         # 初始 KL 系数略低，主循环会自适应（过高会偏保守）

# —— GRPO 专属 —— #
grpo_group_size = 4         # 显存充裕可设 6；离线池要对应生成 per-prompt 组
mb_size_logits  = 1         # 日常 1 就行；OOM 再调大
ratio_min       = 0.75
ratio_max       = 1.25
kl_token_cap    = 0.5
k3_cap          = 1.5
ent_mask_keep   = 0.20      # 暂不启用 token 熵子采样，保留接口

# —— 采样口径（与 worker/评测一致）—— #
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P       = 0.9
SAMPLE_TOP_K       = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK       = 16

# —— sglang 离线样本池（与 PPO 一致）—— #
SGLANG_ON                 = True
SGLANG_OFFLINE            = True
SGLANG_MODEL_PATH         = "/root/autodl-tmp/actor_exports/current"  # symlink，worker 用 realpath 重载
SGLANG_EXPORT_BASE        = "/root/autodl-tmp/actor_exports"
SGLANG_SYNC_DIR           = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET     = 200
SGLANG_REFILL_BATCH       = 32
SGLANG_MAX_NEW            = 128
ROLL_LOW_WATERMARK_FACTOR = 2
ROLL_REFILL_COUNT         = 40
ROLL_COOLDOWN_SEC         = 6
ROLL_MIN_FREE_MB          = 2500
REFRESH_EVERY_BATCHES     = 26
FRESH_RATIO               = 0.5

# —— 奖励模型 —— #
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
