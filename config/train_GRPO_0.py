# configs/GRPO_config.py
import os
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

# —— 算法开关 —— #
use_ppo  = False
use_grpo = True

# —— 训练基本面（与 PPO 保持一致，必要时覆盖）—— #
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

init_from   = "gpt2-large"
block_size  = 256
bias        = False
dropout     = 0.0

batch_size                  = 4
gradient_accumulation_steps = 1
RL_learning_rate            = 1.7e-6
weight_decay                = 5e-3
beta1, beta2                = 0.9, 0.95
max_grad_norm               = 0.5

# —— PPO 专属（GRPO 不用 critic，留空值不影响）—— #
vf_clip                     = None
ppo_clip                    = 0.2
entropy_coef                = 0.005
kl_ctl_init                 = 1.0

# —— GRPO 专属 —— #
grpo_group_size             = 4          # 诊断期望值（实际组大小看日志 grpo/group_eff_mean）
mb_size_logits              = 1
ratio_min                   = 0.75
ratio_max                   = 1.25
kl_token_cap                = 0.5
k3_cap                      = 1.5
ent_mask_keep               = 0.20

# —— 采样口径（评测/在线补齐一致）—— #
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P       = 0.9
SAMPLE_TOP_K       = 0
SAMPLE_REP_PENALTY = 1.1
SAMPLE_STOPS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK       = 16

# —— sglang 离线样本池（与你 PPO 配置一致即可）—— #
SGLANG_ON              = True
SGLANG_OFFLINE         = True
SGLANG_MODEL_PATH      = "/root/autodl-tmp/actor_exports/current"
SGLANG_EXPORT_BASE     = "/root/autodl-tmp/actor_exports"
SGLANG_SYNC_DIR        = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET  = 200
SGLANG_REFILL_BATCH    = 32
SGLANG_MAX_NEW         = 128
ROLL_LOW_WATERMARK_FACTOR = 2
ROLL_REFILL_COUNT      = 40
ROLL_COOLDOWN_SEC      = 6
ROLL_MIN_FREE_MB       = 2500
REFRESH_EVERY_BATCHES  = 26
FRESH_RATIO            = 0.5

REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"
