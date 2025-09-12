# ================== Offline rollouts (sglang) ==================
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"
SGLANG_SYNC_DIR   = "/root/autodl-tmp/sgl_pool"

# 保持池子新鲜，避免 KL 尖峰
SGLANG_ROLLOUT_TARGET = 96      # 目标文件条数
SGLANG_REFILL_BATCH  = 48       # 每次补货
SGLANG_MAX_NEW       = 48       # 生成上限（DAPO/GRPO 初期保守些）

# 补货调度阈值
ROLL_LOW_WATERMARK_FACTOR = 3   # 低水位= batch_size * 该系数
ROLL_REFILL_COUNT = 24
ROLL_COOLDOWN_SEC = 18
ROLL_MIN_FREE_MB  = 7000

# ================== Training / Model ==================
init_from  = "gpt2-large"
block_size = 384

# Data/Batch
batch_size = 4                  # 建议为 group_size 的整数倍
gradient_accumulation_steps = 2

# Optim
RL_learning_rate = 1.5e-6       # policy-only，略高于 PPO 也可；先稳住
weight_decay = 5e-3
beta1 = 0.9
beta2 = 0.95

# ================== DAPO 专用 ==================
# 注：这些名字与 train_RL_only.py 中 DAPOTrainer(...) 的入参变量一一对应
group_size   = 4                # 每个 prompt 组内样本数
kl_coef      = 0.01             # 句级 KL shaping 系数（加到 reward 上）
beta         = 0.8              # token 级 k3 系数（加到 per-token loss）
adv_norm     = "zscore"         # ["zscore" | "center" | "none"]
adv_clip     = 5.0              # 句级优势裁剪 |A|<=adv_clip
MB_SIZE_LOGITS = 2              # logits 计算的 micro-batch
DAPO_MAX_NEW   = SGLANG_MAX_NEW # 采样最长新 token（可独立于池子上限）
MIN_RESP_TOK   = 16             # 最短响应 token（与离线/在线一致）

# ================== 开关 ==================
use_ppo  = False
use_grpo = False
use_dapo = True
use_token_entropy = False
ent_keep_ratio=0.2

# ================== Reward model ==================
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ================== Logging / runtime ==================
out_dir = "/root/autodl-tmp/Results"
eval_interval = 8
max_iters = 1000
wandb_log = False
wandb_project = "hlhf"
wandb_run_name = "dapo_gpt2l"
