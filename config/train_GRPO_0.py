# ================== Offline rollouts (sglang) ==================
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"
SGLANG_SYNC_DIR = "/root/autodl-tmp/sgl_pool"
SGLANG_ROLLOUT_TARGET = 96
SGLANG_REFILL_BATCH = 24
SGLANG_MAX_NEW = 96

# ===== 补货调度阈值 =====
ROLL_LOW_WATERMARK_FACTOR = 2
ROLL_REFILL_COUNT = 16
ROLL_COOLDOWN_SEC = 20
ROLL_MIN_FREE_MB = 6000

# ================== Training ==================
init_from = "gpt2-large"
block_size = 384

# ——GRPO 更依赖“组内相对”稳定性，建议 batch 是 group_size 的整数倍——
batch_size = 4                     # 和 group_size=4 对齐
gradient_accumulation_steps = 2

# ——无 critic；policy 更“轻”，LR 可略高于 PPO（别太激进）——
RL_learning_rate = 2.5e-6          # 你的 PPO 是 2e-6，这里小幅上调

# ——GRPO 专用超参（新增）——
GRPO_GROUP_SIZE   = 4              # 每组样本数；>=4 更稳
GRPO_KL_COEF      = 0.15           # 句级 KL 正则初值（训练中会被主循环动态微调）
GRPO_CLIP_REWARD  = 3.0            # 组内标准化后的权重裁剪（|w|<=3）
MB_SIZE_LOGITS    = 2              # 计算 logits 时的 micro-batch，大模型可再加大

# ——PPO 专属项在 GRPO 中无效；显式设为 0 以免误读——
kl_ctl        = GRPO_KL_COEF       # 复用变量名，便于主循环里的自适应逻辑

use_ppo  = False
use_grpo = True
use_dapo = False
use_token_entropy = False
ent_keep_ratio=0.2

# ——在线新鲜样本占比——
FRESH_RATIO = 0.5

# ================== Reward model (EN) ==================
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ================== Logging / runtime ==================
out_dir = "/root/autodl-tmp/Results"
eval_interval = 8
max_iters = 1000
always_save_checkpoint = False
wandb_log = False
