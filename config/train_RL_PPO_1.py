# 根据输出调整整套参数+train_Rl_only.py的参数
# ================== Offline rollouts (sglang) ==================
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"                 # 也可换更快的推理模型
SGLANG_SYNC_DIR = "/root/autodl-tmp/sgl_pool"    # JSONL 样本池（放大盘）
SGLANG_ROLLOUT_TARGET = 96     # 128                # 目标池容量
SGLANG_REFILL_BATCH = 32       # 64                 # 每次补货生成条数
SGLANG_MAX_NEW = 80                             # 每条最大新 token（response）

# ===== 补货调度阈值（避免与训练争显存）=====
# 低水位阈值：只有池子低到这么少才补货（越小越不容易重叠）
ROLL_LOW_WATERMARK_FACTOR = 2     # 实际低水位 = batch_size * 2
# 单次补货条数（配合上面的低水位，小颗粒更安全）
ROLL_REFILL_COUNT = 16   # 16         # 等价于 SGLANG_REFILL_BATCH 的小颗粒重载
# 两次补货之间最少间隔（秒）
ROLL_COOLDOWN_SEC = 20  #25
# 触发补货前，GPU 至少需要这么多空闲显存（MB）
ROLL_MIN_FREE_MB = 4000          # 例如 12 GB；V100-32GB 上这个值很保守

# ================== Training (single 32GB GPU suggested) ==================
init_from = "gpt2-large"                         # actor/ref 初始化
block_size = 384
batch_size = 4
gradient_accumulation_steps = 2   # 4
RL_learning_rate = 3e-6  # 2e-6
kl_ctl = 0.5                    # 0.4                # 初始 KL 系数（训练中会自适应微调）
ppo_clip = 0.12  #0.12
entropy_coef = 0.003   
use_ppo = True
use_grpo = False
use_dapo  = False
use_token_entropy = False

# ——更高比例在线新鲜样本（更稳，稍慢）——
FRESH_RATIO = 0.5

# ================== Reward model (EN) ==================
# 重要：切换到英文奖励模型，和 hh-rlhf 英文 prompts 对齐
# 可以用 "OpenAssistant/reward-model-deberta-v3-base"（更省显存）
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# ================== Logging / runtime ==================
out_dir = "/root/autodl-tmp/Results"             # 结果目录（放大盘）
eval_interval = 8                                # 每步打日志，便于观察
max_iters = 1000
always_save_checkpoint = False
wandb_log = False