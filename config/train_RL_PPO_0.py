# ================== Offline rollouts (sglang) ==================
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"                 # 也可换更快的推理模型
SGLANG_SYNC_DIR = "/root/autodl-tmp/sgl_pool"    # JSONL 样本池（放大盘）
SGLANG_ROLLOUT_TARGET = 104     # 128                # 目标池容量
SGLANG_REFILL_BATCH = 24       # 64                 # 每次补货生成条数
SGLANG_MAX_NEW = 80                             # 每条最大新 token（response）

# ===== 补货调度阈值（避免与训练争显存）=====
# 低水位阈值：只有池子低到这么少才补货（越小越不容易重叠）
ROLL_LOW_WATERMARK_FACTOR = 2     # 实际低水位 = batch_size * 2
# 单次补货条数（配合上面的低水位，小颗粒更安全）
ROLL_REFILL_COUNT = 12   # 16         # 等价于 SGLANG_REFILL_BATCH 的小颗粒重载
# 两次补货之间最少间隔（秒）
ROLL_COOLDOWN_SEC = 15  #25
# 触发补货前，GPU 至少需要这么多空闲显存（MB）
ROLL_MIN_FREE_MB = 3000          # 例如 12 GB；V100-32GB 上这个值很保守

# ================== Training (single 32GB GPU suggested) ==================
init_from = "gpt2-large"                         # actor/ref 初始化
block_size = 384
batch_size = 4
gradient_accumulation_steps = 2   # 4
RL_learning_rate = 1.5e-6  # 2e-6
kl_ctl = 0.9                    # 0.4                # 初始 KL 系数（训练中会自适应微调）
ppo_clip = 0.12  #0.12
entropy_coef = 0.003   
use_ppo = True
use_grpo = False
use_dapo  = False
use_token_entropy = False
ent_keep_ratio=0.2

# ——更高比例在线新鲜样本（更稳，稍慢）——
FRESH_RATIO = 0.4

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


# ===== KL 迟滞（冻结/解冻）相关，可选：未在 train.py 暴露成全局读取，仅作注释参考 =====
# KL_HALT   = 1.0  # 当报告的平均 KL 连续 >= HALT_STREAK 次超过该阈值 → 冻结 actor（只训 critic）。
#                   # 设得越大越“宽松”（更不容易冻结），设得越小越“保守”（更容易冻结）。
# KL_RESUME = 0.6  # 当报告的平均 KL 连续 >= RESUME_STREAK 次低于该阈值 → 解冻 actor（恢复训 actor）。
#                   # 设得越大越“激进”（更容易解冻），设得越小越“保守”（更不容易解冻）。
# HALT_STREAK   = 2  # 触发冻结所需的连续“超阈值”次数，用来抑制抖动。
# RESUME_STREAK = 2  # 触发解冻所需的连续“低于阈值”次数，同样抑制抖动。
# KL_STOP = 1.0      # 临时“按 KL 降学习率”的枢轴点：当 KL>KL_STOP，actor 的 LR 会按误差缩放降低。
#                    # 设得越大，LR 降得越“晚”；越小，LR 更早开始收缩。

# # ================== Offline rollouts (sglang) ==================
# SGLANG_ON = True
# SGLANG_OFFLINE = True
# SGLANG_MODEL_PATH = "gpt2-large"                 # 也可换更快的推理模型
# SGLANG_SYNC_DIR = "/root/autodl-tmp/sgl_pool"    # JSONL 样本池（放大盘）
# SGLANG_ROLLOUT_TARGET = 128     # 128                # 目标池容量
# SGLANG_REFILL_BATCH = 12       # 64                 # 每次补货生成条数
# SGLANG_MAX_NEW = 104                             # 每条最大新 token（response）

# # ===== 补货调度阈值（避免与训练争显存）=====
# # 低水位阈值：只有池子低到这么少才补货（越小越不容易重叠）
# ROLL_LOW_WATERMARK_FACTOR = 3     # 实际低水位 = batch_size * 2
# # 单次补货条数（配合上面的低水位，小颗粒更安全）
# ROLL_REFILL_COUNT = 8   # 16         # 等价于 SGLANG_REFILL_BATCH 的小颗粒重载
# # 两次补货之间最少间隔（秒）
# ROLL_COOLDOWN_SEC = 8  #25
# # 触发补货前，GPU 至少需要这么多空闲显存（MB）
# ROLL_MIN_FREE_MB = 4000          # 例如 12 GB；V100-32GB 上这个值很保守

# # ================== Training (single 32GB GPU suggested) ==================
# init_from = "gpt2-large"                         # actor/ref 初始化
# block_size = 384
# batch_size = 4
# gradient_accumulation_steps = 2   # 4
# RL_learning_rate = 2e-6  # 2e-6
# kl_ctl = 0.7                    # 0.4                # 初始 KL 系数（训练中会自适应微调）
# ppo_clip = 0.17  #0.12
# entropy_coef = 0.003   
# use_ppo = True
# use_grpo = False
# use_dapo  = False
# use_token_entropy = False

# # ——更高比例在线新鲜样本（更稳，稍慢）——
# FRESH_RATIO = 0.4

# # ================== Reward model (EN) ==================
# # 重要：切换到英文奖励模型，和 hh-rlhf 英文 prompts 对齐
# # 可以用 "OpenAssistant/reward-model-deberta-v3-base"（更省显存）
# REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# # ================== Logging / runtime ==================
# out_dir = "/root/autodl-tmp/Results"             # 结果目录（放大盘）
# eval_interval = 8                                # 每步打日志，便于观察
# max_iters = 1000
# always_save_checkpoint = False
# wandb_log = False


# # ===== KL 迟滞（冻结/解冻）相关，可选：未在 train.py 暴露成全局读取，仅作注释参考 =====
# # KL_HALT   = 1.0  # 当报告的平均 KL 连续 >= HALT_STREAK 次超过该阈值 → 冻结 actor（只训 critic）。
# #                   # 设得越大越“宽松”（更不容易冻结），设得越小越“保守”（更容易冻结）。
# # KL_RESUME = 0.6  # 当报告的平均 KL 连续 >= RESUME_STREAK 次低于该阈值 → 解冻 actor（恢复训 actor）。
# #                   # 设得越大越“激进”（更容易解冻），设得越小越“保守”（更不容易解冻）。
# # HALT_STREAK   = 2  # 触发冻结所需的连续“超阈值”次数，用来抑制抖动。
# # RESUME_STREAK = 2  # 触发解冻所需的连续“低于阈值”次数，同样抑制抖动。
# # KL_STOP = 1.0      # 临时“按 KL 降学习率”的枢轴点：当 KL>KL_STOP，actor 的 LR 会按误差缩放降低。
# #                    # 设得越大，LR 降得越“晚”；越小，LR 更早开始收缩。