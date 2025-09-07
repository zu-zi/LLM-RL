# —— 核心：离线 rollouts 池与 sglang 参数 ——
SGLANG_ON = True
SGLANG_OFFLINE = True
SGLANG_MODEL_PATH = "gpt2-large"                 # 可换成你的推理模型
SGLANG_SYNC_DIR = "/root/autodl-tmp/sgl_pool"    # 样本池（JSONL，放大盘）
SGLANG_ROLLOUT_TARGET = 1024                     # 目标池容量
SGLANG_REFILL_BATCH = 256                        # 每次补货生成条数
SGLANG_MAX_NEW = 192                             # 每条最大新 token 数（response）

# —— 训练参数（32GB 单卡建议） ——
init_from = "gpt2-large"                         # actor/ref 初始化
block_size = 512
batch_size = 4
gradient_accumulation_steps = 1
RL_learning_rate = 5e-6
kl_ctl = 0.05                                    # PPO KL 系数

use_ppo = True
use_grpo = False
use_dapo  = False
use_token_entropy = False

out_dir = "/root/autodl-tmp/Results"             # 结果目录（放大盘）
eval_interval = 1                                # 方便观察每步日志
max_iters = 200
always_save_checkpoint = False
wandb_log = False
