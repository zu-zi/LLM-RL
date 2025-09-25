# train_PPO.py
import os, time, json, random
from typing import List
import torch
from torch.amp import autocast

import tiktoken  # GPT-2 tokenizer（与 nanoGPT 模型一致）
from transformers import AutoTokenizer  # 仅用于 RM（CPU）

from prepare import DataConfig, load_or_build_data, init_wandb_offline, get_device
from model import GPT, GPTConfig
from PPO import PPOConfig, PPOTrainer, RewardModelCPU

# =============== 超参数（全部在此集中可见） ===============
OUT_DIR               = "./runs"
SEED_BASE             = 1337
PROJECT               = "ppo_gpt2_large_rlhf"
RUN_NAME              = "gpt2l_32g_vgpu"

# 模型与序列
MODEL_TYPE            = "gpt2-large"
BLOCK_SIZE            = 384    # RL 阶段裁到 384~512 左右即可
DROPOUT               = 0.0

# 训练节奏
MAX_UPDATES           = 1000
EVAL_INTERVAL_UPDATES = 8      # 每 ~8 次更新评测一次（固定 32 条）
PRINT_INTERVAL        = 1

# batch / 优化
BATCH_SIZE            = 32     # 每次更新的样本数（可视显存调整 32~64）
PPO_EPOCHS            = 2
MINIBATCH_SIZE        = 16
LR                    = 2e-6

# 生成策略
MIN_NEW_TOKENS        = 16
MAX_NEW_TOKENS        = 128
TEMPERATURE           = 1.0
TOP_P                 = 0.9

# 其他
USE_BNB8BIT           = True
WANDB_OFFLINE         = True

# ================== 固定随机种子 ==================
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ================== 评测 ==================
@torch.no_grad()
def evaluate(trainer: PPOTrainer, eval_prompts: List[str], block_size: int) -> dict:
    BATCH = 8
    scores = []
    for i in range(0, len(eval_prompts), BATCH):
        batch = eval_prompts[i:i+BATCH]
        batch_data, P, R = trainer.build_rollout(batch, block_size)  # 内部会生成&打分
        # 仅取 RM 打分（rewards 的最后 token 处值 + KL 惩罚，这里我们改为直接 RM 原始分数便于观测）
        rm_only = trainer.rm.score(P, R)
        scores.extend(rm_only)
    return {
        "eval_rm_mean": float(sum(scores)/len(scores)),
        "eval_rm_median": float(sorted(scores)[len(scores)//2]),
        "eval_num": len(scores),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED_BASE)
    device = get_device()
    print(f"[device] {device}")

    # ========== 数据集 ==========
    data_cfg = DataConfig(data_dir="./data", train_size=20000, eval_size=32, seed=SEED_BASE)
    pool, eval_prompts = load_or_build_data(data_cfg)
    print(f"[data] train_pool={len(pool.prompts)} eval={len(eval_prompts)}")

    # ========== tokenizer ==========
    # 策略/参考：tiktoken GPT-2（与 nanoGPT 一致）
    tok_gpt2 = tiktoken.get_encoding("gpt2")
    # RM：在 RewardModelCPU 内部初始化

    # ========== 模型 ==========
    policy = GPT.from_pretrained(MODEL_TYPE, override_args={"dropout": DROPOUT, "block_size": BLOCK_SIZE})
    ref    = GPT.from_pretrained(MODEL_TYPE, override_args={"dropout": DROPOUT, "block_size": BLOCK_SIZE})
    policy.crop_block_size(BLOCK_SIZE); ref.crop_block_size(BLOCK_SIZE)
    policy.to(device); ref.to(device).eval()
    for p in ref.parameters(): p.requires_grad = False

    # ========== RM（CPU） ==========
    rm = RewardModelCPU("OpenAssistant/reward-model-deberta-v3-large-v2")

    # ========== PPO ==========
    ppo_cfg = PPOConfig(
        gamma=1.0, lam=0.95, cliprange=0.2, vf_coef=0.5, ent_coef=0.005,
        max_grad_norm=1.0,
        min_new_tokens=MIN_NEW_TOKENS, max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE, top_p=TOP_P,
        lr=LR, betas=(0.9, 0.999), weight_decay=0.1, use_bnb8bit=USE_BNB8BIT,
        batch_size=BATCH_SIZE, minibatch_size=MINIBATCH_SIZE, ppo_epochs=PPO_EPOCHS,
        amp_dtype=torch.float16 if device=="cuda" else None
    )
    trainer = PPOTrainer(policy, ref, tok_gpt2, rm, ppo_cfg, device)

    # ========== W&B（离线） ==========
    if WANDB_OFFLINE:
        run = init_wandb_offline(project=PROJECT, run_name=RUN_NAME, config={
            "model": MODEL_TYPE, "block_size": BLOCK_SIZE,
            "batch_size": BATCH_SIZE, "ppo_epochs": PPO_EPOCHS, "minibatch": MINIBATCH_SIZE,
            "lr": LR, "new_tokens":[MIN_NEW_TOKENS, MAX_NEW_TOKENS]
        })
    else:
        run = None

    # ========== 训练主循环 ==========
    step = 0
    t0 = time.time()
    while step < MAX_UPDATES:
        # === 采样一个更新批 ===
        prompts = pool.sample_batch(BATCH_SIZE)
        batch, P_txt, R_txt = trainer.build_rollout(prompts, BLOCK_SIZE)

        # === PPO 更新 ===
        stats = trainer.ppo_update(batch)

        # === 打印/记录 ===
        if step % PRINT_INTERVAL == 0:
            # 关键指标简洁打印
            # 近似 KL：用 (logp - ref_logp) 在 response 上的负均值
            resp = batch.response_mask
            with torch.no_grad():
                kl_est = float((-(batch.logprobs - batch.ref_logprobs)[resp].mean()).item()) if resp.any() else 0.0
                rew_mean = float(batch.rewards[resp].mean().item()) if resp.any() else 0.0
                val_mean = float(batch.values[resp].mean().item()) if resp.any() else 0.0
            msg = (f"[{step:04d}] "
                   f"loss_pi={stats['loss_pi']:.4f} loss_v={stats['loss_v']:.4f} "
                   f"entropy={stats['entropy']:.3f} clip_frac={stats['clip_frac']:.3f} "
                   f"KL~{kl_est:.4f} reward_mean~{rew_mean:.4f} V_mean~{val_mean:.4f} "
                   f"speed={(step+1)/(time.time()-t0):.2f} it/s")
            print(msg)

            if run is not None:
                run.log({
                    "loss_pi": stats["loss_pi"],
                    "loss_v": stats["loss_v"],
                    "entropy": stats["entropy"],
                    "clip_frac": stats["clip_frac"],
                    "approx_kl": kl_est,
                    "reward_mean": rew_mean,
                    "value_mean": val_mean,
                    "updates": step
                })

        # === 周期性评测（固定 32 条） ===
        if (step+1) % EVAL_INTERVAL_UPDATES == 0:
            eval_res = evaluate(trainer, eval_prompts, BLOCK_SIZE)
            print(f"[eval@{step+1}] rm_mean={eval_res['eval_rm_mean']:.4f} "
                  f"rm_median={eval_res['eval_rm_median']:.4f} n={eval_res['eval_num']}")
            if run is not None:
                run.log(eval_res | {"updates": step+1})

        step += 1

    print("done.")

if __name__ == "__main__":
    main()
