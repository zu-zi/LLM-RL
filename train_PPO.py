# train_PPO.py
# 不同于原nanoGPT：不要外部 config 覆盖；减少超参
import os, sys, time, json, glob, shutil, random
from datetime import datetime
import numpy as np
import torch, tiktoken
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from utils.rollout_pool import dequeue_items, estimate_size, ensure_dir

from RL.PPO import PPOTrainer, Critic, Samples, normalize_for_reward
from RL.common.tokenizers import GPT2Tok
from RL.common.sampling import decode_with_sampling, pack_samples
from RL.common.device_utils import cuda_free_mb
from RL.common.export_sglang import export_actor_for_sglang
from RL.common.train_utils import set_seed, greedy_eval_reward

# W&B（离线）
WANDB_ON = True  # 需要时可改为 False 完全关闭
os.environ.setdefault("WANDB_MODE", "offline")  # 强制离线
os.environ.setdefault("WANDB_SILENT", "true")   # 安静模式（仍然保留print）
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


# 路径
OUT_DIR         = "/root/autodl-tmp/Results"
DEVICE          = "cuda"
BLOCK_SIZE      = 256
SEED            = 1337
EVAL_INTERVAL   = 10
MAX_ITERS       = 1000

# 模型
INIT_FROM       = "gpt2-large"
DROPOUT         = 0.0
BIAS            = False

# 
BATCH_SIZE      = 8
POLICY_EPOCHS = 1
LR_ACTOR        = 7e-7
LR_CRITIC       = 1.4e-6
WD_ACTOR        = 3e-3
WD_CRITIC       = 1e-3
BETA1, BETA2    = 0.9, 0.95
MAX_GRAD_NORM   = 0.3
VF_CLIP         = 0.1
PPO_CLIP        = 0.07  # 0.07
ENTROPY_COEF    = 0.0
KL_CTL_INIT     = 0.6  # 0.6

# 
TEMP            = 0.55
TOP_P           = 0.9
TOP_K           = 0
REP_PENALTY     = 1
STOP_STRS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK    = 24
MAX_NEW_TOK     = 96

# sglang
SGLANG_ON       = True
SGLANG_MODEL_SYMLINK = "/root/autodl-tmp/actor_exports/current"
SGLANG_EXPORT_BASE   = "/root/autodl-tmp/actor_exports"
SGLANG_SYNC_DIR      = "/root/autodl-tmp/sgl_pool"
SGLANG_REFILL_CHUNK  = 64          # 48
ROLL_LOW_WATERMARK   = BATCH_SIZE*3
ROLL_MIN_FREE_MB     = 3000 # 6000
ROLL_COOLDOWN_SEC    = 7  # 12

# RM
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# 固定 prompts
PROMPT_BIN = os.path.join(os.path.dirname(__file__), "data/RL_dataset/prompt.bin")

# utils
def spawn_rollout(prompt_bin_path, count, sync_dir, max_new):
    ensure_dir(sync_dir)
    os.makedirs(os.path.join(OUT_DIR, "rollout_logs"), exist_ok=True)
    logf = os.path.join(OUT_DIR, "rollout_logs", f"rollout_{int(time.time())}.log")
    cmd = [
        "python", "-u", "rollout_worker.py",
        "--model", SGLANG_MODEL_SYMLINK,
        "--prompt-bin", prompt_bin_path,
        "--out-dir", sync_dir,
        "--count", str(int(count)),
        "--max-new", str(int(max_new)),
        "--block-size", str(int(BLOCK_SIZE)),
        "--mb", "6",
        "--use-only-train",
        "--min-resp",str(int(MIN_RESP_TOK)),
        "--refresh-every-batches","30",
        "--reload-strategy", "realpath",
        "--min-free-mb", str(int(ROLL_MIN_FREE_MB)),
    ]
    with open(logf, "a") as f:
        import subprocess
        ret = subprocess.call(cmd, stdout=f, stderr=f)
    if ret != 0:
        print(f"[rollout] worker exit code {ret}", flush=True)


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_dir(SGLANG_SYNC_DIR)

    # W&B 初始化
    run = None
    if WANDB_ON and _HAS_WANDB:
        try:
            run_name = f"ppo_gpt2l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = wandb.init(
                project="hlhf-ppo",
                name=run_name,
                dir=OUT_DIR,
                mode="offline",
                config={
                    "seed": SEED,
                    "device": DEVICE,
                    "block_size": BLOCK_SIZE,
                    "max_iters": MAX_ITERS,
                    "eval_interval": EVAL_INTERVAL,
                    "init_from": INIT_FROM,
                    "dropout": DROPOUT,
                    "bias": BIAS,
                    "batch_size": BATCH_SIZE,
                    "policy_epochs": POLICY_EPOCHS,
                    "lr_actor": LR_ACTOR,
                    "lr_critic": LR_CRITIC,
                    "wd_actor": WD_ACTOR,
                    "wd_critic": WD_CRITIC,
                    "beta1": BETA1,
                    "beta2": BETA2,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "vf_clip": VF_CLIP,
                    "ppo_clip": PPO_CLIP,
                    "entropy_coef": ENTROPY_COEF,
                    "kl_ctl_init": KL_CTL_INIT,
                    "gen_temp": TEMP,
                    "gen_top_p": TOP_P,
                    "gen_top_k": TOP_K,
                    "rep_penalty": REP_PENALTY,
                    "min_resp_tok": MIN_RESP_TOK,
                    "max_new_tok": MAX_NEW_TOK,
                    "sglang_on": SGLANG_ON,
                    "sglang_refill_chunk": SGLANG_REFILL_CHUNK,
                    "roll_low_watermark": ROLL_LOW_WATERMARK,
                    "roll_min_free_mb": ROLL_MIN_FREE_MB,
                    "roll_cooldown_sec": ROLL_COOLDOWN_SEC,
                    "reward_model": REWARD_MODEL_NAME,
                },
            )
            print(f"[wandb] offline run started -> {run_name} (dir={OUT_DIR})", flush=True)
        except Exception as e:
            print(f"[wandb][warn] init failed: {e}; continue without wandb.", flush=True)
            run = None

    # 加载固定 prompts
    print(f"[data] loading prompts from {PROMPT_BIN}")
    blob = torch.load(PROMPT_BIN, map_location="cpu")
    PROMPTS_TEXT = blob["prompts"]
    PROMPT_TOKEN_IDS = blob["gpt2_token_ids"]
    EOS_ID = blob["eos_id"]
    TRAIN_INDICES = blob.get("train_indices", None)
    EVAL_INDICES  = blob.get("eval_indices", None)

    tok = GPT2Tok()
    pad_id = tok.eos_id

    ALL_IDX = list(range(len(PROMPT_TOKEN_IDS)))
    if isinstance(TRAIN_INDICES, list) and TRAIN_INDICES:
        TRAIN_IDX = sorted(i for i in TRAIN_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS))
    else:
        if isinstance(EVAL_INDICES, list) and EVAL_INDICES:
            EVAL_SET = set(i for i in EVAL_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS))
            TRAIN_IDX = [i for i in ALL_IDX if i not in EVAL_SET]
        else:
            TRAIN_IDX = ALL_IDX[:]
    TRAIN_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in TRAIN_IDX]

    if isinstance(EVAL_INDICES, list) and EVAL_INDICES:
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[i] for i in EVAL_INDICES if 0 <= i < len(PROMPT_TOKEN_IDS)]
    else:
        rng = np.random.RandomState(SEED)
        choice = rng.choice(len(PROMPT_TOKEN_IDS), size=min(16, len(PROMPT_TOKEN_IDS)), replace=False) if len(PROMPT_TOKEN_IDS)>0 else []
        EVAL_PROMPT_IDS = [PROMPT_TOKEN_IDS[int(i)] for i in choice]

    # 初始化 actor/ref/critic
    from model import GPT, GPTConfig
    print(f"[model] init from {INIT_FROM}")
    m = GPT.from_pretrained(INIT_FROM, dict(dropout=DROPOUT))
    model_args = dict(
        n_layer=m.config.n_layer, n_head=m.config.n_head, n_embd=m.config.n_embd,
        block_size=min(BLOCK_SIZE, m.config.block_size),
        bias=m.config.bias, dropout=DROPOUT, vocab_size=m.config.vocab_size
    )
    if BLOCK_SIZE < m.config.block_size:
        m.crop_block_size(BLOCK_SIZE)
    base_sd = m.state_dict(); del m

    dev = torch.device(DEVICE)
    ref = GPT(GPTConfig(**model_args)).to(dev); ref.load_state_dict(base_sd)
    for p in ref.parameters(): p.requires_grad=False
    ref.eval()
    ref = ref.to(dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16))

    actor = GPT(GPTConfig(**model_args)).to(dev); actor.load_state_dict(base_sd)
    raw_actor = actor 

    critic = Critic(raw_actor).to(dev)

    # 初始导出给 sglang
    if SGLANG_ON:
        try:
            os.makedirs(os.path.dirname(SGLANG_MODEL_SYMLINK), exist_ok=True)
            export_actor_for_sglang(raw_actor, INIT_FROM, SGLANG_EXPORT_BASE, SGLANG_MODEL_SYMLINK)
        except Exception as e:
            print(f"[export][warn] initial export failed: {e}", flush=True)

    # 优化器（优先 8bit）
    try:
        from bitsandbytes.optim import AdamW8bit
        opt_a = AdamW8bit(raw_actor.parameters(), lr=LR_ACTOR, betas=(BETA1,BETA2), weight_decay=WD_ACTOR)
        opt_c = AdamW8bit(critic.parameters(),    lr=LR_CRITIC, betas=(BETA1,BETA2), weight_decay=WD_CRITIC)
        print("[optim] AdamW8bit")
    except Exception:
        opt_a = torch.optim.AdamW(raw_actor.parameters(), lr=LR_ACTOR, betas=(BETA1,BETA2), weight_decay=WD_ACTOR)
        opt_c = torch.optim.AdamW(critic.parameters(),    lr=LR_CRITIC, betas=(BETA1,BETA2), weight_decay=WD_CRITIC)
        print("[optim] torch.optim.AdamW")

    # RM（CPU）
    print(f"[reward] loading {REWARD_MODEL_NAME} on CPU...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, device_map="cpu", torch_dtype=torch.float32).eval()
    reward_tok = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, use_fast=True)
    if getattr(reward_tok, "pad_token", None) is None and getattr(reward_tok, "eos_token", None) is not None:
        reward_tok.pad_token = reward_tok.eos_token
    try: reward_tok.padding_side = "right"
    except Exception: pass

    trainer = PPOTrainer(
        actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
        critic_model=critic, actor_tokenizer=tok, reward_tokenizer=reward_tok,
        optimizer_actor=opt_a, optimizer_critic=opt_c,
        device=dev, mb_size_logits=1, mb_size_values=1,
        kl_ctl=KL_CTL_INIT, ppo_clip=PPO_CLIP, vf_clip=VF_CLIP,
        entropy_coef=ENTROPY_COEF, max_grad_norm=MAX_GRAD_NORM,
    )

    # baseline：iter0 贪心奖励
    if len(EVAL_PROMPT_IDS) > 0:
        r0 = greedy_eval_reward(raw_actor, tok, EVAL_PROMPT_IDS, reward_tok, reward_model, BLOCK_SIZE, max_new_eval=min(64, MAX_NEW_TOK))
        print(f"[baseline] iter0_reward_greedy={r0:.4f}", flush=True)
        # W&B 记录 baseline
        if run is not None:
            wandb.log({"baseline/iter0_reward_greedy": float(r0)}, step=0)

    # 日志 CSV
    METRICS_CSV = os.path.join(OUT_DIR, "metrics.csv")
    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, "w") as f:
            f.write("iter,policy_loss,value_loss,avg_kl,reward_raw,reward_eval_greedy,resp_p50,lr\n")

    last_export_it = 0
    last_roll_t = 0.0

    for it in range(1, MAX_ITERS+1):
        # 补货：只有在池量不足时触发
        pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_CHUNK) if SGLANG_ON else -1
        if SGLANG_ON and pool_est < ROLL_LOW_WATERMARK:
            now = time.time()
            if (now - last_roll_t) >= ROLL_COOLDOWN_SEC and cuda_free_mb(DEVICE) >= ROLL_MIN_FREE_MB:
                spawn_rollout(PROMPT_BIN, SGLANG_REFILL_CHUNK, SGLANG_SYNC_DIR, MAX_NEW_TOK)
                last_roll_t = now

        # 取样：先从池拿，不够就在线补齐
        batch = dequeue_items(SGLANG_SYNC_DIR, BATCH_SIZE) if SGLANG_ON else []
        dev_t = torch.device(DEVICE)

        @torch.no_grad()
        def _gen_one():
            ids = random.choice(TRAIN_PROMPT_IDS)
            x = torch.tensor(ids, dtype=torch.long, device=dev_t).unsqueeze(0)
            room = BLOCK_SIZE - x.size(1) - 1
            if room <= 0: return None
            out = decode_with_sampling(
                raw_actor, x, max_new=min(room, MAX_NEW_TOK), eos_id=tok.eos_id, block_size=BLOCK_SIZE,
                temperature=TEMP, top_p=TOP_P, top_k=TOP_K, rep_penalty=REP_PENALTY,
                stop_strs=STOP_STRS, tokenizer_decode=tok.decode, min_resp=MIN_RESP_TOK
            )
            if (out.size(1) - x.size(1)) < MIN_RESP_TOK: return None
            return {"prompt_ids": ids, "full_ids": out[0].tolist()}

        while len(batch) < BATCH_SIZE:
            g = _gen_one()
            if g is None: break
            batch.append(g)

        samples = pack_samples(batch, pad_id=tok.eos_id, block_size=BLOCK_SIZE, device=dev_t)
        if int(samples.action_mask.sum().item()) == 0:
            print(f"[iter {it:04d}] skip(empty batch) pool={pool_est}", flush=True)
            if run is not None:
                wandb.log({"iter/skip_empty": 1, "pool/size_est": pool_est}, step=it)
            continue

        # 评估经验
        experiences, report_kl, r_raw, _, _, _ = trainer.evaluate_experience(samples)

        # 训练
        pl, vl = [], []
        for _ in range(int(POLICY_EPOCHS)):
            for exp in experiences:
                p, v = trainer.train_on_experience(exp, use_token_entropy=False)
                pl.append(float(p)); vl.append(float(v))

        mean_p = float(np.mean(pl)) if pl else float("nan")
        mean_v = float(np.mean(vl)) if vl else float("nan")

        # 固定评测：贪心
        r_eval_greedy = float("nan")
        if (it % EVAL_INTERVAL == 0) and len(EVAL_PROMPT_IDS) > 0:
            r_eval_greedy = greedy_eval_reward(raw_actor, tok, EVAL_PROMPT_IDS, reward_tok, reward_model, BLOCK_SIZE, max_new_eval=min(64, MAX_NEW_TOK))

        # 日志
        resp_lengths = samples.response_length.detach().cpu().numpy().tolist()
        p50 = float(np.percentile(resp_lengths, 50)) if resp_lengths else 0.0
        cur_lr = opt_a.param_groups[0]['lr']
        print(f"[iter {it:04d}] p={mean_p:.4f} v={mean_v:.4f} kl={report_kl:.6f} r_raw={r_raw:.4f} r_eval_greedy={r_eval_greedy:.4f} resp_p50={p50:.1f} lr={cur_lr:.2e} pool={pool_est}", flush=True)
        with open(METRICS_CSV, "a") as f:
            f.write(f"{it},{mean_p},{mean_v},{report_kl},{r_raw},{r_eval_greedy},{p50},{cur_lr}\n")

        if run is not None:
            wandb.log({
                "loss/policy": mean_p,
                "loss/value": mean_v,
                "kl/avg": report_kl,
                "reward/raw": r_raw,
                "reward/eval_greedy": r_eval_greedy,
                "resp/p50": p50,
                "optim/lr_actor": cur_lr,
                "pool/size_est": pool_est,
                "iter": it,
            }, step=it)

        # 定期导出给 sglang
        if SGLANG_ON and (it - last_export_it) >= 40:
            try:
                export_actor_for_sglang(raw_actor, INIT_FROM, SGLANG_EXPORT_BASE, SGLANG_MODEL_SYMLINK)
                last_export_it = it
            except Exception as e:
                print(f"[export][warn] export failed: {e}", flush=True)

        if it % EVAL_INTERVAL == 0:
            ckpt = {
                "iter": it,
                "actor": raw_actor.state_dict(),
                "critic": critic.state_dict(),
                "ref": ref.state_dict(),
                "opt_actor": opt_a.state_dict(),
                "opt_critic": opt_c.state_dict(),
            }
            torch.save(ckpt, os.path.join(OUT_DIR, "PPO_ckpt.pt"))

    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
