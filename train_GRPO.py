# train_GRPO.py
import os, sys, time, json, glob, shutil, random
from datetime import datetime
import numpy as np
import torch, tiktoken
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from utils.rollout_pool import dequeue_groups, estimate_size, ensure_dir

from RL.GRPO import GRPOTrainer
from RL.PPO import Samples, normalize_for_reward
from RL.common.tokenizers import GPT2Tok
from RL.common.sampling import decode_with_sampling
from RL.common.device_utils import cuda_free_mb
from RL.common.export_sglang import export_actor_for_sglang
from RL.common.train_utils import set_seed, greedy_eval_reward

# W&B（离线）
WANDB_ON = True
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
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

# GRPO
GROUP_SIZE      = 4            # 每 prompt 候选数
NUM_GROUPS      = 4            # 迭代时组数（需求=GROUP_SIZE*NUM_GROUPS）
LR_ACTOR        = 4e-6
WD_ACTOR        = 3e-3
BETA1, BETA2    = 0.9, 0.95
MAX_GRAD_NORM   = 0.5
TAU             = 0.2         # soft target 温度
KL_CTL_INIT     = 0.1
LENGTH_NORM     = True

# 生成
TEMP            = 0.8
TOP_P           = 0.9
TOP_K           = 0
REP_PENALTY     = 1.0
STOP_STRS       = ["\nHuman:", "\n\nHuman:"]
MIN_RESP_TOK    = 24
MAX_NEW_TOK     = 96

# sglang 池
SGLANG_ON       = True
SGLANG_MODEL_SYMLINK = "/root/autodl-tmp/actor_exports/current"
SGLANG_EXPORT_BASE   = "/root/autodl-tmp/actor_exports"
SGLANG_SYNC_DIR      = "/root/autodl-tmp/sgl_pool"
SGLANG_REFILL_CHUNK  = 64          # 72
ROLL_MIN_FREE_MB     = 3000        # 6000
ROLL_COOLDOWN_SEC    = 7           # 7

# RM
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large-v2"

# data
PROMPT_BIN = os.path.join(os.path.dirname(__file__), "data/RL_dataset/prompt.bin")

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
        "--mb", "3",
        "--use-only-train",
        "--min-resp", str(int(MIN_RESP_TOK)),
        "--refresh-every-batches","30",
        "--reload-strategy", "realpath",
        "--min-free-mb", str(int(ROLL_MIN_FREE_MB)),
    ]
    with open(logf, "a") as f:
        import subprocess
        ret = subprocess.call(cmd, stdout=f, stderr=f)
    if ret != 0:
        print(f"[rollout] worker exit code {ret}", flush=True)
        
def _rebucket_and_topup(groups, G, dev, raw_actor, tok):
    flat = []
    for g in groups:
        flat.extend(g)

    from collections import defaultdict
    buckets = defaultdict(list)
    for item in flat:
        k = tuple(item["prompt_ids"])  
        buckets[k].append(item)

    regrouped = []
    singles = []
    for k, lst in buckets.items():
        if len(lst) >= 2:
            for i in range(0, len(lst), G):
                chunk = lst[i:i+G]
                if len(chunk) >= 2:
                    regrouped.append(chunk)
        else:
            singles.append(k)

    for k in singles:
        base = list(k)
        need = G - 1
        new_group = [buckets[k][0]]
        for _ in range(need * 3):
            x = torch.tensor(base, dtype=torch.long, device=dev).unsqueeze(0)
            room = BLOCK_SIZE - x.size(1) - 1
            if room <= 0:
                break
            out = decode_with_sampling(
                raw_actor, x, max_new=min(room, MAX_NEW_TOK), eos_id=tok.eos_id, block_size=BLOCK_SIZE,
                temperature=TEMP, top_p=TOP_P, top_k=TOP_K, rep_penalty=REP_PENALTY,
                stop_strs=STOP_STRS, tokenizer_decode=tok.decode, min_resp=MIN_RESP_TOK
            )
            if (out.size(1) - x.size(1)) < MIN_RESP_TOK:
                continue
            full_ids = out[0].tolist()
            key = ",".join(map(str, full_ids[len(base):]))  
            if any(",".join(map(str, it["full_ids"][len(base):])) == key for it in new_group):
                continue
            new_group.append({"prompt_ids": base, "full_ids": full_ids})
            if len(new_group) >= 2:  
                regrouped.append(new_group[:G])
                break

    return regrouped[:NUM_GROUPS], {"buckets": len(buckets), "rebuilt_groups": len(regrouped)}
            
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    ensure_dir(SGLANG_SYNC_DIR)

    run = None
    if WANDB_ON and _HAS_WANDB:
        try:
            run_name = f"grpo_gpt2l_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = wandb.init(
                project="hlhf-grpo",
                name=run_name,
                dir=OUT_DIR,
                mode="offline",
                config={
                    "seed": SEED, "device": DEVICE, "block_size": BLOCK_SIZE,
                    "max_iters": MAX_ITERS, "eval_interval": EVAL_INTERVAL,
                    "init_from": INIT_FROM, "dropout": DROPOUT, "bias": BIAS,
                    "group_size": GROUP_SIZE, "num_groups": NUM_GROUPS,
                    "lr_actor": LR_ACTOR, "wd_actor": WD_ACTOR,
                    "beta1": BETA1, "beta2": BETA2,
                    "max_grad_norm": MAX_GRAD_NORM, "tau": TAU,
                    "kl_ctl_init": KL_CTL_INIT, "length_norm": LENGTH_NORM,
                    "gen_temp": TEMP, "gen_top_p": TOP_P, "gen_top_k": TOP_K,
                    "rep_penalty": REP_PENALTY, "min_resp_tok": MIN_RESP_TOK, "max_new_tok": MAX_NEW_TOK,
                    "sglang_on": SGLANG_ON, "sglang_refill_chunk": SGLANG_REFILL_CHUNK,
                    "roll_min_free_mb": ROLL_MIN_FREE_MB, "roll_cooldown_sec": ROLL_COOLDOWN_SEC,
                    "reward_model": REWARD_MODEL_NAME,
                },
            )
            print(f"[wandb] offline run started -> {run_name} (dir={OUT_DIR})", flush=True)
        except Exception as e:
            print(f"[wandb][warn] init failed: {e}; continue without wandb.", flush=True)
            run = None

    # 数据
    print(f"[data] loading prompts from {PROMPT_BIN}")
    blob = torch.load(PROMPT_BIN, map_location="cpu")
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

    # 模型
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

    # 初始导出给 sglang
    if SGLANG_ON:
        try:
            os.makedirs(os.path.dirname(SGLANG_MODEL_SYMLINK), exist_ok=True)
            export_actor_for_sglang(raw_actor, INIT_FROM, SGLANG_EXPORT_BASE, SGLANG_MODEL_SYMLINK)
        except Exception as e:
            print(f"[export][warn] initial export failed: {e}", flush=True)

    # 优化器
    try:
        from bitsandbytes.optim import AdamW8bit
        opt_a = AdamW8bit(raw_actor.parameters(), lr=LR_ACTOR, betas=(BETA1,BETA2), weight_decay=WD_ACTOR)
        print("[optim] AdamW8bit")
    except Exception:
        opt_a = torch.optim.AdamW(raw_actor.parameters(), lr=LR_ACTOR, betas=(BETA1,BETA2), weight_decay=WD_ACTOR)
        print("[optim] torch.optim.AdamW")

    # RM（CPU）
    print(f"[reward] loading {REWARD_MODEL_NAME} on CPU...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_NAME, device_map="cpu", torch_dtype=torch.float32).eval()
    reward_tok = AutoTokenizer.from_pretrained(REWARD_MODEL_NAME, use_fast=True)
    if getattr(reward_tok, "pad_token", None) is None and getattr(reward_tok, "eos_token", None) is not None:
        reward_tok.pad_token = reward_tok.eos_token
    try: reward_tok.padding_side = "right"
    except Exception: pass

    trainer = GRPOTrainer(
        actor_model=raw_actor, ref_model=ref, reward_model=reward_model,
        actor_tokenizer=tok, reward_tokenizer=reward_tok,
        optimizer_actor=opt_a, device=dev, mb_size_logits=1,
        tau=TAU, kl_ctl=KL_CTL_INIT, length_norm=LENGTH_NORM, max_grad_norm=MAX_GRAD_NORM,
    )

    # baseline
    if len(EVAL_PROMPT_IDS) > 0:
        r0 = greedy_eval_reward(raw_actor, tok, EVAL_PROMPT_IDS, reward_tok, reward_model, BLOCK_SIZE, max_new_eval=min(64, MAX_NEW_TOK))
        print(f"[baseline] iter0_reward_greedy={r0:.4f}", flush=True)
        if run is not None:
            wandb.log({"baseline/iter0_reward_greedy": float(r0)}, step=0)

    METRICS_CSV = os.path.join(OUT_DIR, "metrics_grpo.csv")
    if not os.path.exists(METRICS_CSV):
        with open(METRICS_CSV, "w") as f:
            f.write("iter,loss,kl_mean,rm_mean,pstar_entropy,r_eval_greedy,items,lr,pool\n")

    last_export_it = 0
    last_roll_t = 0.0

    # 在线生成
    import hashlib
    def _gen_group(G):
        base = random.choice(TRAIN_PROMPT_IDS)
        g, seen = [], set()
        for _ in range(G * 3):
            x = torch.tensor(base, dtype=torch.long, device=dev).unsqueeze(0)
            room = BLOCK_SIZE - x.size(1) - 1
            if room <= 0: break
            out = decode_with_sampling(
                raw_actor, x, max_new=min(room, MAX_NEW_TOK), eos_id=tok.eos_id, block_size=BLOCK_SIZE,
                temperature=TEMP, top_p=TOP_P, top_k=TOP_K, rep_penalty=REP_PENALTY,
                stop_strs=STOP_STRS, tokenizer_decode=tok.decode, min_resp=MIN_RESP_TOK
            )
            if (out.size(1) - x.size(1)) < MIN_RESP_TOK: continue
            full_ids = out[0].tolist()
            resp_ids = full_ids[len(base):]
            key = hashlib.md5(",".join(map(str, resp_ids)).encode("utf-8")).hexdigest()
            if key in seen: continue
            seen.add(key)
            g.append({"prompt_ids": base, "full_ids": full_ids})
            if len(g) >= G: break
        return g if len(g) >= 2 else []

    # 主循环
    demand = GROUP_SIZE * NUM_GROUPS
    LOW_WATER = max(demand * 3, SGLANG_REFILL_CHUNK)  # 低水位（按需求 *3）
    BURST_MULT = 4                                    # 当池空时，突发并发倍数

    for it in range(1, MAX_ITERS+1):
        pool_est = estimate_size(SGLANG_SYNC_DIR, SGLANG_REFILL_CHUNK) if SGLANG_ON else -1
        if SGLANG_ON:
            need_burst = (pool_est <= 0)
            should_refill = need_burst or (pool_est < LOW_WATER)
            if should_refill and cuda_free_mb(DEVICE) >= ROLL_MIN_FREE_MB:
                n_jobs = (BURST_MULT if need_burst else 1)
                now = time.time()
                if need_burst or (now - last_roll_t) >= ROLL_COOLDOWN_SEC:
                    for _ in range(n_jobs):
                        spawn_rollout(PROMPT_BIN, SGLANG_REFILL_CHUNK, SGLANG_SYNC_DIR, MAX_NEW_TOK)
                    last_roll_t = now

        groups = dequeue_groups(SGLANG_SYNC_DIR, GROUP_SIZE, NUM_GROUPS, allow_partial=True) if SGLANG_ON else []
        while len(groups) < NUM_GROUPS:
            g = _gen_group(GROUP_SIZE)
            if not g: break
            groups.append(g)

        groups, gstats = _rebucket_and_topup(groups, GROUP_SIZE, dev, raw_actor, tok)
        print(f"[iter {it:04d}] regrouped_groups={gstats['rebuilt_groups']} buckets={gstats['buckets']} pool={pool_est}", flush=True)

        total_items = sum(len(g) for g in groups)
        if total_items < 2:
            print(f"[iter {it:04d}] skip(empty) pool={pool_est}", flush=True)
            if run is not None:
                wandb.log({"iter/skip_empty": 1, "pool/size_est": pool_est}, step=it)
            continue

        stats = trainer.step_on_groups(groups, BLOCK_SIZE)

        # 评测：固定间隔 & 固定集合
        r_eval_greedy = float("nan")
        if (it % EVAL_INTERVAL == 0) and len(EVAL_PROMPT_IDS) > 0:
            r_eval_greedy = greedy_eval_reward(raw_actor, tok, EVAL_PROMPT_IDS, reward_tok, reward_model, BLOCK_SIZE, max_new_eval=min(64, MAX_NEW_TOK))

        cur_lr = [pg['lr'] for pg in opt_a.param_groups][0]
        loss = stats.get("loss", float("nan"))
        klm  = stats.get("kl_mean", float("nan"))
        rmm  = stats.get("rm_mean", float("nan"))
        pent = stats.get("pstar_entropy", float("nan"))
        items= int(stats.get("items", total_items))

        print(f"[iter {it:04d}] loss={loss:.4f} kl={klm:.6f} rm={rmm:.4f} p*H={pent:.3f} r_eval_greedy={r_eval_greedy:.4f} items={items} lr={cur_lr:.2e} pool={pool_est}", flush=True)
        with open(METRICS_CSV, "a") as f:
            f.write(f"{it},{loss},{klm},{rmm},{pent},{r_eval_greedy},{items},{cur_lr},{pool_est}\n")

        if run is not None:
            wandb.log({
                "loss/total": loss,
                "kl/avg": klm,
                "reward/rm_mean": rmm,
                "pstar/entropy": pent,
                "reward/eval_greedy": r_eval_greedy,
                "optim/lr_actor": cur_lr,
                "pool/size_est": pool_est,
                "iter": it,
            }, step=it)

        # 定期导出给 sglang
        if SGLANG_ON and (it - last_export_it) >= 40:  #40
            try:
                export_actor_for_sglang(raw_actor, INIT_FROM, SGLANG_EXPORT_BASE, SGLANG_MODEL_SYMLINK)
                last_export_it = it
            except Exception as e:
                print(f"[export][warn] export failed: {e}", flush=True)

        if it % EVAL_INTERVAL == 0:
            ckpt = {
                "iter": it,
                "actor": raw_actor.state_dict(),
                "ref": ref.state_dict(),
                "opt_actor": opt_a.state_dict(),
            }
            torch.save(ckpt, os.path.join(OUT_DIR, "GRPO_ckpt.pt"))

        if stats.get("skip"):
            print(f"[iter {it:04d}] skip(step) pool={pool_est} r_eval_greedy={r_eval_greedy:.4f}", flush=True)
            if run is not None and not math.isnan(r_eval_greedy):
                wandb.log({"reward/eval_greedy": r_eval_greedy, "iter": it, "pool/size_est": pool_est}, step=it)
            continue

    if run is not None:
        try: wandb.finish()
        except Exception: pass

if __name__ == "__main__":
    main()