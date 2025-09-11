# RL/GRPO.py
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    Samples,
    Experience,
    model_all_logits,
    token_logprobs_from_logits,
    compute_actor_ref_logprobs,
    compute_approx_kl,
    kl_k3_from_logratio,
    masked_mean,
    entropy_from_logits,
    get_advantages_and_returns,  # 仅为接口一致，这里不用 critic
    scatter_last_token_rewards,
    ratio_stats,
    adv_abs_mean,
    normalize_for_reward,
)

class GRPOTrainer:
    """
    Group Relative Policy Optimization（无 critic）：
    - 将一个 batch 按 group_size 划分为多个组，组内奖励做去均值/标准化，作为样本权重。
    - 目标：最大化 Σ_i (w_i * logπ(a_i | s_i))，可叠加 KL 正则项。
    - 不训练 value/critic，evaluate_experience 只产出 policy 所需字段。
    """
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        actor_tokenizer,           # 需提供 .decode(ids) & eos token 兼容
        reward_tokenizer,          # HF tokenizer
        optimizer_actor: torch.optim.Optimizer,
        group_size: int = 4,
        kl_coef: float = 0.0,
        clip_reward: float = 5.0,
        device: str = "cuda",
        mb_size_logits: int = 0,   # 计算 logits 的 micro-batch
    ):
        self.actor = actor_model
        self.ref   = ref_model
        self.reward_model = reward_model
        self.actor_tok    = actor_tokenizer
        self.reward_tok   = reward_tokenizer
        self.opt_actor    = optimizer_actor

        self.group_size   = max(int(group_size), 1)
        self.kl_coef_base = float(kl_coef)
        self.kl_ctl       = float(kl_coef)   # 训练循环里会被动态调整（与 PPO 接口统一）
        self.clip_reward  = float(clip_reward)
        self.device       = device

        self.mb_logits    = int(mb_size_logits) if mb_size_logits else 0
        self.device_type  = "cuda" if (torch.cuda.is_available() and "cuda" in device) else "cpu"

        # 统计字段（供外层日志）
        self.last_stats = {}

    # ---------------- Reward side ----------------

    @torch.no_grad()
    def _score_with_reward_model(self, seq_texts: List[str]) -> torch.Tensor:
        """
        给定完整的序列文本（含 prompt+response），用 RM 在 CPU 打分。
        返回 [B] float tensor。
        """
        # 归一化：减少 RM 无关差异
        texts = [normalize_for_reward(t, self.reward_tok) for t in seq_texts]
        toks = self.reward_tok(
            texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        )
        outs = self.reward_model(**toks)
        logits = getattr(outs, "logits", None)
        if logits is None:
            return torch.full((len(texts),), float("nan"))
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits.detach().float().cpu()

    @torch.no_grad()
    def _texts_from_samples(self, samples: Samples) -> List[str]:
        B = samples.seqs.size(0)
        texts = []
        for i in range(B):
            L = int(samples.attention_mask[i].sum().item())
            ids = samples.seqs[i, :L].detach().cpu().tolist()
            texts.append(self.actor_tok.decode(ids))
        return texts

    # ---------------- Experience builder ----------------

    @torch.no_grad()
    def evaluate_experience(self, samples: Samples) -> Tuple[List[Experience], float, float, float, float, float]:
        """
        返回：
          - experiences: List[Experience]（仅用于 policy）
          - report_kl: masked k3 KL 均值（日志）
          - r_raw_mean: 未中心化奖励均值（日志）
          - r_shaped_mean: 加 KL 惩罚后的奖励均值（日志）
          - r_ctr_mean: 中心化后（组内去均值/归一化）均值（应接近 0）
          - safe_kl: 与 report_kl 一致（接口兼容）
        """
        device = samples.seqs.device

        # 1) 计算 actor/ref 逐 token logprob（与 seq 对齐到 [B, T-1]）
        lp_actor, lp_ref, mask_tgt = compute_actor_ref_logprobs(
            self.actor, self.ref, samples.seqs, samples.action_mask,
            device_type=self.device_type, ptdtype=None, micro_batch_size=self.mb_logits
        )  # [B, T-1], [B, T-1], [B, T-1]
        log_ratio = lp_actor - lp_ref
        k3 = kl_k3_from_logratio(log_ratio)                    # [B, T-1]
        report_kl = float(masked_mean(k3, mask_tgt.float()).item())
        safe_kl   = report_kl

        # 2) 计算熵（可选，仅日志）
        #    需要 actor 的 full logits，一次性做；显存不够可以分块（已由 model_all_logits 控制）
        with torch.no_grad():
            logits_actor = model_all_logits(
                self.actor, samples.seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits
            )  # [B, T, V]
            ent = entropy_from_logits(logits_actor[:, 1:, :], mask_tgt)  # 与 mask_tgt 对齐到 [B, T-1]

        # 3) 句级奖励（RM）：对完整文本打分
        seq_texts = self._texts_from_samples(samples)
        rm_scores = self._score_with_reward_model(seq_texts)   # [B] on cpu
        B = rm_scores.numel()
        rm_scores = rm_scores.to(device=device)

        # 4) KL 惩罚到句级（均值 KL）
        denom = mask_tgt.sum(dim=1).clamp_min(1e-8).float()
        kl_mean_per_seq = (k3 * mask_tgt).sum(dim=1) / denom   # [B]
        r_shaped = rm_scores - self.kl_ctl * kl_mean_per_seq   # [B]

        r_raw_mean    = float(rm_scores.mean().item())
        r_shaped_mean = float(r_shaped.mean().item())

        # 5) 组内中心化权重
        #    将 batch 按 group_size 划分，不足一组的尾巴也组成一组
        gs = max(1, int(self.group_size))
        n_groups = int(math.ceil(B / gs))
        weights = torch.zeros_like(r_shaped, device=device)

        start = 0
        for _ in range(n_groups):
            end = min(start + gs, B)
            g = r_shaped[start:end]
            if g.numel() == 0:
                break
            g_mean = g.mean()
            g_std  = g.std(unbiased=False).clamp_min(1e-6)
            w = (g - g_mean) / g_std                     # 组内零均值、单位方差
            if self.clip_reward > 0:
                w = torch.clamp(w, -self.clip_reward, self.clip_reward)
            weights[start:end] = w
            start = end

        r_ctr_mean = float(weights.mean().item())  # 理论上 ~0

        # 6) 把“样本级权重”扩展到 token 轴，构造 Experience
        experiences: List[Experience] = []
        sel_tokens = int(mask_tgt.sum().item())

        # 统计用
        q50, q90, q99, rmax = ratio_stats(log_ratio, mask_tgt)
        adv_abs = adv_abs_mean(weights.view(-1, 1).expand_as(mask_tgt), mask_tgt)

        for i in range(B):
            # 常量权重 → token 轴（仅在 action 区域为 1）
            adv_t = torch.zeros_like(mask_tgt[i], dtype=lp_actor.dtype)
            adv_t[mask_tgt[i] > 0] = weights[i]

            # GRPO 不用 critic，这里 values/returns 填 0 来满足接口
            T = mask_tgt.size(1)
            zeros = torch.zeros((T,), dtype=lp_actor.dtype, device=device)

            exp = Experience(
                seqs=samples.seqs[i:i+1],
                action_log_probs=lp_actor[i:i+1, :],
                values=zeros.view(1, T),
                returns=zeros.view(1, T),
                advantages=adv_t.view(1, T),
                attention_mask=samples.attention_mask[i:i+1, 1:],
                action_mask=mask_tgt[i:i+1, :],
                reward=r_shaped[i:i+1].view(1, 1),
                num_actions=samples.num_actions[i:i+1],
                kl=k3[i:i+1, :],
            )
            experiences.append(exp)

        # 汇总日志统计
        self.last_stats = {
            "entropy": float(ent.item()) if torch.is_tensor(ent) else float("nan"),
            "clip_frac": float("nan"),              # PPO 专有，这里留空
            "approx_kl_pi": report_kl,              # 兼容字段名
            "v_mae": float("nan"),
            "explained_var": float("nan"),
            "ratio_q50_q90_q99": (q50, q90, q99),
            "ratio_max": rmax,
            "adv_abs_mean": adv_abs,
            "sel_tokens": sel_tokens,
            "ppo_clip": float("nan"),
            "kl_ctl_now": float(self.kl_ctl),
        }

        return experiences, report_kl, r_raw_mean, r_shaped_mean, r_ctr_mean, safe_kl

    # ---------------- Train step ----------------

    def train_on_experience(self, exp: Experience, use_token_entropy: bool = False) -> torch.Tensor:
        """
        单个 Experience 的 GRPO 更新：
        L = - mean( advantages * logπ(a|s) ) + β * mean(KL)
        其中 advantages 为组内标准化奖励（常量权重，已展开到 token 维）。
        """
        self.actor.train()

        # 只更新 actor
        self.opt_actor.zero_grad(set_to_none=True)

        # exp.action_log_probs 已是与 seq 对齐到 [T-1] 的逐 token logπ
        logp = exp.action_log_probs  # [1, T-1]
        mask = exp.action_mask.float()  # [1, T-1]
        adv  = exp.advantages          # [1, T-1]（常量权重或其剪裁）

        # policy loss
        denom = mask.sum().clamp_min(1e-8)
        pg = - (logp * adv * mask).sum() / denom

        # KL 正则（基于 k3）
        if self.kl_ctl and torch.is_tensor(exp.kl):
            kl_term = masked_mean(exp.kl, mask)
            loss = pg + self.kl_ctl * kl_term
        else:
            loss = pg

        loss.backward()
        self.opt_actor.step()

        # 返回 policy loss（日志用）
        return loss.detach()
