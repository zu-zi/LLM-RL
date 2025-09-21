# RL/DAPO.py
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import math
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    Samples, Experience,
    model_all_logits, token_logprobs_from_logits,
    masked_mean, normalize_for_reward,
    scatter_uniform_rewards,
)

# --------- helpers ---------
def _clean_logp(x: torch.Tensor, fallback: torch.Tensor = None):
    if fallback is not None and fallback.shape == x.shape:
        return torch.where(torch.isfinite(x), x, fallback)
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

def _prompt_end_index(action_mask_row: torch.Tensor) -> int:
    idx = (action_mask_row > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return action_mask_row.numel() - 1
    first_resp = int(idx[0].item())
    return max(0, first_resp - 1)

def _hash_prompt_ids(ids: torch.Tensor) -> str:
    if ids.numel() == 0:
        return "empty"
    if ids.is_cuda:
        ids = ids.detach().cpu()
    b = ",".join(map(str, ids.tolist())).encode("utf-8")
    return hashlib.sha1(b).hexdigest()

# --------- DAPO Trainer ---------
class DAPOTrainer:
    """
    Direct Advantage Policy Optimization（无 critic）
    - 奖励：句级 reward
    - 基线：按 prompt 哈希的 EMA 基线 b(h)；优势 ~ r_i - b(h)
      （等价于对“同一 prompt 的历史分布”做中心化；与 GRPO 的“组内中心化”互补）
    - 优势摊到 response token（与 PPO/GRPO 口径一致）
    - KL：沿用项目里 PPO/GRPO 的 k3 KL 作为 policy-level 正则
    """
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        device: str = "cuda",
        mb_size_logits: int = 1,
        # main RL hparams
        kl_ctl: float = 1.0,
        ppo_clip: float = 0.2,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        # safety caps（与 PPO/GRPO 对齐）
        ratio_min: float = 0.75,
        ratio_max: float = 1.25,
        kl_token_cap: float = 0.5,
        k3_cap: float = 1.5,
        ent_mask_keep: float = 0.20,
        # DAPO 专属
        ema_alpha: float = 0.10,           # prompt 级 EMA 的更新系数
        ema_warmup: int = 1,               # 每个 prompt 至少看到多少次再“生效扣除”
        use_batch_center_fallback: bool = True,  # 看不到历史时回退到 batch-mean
    ):
        self.actor = actor_model
        self.ref   = ref_model
        self.reward_model = reward_model
        self.actor_tok = actor_tokenizer
        self.reward_tok = reward_tokenizer

        if getattr(self.reward_tok, "pad_token_id", None) is None:
            if getattr(self.reward_tok, "eos_token", None) is not None:
                self.reward_tok.pad_token = self.reward_tok.eos_token
        try:
            self.reward_tok.padding_side = "right"
        except Exception:
            pass

        self.opt_actor = optimizer_actor

        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.mb_logits = max(1, int(mb_size_logits))

        self.kl_ctl = float(kl_ctl)
        self.ppo_clip = float(ppo_clip)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.ratio_min = max(0.0, float(ratio_min))
        self.ratio_max = max(0.0, float(ratio_max))
        if 0.0 < self.ratio_max < 1.0:
            self.ratio_max = 10.0
        self.kl_token_cap = float(kl_token_cap)
        self.k3_cap = float(k3_cap)
        self.ent_mask_keep = float(min(max(ent_mask_keep, 0.0), 1.0))

        self.ema_alpha = float(ema_alpha)
        self.ema_warmup = int(ema_warmup)
        self.use_batch_center_fallback = bool(use_batch_center_fallback)

        # prompt 基线状态
        self._ema_count: Dict[str, int] = {}
        self._ema_mean: Dict[str, float] = {}

        self.last_stats: Dict[str, float] = {
            "safety_ratio_min": float(self.ratio_min),
            "safety_ratio_max": float(self.ratio_max),
            "safety_logratio_cap": float(self.kl_token_cap),
            "safety_k3_cap": float(self.k3_cap),
            "safety_ent_keep": float(self.ent_mask_keep),
            # DAPO 诊断
            "dapo/baseline_mae": float("nan"),
            "dapo/baseline_used_frac": float("nan"),
        }

    # ------------------- Reward scoring -------------------
    @torch.no_grad()
    def _decode_and_score(self, seqs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = seqs.size()
        texts = []
        for i in range(B):
            L_i = int(attention_mask[i].sum().item())
            ids = seqs[i, :L_i].detach().cpu().tolist()
            if hasattr(self.actor_tok, "decode"):
                raw = self.actor_tok.decode(ids)
            else:
                raw = self.actor_tok.batch_decode([ids])[0]
            txt = normalize_for_reward(raw, reward_tokenizer=self.reward_tok)
            if "Assistant:" not in txt and "Human:" in txt:
                txt = txt.rstrip() + "\n\nAssistant:"
            texts.append(txt)

        toks = self.reward_tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        outs = self.reward_model(**{k: v for k, v in toks.items()})
        logits = getattr(outs, "logits", None)
        if logits is None:
            return torch.zeros(B, dtype=torch.float32, device=self.device)
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits.detach().to(self.device).float()

    # ------------------- Step 1: rollouts -> Experience -------------------
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples):
        """
        返回：
          experiences: List[Experience]（values/returns 不参与 value 训练）
          report_kl, r_raw_mean, r_shaped_mean, r_centered_mean, safe_kl
        """
        seqs = samples.seqs.to(self.device)
        attn = samples.attention_mask.to(self.device)
        amsk = samples.action_mask.to(self.device)

        assert amsk.size(1) == seqs.size(1)
        mask_tgt = amsk[:, 1:]
        assert int(mask_tgt.sum().item()) == int(samples.num_actions.sum().item())

        # 1) actor/ref token-logprobs（沿用 actor 参数 dtype）
        logits_actor = model_all_logits(self.actor, seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
        logits_ref   = model_all_logits(self.ref,   seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
        lp_actor_full = token_logprobs_from_logits(logits_actor, seqs)
        lp_ref_full   = token_logprobs_from_logits(logits_ref,   seqs)
        lp_actor_full = _clean_logp(lp_actor_full, fallback=lp_ref_full)
        lp_ref_full   = _clean_logp(lp_ref_full,   fallback=lp_actor_full)

        # 2) KL 报告 / 安全 KL
        log_ratio_rep = (lp_actor_full - lp_ref_full).clamp_(-8.0, 8.0)
        k3_report = torch.expm1(log_ratio_rep) - log_ratio_rep
        k3_report = torch.clamp(k3_report, 0.0, 50.0)
        report_kl = float(masked_mean(k3_report * mask_tgt, mask_tgt.float()).detach().item())

        log_ratio = (lp_actor_full - lp_ref_full)
        if self.kl_token_cap > 0.0:
            log_ratio = log_ratio.clamp(-self.kl_token_cap, self.kl_token_cap)
        k3 = torch.expm1(log_ratio) - log_ratio
        if self.k3_cap > 0.0:
            k3 = torch.clamp(k3, 0.0, self.k3_cap)
        k3 = (k3 * mask_tgt)
        denom = mask_tgt.sum(dim=1).clamp_min(1e-8).float()
        safe_kl_seq = k3.sum(dim=1) / denom
        safe_kl = float(safe_kl_seq.mean().detach().item())

        # 3) 句级 raw 奖励
        r_seq = self._decode_and_score(seqs, attn)  # [B]
        r_raw_mean = float(r_seq.mean().detach().item())

        # 4) —— prompt-EMA 基线 —— #
        B, T = seqs.size()
        prompt_hash = []
        for i in range(B):
            end_idx = _prompt_end_index(amsk[i])
            prompt_hash.append(_hash_prompt_ids(seqs[i, :end_idx+1]))

        # batch 级回退基线（缺少历史时可用）
        batch_mean = r_seq.mean().item() if self.use_batch_center_fallback else 0.0

        r_center = r_seq.clone()
        used_frac_cnt = 0
        abs_err = []

        for i in range(B):
            h = prompt_hash[i]
            cur = float(r_seq[i].item())
            cnt = self._ema_count.get(h, 0)
            mean_prev = self._ema_mean.get(h, batch_mean if cnt == 0 else 0.0)

            # 先决定是否扣除基线
            use_baseline = (cnt >= self.ema_warmup)
            base = mean_prev if use_baseline else (batch_mean if self.use_batch_center_fallback else 0.0)
            r_center[i] = r_seq[i] - base
            abs_err.append(abs(cur - base))
            used_frac_cnt += int(use_baseline)

            # 再更新 EMA
            new_mean = (1.0 - self.ema_alpha) * mean_prev + self.ema_alpha * cur if cnt > 0 else cur
            self._ema_mean[h] = new_mean
            self._ema_count[h] = cnt + 1

        baseline_mae = float(sum(abs_err) / max(1, len(abs_err)))
        baseline_used_frac = float(used_frac_cnt / max(1, B))

        # 5) 优势在 response 轴均匀摊（reward 端不叠 KL）
        rewards_t = scatter_uniform_rewards(r_center, mask_tgt, beta_kl=None)

        # shaped/centered 统计（报告仍用 report_kl 的 k3）
        denom2 = mask_tgt.sum(dim=1).clamp_min(1e-8).float()
        kl_mean_per_seq = (k3_report * mask_tgt).sum(dim=1) / denom2
        r_shaped_seq = r_seq - self.kl_ctl * kl_mean_per_seq
        r_shaped_mean = float(r_shaped_seq.mean().detach().item())
        r_centered_mean = float(r_center.mean().detach().item())

        # 6) 构造 Experience（无 value，returns=advantages=rewards_t）
        experiences: List[Experience] = []
        for i in range(B):
            experiences.append(Experience(
                seqs=seqs[i:i+1],
                action_log_probs=lp_actor_full[i:i+1].detach(),
                values=torch.zeros_like(lp_actor_full[i:i+1]).detach(),
                returns=rewards_t[i:i+1].detach(),
                advantages=rewards_t[i:i+1].detach(),
                attention_mask=attn[i:i+1],
                action_mask=mask_tgt[i:i+1],
                reward=r_seq[i:i+1].detach(),
                num_actions=mask_tgt[i:i+1].sum(dim=1).to(torch.long),
                kl=k3_report[i:i+1].detach(),
            ))

        # 公共诊断
        with torch.no_grad():
            logits_actor = model_all_logits(self.actor, seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
            entropy_tok = -(F.softmax(logits_actor[:, 1:, :], dim=-1).clamp_min(1e-12) *
                            F.log_softmax(logits_actor[:, 1:, :], dim=-1)).sum(-1)
            entropy_val = masked_mean(entropy_tok, mask_tgt.float())
            approx_kl_pi = float(masked_mean(k3_report * mask_tgt, mask_tgt.float()).detach().item())

        self.last_stats.update({
            "approx_kl_pi": float(approx_kl_pi),
            "entropy": float(entropy_val.detach().item()),
            "v_mae": float("nan"),
            "explained_var": float("nan"),
            "dapo/baseline_mae": float(baseline_mae),
            "dapo/baseline_used_frac": float(baseline_used_frac),
        })

        return experiences, report_kl, r_raw_mean, r_shaped_mean, r_centered_mean, safe_kl

    # ------------------- Step 2: train (actor only) -------------------
    def train_on_experience(self, exp: Experience, use_token_entropy: bool = False):
        """
        返回 (policy_loss, None)，与 PPO/GRPO 的返回形态一致。
        """
        seqs = exp.seqs
        action_mask = exp.action_mask
        old_logp = exp.action_log_probs
        adv = exp.advantages

        # 逐样本标准化优势
        with torch.no_grad():
            m = action_mask.float()
            denom = m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mean = (adv * m).sum(dim=1, keepdim=True) / denom
            var  = (((adv - mean) ** 2) * m).sum(dim=1, keepdim=True) / denom
            std  = var.sqrt().clamp_min(1e-6)
            adv.copy_(((adv - mean) / std).clamp_(-5.0, 5.0))

        # ===== Actor update =====
        self.actor.train()
        self.opt_actor.zero_grad(set_to_none=True)

        logits = model_all_logits(self.actor, seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
        logp_all = token_logprobs_from_logits(logits, seqs)
        logp_all = _clean_logp(logp_all, fallback=old_logp)

        # 统一裁剪长度
        L = min(old_logp.size(1), logp_all.size(1), action_mask.size(1))
        logp = logp_all[:, :L]
        old_logp = old_logp[:, :L]
        sel_mask = action_mask[:, :L]

        # 重要性比值（对数域夹紧）
        if self.ratio_min > 0.0 and self.ratio_max > 0.0 and self.ratio_max > self.ratio_min:
            lo = math.log(self.ratio_min); hi = math.log(self.ratio_max)
            delta = torch.clamp(logp - old_logp, lo, hi)
            ratio = torch.exp(delta)
        else:
            ratio = torch.exp(logp - old_logp)

        # PPO clipped objective
        surr1 = ratio * (adv[:, :L] * sel_mask)
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * (adv[:, :L] * sel_mask)
        policy_loss_tok = -torch.min(surr1, surr2)

        if self.entropy_coef != 0.0:
            ent_tok = -(F.softmax(logits[:, 1:, :], dim=-1).clamp_min(1e-12) *
                        F.log_softmax(logits[:, 1:, :], dim=-1)).sum(-1)
            policy_loss_tok = policy_loss_tok - self.entropy_coef * ent_tok[:, :L] * sel_mask

        denom_tok = sel_mask.sum().clamp_min(1e-8)
        policy_loss = policy_loss_tok.sum() / denom_tok

        # KL 正则（k3）
        with torch.no_grad():
            logits_ref = model_all_logits(self.ref, seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
            lp_ref_full = token_logprobs_from_logits(logits_ref, seqs)
            lp_ref_full = _clean_logp(lp_ref_full, fallback=logp_all)

        Lr = min(logp.size(1), lp_ref_full.size(1), sel_mask.size(1))
        lp_ref = lp_ref_full[:, :Lr]
        logp_cut = logp[:, :Lr]
        sel = sel_mask[:, :Lr].float()
        denom_cut = sel.sum().clamp_min(1e-8)

        log_ratio_raw = (logp_cut - lp_ref)
        if self.kl_token_cap > 0.0:
            log_ratio = torch.clamp(log_ratio_raw, -self.kl_token_cap, self.kl_token_cap)
        else:
            log_ratio = log_ratio_raw
        k3 = torch.expm1(log_ratio) - log_ratio
        if self.k3_cap > 0.0:
            k3 = torch.clamp(k3, 0.0, self.k3_cap)
        kl_mean = (k3 * sel).sum() / denom_cut
        policy_loss = policy_loss + self.kl_ctl * kl_mean

        policy_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), self.max_grad_norm)
        self.opt_actor.step()

        # 统计（键名与主循环对齐）
        with torch.no_grad():
            over = (torch.abs(ratio - 1.0) > self.ppo_clip).to(ratio.dtype)
            clipped = over * sel_mask[:, :ratio.size(1)].to(ratio.dtype)
            clip_frac = float((clipped.sum() / denom_tok).item())

            rdiff = (ratio - 1.0).abs() * sel_mask[:, :ratio.size(1)].float()
            if rdiff.sum() > 0:
                q = torch.quantile(rdiff[rdiff > 0], torch.tensor([0.5, 0.9, 0.99], device=rdiff.device))
                q50, q90, q99 = q.tolist()
                rmax = float(rdiff.max().item())
                adv_abs_mean = float(((adv.abs() * sel_mask.float()).sum() /
                                      sel_mask.float().sum()).item())
                sel_tokens = int(sel_mask.sum().item())
            else:
                q50 = q90 = q99 = rmax = adv_abs_mean = 0.0
                sel_tokens = 0

        self.last_stats.update({
            "clip_frac": float(clip_frac),
            "ratio_q50_q90_q99": (float(q50), float(q90), float(q99)),
            "ratio_max_stat": float(rmax),
            "adv_abs_mean": float(adv_abs_mean),
            "sel_tokens": int(sel_tokens),
            "ppo_clip": float(self.ppo_clip),
            "kl_ctl_now": float(self.kl_ctl),
        })

        return policy_loss.detach(), None
