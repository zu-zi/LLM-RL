from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    Samples, Experience,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs, gae_compute, masked_mean,
    entropy_from_logits, apply_entropy_mask,
    clip_by_global_norm, normalize_for_reward,
    forward_values_via_actor, scatter_uniform_rewards
)

# ===================== helpers: numeric stability =====================
def _clean_logp(x: torch.Tensor, fallback: torch.Tensor = None):
    """
    用 fallback 替换非有限值；fallback 不给则置 0。
    """
    if fallback is not None and fallback.shape == x.shape:
        return torch.where(torch.isfinite(x), x, fallback)
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


# ===================== Critic with independent adapter =====================
class Critic(nn.Module):
    """
    独立的 critic 分支（小 MLP 适配器），输入为 actor 的隐藏状态（可 detach），
    但参数不与 actor 共享；适配 V 回归任务，缓解“线性头过弱 + 表征漂移”。
    """
    def __init__(self, actor_like: nn.Module, width: int = 2, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        h = getattr(getattr(actor_like, "config", None), "n_embd", None) \
            or getattr(getattr(actor_like, "config", None), "hidden_size", None)
        if h is None:
            raise ValueError("Cannot infer hidden size from actor model config.")
        layers = []
        for _ in range(int(depth)):
            layers += [nn.LayerNorm(h), nn.Linear(h, width * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * h, h)]
        self.adapter = nn.Sequential(*layers) if layers else nn.Identity()
        self.value_head = nn.Linear(h, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, H]  (来自 actor 的 token 表征)
        return: [B, T, 1]
        """
        z = self.adapter(hidden_states)
        return self.value_head(z)


# ===================== PPO Trainer =====================
class PPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        critic_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        optimizer_critic,
        device: str = "cuda",
        mb_size_logits: int = 1,
        mb_size_values: int = 1,
        # --- main RL hparams ---
        kl_ctl: float = 0.3,          # KL 系数（唯一 KL：仅在 policy loss 中）
        ppo_clip: float = 0.2,
        vf_clip: Optional[float] = 0.2,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        # --- safety caps（不依赖环境变量，可由 config 覆盖） ---
        ratio_min: float = 0.75,
        ratio_max: float = 1.25,
        kl_token_cap: float = 0.5,    # 对 Δlogp 的逐 token 上限
        k3_cap: float = 1.5,          # 对 k3 的逐 token 上限
        ent_mask_keep: float = 0.20,  # 熵子采样比例（use_token_entropy=True 时生效）
    ):
        self.actor = actor_model
        self.ref   = ref_model
        self.critic = critic_model
        self.reward_model = reward_model
        self.actor_tok = actor_tokenizer
        self.reward_tok = reward_tokenizer

        # 奖励分词器：pad/eos/right padding
        if getattr(self.reward_tok, "pad_token_id", None) is None:
            if getattr(self.reward_tok, "eos_token", None) is not None:
                self.reward_tok.pad_token = self.reward_tok.eos_token
        try:
            self.reward_tok.padding_side = "right"
        except Exception:
            pass

        self.opt_actor = optimizer_actor
        self.opt_critic = optimizer_critic

        self.device = device
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.mb_logits = max(1, int(mb_size_logits))
        self.mb_values = max(1, int(mb_size_values))

        # 超参
        self.kl_ctl = float(kl_ctl)
        self.ppo_clip = float(ppo_clip)
        self.vf_clip = vf_clip
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)

        # safety caps
        self.ratio_min = max(0.0, float(ratio_min))
        self.ratio_max = max(0.0, float(ratio_max))
        if 0.0 < self.ratio_max < 1.0:
            self.ratio_max = 10.0
        self.kl_token_cap = float(kl_token_cap)
        self.k3_cap = float(k3_cap)
        self.ent_mask_keep = float(min(max(ent_mask_keep, 0.0), 1.0))

        self.last_stats = {
            "safety_ratio_min": float(self.ratio_min),
            "safety_ratio_max": float(self.ratio_max),
            "safety_logratio_cap": float(self.kl_token_cap),
            "safety_k3_cap": float(self.k3_cap),
            "safety_ent_keep": float(self.ent_mask_keep),
        }

    # ------------------- Reward scoring -------------------
    @torch.no_grad()
    def _decode_dialogue_and_score(self, seqs: torch.Tensor,
                                   attention_mask: torch.Tensor) -> torch.Tensor:
        """
        解码到有效长度（含 Human: ... + Assistant: + response），再送入奖励模型。返回 [B] logits。
        """
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
            txt = txt.replace("\ufffd", "")
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
        将一个 batch 的 Samples 转成训练所需 Experience，并计算日志指标。
        返回：
          experiences (List[Experience]),
          report_kl (float),
          r_raw_mean (float),
          r_shaped_mean (float),
          r_centered_mean (float),
          safe_kl (float)
        """
        seqs = samples.seqs.to(self.device)
        attn = samples.attention_mask.to(self.device)
        amsk = samples.action_mask.to(self.device)

        assert amsk.size(1) == seqs.size(1)
        mask_tgt = amsk[:, 1:]
        assert int(mask_tgt.sum().item()) == int(samples.num_actions.sum().item())

        # 1) actor/ref token-logprobs（float32 更稳）
        lp_actor_full, lp_ref_full, mask_target = compute_actor_ref_logprobs(
            self.actor, self.ref, seqs, amsk, self.device_type,
            ptdtype=torch.float32, micro_batch_size=self.mb_logits
        )
        lp_actor_full = _clean_logp(lp_actor_full, fallback=lp_ref_full)
        lp_ref_full   = _clean_logp(lp_ref_full,   fallback=lp_actor_full)

        # 2) 报告用 KL（温和夹紧）
        log_ratio_rep = (lp_actor_full - lp_ref_full).clamp_(-8.0, 8.0)
        k3_report = torch.expm1(log_ratio_rep) - log_ratio_rep
        k3_report = torch.clamp(k3_report, 0.0, 50.0)
        report_kl = float(masked_mean(k3_report * mask_target, mask_target.float()).detach().item())

        # 2') 训练/塑形同款 k3（先对 Δlogp 夹紧；对 k3 再可夹顶），用作 safe_kl 统计
        if self.kl_token_cap > 0.0:
            log_ratio = (lp_actor_full - lp_ref_full).clamp(-self.kl_token_cap, self.kl_token_cap)
        else:
            log_ratio = (lp_actor_full - lp_ref_full)
        k3 = torch.expm1(log_ratio) - log_ratio
        if self.k3_cap > 0.0:
            k3 = torch.clamp(k3, 0.0, self.k3_cap)
        k3 = k3 * mask_target

        denom = mask_target.sum(dim=1).clamp_min(1e-8).float()
        safe_kl_seq = k3.sum(dim=1) / denom
        safe_kl = float(safe_kl_seq.mean().detach().item())

        # 3) critic 值函数（完整时间轴，随后按 action 轴裁切）
        values_full = forward_values_via_actor(
            self.actor, self.critic, seqs, self.device_type,
            ptdtype=torch.float32, micro_batch_size=self.mb_values, detach_hidden=False
        )
        values = values_full[:, 1:]
        action_mask = mask_target

        # 4) 句级 raw 奖励
        r_seq = self._decode_dialogue_and_score(seqs, attn)
        r_raw_mean = float(r_seq.mean().detach().item())

        # 5) 均匀分摊奖励（不在奖励端叠 KL；KL 仅在 policy loss 中）
        rewards_t = scatter_uniform_rewards(r_seq, action_mask, beta_kl=None)

        # shaped/centered 统计（报告仍用 report_kl 的 k3）
        denom2 = action_mask.sum(dim=1).clamp_min(1e-8).float()
        kl_mean_per_seq = (k3_report * action_mask).sum(dim=1) / denom2
        r_shaped_seq = r_seq - self.kl_ctl * kl_mean_per_seq
        r_shaped_mean = float(r_shaped_seq.mean().detach().item())
        r_centered_mean = float((r_seq - r_seq.mean()).mean().detach().item())

        # 6) GAE（按 action 轴）
        returns, advantages = gae_compute(
            values=values, rewards=rewards_t, mask_time=action_mask,
            gamma=self.gae_gamma, lam=self.gae_lambda, use_last_as_terminal=True
        )

        # 拆成单条 Experience（PPO 里好用）
        experiences = []
        B = seqs.size(0)
        for i in range(B):
            experiences.append(Experience(
                seqs=seqs[i:i+1],
                action_log_probs=lp_actor_full[i:i+1].detach(),
                values=values[i:i+1].detach(),
                returns=returns[i:i+1].detach(),
                advantages=advantages[i:i+1].detach(),
                attention_mask=attn[i:i+1],
                action_mask=action_mask[i:i+1],
                reward=r_seq[i:i+1].detach(),
                num_actions=action_mask[i:i+1].sum(dim=1).to(torch.long),
                kl=k3_report[i:i+1].detach(),
            ))

        # 额外日志统计
        with torch.no_grad():
            approx_kl_pi = float(masked_mean(k3_report * action_mask, action_mask.float()).detach().item())
            logits_actor = model_all_logits(
                self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits
            )
            entropy_tok = float(entropy_from_logits(logits_actor[:, 1:, :], action_mask).detach().item())
            v_mae = float(((values - returns).abs() * action_mask).sum() / action_mask.sum().clamp_min(1e-8))

            # explained variance（按 response 轴）
            m = action_mask.float()
            y = returns
            y_mean = (y * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            var_y = (((y - y_mean) ** 2) * m).sum() / m.sum().clamp_min(1e-8)
            err = (returns - values)
            e_mean = (err * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            var_err = (((err - e_mean) ** 2) * m).sum() / m.sum().clamp_min(1e-8)
            explained_var = float(1.0 - (var_err / var_y.clamp_min(1e-8)))

        self.last_stats.update({
            "approx_kl_pi": float(approx_kl_pi),
            "entropy": float(entropy_tok),
            "v_mae": float(v_mae),
            "explained_var": float(explained_var),
        })

        return experiences, report_kl, r_raw_mean, r_shaped_mean, r_centered_mean, safe_kl

    # ------------------- Step 2: train (actor + critic) -------------------
    def train_on_experience(self, exp: Experience, use_token_entropy: bool = False):
        """
        单次参数更新：返回 (policy_loss, value_loss) 两个张量（供日志打印）。
        """
        seqs = exp.seqs
        attn = exp.attention_mask
        action_mask = exp.action_mask
        old_logp = exp.action_log_probs
        returns = exp.returns
        old_values = exp.values
        adv = exp.advantages

        # 归一化优势（逐样本）
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

        logits = model_all_logits(
            self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits
        )  # [B, T, V]
        logp_all = token_logprobs_from_logits(logits, seqs)  # [B, T-1]
        logp_all = _clean_logp(logp_all)

        L = old_logp.size(1)
        logp = logp_all[:, :L]
        action_mask = action_mask[:, :L]
        attn = attn[:, :L]

        # token-entropy 子采样（只改 loss 掩码）
        if use_token_entropy:
            keep = self.ent_mask_keep if (0.0 < self.ent_mask_keep <= 1.0) else 0.25
            sel_mask = apply_entropy_mask(logits[:, 1:, :].detach(), action_mask, keep_ratio=keep)
        else:
            sel_mask = action_mask

        # 重要性采样比值（对数域夹紧更稳）
        raw_delta = logp - old_logp
        if self.ratio_min > 0.0 and self.ratio_max > 0.0 and self.ratio_max > self.ratio_min:
            lo = math.log(self.ratio_min)
            hi = math.log(self.ratio_max)
            delta = torch.clamp(raw_delta, lo, hi)
            ratio = torch.exp(delta)
        else:
            ratio = torch.exp(raw_delta)

        # 统计
        with torch.no_grad():
            sel = sel_mask.float()
            rdiff = (ratio - 1.0).abs() * sel
            if sel.sum() > 0:
                q = torch.quantile(
                    rdiff[sel.bool()], torch.tensor([0.5, 0.9, 0.99], device=rdiff.device)
                )
                q50, q90, q99 = q.tolist()
                rmax = float(rdiff.max().item())
                adv_abs_mean = float(((adv.abs() * sel).sum() / sel.sum()).item())
                sel_tokens = int(sel.sum().item())
            else:
                q50 = q90 = q99 = rmax = adv_abs_mean = 0.0
                sel_tokens = 0

            self.last_stats["ratio_q50_q90_q99"] = (float(q50), float(q90), float(q99))
            self.last_stats["ratio_max"] = float(rmax)
            self.last_stats["adv_abs_mean"] = float(adv_abs_mean)
            self.last_stats["sel_tokens"] = int(sel_tokens)
            self.last_stats["ppo_clip"] = float(self.ppo_clip)
            self.last_stats["kl_ctl_now"] = float(self.kl_ctl)

        # PPO clipped objective（仅选中 token）
        surr1 = ratio * (adv * sel_mask)
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * (adv * sel_mask)
        policy_loss_tok = -torch.min(surr1, surr2)

        if self.entropy_coef != 0.0:
            ent_tok = -(F.softmax(logits[:, 1:, :], dim=-1).clamp_min(1e-12) *
                        F.log_softmax(logits[:, 1:, :], dim=-1)).sum(-1)
            policy_loss_tok = policy_loss_tok - self.entropy_coef * ent_tok * sel_mask

        policy_loss = policy_loss_tok.sum() / sel_mask.sum().clamp_min(1e-8)

        # KL （唯一的 KL，放在 policy loss 中；对 Δlogp/ k3 做安全夹紧）
        with torch.no_grad():
            logits_ref = model_all_logits(
                self.ref, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits
            )
            lp_ref_full = token_logprobs_from_logits(logits_ref, seqs)
            lp_ref_full = _clean_logp(lp_ref_full, fallback=logp_all)

        Lr = min(logp.size(1), lp_ref_full.size(1))
        lp_ref = lp_ref_full[:, :Lr]
        logp_cut = logp[:, :Lr]
        sel = sel_mask[:, :Lr].float()

        log_ratio_raw = (logp_cut - lp_ref)
        if self.kl_token_cap > 0.0:
            log_ratio = torch.clamp(log_ratio_raw, -self.kl_token_cap, self.kl_token_cap)
        else:
            log_ratio = log_ratio_raw
        k3 = torch.expm1(log_ratio) - log_ratio
        if self.k3_cap > 0.0:
            k3 = torch.clamp(k3, 0.0, self.k3_cap)
        kl_mean = (k3 * sel).sum() / sel.sum().clamp_min(1e-8)
        policy_loss = policy_loss + self.kl_ctl * kl_mean

        policy_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            clip_by_global_norm(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        # clip_frac
        with torch.no_grad():
            Lm = ratio.size(1)
            sel = sel_mask[:, :Lm].to(ratio.dtype)
            over = (torch.abs(ratio - 1.0) > self.ppo_clip).to(ratio.dtype)
            clipped = over * sel
            denom = sel.sum().clamp_min(1e-8)
            clip_frac = float((clipped.sum() / denom).item())
        self.last_stats["clip_frac"] = clip_frac

        # ===== Critic update =====
        self.critic.train()
        self.opt_critic.zero_grad(set_to_none=True)

        values_full = forward_values_via_actor(
            self.actor, self.critic, seqs, self.device_type,
            ptdtype=torch.float32, micro_batch_size=self.mb_values, detach_hidden=True
        )  # [B, T]
        values_new = values_full[:, 1:]  # [B, T-1]

        if self.vf_clip is not None:
            values_clipped = old_values + (values_new - old_values).clamp(-self.vf_clip, self.vf_clip)
            vloss1 = (values_new - returns) ** 2
            vloss2 = (values_clipped - returns) ** 2
            v_loss_tok = torch.max(vloss1, vloss2)
        else:
            v_loss_tok = F.smooth_l1_loss(values_new, returns, beta=1.0, reduction="none")

        v_loss = (v_loss_tok * action_mask).sum() / action_mask.sum().clamp_min(1e-8)
        v_loss.backward()
        clip_by_global_norm(self.critic.parameters(), 0.5)
        self.opt_critic.step()

        return policy_loss.detach(), v_loss.detach()
