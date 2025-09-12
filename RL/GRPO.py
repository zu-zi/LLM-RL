# RL/GRPO.py
import os, math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.common import (
    Samples, Experience,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs, compute_approx_kl, kl_k3_from_logratio,
    entropy_from_logits, masked_mean,
    get_advantages_and_returns,  # 仅为签名一致；GRPO不用critic
    ratio_stats, adv_abs_mean,
    normalize_for_reward,
    apply_entropy_mask,
)

# 环境变量（与 PPO 保持一致的“硬夹”，避免尖峰）
PPO_RATIO_MIN = float(os.getenv("PPO_RATIO_MIN", "0.75"))
PPO_RATIO_MAX = float(os.getenv("PPO_RATIO_MAX", "1.25"))
PPO_KL_TOKEN_CAP = float(os.getenv("PPO_KL_TOKEN_CAP", "0.5"))
PPO_K3_CAP = float(os.getenv("PPO_K3_CAP", "1.5"))
ENT_MASK_KEEP = float(os.getenv("ENT_MASK_KEEP", "0.2"))

# 采样解码的默认参数（和 train_RL_only 的一致）
SAMPLE_TEMPERATURE = float(os.getenv("SAMPLE_TEMPERATURE", "0.8"))
SAMPLE_TOP_P = float(os.getenv("SAMPLE_TOP_P", "0.9"))
SAMPLE_TOP_K = int(os.getenv("SAMPLE_TOP_K", "0"))
SAMPLE_REP_PENALTY = float(os.getenv("SAMPLE_REP_PENALTY", "1.1"))

MIN_RESP_TOK = int(os.getenv("ROLL_MIN_RESP_TOKENS", "16"))

# ------------------------------
# 简单 nucleus 采样一个 token（与 train_RL_only 内部实现一致口径）
# ------------------------------
@torch.no_grad()
def _sample_next_token(last_logits: torch.Tensor, prev_ids: torch.Tensor,
                       top_p=0.9, temperature=0.8, top_k=0, repetition_penalty=1.1):
    if repetition_penalty and prev_ids is not None and prev_ids.numel() > 0:
        uniq = torch.unique(prev_ids)
        last_logits[:, uniq] = last_logits[:, uniq] / float(repetition_penalty)
    last_logits = last_logits / max(float(temperature), 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(last_logits, k=min(int(top_k), last_logits.size(-1)), dim=-1).values[..., -1:]
        last_logits = torch.where(last_logits < kth, torch.full_like(last_logits, -1e10), last_logits)
    probs = torch.softmax(last_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cumsum > float(top_p)).float().argmax(dim=-1, keepdim=True)
    mask = torch.arange(probs.size(-1), device=probs.device).view(1, -1) <= cutoff
    kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
    kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    next_sorted = torch.multinomial(kept, num_samples=1)
    next_id = sorted_idx.gather(1, next_sorted)  # [1,1]
    return next_id


@torch.no_grad()
def _decode_with_sampling(model, idx, max_new_tokens, eos_id, block_size,
                          temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P, top_k=SAMPLE_TOP_K,
                          repetition_penalty=SAMPLE_REP_PENALTY, min_resp=MIN_RESP_TOK):
    """单样本按采样生成；仅用于 GRPO 组内补样。"""
    device = idx.device
    was_train = model.training
    model.eval()
    try:
        out = idx
        start_len = out.size(1)
        for _ in range(int(max_new_tokens)):
            idx_cond = out[:, -int(block_size):]
            logits = model(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[0]
            last = logits[:, -1, :]

            next_id = _sample_next_token(last, out, top_p=top_p, temperature=temperature,
                                         top_k=top_k, repetition_penalty=repetition_penalty)

            # 强制最短回复前不允许 EOS
            if (out.size(1) - start_len) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                # 退而取第二大
                probs = torch.softmax(last / max(float(temperature), 1e-6), dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                alt = sorted_idx[:, 1:2] if sorted_idx.size(1) > 1 else next_id
                next_id = alt

            out = torch.cat((out, next_id.to(device)), dim=1)

            if (out.size(1) - start_len) >= int(min_resp):
                if eos_id is not None and int(next_id.item()) == int(eos_id):
                    break
        return out
    finally:
        if was_train:
            model.train()


class GRPOTrainer:
    """
    组相对策略优化（Group Relative Policy Optimization）
    - 不用 value/critic，优势由“组内标准化奖励”得到。
    - 句级优势 → 复制到每个 action token。
    - 可选：对 ref 使用 KL k3 惩罚。
    - 可选：PPO 风格的 ratio 裁剪。
    """
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: nn.Module,               # 标量 RM（如 OASST RM）；或自定义函数也可，在外面包一下
        actor_tokenizer,                       # 你项目里的 GPT2Tok
        reward_tokenizer,                      # HF tokenizer（CPU）
        optimizer_actor: torch.optim.Optimizer,
        device: str = "cuda",
        group_size: int = 4,
        kl_coef: float = 0.0,
        clip_reward: float = 5.0,
        mb_size_logits: int = 1,               # micro-batch for logits
        block_size: int = 384,
        max_new_tokens: int = 6,              # 在线补样时的最大生成
        # use_token_entropy: bool = False,
        # ent_keep_ratio: float = 0.2,
    ):
        self.actor = actor_model
        self.ref = ref_model
        self.reward_model = reward_model
        self.tok = actor_tokenizer
        self.rtok = reward_tokenizer
        self.opt_actor = optimizer_actor

        self.device_type = "cuda" if ("cuda" in str(device)) else "cpu"
        self.mb_logits = int(mb_size_logits) if mb_size_logits else 0

        self.group_size = int(group_size)
        self.kl_coef = float(kl_coef)
        self.clip_reward = float(clip_reward)
        self.block_size = int(block_size)
        self.max_new_tokens = int(max_new_tokens)
        # self.use_token_entropy = bool(use_token_entropy)
        # self.ent_keep_ratio = float(ent_keep_ratio)

        self.last_stats: Dict[str, Any] = {}

        # 日志辅助
        self.kl_ctl = self.kl_coef  # 为了和主循环打印字段名对齐
        self._eps = 1e-8

    # --------------------------
    # 工具：从 batch 中恢复 prompt ids（通过 action_mask 边界）
    # --------------------------
    @staticmethod
    def _prompt_len_from_mask(action_mask_row: torch.Tensor) -> int:
        # action_mask=1 的第一位是 response 起点
        nz = (action_mask_row > 0).nonzero(as_tuple=False)
        return int(nz[0].item()) if nz.numel() > 0 else int(action_mask_row.numel())

    def _split_prompt_response(self, seqs: torch.Tensor, action_mask: torch.Tensor) -> Tuple[List[List[int]], List[List[int]]]:
        B, T = seqs.shape
        prompts, responses = [], []
        for i in range(B):
            p_len = self._prompt_len_from_mask(action_mask[i])
            L = int((action_mask[i] > 0).nonzero(as_tuple=False))
            tlen = int((action_mask[i] > 0).sum().item() + p_len)
            # 用 attention_mask 会更准，这里取 action_mask 范围对齐
            full = seqs[i].tolist()
            prompts.append(full[:p_len])
            responses.append(full[p_len:tlen])
        return prompts, responses

    # --------------------------
    # 评分：把 prompt+response 文本走 RM
    # --------------------------
    @torch.no_grad()
    def _score_with_rm(self, full_texts: List[str]) -> torch.Tensor:
        if len(full_texts) == 0:
            return torch.zeros(0, device="cpu")
        texts = [normalize_for_reward(t, self.rtok) for t in full_texts]
        toks = self.rtok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        outs = self.reward_model(**{k: v for k, v in toks.items()})
        logits = getattr(outs, "logits", None)
        if logits is None:
            return torch.zeros(len(full_texts), device="cpu")
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits.detach().float()

    # --------------------------
    # 组内补样：基于已有 prompt，再生成 (group_size-1) 个 response
    # --------------------------
    @torch.no_grad()
    def _augment_group_for_prompt(self, prompt_ids: List[int], use_actor=True) -> List[List[int]]:
        dev = next(self.actor.parameters()).device
        base = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0)
        eos_id = self.tok.eos_id
        room = self.block_size - base.size(1) - 1
        if room <= 0:
            return [prompt_ids[:self.block_size]]

        K = max(1, int(self.group_size))
        seqs: List[List[int]] = []
        # 包含“现成的一条”（由外部 samples 带入），其余 K-1 条在线生成补齐
        # 这里只做“全 K 条都采样”，由 evaluate_experience 负责把 batch 里的那条并入组
        for _ in range(K):
            out = _decode_with_sampling(
                self.actor if use_actor else self.ref, base,
                max_new_tokens=min(self.max_new_tokens, room),
                eos_id=eos_id, block_size=self.block_size,
                temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
                top_k=SAMPLE_TOP_K, repetition_penalty=SAMPLE_REP_PENALTY,
                min_resp=MIN_RESP_TOK
            )
            seqs.append(out[0].tolist()[:self.block_size])
        return seqs

    # --------------------------
    # 主入口：评估经验（不更新），返回 Experience 列表 + 各指标
    # --------------------------
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples):
        """
        输入：一个 batch 的样本（每条只有一条 response）。
        本实现会按“同一 prompt”为单位，在线补足若干条 response，形成组（group_size）。
        然后在组内做奖励标准化，得到句级优势；优势复制到每个 token。
        """
        dev = next(self.actor.parameters()).device
        B, T = samples.seqs.shape
        # 先拿到本 batch 的 prompt 边界与文本
        # prompt_ids_list, response_ids_list = self._split_prompt_response(samples.seqs, samples.action_mask)

        # ——更稳的方式：直接根据掩码切 —— #
        prompt_ids_list = []
        base_full_list = []  # 把 batch 里已有的“完整一条”放入组
        for i in range(B):
            seq = samples.seqs[i]
            mask = samples.action_mask[i]
            p_len = self._prompt_len_from_mask(mask)
            # full 的有效长度（用 attention_mask 更准）
            t_len = int(samples.attention_mask[i].sum().item())
            prompt_ids = seq[:p_len].tolist()
            full_ids   = seq[:t_len].tolist()
            prompt_ids_list.append(prompt_ids)
            base_full_list.append(full_ids)

        # === 为每条样本补齐 group ===
        group_full_ids: List[List[List[int]]] = []  # [B][K][T*]
        for i in range(B):
            # 在线生成 K 条，再把 batch 自带的那条替换掉其中一条（避免重复费时）
            g = self._augment_group_for_prompt(prompt_ids_list[i], use_actor=True)  # K 条
            # 把第 0 条替换为 batch 自带那条（保证“现成的”也被纳入评分）
            g[0] = base_full_list[i]
            group_full_ids.append(g)

        # === 对每条 group，计算：
        # - actor/ref logprobs（逐 token，方便做 KL 和 ratio）
        # - action_mask（对齐到 response 段 = 从 prompt_len 开始到句末）
        # - 奖励（句级）
        experiences: List[Experience] = []
        all_rewards = []
        all_k3_means = []

        sel_tokens_total = 0
        ratio_qs_total = []
        ratio_max_total = []

        for i in range(B):
            g_full = group_full_ids[i]             # K 条完整 ids
            K = len(g_full)
            # 统一右 pad 到本组的最大长度，构造 Samples-like 结构
            Lmax = min(self.block_size, max(len(x) for x in g_full))
            seqs = torch.full((K, Lmax), self.tok.eos_id, dtype=torch.long, device=dev)
            attn = torch.zeros((K, Lmax), dtype=torch.long, device=dev)
            amsk = torch.zeros((K, Lmax), dtype=torch.long, device=dev)

            # 基于“同一个 prompt”，起点是一致的
            p_len = self._prompt_len_from_mask(samples.action_mask[i])

            for k in range(K):
                ids = g_full[k][:Lmax]
                t = len(ids)
                seqs[k, :t] = torch.tensor(ids, dtype=torch.long, device=dev)
                attn[k, :t] = 1
                if p_len < t:
                    amsk[k, p_len:t] = 1

            # 计算 actor/ref 的逐 token logprob
            lp_actor, lp_ref, mask_tgt = compute_actor_ref_logprobs(
                self.actor, self.ref if (self.ref is not None) else self.actor,
                seqs, amsk, device_type=self.device_type, ptdtype=None, micro_batch_size=self.mb_logits
            )  # [K, T-1], [K, T-1], [K, T-1]
            
            # === Token-entropy 掩码（2/8 高熵保留）===
            if self.use_token_entropy:
                logits_actor_full = model_all_logits(
                    self.actor, seqs, device_type=self.device_type, ptdtype=None, micro_batch_size=self.mb_logits
                )
                new_mask_tgt = apply_entropy_mask(
                    logits_actor_full[:, 1:, :].detach(),  # 与 mask_tgt 对齐
                    mask_tgt,
                    keep_ratio=float(ent_keep_ratio)
                )
                if new_mask_tgt.sum() > 0:  # 防极端回退
                    mask_tgt = new_mask_tgt


            # 句级奖励（RM）
            # 注意：GRPO 一般用“prompt+response 全文”送 RM
            texts = []
            for k in range(K):
                tlen = int(attn[k].sum().item())
                texts.append(self.tok.decode(seqs[k, :tlen]))
            rw = self._score_with_rm(texts).to(dev)  # [K]
            if self.clip_reward > 0:
                rw = torch.clamp(rw, -self.clip_reward, self.clip_reward)

            # 组内标准化优势（句级）
            mean_r = rw.mean()
            std_r = rw.std(unbiased=False)
            adv_sent = (rw - mean_r) / (std_r + self._eps)  # [K]

            # k3 KL（对 ref；若不提供 ref，就把 ref=actor，相当于 0）
            log_ratio = (lp_actor - lp_ref) * mask_tgt
            # 夹紧 Δlogp，避免尖峰
            if PPO_KL_TOKEN_CAP > 0:
                log_ratio = torch.clamp(log_ratio, -PPO_KL_TOKEN_CAP, PPO_KL_TOKEN_CAP)
            k3 = kl_k3_from_logratio(log_ratio)
            if PPO_K3_CAP > 0:
                k3 = torch.clamp(k3, max=PPO_K3_CAP)

            # 重要性比 r = exp(Δlogp)
            ratio = torch.exp(log_ratio)
            ratio = torch.clamp(ratio, PPO_RATIO_MIN, PPO_RATIO_MAX)

            # 复制句级优势到 token
            adv_tok = adv_sent.view(-1, 1).expand_as(ratio) * mask_tgt

            # GRPO 的 per-token 损失（句级优势 * 比率，PPO-style 裁剪）
            unclipped =  ratio * adv_tok
            clipped   = torch.clamp(ratio, 1.0 - (PPO_RATIO_MAX-1.0), 1.0 + (PPO_RATIO_MAX-1.0)) * adv_tok
            per_tok_loss = -torch.minimum(unclipped, clipped)
            # KL 惩罚
            if self.kl_coef and self.kl_coef > 0:
                per_tok_loss = per_tok_loss + self.kl_coef * k3

            # 聚合为 per-seq，再做 batch mean（训练时）
            # 这里只是准备 Experience；真正的 backward 在 train_on_experience
            # old_action_log_probs 取当前 lp_actor（detach）做“快照”
            # advantage 存句级（复用 Experience.advantages 字段的 [K, T] 形式方便）
            # returns/values 在 GRPO 不用，放 0
            per_seq_mask = mask_tgt
            # 统计
            k3_mean = masked_mean(k3, mask_tgt.float()).item()
            all_k3_means.append(k3_mean)
            all_rewards.extend(rw.detach().tolist())

            r_q50, r_q90, r_q99, r_max = ratio_stats(log_ratio, mask_tgt)
            ratio_qs_total.append((r_q50, r_q90, r_q99))
            ratio_max_total.append(r_max)
            sel_tokens_total += int(mask_tgt.sum().item())

            # 打包 Experience
            exp = Experience(
                seqs=seqs,                                   # [K, Lmax]
                action_log_probs=lp_actor.detach(),          # 作为 old_pi
                values=torch.zeros_like(lp_actor),           # 占位
                returns=torch.zeros_like(lp_actor),          # 占位
                advantages=adv_tok.detach(),                 # [K, T-1]
                attention_mask=attn,
                action_mask=mask_tgt,
                reward=rw.view(-1, 1),                       # 句级 reward（仅日志）
                num_actions=mask_tgt.sum(dim=1).long(),
                kl=k3,                                       # 保存逐 token k3，训练期也可用
            )
            experiences.append(exp)

        # 汇总日志指标（模仿 PPOTrainer）
        approx_kl_pi = float(sum(all_k3_means) / max(len(all_k3_means), 1))
        r_raw = float(torch.tensor(all_rewards).mean().item()) if all_rewards else float("nan")
        r_shaped = r_raw  # GRPO 没有额外 shaping
        r_ctr = 0.0       # 无对比奖励
        safe_kl = approx_kl_pi

        # 存 “last_stats” 供主循环打印
        if len(ratio_qs_total) > 0:
            q50 = float(sum(x[0] for x in ratio_qs_total) / len(ratio_qs_total))
            q90 = float(sum(x[1] for x in ratio_qs_total) / len(ratio_qs_total))
            q99 = float(sum(x[2] for x in ratio_qs_total) / len(ratio_qs_total))
            rmax = float(max(ratio_max_total))
        else:
            q50 = q90 = q99 = rmax = float("nan")

        self.last_stats = {
            "approx_kl_pi": approx_kl_pi,
            "entropy": float("nan"),           # 可选：需要再算一次全 logits；先置空
            "clip_frac": float("nan"),         # 训练步里再记录
            "v_mae": float("nan"),
            "explained_var": float("nan"),
            "ratio_q50_q90_q99": (q50, q90, q99),
            "ratio_max": rmax,
            "adv_abs_mean": 0.0,               # 训练时再记
            "sel_tokens": sel_tokens_total,
            "ppo_clip": float(PPO_RATIO_MAX-1.0),
            "kl_ctl_now": self.kl_ctl,
        }

        return experiences, approx_kl_pi, r_raw, r_shaped, r_ctr, safe_kl

    # --------------------------
    # 训练一步：仅更新 actor
    # --------------------------
    def train_on_experience(self, exp: Experience, use_token_entropy: bool = False):
        self.actor.train()
        self.opt_actor.zero_grad(set_to_none=True)

        # 重新前向拿到当前 logprob（新 pi）
        logits = model_all_logits(self.actor, exp.seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
        lp_new = token_logprobs_from_logits(logits, exp.seqs)  # [K, T-1]
        mask_tgt = exp.action_mask

        lp_new = torch.where(mask_tgt > 0, lp_new, torch.zeros_like(lp_new))
        old_lp = torch.where(mask_tgt > 0, exp.action_log_probs, torch.zeros_like(exp.action_log_probs))

        # Δlogp + ratio（裁剪）
        log_ratio = lp_new - old_lp
        if PPO_KL_TOKEN_CAP > 0:
            log_ratio = torch.clamp(log_ratio, -PPO_KL_TOKEN_CAP, PPO_KL_TOKEN_CAP)
        ratio = torch.exp(log_ratio)
        ratio = torch.clamp(ratio, PPO_RATIO_MIN, PPO_RATIO_MAX)

        adv_tok = exp.advantages  # [K, T-1]，已含 mask
        unclipped =  ratio * adv_tok
        clipped   = torch.clamp(ratio, 1.0 - (PPO_RATIO_MAX-1.0), 1.0 + (PPO_RATIO_MAX-1.0)) * adv_tok
        pg_loss_tok = -torch.minimum(unclipped, clipped)  # [K, T-1]
        pg_loss = (pg_loss_tok * mask_tgt).sum() / mask_tgt.sum().clamp_min(1.0)

        # KL（对 ref；如果 ref=None，相当于 0）
        if self.ref is not None and self.kl_coef > 0:
            with torch.no_grad():
                logits_ref = model_all_logits(self.ref, exp.seqs, self.device_type, ptdtype=None, micro_batch_size=self.mb_logits)
                lp_ref = token_logprobs_from_logits(logits_ref, exp.seqs)
            log_ratio_ref = (lp_new - lp_ref) * mask_tgt
            k3 = kl_k3_from_logratio(log_ratio_ref)
            if PPO_K3_CAP > 0:
                k3 = torch.clamp(k3, max=PPO_K3_CAP)
            kl_loss = (k3 * mask_tgt).sum() / mask_tgt.sum().clamp_min(1.0)
            loss = pg_loss + self.kl_coef * kl_loss
        else:
            loss = pg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # 日志
        with torch.no_grad():
            clip_frac = float(((ratio <= PPO_RATIO_MIN) | (ratio >= PPO_RATIO_MAX)).float()[mask_tgt>0].float().mean().item())
            r_q50, r_q90, r_q99, r_max = ratio_stats(log_ratio, mask_tgt)
            H = entropy_from_logits(logits[:, 1:, :], mask_tgt)  # 对齐 action 轴
            self.last_stats.update({
                "clip_frac": clip_frac,
                "ratio_q50_q90_q99": (r_q50, r_q90, r_q99),
                "ratio_max": r_max,
                "entropy": float(H.item()),
                "adv_abs_mean": adv_abs_mean(adv_tok, mask_tgt),
            })

        return loss
