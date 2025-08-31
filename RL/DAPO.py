# 说明：
# - DAPO: Direct Advantage Policy Optimization（直接把 logπ - logπ_ref 回归到 β*A）
# - 无需 critic。优势 A 由“序列级奖励 - 基线”得到，再广播到 response token（只在 response 段做训练）。
# - 与本项目的 PPO/GRPO 公共接口保持一致；显存友好，支持 logits 前向的 micro-batch。

import torch
import torch.nn.functional as F
from torch import amp
from typing import List, Tuple

from .common import (
    Samples, Experience, normalize_for_reward,
    build_samples_from_generations,
    compute_actor_ref_logprobs,
    model_all_logits, token_logprobs_from_logits,
    masked_mean, clip_by_global_norm
)


class DAPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        reward_model,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        beta: float = 1.0,                # 回归目标中的温度系数 β
        adv_norm: str = "zscore",         # 对优势做标准化：["zscore", "mean0", "none"]
        adv_clip: float = 5.0,            # |A| 的裁剪阈值；防止梯度爆炸
        kl_coef: float = 0.0,             # 额外 KL 惩罚（通常设 0；留作实验）
        ema_baseline_momentum: float = 0.9,  # 移动基线（序列奖励）的动量
        mb_size_logits: int = 0,          # actor/ref logits micro-batch（0 表示不切）
        device: str = "cuda",
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.optimizer_actor = optimizer_actor

        self.beta = float(beta)
        self.adv_norm = adv_norm
        self.adv_clip = float(adv_clip)
        self.kl_coef = float(kl_coef)
        self.ema_m = float(ema_baseline_momentum)
        self.ema_baseline = None  # 运行时更新，用作“序列级奖励”的全局基线

        self.mb_size_logits = int(mb_size_logits)
        self.device = torch.device(device)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler_actor = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 兜底 pad/eos
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50256)

    # ------------------- 逐样本生成（避免 pad 作为上下文） -------------------
    @torch.no_grad()
    def generate_samples(self, inputs: Tuple[torch.Tensor, torch.Tensor], max_length: int, max_new_tokens: int) -> List[Samples]:
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        block_size = self.actor.config.block_size
        B = input_ids.size(0)

        gens = []
        for i in range(B):
            p_len = int(attention_mask[i].sum().item())
            p_len = min(p_len, block_size - 1)
            if p_len <= 0:
                continue
            prompt = input_ids[i:i+1, :p_len]
            room = max(1, min(max_new_tokens, block_size - p_len - 1))

            out = self.actor.generate(
                idx=prompt,
                max_new_tokens=room,
                temperature=1.0,
                top_k=None,
                eos_token_id=self.eos_token_id
            )
            full = out[0]
            resp = full[p_len:]
            gens.append({
                "prompt_ids": prompt[0],
                "full_ids": full,
                "response_ids": resp,
            })

        samples = build_samples_from_generations(
            gens=gens, block_size=block_size, pad_to_multiple_of=8, device=self.device
        )
        return [samples]

    # ------------------- 经验评估：得到优势 A（序列级→token 级广播） -------------------
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples, debug: bool = False):
        seqs = samples.seqs.to(self.device)                 # [B, T]
        attn_mask = samples.attention_mask.to(self.device)  # [B, T]
        act_mask = samples.action_mask.to(self.device)      # [B, T]
        num_actions = samples.num_actions.to(self.device)   # [B]

        B, T = seqs.size()
        if num_actions.sum().item() == 0:
            empty = Experience(
                seqs=seqs,
                action_log_probs=torch.zeros((B, 0), device=self.device),
                values=torch.zeros((B, 0), device=self.device),
                returns=torch.zeros((B, 0), device=self.device),
                advantages=torch.zeros((B, 0), device=self.device),
                attention_mask=attn_mask[:, 1:],
                action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
                reward=torch.zeros((B, 1), device=self.device),
                num_actions=num_actions,
                kl=torch.zeros((B, 0), device=self.device),
            )
            return [empty], 0.0, 0.0

        # 1) actor/ref token logprob（target=seqs[:,1:]），只在 response 段使用
        actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
            actor=self.actor, ref=self.ref,
            seqs=seqs, action_mask=act_mask,
            device_type=self.device.type, ptdtype=seqs.dtype,
            micro_batch_size=self.mb_size_logits,
        )  # [B, T-1]
        diff_lp = actor_lp - ref_lp                 # [B, T-1]
        avg_kl = masked_mean(diff_lp, mask_tgt.float()).item()

        # 2) 奖励模型，得到“序列级奖励”
        ids_cpu = seqs.detach().cpu().tolist()
        if hasattr(self.actor_tokenizer, "batch_decode"):
            texts = self.actor_tokenizer.batch_decode(ids_cpu)
        else:
            texts = [self.actor_tokenizer.decode(ids) for ids in ids_cpu]
        texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

        rinp = self.reward_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=getattr(self.reward_tokenizer, "model_max_length", 2048)
        )
        rinp = {k: v.to(self.device) for k, v in rinp.items()}
        rout = self.reward_model(**rinp)
        reward_scores = rout.logits[:, 0]  # [B]
        avg_reward = reward_scores.mean().item()

        # 3) 基线与优势（序列级）
        with torch.no_grad():
            if self.ema_baseline is None:
                self.ema_baseline = reward_scores.mean()
            else:
                self.ema_baseline = self.ema_m * self.ema_baseline + (1 - self.ema_m) * reward_scores.mean()

        adv_seq = reward_scores - self.ema_baseline  # [B]

        # 归一化/裁剪
        if self.adv_norm == "zscore":
            mean = adv_seq.mean()
            std = adv_seq.std().clamp_min(1e-8)
            adv_seq = (adv_seq - mean) / std
        elif self.adv_norm == "mean0":
            adv_seq = adv_seq - adv_seq.mean()
        # else: "none" -> 不处理

        if self.adv_clip is not None and self.adv_clip > 0:
            adv_seq = adv_seq.clamp(-self.adv_clip, self.adv_clip)

        # 广播到 token 维（只在 response 段有效，其他位置乘 0）
        advantages = (adv_seq.unsqueeze(1) * mask_tgt.float())  # [B, T-1]

        # DAPO 的监督目标：目标对数几率差 target = β * A
        target_delta = self.beta * advantages  # [B, T-1]

        exp = Experience(
            seqs=seqs,                               # [B, T]
            action_log_probs=diff_lp,                # 旧的差分（不是必须，但保留便于日志）
            values=torch.zeros_like(diff_lp),        # 无 critic
            returns=torch.zeros_like(diff_lp),
            advantages=target_delta,                 # 直接存放“监督目标 βA”
            attention_mask=attn_mask[:, 1:],         # [B, T-1]
            action_mask=mask_tgt,                    # [B, T-1]
            reward=reward_scores.unsqueeze(1),       # [B, 1]
            num_actions=num_actions,                 # [B]
            kl=diff_lp * mask_tgt,                   # [B, T-1]
        )
        return [exp], float(avg_kl), float(avg_reward)

    # ------------------- 训练一步：回归 (logπ - logπ_ref) 到 βA -------------------
    def train_on_experience(self, experience: Experience, use_token_entropy: bool = False):
        self.actor.train()

        with amp.autocast(device_type='cuda', enabled=self.use_amp):
            # 当前策略的 token logprob（target=seqs[:,1:]）
            logits_all = model_all_logits(
                model=self.actor, seqs=experience.seqs,
                device_type=self.device.type, ptdtype=experience.seqs.dtype,
                micro_batch_size=self.mb_size_logits
            )  # [B, T, V]
            new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B, T-1]

            # ref 也需要（可选：若想省一次前向，可在 evaluate_experience 缓存；代码简化起见再算一遍）
            ref_logits_all = model_all_logits(
                model=self.ref, seqs=experience.seqs,
                device_type=self.device.type, ptdtype=experience.seqs.dtype,
                micro_batch_size=self.mb_size_logits
            )
            ref_lp_all = token_logprobs_from_logits(ref_logits_all, experience.seqs)  # [B, T-1]

            # 预测的对数几率差：δ = logπ - logπ_ref
            delta = (new_lp_all - ref_lp_all)  # [B, T-1]

            # ====== 熵筛选 ======
            mask = experience.action_mask.float()
            if use_token_entropy:
                from .common import apply_entropy_mask
                mask = apply_entropy_mask(logits_all[:, :-1], mask, keep_ratio=0.2).float()
            # ====================

            # 监督目标：βA 已经存放在 experience.advantages
            target = experience.advantages  # [B, T-1]

            # MSE 回归（只在 response 段）
            mse = (delta - target).pow(2)
            denom = mask.sum(dim=1).clamp_min(1)
            loss_reg = ((mse * mask).sum(dim=1) / denom).mean()

            # 可选：额外的 KL 惩罚（对 token 级差分做 L2）
            if self.kl_coef != 0.0:
                loss_kl = ((delta.pow(2) * mask).sum(dim=1) / denom).mean()
                total_loss = loss_reg + self.kl_coef * loss_kl
            else:
                total_loss = loss_reg

        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(total_loss).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        clip_by_global_norm(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        return total_loss
