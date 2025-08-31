# RL/GRPO.py

import torch
import torch.nn.functional as F
from torch import amp
from typing import List, Tuple

from .common import (
    Samples, Experience, normalize_for_reward,
    build_samples_from_generations,
    compute_actor_ref_logprobs,
    model_all_logits, token_logprobs_from_logits,
    masked_mean, clip_by_global_norm,
)

class GRPOTrainer:
    """
    Group Relative Policy Optimization
    - 每个 prompt 生成 group_size 个候选，使用组内相对奖励作为优势（无 critic）
    - 只在 response 段计算 logprob / loss
    - 显存友好：支持 logits 前向 micro-batch
    """
    def __init__(
        self,
        actor_model,
        ref_model,
        reward_model,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        group_size: int = 4,
        kl_coef: float = 0.0,          # 常用 0；保留接口以便实验
        clip_reward: float = 5.0,
        mb_size_logits: int = 0,       # actor/ref logits 的 micro-batch（0 表示不切）
        device: str = "cuda",
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        self.reward_model = reward_model.to(device)

        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.optimizer_actor = optimizer_actor

        self.group_size = int(group_size)
        self.kl_coef = float(kl_coef)
        self.clip_reward = float(clip_reward)
        self.mb_size_logits = int(mb_size_logits)

        self.device = torch.device(device)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler_actor = amp.GradScaler(enabled=self.use_amp)

        # 兜底 pad/eos
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50256)

    # ------------------- 逐样本逐候选生成 -------------------
    @torch.no_grad()
    def generate_samples(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        max_length: int,
        max_new_tokens: int
    ) -> List[Samples]:
        """
        inputs: (input_ids, attention_mask) 来自固定 prompts
        - 对于 batch 内每条 prompt，生成 group_size 个候选
        - 逐条生成，避免把 pad 当上下文
        - 返回 [Samples]（B*G 拼在一起）
        """
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        block_size = self.actor.config.block_size
        B = input_ids.size(0)

        gens = []
        for b in range(B):
            p_len = int(attention_mask[b].sum().item())
            p_len = min(p_len, block_size - 1)  # 预留一个给 eos
            if p_len <= 0:
                continue
            prompt = input_ids[b:b+1, :p_len]  # [1, T_prompt]
            room = max(1, min(max_new_tokens, block_size - p_len - 1))

            for _ in range(self.group_size):
                out = self.actor.generate(
                    idx=prompt,
                    max_new_tokens=room,
                    temperature=1.0,
                    top_k=None,
                    eos_token_id=self.eos_token_id
                )  # [1, T_prompt + T_resp]
                full = out[0]
                resp = full[p_len:]
                gens.append({
                    "prompt_ids": prompt[0],
                    "full_ids": full,
                    "response_ids": resp,
                })

        samples = build_samples_from_generations(
            gens=gens,
            block_size=block_size,
            pad_to_multiple_of=8,
            device=self.device,
        )
        return [samples]

    # ------------------- 经验评估 -------------------
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples, debug: bool = False):
        """
        - actor/ref token logprob（target=seqs[:,1:]），只在 response 段使用
        - 奖励模型得到序列级 reward，按组归一化为优势，广播到 response 段
        - 返回 (experiences, avg_kl, avg_reward)
        """
        seqs        = samples.seqs.to(self.device)           # [B_rep, T]
        attn_mask   = samples.attention_mask.to(self.device) # [B_rep, T]
        act_mask    = samples.action_mask.to(self.device)    # [B_rep, T]
        num_actions = samples.num_actions.to(self.device)    # [B_rep]

        B_rep, T = seqs.size()
        assert B_rep % self.group_size == 0, "B_rep must be divisible by group_size"
        B = B_rep // self.group_size

        if num_actions.sum().item() == 0:
            empty = Experience(
                seqs=seqs,
                action_log_probs=torch.zeros((B_rep, 0), device=self.device),
                values=torch.zeros((B_rep, 0), device=self.device),
                returns=torch.zeros((B_rep, 0), device=self.device),
                advantages=torch.zeros((B_rep, 0), device=self.device),
                attention_mask=attn_mask[:, :0],
                action_mask=torch.zeros((B_rep, 0), device=self.device, dtype=torch.long),
                reward=torch.zeros((B_rep, 1), device=self.device),
                num_actions=num_actions,
                kl=torch.zeros((B_rep, 0), device=self.device),
            )
            return [empty], 0.0, 0.0

        # 1) actor/ref 的 logprob（target=seqs[:,1:]）——显式传入 action_mask
        actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
            actor=self.actor, ref=self.ref,
            seqs=seqs, action_mask=act_mask,
            device_type=self.device.type,
            ptdtype=None,
            micro_batch_size=self.mb_size_logits,
        )  # [B_rep, T-1] * 3

        # 统一裁剪到最短长度 L，防止 off-by-one
        L = min(actor_lp.size(1), ref_lp.size(1), mask_tgt.size(1))
        actor_lp = actor_lp[:, :L]
        ref_lp   = ref_lp[:, :L]
        mask_tgt = mask_tgt[:, :L]

        diff_lp   = actor_lp - ref_lp              # [B_rep, L]
        kl_tokens = diff_lp * mask_tgt             # [B_rep, L]
        avg_kl    = masked_mean(diff_lp, mask_tgt.float()).item()

        # 2) 奖励模型（标量）
        ids_cpu = seqs.detach().cpu().tolist()
        texts = self.actor_tokenizer.batch_decode(ids_cpu) if hasattr(self.actor_tokenizer, "batch_decode") \
                else [self.actor_tokenizer.decode(ids) for ids in ids_cpu]
        texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

        r_inputs = self.reward_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=getattr(self.reward_tokenizer, "model_max_length", 2048)
        )
        r_inputs = {k: v.to(self.device) for k, v in r_inputs.items()}
        r_out = self.reward_model(**r_inputs)
        reward_scores = r_out.logits[:, 0]  # [B_rep]
        avg_reward = reward_scores.mean().item()

        # 3) 组内归一化优势（序列级 -> token 广播）
        rs = reward_scores.view(B, self.group_size)
        rs_mean = rs.mean(dim=1, keepdim=True)
        rs_std  = rs.std(dim=1, keepdim=True).clamp_min(1e-8)
        rs_norm = ((rs - rs_mean) / rs_std).view(B_rep)          # [B_rep]

        advantages = rs_norm.unsqueeze(1) * mask_tgt.float()     # [B_rep, L]
        if self.kl_coef != 0.0:
            advantages = advantages - self.kl_coef * kl_tokens

        exp = Experience(
            seqs=seqs,                               # [B_rep, T]
            action_log_probs=actor_lp,               # [B_rep, L]（旧 logp）
            values=torch.zeros_like(actor_lp),       # 无 critic
            returns=torch.zeros_like(actor_lp),
            advantages=advantages,                   # [B_rep, L]
            attention_mask=attn_mask[:, 1:][:, :L],  # [B_rep, L]
            action_mask=mask_tgt,                    # [B_rep, L]
            reward=reward_scores.unsqueeze(1),       # [B_rep, 1]
            num_actions=num_actions,                 # [B_rep]
            kl=kl_tokens,                            # [B_rep, L]
        )
        return [exp], float(avg_kl), float(avg_reward)

    # ------------------- PPO 风格策略损失 -------------------
    @staticmethod
    def _policy_loss(new_logp, old_logp, advantages, mask, clip_eps: float = 0.2):
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        denom = mask.sum(dim=1).clamp_min(1)
        return ((loss * mask).sum(dim=1) / denom).mean()

    # ------------------- 训练一步 -------------------
    def train_on_experience(self, experience: Experience, use_token_entropy: bool = False):
        self.actor.train()

        with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            # 重新前向拿当前策略的 token logprob（target=seqs[:,1:]）
            logits_all = model_all_logits(
                model=self.actor, seqs=experience.seqs,
                device_type=self.device.type,
                ptdtype=None,
                micro_batch_size=self.mb_size_logits
            )  # [B_rep, T, V]
            new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B_rep, T-1]

            # 与缓存的旧量对齐到最短长度 L，彻底避免 510/509
            L = min(
                new_lp_all.size(1),
                experience.action_log_probs.size(1),
                experience.advantages.size(1),
                experience.action_mask.size(1),
            )
            new_lp = new_lp_all[:, :L]
            old_lp = experience.action_log_probs[:, :L]
            adv    = experience.advantages[:, :L]
            mask   = experience.action_mask[:, :L].float()

            # 可选：熵筛选（在 L 范围内）
            if use_token_entropy:
                from .common import apply_entropy_mask
                mask = apply_entropy_mask(logits_all[:, :-1][:, :L], mask, keep_ratio=0.2).float()

            p_loss = self._policy_loss(
                new_logp=new_lp,
                old_logp=old_lp,
                advantages=adv,
                mask=mask,
                clip_eps=0.2
            )

        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(p_loss).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        clip_by_global_norm(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        return p_loss
