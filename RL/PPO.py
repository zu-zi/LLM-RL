import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from dataclasses import dataclass
from typing import List, Tuple

from .common import (
    Samples, Experience, ExperienceBuffer,
    normalize_for_reward,
    build_samples_from_generations,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs, compute_values_on_response,
    gae_compute, masked_mean, clip_by_global_norm
)


# -------------------------- Critic --------------------------
class Critic(nn.Module):
    """
    简单的 value 头：接在 actor 的隐藏态上。
    注意：不要仅对 response 片段单独前向 base_model（会丢上下文导致 value 偏差）。
    正确做法：对整条序列取 hidden，然后只在 response 段取 value。
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.n_embd = base_model.config.n_embd
        self.value_head = nn.Linear(self.n_embd, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, C] -> values: [B, T]
        return self.value_head(hidden_states).squeeze(-1)


# ------------------------ PPO Trainer ------------------------
class PPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        critic_model: nn.Module,
        actor_tokenizer,              # GPT2Tok（我们自写）或任意 tokenizer
        reward_tokenizer,             # HF tokenizer（奖励模型）
        optimizer_actor,
        optimizer_critic,
        kl_ctl: float = 0.02,         # KL 系数
        clip_reward: float = 5.0,     # 奖励裁剪
        gamma: float = 1.0,           # RLHF 加折扣一般用 1.0
        lambd: float = 0.95,          # GAE 参数
        mb_size_logits: int = 0,      # actor/ref logits 计算的 micro-batch（0=不切）
        mb_size_values: int = 0,      # critic values 计算的 micro-batch（0=不切）
        device: str = "cuda",
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.critic = critic_model.to(device)

        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer

        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic

        self.kl_ctl = kl_ctl
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.lambd = lambd

        self.mb_size_logits = mb_size_logits
        self.mb_size_values = mb_size_values

        self.device = torch.device(device)
        self.use_amp = (self.device.type == "cuda")
        self.scaler_actor  = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.scaler_critic = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 尝试读取 pad/eos；GPT2Tok 没有 batch_decode，用兜底
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50256)  # gpt2 eos

    # ------------------- 逐样本生成（避免 pad 干扰） -------------------
    @torch.no_grad()
    def generate_samples(self, inputs: Tuple[torch.Tensor, torch.Tensor], max_length: int, max_new_tokens: int) -> List[Samples]:
        """
        inputs: (input_ids, attention_mask)，来自固定 prompts bin
        - 逐样本生成，保证每条的 prompt_len 精确生效，且不把pad当上下文
        - 返回 [Samples]（与你原来接口对齐）
        """
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        B = input_ids.size(0)
        block_size = self.actor.config.block_size

        gens = []
        for i in range(B):
            # 精确的 prompt 长度
            p_len = int(attention_mask[i].sum().item())
            p_len = min(p_len, block_size - 1)  # 给 eos 预留
            if p_len <= 0:
                continue
            prompt = input_ids[i:i+1, :p_len]   # [1, T_prompt]
            room = max(1, min(max_new_tokens, block_size - p_len - 1))

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

        # 对齐到 BxT 的 Samples（右 pad，构造 action_mask/attention_mask/num_actions...）
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
        - 计算 actor/ref token logprob（target=seqs[:,1:]），仅在 response 段聚合
        - 调用奖励模型得到样本奖励（标量），均匀分配到 response token
        - 计算 critic 值函数、GAE
        """
        seqs = samples.seqs.to(self.device)                 # [B, T]
        attn_mask = samples.attention_mask.to(self.device)  # [B, T]
        act_mask = samples.action_mask.to(self.device)      # [B, T]
        num_actions = samples.num_actions.to(self.device)   # [B]

        B, T = seqs.size()
        if num_actions.sum().item() == 0:
            empty_exp = Experience(
                seqs=seqs,
                action_log_probs=torch.zeros((B, 0), device=self.device),
                values=torch.zeros((B, 0), device=self.device),
                returns=torch.zeros((B, 0), device=self.device),
                advantages=torch.zeros((B, 0), device=self.device),
                attention_mask=attn_mask,
                action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
                reward=torch.zeros((B, 1), device=self.device),
                num_actions=num_actions,
                kl=torch.zeros((B, 0), device=self.device),
            )
            return [empty_exp], 0.0, 0.0

        # ---- 1) actor/ref 的 logprob（target=seqs[:,1:]） ----
        actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
            actor=self.actor, ref=self.ref,
            seqs=seqs, action_mask=act_mask,
            device_type=self.device.type, ptdtype=seqs.dtype,
            micro_batch_size=self.mb_size_logits,
        )  # 都是 [B, T-1]

        # 取响应 token 对应的后缀（用 mask 按位筛，不再靠负索引拼猜）
        # 直接保留全长 + 掩码，后面 loss 都会乘 mask
        kl_per_token = (actor_lp - ref_lp) * mask_tgt

        # ---- 2) 奖励模型（标量），均匀分到 response token ----
        # 健壮的 batch 解码
        ids_cpu = seqs.detach().cpu().tolist()
        if hasattr(self.actor_tokenizer, "batch_decode"):
            texts = self.actor_tokenizer.batch_decode(ids_cpu)
        else:
            texts = [self.actor_tokenizer.decode(ids) for ids in ids_cpu]
        texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

        r_inputs = self.reward_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=getattr(self.reward_tokenizer, "model_max_length", 2048)
        )
        r_inputs = {k: v.to(self.device) for k, v in r_inputs.items()}
        r_out = self.reward_model(**r_inputs)
        reward_scores = r_out.logits[:, 0]  # [B]

        # token 奖励：-kl + 标量奖励/len(response)
        # 先做 -kl（逐 token），再把标量 reward 均匀摊到 response 位置
        rewards_t = -self.kl_ctl * (actor_lp - ref_lp) * mask_tgt                 # [B, T-1]
        resp_len = mask_tgt.sum(dim=1).clamp_min(1)                               # [B]
        rewards_t = rewards_t + (reward_scores.clamp(-self.clip_reward, self.clip_reward) / resp_len).unsqueeze(-1) * mask_tgt

        # ---- 3) critic values & GAE（只在 response 段） ----
        # 这里直接用 actor 的隐藏态 + critic（见 common.forward_values_via_actor）
        from .common import forward_values_via_actor  # 避免循环 import 顶层导入
        values_full = forward_values_via_actor(
            model=self.actor, critic=self.critic,
            seqs=seqs, device_type=self.device.type, ptdtype=seqs.dtype,
            micro_batch_size=self.mb_size_values,
        )  # [B, T]

        # 目标是预测 seqs[:,1:]，所以把 values 对齐到 target 维：values[:,1:]
        values_t = values_full[:, 1:]                # [B, T-1]
        mask_time = mask_tgt                         # [B, T-1]

        # GAE（rewards/values/mask 都是 [B, T-1]）
        returns_t, adv_t = gae_compute(
            values=values_t, rewards=rewards_t, mask_time=mask_time,
            gamma=self.gamma, lam=self.lambd, use_last_as_terminal=True
        )

        # scalar 监控
        avg_kl = masked_mean(actor_lp - ref_lp, mask_tgt.float()).item()
        avg_reward = reward_scores.mean().item()

        # 打包 Experience（保持 token 维是 T-1；下游 loss 直接乘 mask）
        exp = Experience(
            seqs=seqs,                               # [B, T]
            action_log_probs=actor_lp,               # [B, T-1]
            values=values_t,                         # [B, T-1]
            returns=returns_t,                       # [B, T-1]
            advantages=adv_t,                        # [B, T-1]
            attention_mask=attn_mask[:, 1:],         # [B, T-1]（与 target 对齐）
            action_mask=mask_tgt,                    # [B, T-1]
            reward=reward_scores.unsqueeze(1),       # [B, 1]
            num_actions=num_actions,                 # [B]
            kl=(actor_lp - ref_lp) * mask_tgt,       # [B, T-1]
        )
        return [exp], float(avg_kl), float(avg_reward)

    # ------------------- Loss -------------------
    @staticmethod
    def _policy_loss(new_logp, old_logp, advantages, mask, clip_eps=0.2):
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)  # maximize
        denom = mask.sum(dim=1).clamp_min(1)
        return ((loss * mask).sum(dim=1) / denom).mean()

    @staticmethod
    def _value_loss(values, old_values, returns, mask, clip_eps=0.2):
        old_values = old_values.detach()
        clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (returns - values).pow(2)
        surr2 = (returns - clipped).pow(2)
        loss = torch.max(surr1, surr2)
        denom = mask.sum(dim=1).clamp_min(1)
        return ((loss * mask).sum(dim=1) / denom).mean()

    # ------------------- 训练一步 -------------------
    def train_on_experience(self, experience: Experience, use_token_entropy: bool = False):
        self.actor.train()
        self.critic.train()

        # ------- Actor update -------
        with amp.autocast(device_type='cuda', enabled=self.use_amp):
            # 重新计算当前策略的 logprob（target=seqs[:,1:]）
            logits_all = model_all_logits(
                model=self.actor, seqs=experience.seqs,
                device_type=self.device.type, ptdtype=experience.seqs.dtype,
                micro_batch_size=self.mb_size_logits
            )  # [B, T, V]

            new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B, T-1]

            # ====== 新增：熵筛选 mask ======
            mask = experience.action_mask.float()
            if use_token_entropy:
                from .common import apply_entropy_mask
                mask = apply_entropy_mask(logits_all[:, :-1], mask, keep_ratio=0.2).float()
            # ==============================

            # policy loss 仅在 response 段
            p_loss = self._policy_loss(
                new_logp=new_lp_all,
                old_logp=experience.action_log_probs,
                advantages=experience.advantages,
                mask=mask,
                clip_eps=0.2
            )

        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(p_loss).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        clip_by_global_norm(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        # ------- Critic update -------
        with amp.autocast(device_type='cuda', enabled=self.use_amp):
            from .common import forward_values_via_actor
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=experience.seqs, device_type=self.device.type, ptdtype=experience.seqs.dtype,
                micro_batch_size=self.mb_size_values
            )  # [B, T]
            values_t = values_full[:, 1:]  # [B, T-1]

            v_loss = self._value_loss(
                values=values_t,
                old_values=experience.values,
                returns=experience.returns,
                mask=experience.action_mask.float(),  # critic 仍然用全 mask
                clip_eps=0.2
            )

        self.optimizer_critic.zero_grad(set_to_none=True)
        self.scaler_critic.scale(v_loss).backward()
        self.scaler_critic.unscale_(self.optimizer_critic)
        clip_by_global_norm(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.optimizer_critic)
        self.scaler_critic.update()

        return p_loss, v_loss

