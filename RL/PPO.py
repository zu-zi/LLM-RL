# RL/PPO.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from typing import List, Tuple

from .common import (
    Samples, Experience, ExperienceBuffer,
    normalize_for_reward,
    build_samples_from_generations,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs, compute_values_on_response,
    gae_compute, masked_mean, clip_by_global_norm,
)

# -------------------------- Critic --------------------------
class Critic(nn.Module):
    """
    简单 value 头：基于 actor 的隐藏态输出每个 token 的 V(s_t)。
    注意不要只对 response 片段单独前向 actor，否则会丢上下文。
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.n_embd = base_model.config.n_embd
        self.value_head = nn.Linear(self.n_embd, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, C]  ->  values: [B, T]
        return self.value_head(hidden_states).squeeze(-1)


# ------------------------ PPO Trainer ------------------------
class PPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        critic_model: nn.Module,
        actor_tokenizer,              # GPT2Tok（自写）或任意 tokenizer
        reward_tokenizer,             # HF tokenizer（奖励模型）
        optimizer_actor,
        optimizer_critic,
        kl_ctl: float = 0.02,         # KL 系数
        clip_reward: float = 5.0,     # 奖励裁剪
        gamma: float = 1.0,           # RLHF 常用 1.0
        lambd: float = 0.95,          # GAE 参数
        mb_size_logits: int = 0,      # actor/ref logits 前向的 micro-batch（0=不切）
        mb_size_values: int = 0,      # critic values 前向的 micro-batch（0=不切）
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
        self.scaler_actor  = amp.GradScaler(enabled=self.use_amp)
        self.scaler_critic = amp.GradScaler(enabled=self.use_amp)

        # pad/eos
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50256)  # gpt2 eos

    # ------------------- 逐样本生成（避免 pad 干扰） -------------------
    @torch.no_grad()
    def generate_samples(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        max_length: int,
        max_new_tokens: int
    ) -> List[Samples]:
        """
        inputs: (input_ids, attention_mask)（来自固定 prompts bin）
        - 精准按各自 prompt_len 生成；不把 pad 当上下文
        - 返回 [Samples]（与上游调用对齐）
        """
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        B = input_ids.size(0)
        block_size = self.actor.config.block_size

        gens = []
        for i in range(B):
            p_len = int(attention_mask[i].sum().item())
            p_len = min(p_len, block_size - 1)  # 预留 EOS
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
        - 计算 actor/ref 的 token logprob（target=seqs[:,1:]）
        - 奖励模型得到样本标量奖励，均匀分配到 response token
        - 用 critic 计算 values，并以 GAE 得到 returns/advantages
        """
        seqs        = samples.seqs.to(self.device)           # [B, T]
        attn_mask   = samples.attention_mask.to(self.device) # [B, T]
        act_mask    = samples.action_mask.to(self.device)    # [B, T]
        num_actions = samples.num_actions.to(self.device)    # [B]

        B, T = seqs.size()
        if num_actions.sum().item() == 0:
            empty = Experience(
                seqs=seqs,
                action_log_probs=torch.zeros((B, 0), device=self.device),
                values=torch.zeros((B, 0), device=self.device),
                returns=torch.zeros((B, 0), device=self.device),
                advantages=torch.zeros((B, 0), device=self.device),
                attention_mask=attn_mask[:, :0],
                action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
                reward=torch.zeros((B, 1), device=self.device),
                num_actions=num_actions,
                kl=torch.zeros((B, 0), device=self.device),
            )
            return [empty], 0.0, 0.0

        # ---- 1) actor/ref token logprob（target=seqs[:,1:]）----
        # 显式传入 action_mask，返回 mask_tgt = action_mask[:, 1:]
        actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
            actor=self.actor, ref=self.ref,
            seqs=seqs, action_mask=act_mask,
            device_type=self.device.type,
            ptdtype=None,
            micro_batch_size=self.mb_size_logits,
        )  # [B, T-1], [B, T-1], [B, T-1]

        # 统一裁到最短长度，防止任何 off-by-one
        L = min(actor_lp.size(1), ref_lp.size(1), mask_tgt.size(1))
        actor_lp = actor_lp[:, :L]
        ref_lp   = ref_lp[:, :L]
        mask_tgt = mask_tgt[:, :L]

        # ---- 2) 奖励模型（标量），均匀分配到 response 段 ----
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
        reward_scores = r_out.logits[:, 0]  # [B]

        # NEW: 先对序列级奖励做去均值，再均匀分配到 response token
        resp_len  = mask_tgt.sum(dim=1).clamp_min(1)  # [B]
        r_center  = reward_scores - reward_scores.mean()
        per_token = (r_center.clamp(-self.clip_reward, self.clip_reward) / resp_len).unsqueeze(1) * mask_tgt
        rewards_t = -self.kl_ctl * (actor_lp - ref_lp) * mask_tgt + per_token

        # ---- 3) critic values & GAE（只在 response 段）----
        from .common import forward_values_via_actor
        values_full = forward_values_via_actor(
            model=self.actor, critic=self.critic,
            seqs=seqs, device_type=self.device.type, ptdtype=None,
            micro_batch_size=self.mb_size_values,
        )  # [B, T]
        values_t = values_full[:, 1:][:, :L]  # [B, L]

        returns_t, adv_t = gae_compute(
            values=values_t, rewards=rewards_t, mask_time=mask_tgt,
            gamma=self.gamma, lam=self.lambd, use_last_as_terminal=True
        )
        # NEW: 仅对 response 段做优势标准化（数值更稳）
        denom = mask_tgt.sum().clamp_min(1)
        mean  = (adv_t * mask_tgt).sum() / denom
        var   = ((adv_t - mean)**2 * mask_tgt).sum() / denom
        adv_t = (adv_t - mean) / torch.sqrt(var + 1e-8)

        avg_kl     = masked_mean(actor_lp - ref_lp, mask_tgt.float()).item()
        avg_reward = reward_scores.mean().item()

        exp = Experience(
            seqs=seqs,                              # [B, T]
            action_log_probs=actor_lp,              # [B, L]
            values=values_t,                        # [B, L]
            returns=returns_t,                      # [B, L]
            advantages=adv_t,                       # [B, L]
            attention_mask=attn_mask[:, 1:][:, :L], # [B, L]
            action_mask=mask_tgt,                   # [B, L]
            reward=reward_scores.unsqueeze(1),      # [B, 1]
            num_actions=num_actions,                # [B]
            kl=(actor_lp - ref_lp) * mask_tgt,      # [B, L]
        )
        return [exp], float(avg_kl), float(avg_reward)


    # ------------------- Loss -------------------
    @staticmethod
    def _policy_loss(new_logp, old_logp, advantages, mask, clip_eps=0.2):
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
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
        with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits_all = model_all_logits(
                model=self.actor, seqs=experience.seqs,
                device_type=self.device.type,
                ptdtype=None,
                micro_batch_size=self.mb_size_logits,
            )  # [B, T, V]

            new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B, T-1]

            # 统一裁到 L，避免与 old_logp/advantages/action_mask 产生 510/509
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

            # 可选：token-entropy 掩码，仅在 response 段筛高熵 token
            if use_token_entropy:
                from .common import apply_entropy_mask
                mask = apply_entropy_mask(logits_all[:, :-1][:, :L], mask, keep_ratio=0.2).float()

            p_loss = self._policy_loss(
                new_logp=new_lp,
                old_logp=old_lp,
                advantages=adv,
                mask=mask,
                clip_eps=0.2,
            )

        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(p_loss).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        clip_by_global_norm(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        # ------- Critic update -------
        with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            from .common import forward_values_via_actor
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=experience.seqs, device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_values,
            )  # [B, T]
            values_t_all = values_full[:, 1:]  # [B, T-1]

            # 与 returns/old_values/action_mask 对齐长度
            Lv = min(
                values_t_all.size(1),
                experience.values.size(1),
                experience.returns.size(1),
                experience.action_mask.size(1),
            )
            values_t = values_t_all[:, :Lv]

            v_loss = self._value_loss(
                values=values_t,
                old_values=experience.values[:, :Lv],
                returns=experience.returns[:, :Lv],
                mask=experience.action_mask[:, :Lv].float(),  # 使用同样裁剪后的 mask
                clip_eps=0.2,
            )

        self.optimizer_critic.zero_grad(set_to_none=True)
        self.scaler_critic.scale(v_loss).backward()
        self.scaler_critic.unscale_(self.optimizer_critic)
        clip_by_global_norm(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.optimizer_critic)
        self.scaler_critic.update()

        return p_loss, v_loss
