# RL/PPO.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from typing import List, Tuple

from .common import (
    Samples, Experience,
    normalize_for_reward,
    build_samples_from_generations,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs,
    gae_compute, masked_mean, clip_by_global_norm,
    forward_values_via_actor,
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
        clip_reward: float = 5.0,     # 奖励裁剪（对序列级奖励裁剪）
        gamma: float = 1.0,           # RLHF 常用 1.0
        lambd: float = 0.95,          # GAE 参数
        mb_size_logits: int = 0,      # actor/ref logits 前向的 micro-batch（0=不切）
        mb_size_values: int = 0,      # critic values 前向的 micro-batch（0=不切）
        device: str = "cuda",
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        # 关键：奖励模型保持 CPU，避免与 r_inputs（CPU）不一致
        self.reward_model = reward_model
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

        # pad/eos（兼容自定义 GPT2Tok 与 HF tokenizer）
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id",
                             getattr(actor_tokenizer, "eos_id", 50256))

    # --------- 小工具：安全解码（兼容自定义/HF tokenizer） ---------
    @staticmethod
    def _safe_decode(tok, ids_tensor) -> str:
        ids = ids_tensor.detach().cpu().tolist()
        # 优先 decode(skip_special_tokens=False)
        if hasattr(tok, "decode"):
            try:
                return tok.decode(ids, skip_special_tokens=False)
            except TypeError:
                try:
                    return tok.decode(ids)
                except Exception:
                    pass
        # 退路 batch_decode
        if hasattr(tok, "batch_decode"):
            try:
                return tok.batch_decode([ids], skip_special_tokens=False)[0]
            except TypeError:
                try:
                    return tok.batch_decode([ids])[0]
                except Exception:
                    pass
        # 退路 tokens->join
        if hasattr(tok, "convert_ids_to_tokens"):
            try:
                return " ".join(tok.convert_ids_to_tokens(ids))
            except Exception:
                pass
        # 最后兜底：id 串
        return " ".join(str(x) for x in ids)

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
        - 奖励模型得到样本标量奖励（序列级），聚合到 response 的最后一个 token
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
        mask_tgt = mask_tgt[:, :L]  # 这是 response 段在 target 对齐维度上的掩码（B, L）

        # ---- 2) 奖励模型（标量），只用有效 token（去除右侧 pad）----
        valid_lens = attn_mask.sum(dim=1).tolist()  # [B]
        texts = []
        for i in range(B):
            Li = int(valid_lens[i])
            toks = seqs[i, :Li]
            texts.append(self._safe_decode(self.actor_tokenizer, toks))

        texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

        r_inputs = self.reward_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=getattr(self.reward_tokenizer, "model_max_length", 2048)
        )
        # 奖励模型保持在 CPU，前向也在 CPU；只把结果搬回 GPU
        r_out = self.reward_model(**r_inputs)
        reward_scores = r_out.logits[:, 0].to(self.device)  # [B]

        # ---- 2.1) 将序列级奖励聚合到 response 最后一个 token；KL 也聚合 ----
        r_center = (reward_scores - reward_scores.mean()).clamp(-self.clip_reward, self.clip_reward)  # [B]

        # 用“平均 KL”而不是“总和 KL”，并把 KL 作为正数直觉
        kl_t    = (actor_lp - ref_lp) * mask_tgt                       # [B, L]
        na      = mask_tgt.sum(dim=1).clamp_min(1)                     # [B] 有效 token 数
        # 先得到负的均值（logπ-logpref 的掩码均值），再取相反数变成正的 KL
        kl_mean = -(kl_t.sum(dim=1) / na)                              # [B] 正的 KL 平均
        # kl_mean = kl_mean.clamp(0.0, 2.0)                              # 夹紧防爆
        shaped  = (r_center - self.kl_ctl * kl_mean).clamp(-self.clip_reward, self.clip_reward)  # [B]

        rewards_t = torch.zeros_like(actor_lp)                          # [B, L]
        if L > 0:
            has_act = (na > 0)
            if has_act.any():
                arange_L = torch.arange(L, device=self.device).view(1, -1)
                last_idx = (mask_tgt * arange_L).amax(dim=1)            # [B]
                rows = torch.nonzero(has_act, as_tuple=False).squeeze(-1)
                rewards_t[rows, last_idx[rows]] = shaped[rows]

        # ---- 3) critic values & GAE（只在 response 段）----
        with torch.no_grad():
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=seqs, device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_values,
                detach_hidden=True,
            )  # [B, T]
        values_t = values_full[:, 1:][:, :L]  # [B, L] 与 target 维对齐

        returns_t, adv_t = gae_compute(
            values=values_t, rewards=rewards_t, mask_time=mask_tgt,
            gamma=self.gamma, lam=self.lambd, use_last_as_terminal=True
        )
        # 仅对 response 段做优势标准化
        denom = mask_tgt.sum().clamp_min(1)
        mean  = (adv_t * mask_tgt).sum() / denom
        var   = ((adv_t - mean)**2 * mask_tgt).sum() / denom
        adv_t = (adv_t - mean) / torch.sqrt(var + 1e-8)

        # 返回“正的 KL”用于日志直观判断
        avg_kl = float(kl_mean.mean().item())
        avg_reward = reward_scores.mean().item()  # 记录 raw RM reward 的平均值

        exp = Experience(
            seqs=seqs,                              # [B, T]
            action_log_probs=actor_lp,              # [B, L]（old logp）
            values=values_t,                        # [B, L]
            returns=returns_t,                      # [B, L]
            advantages=adv_t,                       # [B, L]
            attention_mask=attn_mask[:, 1:][:, :L], # [B, L]
            action_mask=mask_tgt,                   # [B, L]
            reward=reward_scores.unsqueeze(1),      # [B, 1]（便于日志）
            num_actions=num_actions,                # [B]
            kl=kl_t,                                # [B, L]（未取负，保留原始差值方便调试）
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
            # 关键：不要用 no_grad 掐掉 critic 的梯度；只 detach actor 的隐藏态
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=experience.seqs, device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_values,
                detach_hidden=True,   # 阻断 critic→actor 的反传，critic 自身仍可学习
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
