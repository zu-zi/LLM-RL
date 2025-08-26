import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from dataclasses import dataclass
from torch import amp
from common import Samples,Experience,ExperienceBuffer,normalize_for_reward

# Group Relative Policy Optimization (无 critic，组内归一化奖励作为优势)
# 接口与 PPOTrainer 保持一致（generate_samples, evaluate_experience, train_on_experience）
class GRPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        reward_model,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        kl_ctl: float = 0.02,
        clip_reward: float = 5.0,
        device: str = "cuda"
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.optimizer_actor = optimizer_actor
        self.kl_ctl = kl_ctl
        self.clip_reward = clip_reward
        self.device = torch.device(device)
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50304)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler_actor = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

    # 复用 PPO 的 generate_samples 签名（实现一致）
    def generate_samples(self, inputs, max_length, max_new_tokens):
        model = self.actor
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        prompt_len = attention_mask.sum(1).long()
        max_new_tokens = int(min(max_new_tokens, model.config.block_size - 1))
        max_prompt_len = model.config.block_size - max_new_tokens
        prompt_len = torch.clamp(prompt_len, max=max_prompt_len)

        with torch.no_grad():
            end = int(prompt_len.max().item()) if prompt_len.numel() > 0 else 1
            outputs = model.generate(
                input_ids[:, :end],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id
            )

        seqs = outputs[:, :model.config.block_size]
        B, T = seqs.size()
        action_mask = torch.zeros_like(seqs, dtype=torch.long, device=self.device)
        for i in range(B):
            s = int(prompt_len[i].item())
            if s >= T:
                s = T - 1
            action_mask[i, s:T] = (seqs[i, s:T] != self.pad_token_id).long()

        new_attention_mask = (seqs != self.pad_token_id).long()
        num_actions = action_mask.sum(dim=1).long()
        response_length = num_actions
        total_length = new_attention_mask.sum(-1)

        return [Samples(
            seqs=seqs,
            attention_mask=new_attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            response_length=response_length,
            total_length=total_length
        )]

    def evaluate_experience(self, samples: Samples, debug: bool = False):
        seqs = samples.seqs.to(self.device)
        attn_mask = samples.attention_mask.to(self.device)
        act_mask_full = samples.action_mask.to(self.device)
        num_actions = samples.num_actions.to(self.device)
        B, L = seqs.size()

        with torch.no_grad():
            out_act = self.actor(seqs, return_all_logits=True)
            logits_all = out_act[0] if isinstance(out_act, tuple) else out_act
            out_ref = self.ref(seqs, return_all_logits=True)
            ref_logits_all = out_ref[0] if isinstance(out_ref, tuple) else out_ref

            log_probs_all = F.log_softmax(logits_all, dim=-1)
            ref_log_probs_all = F.log_softmax(ref_logits_all, dim=-1)

            indices = seqs[:, 1:].unsqueeze(-1)
            logp_next = log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)
            ref_logp_next = ref_log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)

            na_list = [int(x.item()) for x in num_actions]
            max_na = max(na_list) if len(na_list) > 0 else 0
            if max_na == 0:
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
                    kl=torch.zeros((B, 0), device=self.device)
                )
                return [empty_exp], 0.0

            new_logp = logp_next.new_full((B, max_na), 0.0)
            new_ref_logp = new_logp.clone()
            new_act_mask = torch.zeros((B, max_na), dtype=torch.long, device=self.device)
            for i in range(B):
                na = na_list[i]
                if na <= 0:
                    continue
                new_logp[i, :na] = logp_next[i, -na:]
                new_ref_logp[i, :na] = ref_logp_next[i, -na:]
                new_act_mask[i, :na] = act_mask_full[i, -na:]

            # reward model scoring (same as PPO)
            seqs_cpu = seqs.detach().cpu().tolist()
            texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer)
                     for t in self.actor_tokenizer.batch_decode(seqs_cpu)]
            reward_inputs = self.reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                                  max_length=getattr(self.reward_tokenizer, "model_max_length", 2048))
            reward_inputs = {k: v.to(self.device) for k, v in reward_inputs.items()}
            reward_outputs = self.reward_model(**reward_inputs)
            reward_scores = reward_outputs.logits[:, 0].to(self.device)  # (B,)

            # group relative advantage: normalize reward_scores across batch
            rs_mean = reward_scores.mean()
            rs_std = reward_scores.std().clamp_min(1e-8)
            advantages_scalar = (reward_scores - rs_mean) / rs_std  # (B,)
            # expand to token-level advantage aligned with new_act_mask
            advantages = advantages_scalar.unsqueeze(1).expand(-1, max_na) * new_act_mask.float()

            kl = (new_logp - new_ref_logp) * new_act_mask.float()

            if debug:
                for i in range(B):
                    print(f"GRPO Sample {i}: num_actions={num_actions[i].item()}, reward_score={reward_scores[i].item():.4f}")

        exp = Experience(
            seqs=seqs,
            action_log_probs=new_logp,
            values=torch.zeros_like(new_logp),
            returns=torch.zeros_like(new_logp),
            advantages=advantages,
            attention_mask=attn_mask,
            action_mask=new_act_mask,
            reward=reward_scores.unsqueeze(1),
            num_actions=num_actions,
            kl=kl
        )

        avg_kl = (kl * new_act_mask.float()).sum() / new_act_mask.sum().clamp_min(1).float()
        return [exp], float(avg_kl.item())

    def policy_loss(self, new_log_probs, old_log_probs, advantages, action_mask, clip_eps: float = 0.2):
        # still use clipped surrogate objective, but advantages were computed group-wise
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        denom = action_mask.sum(-1).clamp_min(1)
        return ((loss * action_mask).sum(-1) / denom).mean()

    def train_on_experience(self, experience: Experience):
        self.actor.train()
        with amp.autocast(device_type='cuda', enabled=self.use_amp):
            logits, *_ = self.actor(experience.seqs, return_all_logits=True)
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            indices = experience.seqs[:, 1:].unsqueeze(-1)
            gathered_log_probs = log_probs.gather(-1, indices).squeeze(-1)

            new_log_probs_list = []
            for i in range(experience.seqs.size(0)):
                na = experience.num_actions[i].item()
                if na > 0:
                    new_log_probs_list.append(gathered_log_probs[i, -na:])
                else:
                    new_log_probs_list.append(gathered_log_probs.new_empty(0))
            new_log_probs = torch.nn.utils.rnn.pad_sequence(new_log_probs_list, batch_first=True)

            policy_loss = self.policy_loss(
                new_log_probs,
                experience.action_log_probs,
                experience.advantages,
                experience.action_mask
            )

        self.optimizer_actor.zero_grad(set_to_none=True)
        if policy_loss.requires_grad:
            self.scaler_actor.scale(policy_loss).backward()
        else:
            dummy = sum(p.sum() * 0 for p in self.actor.parameters() if p.requires_grad)
            self.scaler_actor.scale(dummy).backward()

        self.scaler_actor.unscale_(self.optimizer_actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        return policy_loss
