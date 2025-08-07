import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List
from dataclasses import dataclass

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int
    response_length: torch.Tensor
    total_length: torch.Tensor

@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    reward: torch.Tensor
    num_actions: int
    kl: torch.Tensor

class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []

    def append(self, experiences: List[Experience]):
        self.buffer.extend(experiences)
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[-self.limit:]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        hidden = self.base_model(input_ids, attention_mask=attention_mask).last_hidden_state
        value = self.value_head(hidden).squeeze(-1)[:, -num_actions:]
        return value

class PPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        reward_model,
        critic_model,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        optimizer_critic,
        kl_ctl=0.1,
        clip_reward=0.2,
        gamma=0.99,
        lambd=0.95,
        device="cuda"
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
        self.device = device

    def generate_samples(self, prompts: List[str], max_length, max_new_tokens):
        model = self.actor.eval()
        inputs = self.actor_tokenizer(prompts, return_tensors='pt', padding='max_length',
                                      truncation=True, max_length=max_length).to(self.device)
        prompt_len = (inputs['attention_mask'] == 1).sum(1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.actor_tokenizer.eos_token_id,
                pad_token_id=self.actor_tokenizer.pad_token_id,
            )

        seqs = outputs
        attention_mask = (seqs != self.actor_tokenizer.pad_token_id).long()
        action_mask = torch.zeros_like(seqs)
        for i in range(seqs.size(0)):
            action_mask[i, prompt_len[i]:] = (seqs[i, prompt_len[i]:] != self.actor_tokenizer.eos_token_id)

        return [Samples(seqs, attention_mask, action_mask, action_mask.size(1) - prompt_len.min().item(),
                        action_mask.sum(-1), attention_mask.sum(-1))]

    def compute_kl(self, log_probs, ref_log_probs, action_mask):
        kl = (log_probs - ref_log_probs) * action_mask
        return kl

    def compute_rewards(self, kl, reward_scores, action_mask):
        rewards = -self.kl_ctl * kl
        ends = action_mask.sum(1)
        reward_clip = torch.clamp(reward_scores, -self.clip_reward, self.clip_reward)
        for j in range(rewards.size(0)):
            if ends[j] > 0:
                rewards[j, ends[j] - 1] += reward_clip[j]
        return rewards

    def get_advantages(self, values, rewards, action_mask):
        lastgaelam = 0
        advantages = []
        T = rewards.size(1)
        for t in reversed(range(T)):
            nextval = values[:, t+1] if t < T - 1 else 0
            delta = rewards[:, t] + self.gamma * nextval - values[:, t]
            lastgaelam = delta + self.gamma * self.lambd * lastgaelam
            advantages.insert(0, lastgaelam)

        advantages = torch.stack(advantages, dim=1)
        advantages = advantages * action_mask
        returns = (advantages + values) * action_mask
        return advantages.detach(), returns.detach()

    def evaluate_experience(self, samples: Samples):
        seqs, attn_mask, act_mask, num_actions = samples.seqs, samples.attention_mask, samples.action_mask, samples.num_actions
        with torch.no_grad():
            logits = self.actor(seqs, attention_mask=attn_mask).logits
            ref_logits = self.ref(seqs, attention_mask=attn_mask).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)
            indices = seqs[:, 1:].unsqueeze(-1)
            log_probs = log_probs.gather(-1, indices).squeeze(-1)[:, -num_actions:]
            ref_log_probs = ref_log_probs.gather(-1, indices).squeeze(-1)[:, -num_actions:]

            kl = self.compute_kl(log_probs, ref_log_probs, act_mask)
            values = self.critic(seqs, attn_mask, num_actions)
            texts = self.actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            reward_inputs = self.reward_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            reward_scores = self.reward_model(**reward_inputs).logits[:, 0]
            rewards = self.compute_rewards(kl, reward_scores, act_mask)
            advantages, returns = self.get_advantages(values, rewards, act_mask)

        return [
            Experience(
                seqs=seqs,
                action_log_probs=log_probs,
                values=values,
                returns=returns,
                advantages=advantages,
                attention_mask=attn_mask,
                action_mask=act_mask,
                reward=reward_scores.unsqueeze(1),
                num_actions=num_actions,
                kl=kl
            )
        ]

    def policy_loss(self, new_log_probs, old_log_probs, advantages, action_mask, clip_eps=0.2):
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    def value_loss(self, values, old_values, returns, action_mask, clip_eps=0.2):
        old_values = old_values.detach()
        clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (returns - values).pow(2)
        surr2 = (returns - clipped).pow(2)
        loss = torch.max(surr1, surr2)
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    def train_on_experience(self, experience: Experience):
        self.actor.train()
        self.optimizer_actor.zero_grad()
        logits = self.actor(experience.seqs, attention_mask=experience.attention_mask).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        indices = experience.seqs[:, 1:].unsqueeze(-1)
        new_log_probs = log_probs.gather(-1, indices).squeeze(-1)[:, -experience.num_actions:]
        policy_loss = self.policy_loss(new_log_probs, experience.action_log_probs, experience.advantages, experience.action_mask)
        policy_loss.backward()
        self.optimizer_actor.step()

        self.critic.train()
        self.optimizer_critic.zero_grad()
        values = self.critic(experience.seqs, experience.attention_mask, experience.num_actions)
        value_loss = self.value_loss(values, experience.values, experience.returns, experience.action_mask)
        value_loss.backward()
        self.optimizer_critic.step()

        return policy_loss.item(), value_loss.item()
