import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List
from dataclasses import dataclass
from torch import amp

# Samples = 一次完整文本生成的“样本”
# Experience = 在此基础上，结合强化学习信号封装的训练用经历

#存储一个batch的模型输入和相关信息
@dataclass
class Samples:
    seqs: torch.Tensor#文本输入+生成输出的token序列
    attention_mask: torch.Tensor #用于Transformer计算时屏蔽无效token
    action_mask: torch.Tensor#哪些位置是“可动作”的位置
    num_actions: torch.Tensor  # shape: (batch_size,)，每个样本动作token数 #?Union[int, torch.Tensor]
    response_length: torch.Tensor#回复长度，生成的token数
    total_length: torch.Tensor#整个序列的长度，包含prompt+回复

#PPO训练中保存的每条经历数据
@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor#动作的对数概率
    values: torch.Tensor#critic网络给出的状态值
    returns: torch.Tensor#经过折扣的累积奖励，训练critic用
    advantages: torch.Tensor#表示“该动作相对于平均水平好多少”
    attention_mask: torch.Tensor#模型mask操作
    action_mask: torch.Tensor
    reward: torch.Tensor#该经历对应的奖励值
    num_actions: torch.Tensor  # shape: (batch_size,)
    kl: torch.Tensor#PPO中用来控制策略更新的步长，防止偏离原策略太远

#存储一定数量的Experience对象
class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit#最大缓存大小，先进先出
        self.buffer = []

    def append(self, experiences: List[Experience]):
        self.buffer.extend(experiences)
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[-self.limit:]

    #随机采样一批经验，训练时用
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

def contains_chinese(text):
    # 检查是否含中文字符
    return any('\u4e00' <= c <= '\u9fff' for c in text)

def normalize_for_reward(text, reward_tokenizer=None):
    # 确保GPT-2生成文本适配Qwen分词器
    # 1. 替换EOS
    if reward_tokenizer is not None and hasattr(reward_tokenizer, "eos_token"):
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    
    # 2. 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 3. 中文标点→英文
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    
    # 4. 移除控制字符
    return "".join(c for c in text if c.isprintable())

class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.n_embd, 1)

    def forward(self, input_ids, attention_mask, num_actions):
        """
        只计算动作 token 的 value，显存友好。
        """
        device = input_ids.device
        B, L = input_ids.size()

        # prompt长度 = attention_mask.sum - num_actions
        prompt_lens = attention_mask.sum(dim=1).long() - num_actions
        max_na = num_actions.max().item() if num_actions.numel() > 0 else 0

        if max_na == 0:
            return input_ids.new_zeros((B, 0)), input_ids.new_zeros((B, 0), dtype=torch.long)

        # 存储输出
        values_out = input_ids.new_zeros((B, max_na), dtype=torch.float)
        vmask_out = torch.zeros((B, max_na), dtype=torch.long, device=device)

        for i in range(B):
            na = int(num_actions[i].item())
            if na <= 0:
                continue
            start = int(prompt_lens[i].item())
            end = start + na
            # 只拿动作 token 的 hidden
            hidden = self.base_model(input_ids[i:i+1, start:end], return_hidden=True)
            # hidden: (1, na, H)
            if isinstance(hidden, tuple) and len(hidden) >= 3:
                h = hidden[2]
            else:
                h = self.base_model.forward_hidden(input_ids[i:i+1, start:end])
            values_out[i, :na] = self.value_head(h).squeeze(-1)
            vmask_out[i, :na] = 1

        return values_out, vmask_out

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
        kl_ctl=0.02,# 需要调整，初始较小的KL惩罚
        clip_reward=5.0,# 需要调整，更大的奖励裁剪范围
        gamma=0.95,# 折扣因子（计算优势函数）
        lambd=0.95,# GAE参数，平衡偏差和方差
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
        self.device = torch.device(device)
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50304)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler_actor  = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
        self.scaler_critic = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

    def generate_samples(self, inputs, max_length, max_new_tokens):
        # print(f">>>max_new_tokens: {max_new_tokens}")
        model = self.actor
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
    
        prompt_len = attention_mask.sum(1).long()
    
        # 强制 prompt_len <= block_size - max_new_tokens
        max_prompt_len = model.config.block_size - max_new_tokens
        prompt_len = torch.clamp(prompt_len, max=max_prompt_len)
    
        # 生成长度
        max_new_tokens = max_new_tokens
        with torch.no_grad():
            outputs = model.generate(
                input_ids[:, :prompt_len.max()],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id
            )
    
        seqs = outputs[:, :model.config.block_size]
        B, T = seqs.size()
    
        action_mask = torch.zeros_like(seqs, dtype=torch.long, device=self.device)
        for i in range(B):
            start = prompt_len[i].item()
            if start >= T:
                start = T - 1
            action_mask[i, start:T] = (seqs[i, start:T] != self.pad_token_id).long()
    
        new_attention_mask = (seqs != self.pad_token_id).long()
        num_actions = action_mask.sum(dim=1).long()
        response_length = num_actions
        total_length = new_attention_mask.sum(-1)
    
        # print(f">>> seqs.shape: {seqs.shape}, max_new_tokens: {max_new_tokens}")
        # print(f">>> num_actions: {num_actions}")
    
        return [Samples(
            seqs=seqs,
            attention_mask=new_attention_mask,
            action_mask=action_mask,
            num_actions=num_actions,
            response_length=response_length,
            total_length=total_length
        )]

    #计算当前策略(log_probs)和参考策略(ref_log_probs)的KL散度#只计算动作位置
    def compute_kl(self, log_probs, ref_log_probs, action_mask):
        kl = (log_probs - ref_log_probs) * action_mask
        return kl

    #奖励=负的KL惩罚 + 通过奖励模型打分的奖励，控制策略不偏离参考模型太远
    def compute_rewards(self, kl, reward_scores, action_mask):
        rewards = -self.kl_ctl * kl
        # 更平滑的奖励分配
        reward_clip = torch.clamp(reward_scores, -self.clip_reward, self.clip_reward)
        response_lengths = action_mask.sum(1)
        for j in range(rewards.size(0)):
            if response_lengths[j] > 0:
                # 将奖励均匀分配到response的每个token
                rewards[j] += reward_clip[j] / response_lengths[j]
        return rewards

    def get_advantages(self, values, rewards, action_mask):
        # 全部 (B, T)
        T = min(values.size(1), rewards.size(1), action_mask.size(1))
        if T == 0:
            B = values.size(0)
            z = torch.zeros(B, 0, device=values.device)
            return z, z

        values  = values[:, :T]
        rewards = rewards[:, :T]
        masks   = action_mask[:, :T].float()

        B = values.size(0)
        lastgaelam = torch.zeros(B, device=values.device)
        adv = torch.zeros_like(values)
        for t in reversed(range(T)):
            nextval = values[:, t+1] if t < T-1 else torch.zeros(B, device=values.device)
            delta = rewards[:, t] + self.gamma * nextval - values[:, t]
            lastgaelam = delta + self.gamma * self.lambd * lastgaelam
            adv[:, t] = lastgaelam

        adv = adv * masks
        ret = (adv + values) * masks
        return adv.detach(), ret.detach()

    # def evaluate_experience(self, samples: Samples):
    #     """
    #     输入 samples（单个 Samples，batch 的张量在其内），返回 ( [Experience(...)], avg_kl )
    #     Experience 里的字段保持你原来的批量形式（tensor batch）
    #     """
    #     seqs = samples.seqs.to(self.device)           # (B, L)
    #     attn_mask = samples.attention_mask.to(self.device)
    #     act_mask_full = samples.action_mask.to(self.device)  # (B, L)
    #     num_actions = samples.num_actions.to(self.device)    # (B,)

    #     B, L = seqs.size()

    #     with torch.no_grad():
    #         # 1) actor 和 ref 的 token-level log_probs（对 next-token）
    #         out_act = self.actor(seqs, return_all_logits=True)
    #         logits_all = out_act[0] if isinstance(out_act, tuple) else out_act  # (B, L, V)
    #         out_ref = self.ref(seqs, return_all_logits=True)
    #         ref_logits_all = out_ref[0] if isinstance(out_ref, tuple) else out_ref

    #         log_probs_all = F.log_softmax(logits_all, dim=-1)      # (B, L, V)
    #         ref_log_probs_all = F.log_softmax(ref_logits_all, dim=-1)

    #         # gather next-token logprobs
    #         # indices are the actual token ids we observed (next-token prediction)
    #         indices = seqs[:, 1:].unsqueeze(-1)  # (B, L-1, 1)
    #         logp_next = log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)      # (B, L-1)
    #         ref_logp_next = ref_log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)

    #         # We want token-level log_probs aligned to seq positions of tokens produced (exclude first prompt token)
    #         # For simplicity, we'll align actions to the tail: take last na tokens from logp_next
    #         na_list = [int(x.item()) for x in num_actions]
    #         max_na = max(na_list) if len(na_list) > 0 else 0

    #         if max_na == 0:
    #             # 没有动作，全返回空 Experience（避免 crash）
    #             empty_exp = Experience(
    #                 seqs=seqs,
    #                 action_log_probs=torch.zeros((B, 0), device=self.device),
    #                 values=torch.zeros((B, 0), device=self.device),
    #                 returns=torch.zeros((B, 0), device=self.device),
    #                 advantages=torch.zeros((B, 0), device=self.device),
    #                 attention_mask=attn_mask,
    #                 action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
    #                 reward=torch.zeros((B, 1), device=self.device),
    #                 num_actions=num_actions,
    #                 kl=torch.zeros((B, 0), device=self.device)
    #             )
    #             return [empty_exp], 0.0

    #         # allocate left-aligned tensors (B, max_na)
    #         new_logp = logp_next.new_full((B, max_na), fill_value=0.0)
    #         new_ref_logp = new_logp.clone()
    #         new_act_mask = torch.zeros((B, max_na), dtype=torch.long, device=self.device)

    #         for i in range(B):
    #             na = na_list[i]
    #             if na <= 0:
    #                 continue
    #             # take last na tokens from logp_next[i]
    #             new_logp[i, :na] = logp_next[i, -na:]
    #             new_ref_logp[i, :na] = ref_logp_next[i, -na:]
    #             # corresponding action mask: take last na of act_mask_full
    #             new_act_mask[i, :na] = act_mask_full[i, -na:]

    #         # KL (token-level) on the actions
    #         kl = (new_logp - new_ref_logp) * new_act_mask.float()  # (B, max_na)  (masked)

    #         # 2) reward model: decode texts from seqs (move to CPU for tokenizer)
    #         seqs_cpu = seqs.detach().cpu().tolist()
    #         texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer)
    #                 for t in self.actor_tokenizer.batch_decode(seqs_cpu)]
    #         reward_inputs = self.reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
    #                                             max_length=getattr(self.reward_tokenizer, "model_max_length", 2048))
    #         reward_inputs = {k: v.to(self.device) for k, v in reward_inputs.items()}
    #         reward_outputs = self.reward_model(**reward_inputs)
    #         # 假设 reward_model.logits[:,0] 给分数
    #         reward_scores = reward_outputs.logits[:, 0].to(self.device)  # (B,)

    #         # 3) compute token-wise rewards (B, max_na)
    #         rewards = self.compute_rewards(kl, reward_scores, new_act_mask)

    #         # 4) critic values: 得到 (B, max_na) 和 mask
    #         values, value_mask = self.critic(seqs, attn_mask, num_actions)  # values: (B, max_na), value_mask: (B, max_na)

    #         # 5) advantages & returns
    #         advantages, returns = self.get_advantages(values, rewards, new_act_mask)

    #         # 6) build Experience object (batched)
    #         exp = Experience(
    #             seqs=seqs,
    #             action_log_probs=new_logp,
    #             values=values,
    #             returns=returns,
    #             advantages=advantages,
    #             attention_mask=attn_mask,
    #             action_mask=new_act_mask,
    #             reward=reward_scores.unsqueeze(1),
    #             num_actions=num_actions,
    #             kl=kl
    #         )

    #         # avg_kl for logging (scalar)
    #         denom = new_act_mask.sum().clamp_min(1).float()
    #         avg_kl = (kl * new_act_mask.float()).sum() / denom

    #     return [exp], float(avg_kl.item())

    def evaluate_experience(self, samples: Samples, debug=False):
        seqs = samples.seqs.to(self.device)           # (B, L)
        attn_mask = samples.attention_mask.to(self.device)
        act_mask_full = samples.action_mask.to(self.device)  # (B, L)
        num_actions = samples.num_actions.to(self.device)    # (B,)
    
        B, L = seqs.size()
    
        with torch.no_grad():
            # --- actor / ref log_probs ---
            out_act = self.actor(seqs, return_all_logits=True)
            logits_all = out_act[0] if isinstance(out_act, tuple) else out_act
            out_ref = self.ref(seqs, return_all_logits=True)
            ref_logits_all = out_ref[0] if isinstance(out_ref, tuple) else out_ref
    
            log_probs_all = F.log_softmax(logits_all, dim=-1)
            ref_log_probs_all = F.log_softmax(ref_logits_all, dim=-1)
    
            indices = seqs[:, 1:].unsqueeze(-1)
            logp_next = log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)
            ref_logp_next = ref_log_probs_all[:, :-1].gather(-1, indices).squeeze(-1)
    
            # --- left-align actions ---
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
    
            # --- reward model ---
            seqs_cpu = seqs.detach().cpu().tolist()
            texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer)
                     for t in self.actor_tokenizer.batch_decode(seqs_cpu)]
            reward_inputs = self.reward_tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                                  max_length=getattr(self.reward_tokenizer, "model_max_length", 2048))
            reward_inputs = {k: v.to(self.device) for k, v in reward_inputs.items()}
            reward_outputs = self.reward_model(**reward_inputs)
            reward_scores = reward_outputs.logits[:, 0].to(self.device)
    
            # --- compute token-wise rewards ---
            rewards = self.compute_rewards(kl=new_logp - new_ref_logp, reward_scores=reward_scores, action_mask=new_act_mask)
    
            # --- critic ---
            values, value_mask = self.critic(seqs, attn_mask, num_actions)
            advantages, returns = self.get_advantages(values, rewards, new_act_mask)
    
            # --- debug 打印 ---
            if debug:
                for i in range(B):
                    print(f"Sample {i}: num_actions={num_actions[i].item()}, "
                          f"action_mask.sum={new_act_mask[i].sum().item()}, "
                          f"reward_score={reward_scores[i].item():.4f}, "
                          f"logp_next min={logp_next[i,-na_list[i]:].min().item() if na_list[i]>0 else 0:.4f}, "
                          f"max={logp_next[i,-na_list[i]:].max().item() if na_list[i]>0 else 0:.4f}")
    
        # --- build Experience object ---
        exp = Experience(
            seqs=seqs,
            action_log_probs=new_logp,
            values=values,
            returns=returns,
            advantages=advantages,
            attention_mask=attn_mask,
            action_mask=new_act_mask,
            reward=reward_scores.unsqueeze(1),
            num_actions=num_actions,
            kl=new_logp - new_ref_logp
        )
    
        avg_kl = ((new_logp - new_ref_logp) * new_act_mask.float()).sum() / new_act_mask.sum().clamp_min(1).float()
        return [exp], float(avg_kl.item())

    #
    def policy_loss(self, new_log_probs, old_log_probs, advantages, action_mask, clip_eps=0.2):
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)#目标是最大化策略目标
        # return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
        denom = action_mask.sum(-1).clamp_min(1)
        return ((loss * action_mask).sum(-1) / denom).mean()

    def value_loss(self, values, old_values, returns, action_mask, clip_eps=0.2):
        old_values = old_values.detach()
        clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (returns - values).pow(2)
        surr2 = (returns - clipped).pow(2)
        loss = torch.max(surr1, surr2)
        # return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()
        denom = action_mask.sum(-1).clamp_min(1)
        return ((loss * action_mask).sum(-1) / denom).mean()

    def train_on_experience(self, experience: Experience):
        self.actor.train()
        # ----------- Actor update -----------
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
            # dummy loss 防止 AMP 报 No inf checks
            dummy = sum(p.sum() * 0 for p in self.actor.parameters() if p.requires_grad)
            self.scaler_actor.scale(dummy).backward()
    
        self.scaler_actor.unscale_(self.optimizer_actor)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()
    
        # ----------- Critic update -----------
        self.critic.train()
        with amp.autocast(device_type='cuda', enabled=self.use_amp):
            values, _ = self.critic(experience.seqs, experience.attention_mask, experience.num_actions)
            value_loss = self.value_loss(values, experience.values, experience.returns, experience.action_mask)
    
        self.optimizer_critic.zero_grad(set_to_none=True)
        if value_loss.requires_grad:
            self.scaler_critic.scale(value_loss).backward()
        else:
            dummy = sum(p.sum() * 0 for p in self.critic.parameters() if p.requires_grad)
            self.scaler_critic.scale(dummy).backward()
    
        self.scaler_critic.unscale_(self.optimizer_critic)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.optimizer_critic)
        self.scaler_critic.update()
    
        return policy_loss, value_loss