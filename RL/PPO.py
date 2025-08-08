import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List
from dataclasses import dataclass

# Samples = 一次完整文本生成的“样本”
# Experience = 在此基础上，结合强化学习信号封装的训练用经历

#存储一个batch的模型输入和相关信息
@dataclass
class Samples:
    seqs: torch.Tensor#文本输入+生成输出的token序列
    attention_mask: torch.Tensor #用于Transformer计算时屏蔽无效token
    action_mask: torch.Tensor#哪些位置是“可动作”的位置
    num_actions: int#动作数，生成序列中实际动作（token）的数量
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
    num_actions: int#动作数量
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

@staticmethod
def contains_chinese(text):
    #检查是否含中文字符
    return any('\u4e00' <= c <= '\u9fff' for c in text)

def normalize_for_reward(self, text):
    #确保GPT-2生成文本适配Qwen分词器
    # 1. 替换EOS（必须）
    if hasattr(self.reward_tokenizer, "eos_token"):
        text = text.replace("<|endoftext|>", self.reward_tokenizer.eos_token)
    
    # 2. 统一换行符（预防性）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # 3. 中文标点→英文（可选，依数据集而定）
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    
    # 4. 移除控制字符（必须）
    return "".join(c for c in text if c.isprintable())

class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.transformer = base_model.transformer  # 共享底层transformer
        self.value_head = nn.Linear(base_model.config.n_embd, 1)
        
    def forward(self, input_ids, attention_mask, num_actions):
        hidden = self.transformer(input_ids, attention_mask=attention_mask)[0]
        values = self.value_head(hidden).squeeze(-1)
        prompt_len = attention_mask.sum(1) - num_actions
        values = torch.stack([values[i, prompt_len[i]:] for i in range(len(prompt_len))])
        return values

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
        gamma=0.9,# 折扣因子（计算优势函数）
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
        self.device = device
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id", 50304)

    # 用actor模型生成文本，从input_ids和attention_mask中给定的prompt开始生成max_new_tokens个token
    def generate_samples(self, inputs, max_length, max_new_tokens):
        # inputs: (input_ids, attention_mask) 来自 prepare.py 预处理的 prompt.bin
        model = self.actor.eval()

        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # prompt 长度（非 pad 部分的长度）
        prompt_len = attention_mask.sum(1)

        with torch.no_grad():# 要改一下，跟nanogpt对齐
            outputs = model.generate(
                ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id
            )

        seqs = outputs
        # 新的attention_mask：生成后的序列中，非pad部分标 1
        new_attention_mask = (seqs != self.pad_token_id).long()

        # action_mask：从prompt_len之后的部分为动作
        action_mask = torch.zeros_like(seqs)
        for i in range(seqs.size(0)):
            action_mask[i, prompt_len[i]:] = (seqs[i, prompt_len[i]:] != self.eos_token_id)

        return [
            Samples(
                seqs=seqs,
                attention_mask=new_attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1) - prompt_len.min().item(),
                response_length=action_mask.sum(-1),
                total_length=new_attention_mask.sum(-1)
            )
        ]#返回一个Samples对象，封装生成的序列和对应mask信息

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

    #GAE计算优势函数advantages
    def get_advantages(self, values, rewards, action_mask):
        lastgaelam = 0
        advantages = []
        T = rewards.size(1)
        for t in reversed(range(T)):
            nextval = values[:, t+1] if t < T - 1 else 0
            #delta = reward + gamma * next_value - value 是TD误差
            delta = rewards[:, t] + self.gamma * nextval - values[:, t]
            lastgaelam = delta + self.gamma * self.lambd * lastgaelam
            advantages.insert(0, lastgaelam)

        advantages = torch.stack(advantages, dim=1)
        advantages = advantages * action_mask
        returns = (advantages + values) * action_mask#训练critic的目标
        return advantages.detach(), returns.detach()

    #根据生成的样本samples计算各种训练所需的信号
    def evaluate_experience(self, samples: Samples):
        seqs, attn_mask, act_mask, num_actions = samples.seqs, samples.attention_mask, samples.action_mask, samples.num_actions
        with torch.no_grad():
            #先通过actor和ref模型算出动作对应的log概率，用来计算KL
            logits = self.actor(seqs, attention_mask=attn_mask).logits
            ref_logits = self.ref(seqs, attention_mask=attn_mask).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            ref_log_probs = F.log_softmax(ref_logits[:, :-1], dim=-1)
            indices = seqs[:, 1:].unsqueeze(-1)
            log_probs = log_probs.gather(-1, indices).squeeze(-1)[:, -num_actions:]
            ref_log_probs = ref_log_probs.gather(-1, indices).squeeze(-1)[:, -num_actions:]
            kl = self.compute_kl(log_probs, ref_log_probs, act_mask)

            #用critic模型计算价值估计
            values = self.critic(seqs, attn_mask, num_actions)
            # 用GPT-2解码
            # texts = self.actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            texts = [
                normalize_for_reward(t) 
                for t in self.actor_tokenizer.batch_decode(seqs)
            ]
            # 用Qwen重新编码
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

    #
    def policy_loss(self, new_log_probs, old_log_probs, advantages, action_mask, clip_eps=0.2):
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)#目标是最大化策略目标
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    def value_loss(self, values, old_values, returns, action_mask, clip_eps=0.2):
        old_values = old_values.detach()
        clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (returns - values).pow(2)
        surr2 = (returns - clipped).pow(2)
        loss = torch.max(surr1, surr2)
        return ((loss * action_mask).sum(-1) / action_mask.sum(-1)).mean()

    def train_on_experience(self, experience: Experience):
        # 训练actor
        self.actor.train()
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            logits = self.actor(experience.seqs, attention_mask=experience.attention_mask).logits
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            indices = experience.seqs[:, 1:].unsqueeze(-1)
            new_log_probs = log_probs.gather(-1, indices).squeeze(-1)[:, -experience.num_actions:]
            policy_loss = self.policy_loss(new_log_probs, experience.action_log_probs, 
                                        experience.advantages, experience.action_mask)
        
        self.optimizer_actor.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # 在step之前裁剪，可尝试更宽松的裁剪
        self.optimizer_actor.step()

        # 训练critic
        self.critic.train()
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            values = self.critic(experience.seqs, experience.attention_mask, experience.num_actions)
            value_loss = self.value_loss(values, experience.values, 
                                    experience.returns, experience.action_mask)
        
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # 在step之前裁剪
        self.optimizer_critic.step()

        return policy_loss.item(), value_loss.item()
