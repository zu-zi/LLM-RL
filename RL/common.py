# common.py
# 说明：
# - 面向所有 RL 算法（PPO/GRPO/DAPO/Token-Entropy）可复用的工具。
# - 关注显存：提供 batch 维 micro-batch 前向、只对 response 段建掩码计算指标、可选按 8 对齐 padding。
# - 与 train.py / model.py 对齐：model.forward 支持 return_all_logits / return_hidden。
# - 注意：这里不直接依赖具体的 Trainer，实现尽量通用。

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import random


# 数据结构
@dataclass
class Samples:
    """
    一个 batch 的已对齐样本（把若干条生成结果 pad 成 BxT）
    - seqs:        B x T，完整序列（prompt + response [+ eos]），右侧 pad
    - attention_mask: B x T，1 表示有效 token，0 表示 pad
    - action_mask:    B x T，1 表示该位置的 token 属于 response 且位置>0（用于计算 logprob 的 target 位置）
                      注意：在计算时通常会用 action_mask[:, 1:] 去对齐 CE 的 target 维度
    - num_actions:    B，response token 个数（= action_mask.sum(dim=1)）
    - response_length: B，response 长度
    - total_length:    B，实际序列长度（不含 pad）
    """
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor


@dataclass
class Experience:
    """
    训练用经历条目（可按算法自行扩展）
    - action_log_probs: B x (T-1)，与交叉熵 target 维对齐（通常只在 response 段有效）
    - values:    B x T 或 B x (T-1)，视你的 critic 设计而定（这里倾向于 B x T，之后切片）
    - returns / advantages: 与 values 对齐的张量（mask 到 response 段）
    - kl:        B 或 B x (T-1)，按算法记录 KL（支持标量/逐 token）
    """
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    reward: torch.Tensor
    num_actions: torch.Tensor
    kl: torch.Tensor


class ExperienceBuffer:
    """简易经验回放缓冲区（可用于 DAPO 或多轮 PPO 累积）"""
    def __init__(self, limit: int):
        self.limit = limit
        self.buffer: List[Experience] = []

    def append(self, experiences: List[Experience]):
        self.buffer.extend(experiences)
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[-self.limit:]

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []


# 文本归一化（奖励模型用）
def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def normalize_for_reward(text: str, reward_tokenizer=None) -> str:
    """
    把 actor 的 eos 符号替换成 reward 模型的 eos，做简单清洗。
    注意：这里只做轻量处理，复杂清洗应在上游做。
    """
    if reward_tokenizer is not None and hasattr(reward_tokenizer, "eos_token") and reward_tokenizer.eos_token:
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    return "".join(c for c in text if c.isprintable())


# Ragged → Padded（构造 Samples）
def _pad_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    r = length % multiple
    return length if r == 0 else (length + multiple - r)


def build_samples_from_generations(
    gens: List[dict],
    block_size: int,
    pad_to_multiple_of: int = 8,
    device: Optional[torch.device] = None,
) -> Samples:
    """
    把 list[{"prompt_ids":Tensor[Ti], "full_ids":Tensor[Ti+Tj], "response_ids":Tensor[Tj]}]
    转成对齐后的 Samples（BxT）。
    - 右 pad；T 为本 batch 的最大实际长度（截断不超过 block_size）
    - action_mask 只标记 response token（且排除 position=0），方便与 CE target 对齐
    """
    # 收集长度
    total_lens = []
    resp_lens = []
    for g in gens:
        total_lens.append(int(g["full_ids"].numel()))
        resp_lens.append(int(g["response_ids"].numel()))
    B = len(gens)
    T_max = min(max(total_lens), block_size)
    if pad_to_multiple_of and T_max < block_size:
        # 为了 Tensor Core 对齐，尽量 pad 到 8 的倍数，但不超过 block_size
        T_max = min(_pad_to_multiple(T_max, pad_to_multiple_of), block_size)

    # 准备承载张量
    seqs = gens[0]["full_ids"].new_full((B, T_max), fill_value=0)  # 这里用 0 作为 pad_id（gpt2 的 pad 不重要，反正不参与 loss）
    attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    action_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    response_length = torch.tensor(resp_lens, dtype=torch.long, device=device)
    total_length = torch.tensor([min(L, T_max) for L in total_lens], dtype=torch.long, device=device)

    # 填充数据与掩码
    for i, g in enumerate(gens):
        full_ids = g["full_ids"]
        T_i = min(full_ids.numel(), T_max)
        seqs[i, :T_i] = full_ids[:T_i]
        attention_mask[i, :T_i] = 1

        # 响应掩码（token 维度）：位置 j 属于 response（且 j < T_i）
        T_prompt = int(g["prompt_ids"].numel())
        # response token 在 [T_prompt, T_i-1]
        start = min(T_prompt, T_i - 1)  # 防越界
        if T_i > start:
            action_mask[i, start:T_i] = 1  # 注意：第 0 位不会被用作 target

    # num_actions = 每条 response token 数（也可用 action_mask.sum(dim=1) 得到）
    num_actions = (action_mask.sum(dim=1)).to(torch.long)

    if device is not None:
        seqs = seqs.to(device)

    return Samples(
        seqs=seqs,
        attention_mask=attention_mask,
        action_mask=action_mask,
        num_actions=num_actions,
        response_length=response_length,
        total_length=total_length,
    )


# 显存友好：micro-batch 前向
def iter_batch_indices(n_items: int, micro_batch_size: int) -> Iterable[Tuple[int, int]]:
    """按 batch 维切分，避免一次前向过大"""
    if micro_batch_size <= 0 or micro_batch_size >= n_items:
        yield 0, n_items
        return
    s = 0
    while s < n_items:
        e = min(s + micro_batch_size, n_items)
        yield s, e
        s = e


def model_all_logits(
    model: nn.Module,
    seqs: torch.Tensor,               # B x T（右 pad）
    device_type: str,
    ptdtype: torch.dtype,
    micro_batch_size: int = 0,        # 0 表示不切
) -> torch.Tensor:
    """
    计算所有时间步的 logits（B x T x V），按 batch 维分块以省显存。
    注意：不做 attention mask 到模型内部，模型是 causal 的；外部用 action_mask/attention_mask 控制 loss。
    """
    B = seqs.size(0)
    logits_chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        with amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext():
            logits_chunk, _ = model(seqs[s:e], return_all_logits=True)  # (b, T, V)
        logits_chunks.append(logits_chunk)
        # 释放中间显存
        del logits_chunk
        torch.cuda.empty_cache() if device_type.startswith('cuda') else None
    return torch.cat(logits_chunks, dim=0)


def token_logprobs_from_logits(
    logits: torch.Tensor,     # B x T x V
    seqs: torch.Tensor,       # B x T
) -> torch.Tensor:
    """
    把“下一个 token 的交叉熵”形式对齐：
    - 我们要预测 seqs[:, 1:]（作为 target），logits 对应位置是 logits[:, :-1, :]
    - 返回 B x (T-1) 的 logprob
    """
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)                     # B x (T-1) x V
    tgt = seqs[:, 1:].unsqueeze(-1)                                     # B x (T-1) x 1
    lp = torch.gather(logp, dim=-1, index=tgt).squeeze(-1).contiguous() # B x (T-1)
    return lp


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """对齐形状后做掩码平均；支持标量返回"""
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


# KL / 熵 / 值函数

def approx_kl_from_logprobs(
    actor_logprobs: torch.Tensor,   # B x (T-1)
    ref_logprobs: torch.Tensor,     # B x (T-1)
    mask_target: torch.Tensor,      # B x (T-1)（通常 = action_mask[:, 1:]）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    近似 KL：E[ log pi(a|s) - log ref(a|s) ]，仅在响应段计算。
    返回：(kl_scalar_mean, kl_per_token_masked_mean)
    """
    diff = (actor_logprobs - ref_logprobs)
    kl_mean = masked_mean(diff, mask_target.float())
    # per-token 平均也返回一个标量（便于日志）
    return kl_mean, masked_mean(diff, mask_target.float())


def entropy_from_logits(
    logits: torch.Tensor,        # B x T x V（或 B x (T-1) x V）
    mask_time: torch.Tensor,     # B x T（或 B x (T-1)）
) -> torch.Tensor:
    """
    计算 token 熵：-sum p log p。注意这是“全分布”计算，显存较重，谨慎启用。
    返回标量平均（masked）。
    """
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)           # B x T
    return masked_mean(ent, mask_time.float())


def critic_values_from_hidden(
    critic: nn.Module,
    hidden_states: torch.Tensor,       # B x T x C（来自 model forward(return_hidden=True)）
    mask_time: Optional[torch.Tensor] = None,  # B x T（可选）
) -> torch.Tensor:
    """
    从 actor 的隐藏态取 value（建议 critic 是简单线性头，避免重复前向）
    返回 B x T
    """
    v = critic(hidden_states, mask=mask_time)  # 你的 Critic.forward 可以忽略 mask 或内部使用
    return v


def forward_values_via_actor(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,                 # B x T
    device_type: str,
    ptdtype: torch.dtype,
    micro_batch_size: int = 0,
) -> torch.Tensor:
    """
    为了拿 critic 的值，在前向时让 actor 返回 hidden，然后过 critic。
    按 batch 维做 micro-batch，节省显存。
    返回 B x T 的 values。
    """
    B = seqs.size(0)
    values_chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        with amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext():
            _, _, h = model(seqs[s:e], return_hidden=True)  # (B, T, C)
        v = critic(h)                                      # (B, T)
        values_chunks.append(v)
        # 释放
        del h, v
        torch.cuda.empty_cache() if device_type.startswith('cuda') else None
    return torch.cat(values_chunks, dim=0)


# 优势/回报（GAE）
def gae_compute(
    values: torch.Tensor,            # B x T
    rewards: torch.Tensor,           # B x T 或 B（若是标量奖励则会广播到最后一个 response token）
    mask_time: torch.Tensor,         # B x T（1 表示有效 token；通常只在 response 段为 1）
    gamma: float = 1.0,
    lam: float = 0.95,
    use_last_as_terminal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    通用 GAE 计算。这里默认 response 段的最后一个 token 为“终止”（终止处没有 bootstrap）。
    - values/rewards/mask_time 形状对齐到 B x T（若 rewards 是 B，会广播到每条最后一个响应 token）
    - 返回：returns, advantages（B x T），未掩码的位置为 0
    """
    B, T = values.shape
    device = values.device

    # 若 rewards 是 B，广播到每条“最后一个 mask=1 的位置”
    if rewards.dim() == 1:
        r_full = torch.zeros_like(values)
        last_idx = (mask_time * torch.arange(T, device=device).view(1, T)).argmax(dim=1)  # 每条最后一个为1的索引
        r_full[torch.arange(B, device=device), last_idx] = rewards
        rewards = r_full
    else:
        rewards = rewards * mask_time  # 非 response 段置 0

    # delta_t = r_t + gamma * V_{t+1} - V_t（终止处 V_{t+1} = 0）
    V = values
    V_next = torch.zeros_like(V)
    V_next[:, :-1] = V[:, 1:]
    if use_last_as_terminal:
        # 最后一步当作 terminal，V_next[:, -1] = 0（默认已是 0）
        pass

    deltas = (rewards + gamma * V_next - V) * mask_time

    advantages = torch.zeros_like(V)
    gae = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        mask_t = mask_time[:, t]
        gae = deltas[:, t] + gamma * lam * gae * mask_t  # 只在有效位置递推
        advantages[:, t] = gae

    returns = advantages + V
    # 无效位置清零
    advantages = advantages * mask_time
    returns = returns * mask_time
    return returns, advantages


# 便捷打包：actor/ref logprob / KL / values

def compute_actor_ref_logprobs(
    actor: nn.Module,
    ref: nn.Module,
    seqs: torch.Tensor,                 # B x T
    action_mask: torch.Tensor,          # B x T（注意：target 维将用 [:,1:]）
    device_type: str,
    ptdtype: torch.dtype,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 actor/ref 在 target 维（预测 seqs[:,1:]）上的 logprob，以及 mask。
    返回：
      actor_lp: B x (T-1)
      ref_lp:   B x (T-1)
      mask_tgt: B x (T-1)  （通常 = action_mask[:,1:]）
    """
    # logits（显存友好：按 batch 维分块）
    logits_actor = model_all_logits(actor, seqs, device_type, ptdtype, micro_batch_size)
    logits_ref   = model_all_logits(ref,   seqs, device_type, ptdtype, micro_batch_size)

    # token logprobs
    actor_lp = token_logprobs_from_logits(logits_actor, seqs)  # B x (T-1)
    ref_lp   = token_logprobs_from_logits(logits_ref,   seqs)  # B x (T-1)

    # target 维的 mask：仅在响应 token 上计算
    mask_tgt = action_mask[:, 1:].contiguous().to(actor_lp.dtype)

    # 及时释放 logits
    del logits_actor, logits_ref
    torch.cuda.empty_cache() if device_type.startswith('cuda') else None

    return actor_lp, ref_lp, mask_tgt


def compute_values_on_response(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,                 # B x T
    action_mask: torch.Tensor,          # B x T
    device_type: str,
    ptdtype: torch.dtype,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 values，并返回（values, mask_time），两者均为 B x T。
    实际训练时通常只在 response 段（mask_time=1）使用。
    """
    values = forward_values_via_actor(model, critic, seqs, device_type, ptdtype, micro_batch_size)  # B x T
    mask_time = action_mask.clone()  # 我们约定 action_mask 的 1 正好覆盖 response token
    return values, mask_time


# 其他小工具

def to_device_rec(obj, device):
    """把张量递归搬到 device（dict/list/tuple 简单支持）"""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device_rec(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device_rec(v, device) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj


def detach_rec(obj):
    """把张量从计算图中分离出去（dict/list/tuple 简单支持）"""
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: detach_rec(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [detach_rec(v) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj


def clip_by_global_norm(params: Iterable[torch.Tensor], max_norm: float) -> float:
    """按全局范数裁剪，返回裁剪后的全局范数（可用于日志）"""
    total_norm = torch.nn.utils.clip_grad_norm_(list(params), max_norm)
    return float(total_norm)

def apply_entropy_mask(logits: torch.Tensor, action_mask: torch.Tensor, keep_ratio: float = 0.2) -> torch.Tensor:
    """
    基于 token 的熵筛选 action_mask，只保留高熵 token（例如 top-20%）。
    - logits: (B, T, V)，模型在所有 token 的输出分布
    - action_mask: (B, T)，原始的 action mask（只在 response 段为1）
    - keep_ratio: 保留比例（默认0.2，表示保留top 20%熵最高的token）
    return:
        new_action_mask: (B, T)，高熵token保留，其余置0
    """
    device = logits.device
    B, T, V = logits.shape

    # softmax 得到概率分布
    probs = F.softmax(logits, dim=-1)  # (B, T, V)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)

    new_mask = torch.zeros_like(action_mask, dtype=torch.long, device=device)

    for i in range(B):
        act_pos = (action_mask[i] > 0).nonzero(as_tuple=False).squeeze(-1)
        if act_pos.numel() == 0:
            continue
        ent_vals = entropy[i, act_pos]

        # 选 top-k 熵最大的 token
        k = max(1, int(len(ent_vals) * keep_ratio))
        topk_idx = torch.topk(ent_vals, k=k, largest=True).indices
        selected_pos = act_pos[topk_idx]

        new_mask[i, selected_pos] = 1

    return new_mask

__all__ = [
    "Samples",
    "Experience",
    "ExperienceBuffer",
    "build_samples_from_generations",
    "model_all_logits",
    "token_logprobs_from_logits",
    "compute_actor_ref_logprobs",
    "approx_kl_from_logprobs",
    "entropy_from_logits",
    "critic_values_from_hidden",
    "forward_values_via_actor",
    "compute_values_on_response",
    "gae_compute",
    "masked_mean",
    "to_device_rec",
    "detach_rec",
    "clip_by_global_norm",
    "normalize_for_reward",
    "contains_chinese",
    "apply_entropy_mask", 
]
