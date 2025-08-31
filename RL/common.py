# common.py
# 说明：
# - 面向所有 RL 算法（PPO/GRPO/DAPO/Token-Entropy）可复用的工具。
# - 关注显存：提供 batch 维 micro-batch 前向、只对 response 段建掩码计算指标、可选按 8 对齐 padding。
# - 与 train.py / model.py 对齐：model.forward 支持 return_all_logits / return_hidden。
# - 注意：这里不直接依赖具体的 Trainer，实现尽量通用。

from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Union

from contextlib import nullcontext
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp


# =========================
# 数据结构
# =========================

@dataclass
class Samples:
    """
    一个 batch 的已对齐样本（把若干条生成结果 pad 成 BxT）
    - seqs:            B x T，完整序列（prompt + response [+ eos]），右侧 pad
    - attention_mask:  B x T，1 表示有效 token，0 表示 pad
    - action_mask:     B x T，1 表示该位置的 token 属于 response（用于 target 对齐，通常后续会用 [:,1:]）
    - num_actions:     B，response token 个数（= action_mask.sum(dim=1)）
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
    - values/returns/advantages: 与时间维对齐（通常是 B x (T-1)，也可 B x T，视实现切片）
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


# =========================
# 文本归一化（奖励模型用）
# =========================

def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def normalize_for_reward(text: str, reward_tokenizer=None) -> str:
    """
    把 actor 的 eos 符号替换成 reward 模型的 eos，做简单清洗。
    注意：这里只做轻量处理，复杂清洗应在上游做。
    """
    if reward_tokenizer is not None and getattr(reward_tokenizer, "eos_token", None):
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    return "".join(c for c in text if c.isprintable())


# =========================
# Ragged → Padded（构造 Samples）
# =========================

def _pad_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return length
    r = length % multiple
    return length if r == 0 else (length + multiple - r)


def build_samples_from_generations(
    gens: List[dict],
    block_size: int,
    pad_to_multiple_of: int = 8,
    device: Optional[Union[str, torch.device]] = None,
) -> Samples:
    """
    把 list[{"prompt_ids":Tensor[Ti], "full_ids":Tensor[Ti+Tj], "response_ids":Tensor[Tj]}]
    转成对齐后的 Samples（BxT）。
    - 右 pad；T 为本 batch 的最大实际长度（截断不超过 block_size）
    - action_mask 只标记 response token（后续通常会用 action_mask[:,1:] 与 target 维对齐）
    """
    assert len(gens) > 0, "gens 为空"
    if device is None:
        device = gens[0]["full_ids"].device

    # 收集长度
    total_lens = [int(g["full_ids"].numel()) for g in gens]
    resp_lens  = [int(g["response_ids"].numel()) for g in gens]
    B = len(gens)
    T_max = min(max(total_lens), block_size)
    if pad_to_multiple_of and T_max < block_size:
        # 为了 Tensor Core，对齐到 8 的倍数，但不超过 block_size
        T_max = min(_pad_to_multiple(T_max, pad_to_multiple_of), block_size)

    # 承载张量
    pad_id = 0  # GPT-2 无 pad_token_id；此处用 0 作为 padding，不参与 loss
    dtype_ids = gens[0]["full_ids"].dtype
    seqs = torch.full((B, T_max), pad_id, dtype=dtype_ids, device=device)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    action_mask    = torch.zeros((B, T_max), dtype=torch.long, device=device)
    response_length = torch.tensor(resp_lens, dtype=torch.long, device=device)
    total_length    = torch.tensor([min(L, T_max) for L in total_lens], dtype=torch.long, device=device)

    # 填充
    for i, g in enumerate(gens):
        full_ids = g["full_ids"].to(device)
        T_i = min(full_ids.numel(), T_max)
        seqs[i, :T_i] = full_ids[:T_i]
        attention_mask[i, :T_i] = 1

        # 响应掩码（token 维度）
        T_prompt = int(g["prompt_ids"].numel())
        start = min(T_prompt, T_i - 1)  # 防越界
        if T_i > start:
            action_mask[i, start:T_i] = 1

    num_actions = action_mask.sum(dim=1).to(torch.long)

    return Samples(
        seqs=seqs,
        attention_mask=attention_mask,
        action_mask=action_mask,
        num_actions=num_actions,
        response_length=response_length,
        total_length=total_length,
    )


# =========================
# 显存友好：micro-batch 前向
# =========================

def iter_batch_indices(n_items: int, micro_batch_size: int) -> Iterable[Tuple[int, int]]:
    """按 batch 维切分，避免一次前向过大"""
    if micro_batch_size is None or micro_batch_size <= 0 or micro_batch_size >= n_items:
        yield 0, n_items
        return
    s = 0
    while s < n_items:
        e = min(s + micro_batch_size, n_items)
        yield s, e
        s = e


def model_all_logits(
    model: nn.Module,
    seqs: torch.Tensor,               # [B, T] (Long)
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,        # 0/None 表示不切
) -> torch.Tensor:
    """
    计算所有时间步的 logits（[B, T, V]），按 batch 维分块以省显存。
    注意：不在模型内部传 mask；外部用 action_mask/attention_mask 控制 loss。
    """
    if ptdtype is None:
        ptdtype = next(model.parameters()).dtype  # 关键：用模型 dtype，而不是 seqs.dtype (Long)

    B = seqs.size(0)
    chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()
        with ctx:
            logits_chunk, _ = model(seqs[s:e], return_all_logits=True)  # (b, T, V)
        chunks.append(logits_chunk)
        del logits_chunk
    return torch.cat(chunks, dim=0)


# =========================
# token logprob / mask 计算
# =========================

def token_logprobs_from_logits(
    logits: torch.Tensor,     # [B, T, V]
    seqs: torch.Tensor,       # [B, T]
) -> torch.Tensor:
    """
    把“下一个 token 的交叉熵”形式对齐：
    - 我们要预测 seqs[:, 1:]（作为 target），logits 对应位置是 logits[:, :-1, :]
    - 返回 [B, (T-1)] 的 logprob
    """
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)                     # [B, T-1, V]
    tgt = seqs[:, 1:].unsqueeze(-1)                                     # [B, T-1, 1]
    lp = torch.gather(logp, dim=-1, index=tgt).squeeze(-1).contiguous() # [B, T-1]
    return lp


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """掩码平均（返回标量）"""
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom


# =========================
# KL / 熵 / 值函数
# =========================

def approx_kl_from_logprobs(
    actor_logprobs: torch.Tensor,   # [B, T-1]
    ref_logprobs: torch.Tensor,     # [B, T-1]
    mask_target: torch.Tensor,      # [B, T-1]（通常 = action_mask[:, 1:]）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    近似 KL：E[ log pi(a|s) - log ref(a|s) ]，仅在响应段计算。
    返回：(kl_scalar_mean, kl_per_token_masked_mean) —— 两者都是标量
    """
    diff = (actor_logprobs - ref_logprobs)
    kl_mean = masked_mean(diff, mask_target.float())
    return kl_mean, masked_mean(diff, mask_target.float())


def entropy_from_logits(
    logits: torch.Tensor,        # [B, T, V] 或 [B, T-1, V]
    mask_time: torch.Tensor,     # [B, T] 或 [B, T-1]
) -> torch.Tensor:
    """
    计算 token 熵：-sum p log p。注意这是“全分布”计算，显存较重，谨慎启用。
    返回标量平均（masked）。
    """
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)           # [B, T]
    return masked_mean(ent, mask_time.float())


def critic_values_from_hidden(
    critic: nn.Module,
    hidden_states: torch.Tensor,       # [B, T, C]（来自 actor: forward(..., return_hidden=True)）
    mask_time: Optional[torch.Tensor] = None,  # [B, T]（可选）
) -> torch.Tensor:
    """
    从 actor 的隐藏态取 value。要求 critic(hidden_states) -> [B, T] 或 [B, T, 1]。
    """
    v = critic(hidden_states)
    if v.dim() == 3 and v.size(-1) == 1:
        v = v.squeeze(-1)
    if mask_time is not None:
        v = v * mask_time
    return v


def forward_values_via_actor(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,                 # [B, T]
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
) -> torch.Tensor:
    """
    为了拿 critic 的值，在前向时让 actor 返回 hidden，然后过 critic。
    按 batch 维做 micro-batch，节省显存。
    返回 [B, T] 的 values。
    """
    if ptdtype is None:
        ptdtype = next(model.parameters()).dtype

    B = seqs.size(0)
    chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()
        with ctx:
            _, _, h = model(seqs[s:e], return_hidden=True)  # (b, T, C)
        v = critic(h)                                      # (b, T) or (b, T, 1)
        if v.dim() == 3 and v.size(-1) == 1:
            v = v.squeeze(-1)
        chunks.append(v)
        del h, v
    return torch.cat(chunks, dim=0)


# =========================
# 优势/回报（GAE）
# =========================

def gae_compute(
    values: torch.Tensor,            # [B, T]
    rewards: torch.Tensor,           # [B, T] 或 [B]
    mask_time: torch.Tensor,         # [B, T]（1 表示有效 token；通常只在 response 段为 1）
    gamma: float = 1.0,
    lam: float = 0.95,
    use_last_as_terminal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    通用 GAE 计算。默认 response 段的最后一个 token 为“终止”（终止处没有 bootstrap）。
    - 若 rewards 是 [B]，则广播到每条“最后一个 mask=1 的时间步”
    - 返回：returns, advantages（[B, T]），未掩码的位置为 0
    """
    B, T = values.shape
    device = values.device

    # 若 rewards 是 [B]，广播到每条最后一个有效位置
    if rewards.dim() == 1:
        r_full = torch.zeros_like(values)
        # 每条“最后一个 mask=1 的位置”
        last_idx = (mask_time * torch.arange(T, device=device).view(1, T)).argmax(dim=1)
        r_full[torch.arange(B, device=device), last_idx] = rewards
        rewards = r_full
    else:
        rewards = rewards * mask_time  # 非 response 段置 0

    # delta_t = r_t + gamma * V_{t+1} - V_t（终止处 V_{t+1} = 0）
    V = values
    V_next = torch.zeros_like(V)
    V_next[:, :-1] = V[:, 1:]
    if use_last_as_terminal:
        pass  # 末位默认 0，不做 bootstrap

    deltas = (rewards + gamma * V_next - V) * mask_time

    advantages = torch.zeros_like(V)
    gae = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        mask_t = mask_time[:, t]
        gae = deltas[:, t] + gamma * lam * gae * mask_t
        advantages[:, t] = gae

    returns = advantages + V
    advantages = advantages * mask_time
    returns = returns * mask_time
    return returns, advantages


# =========================
# 便捷打包：actor/ref logprob / KL / values
# =========================

@torch.no_grad()
def _ref_logits(ref: nn.Module, seqs: torch.Tensor, device_type: str, micro_batch_size: int):
    # 只对 ref 做 no_grad + micro-batch 前向，省显存
    return model_all_logits(ref, seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)

@torch.no_grad()
def compute_actor_ref_logprobs(
    actor: nn.Module,
    ref: nn.Module,
    seqs: torch.Tensor,                 # [B, T]
    action_mask: torch.Tensor,          # [B, T]
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1) 前向拿 logits（显存友好）
    logits_actor = model_all_logits(actor, seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)
    logits_ref   = model_all_logits(ref,   seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)

    # 2) 转成 next-token logprob（target=seqs[:,1:]）
    lp_actor = token_logprobs_from_logits(logits_actor, seqs)  # [B, T-1]
    lp_ref   = token_logprobs_from_logits(logits_ref,   seqs)  # [B, T-1]

    # 3) 只在 response 段训练：mask_tgt = action_mask[:,1:]
    mask_tgt = action_mask[:, 1:].contiguous().long()         # [B, T-1]（理论上）

    # 4) 强力对齐（防止任何上游切块/裁剪造成的 off-by-one）
    L = min(lp_actor.size(1), lp_ref.size(1), mask_tgt.size(1))
    if (lp_actor.size(1) != L) or (lp_ref.size(1) != L) or (mask_tgt.size(1) != L):
        lp_actor = lp_actor[:, :L]
        lp_ref   = lp_ref[:, :L]
        mask_tgt = mask_tgt[:, :L]

    return lp_actor, lp_ref, mask_tgt


def compute_values_on_response(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,                 # [B, T]
    action_mask: torch.Tensor,          # [B, T]
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 values，并返回（values, mask_time），两者均为 [B, T]。
    实际训练时通常只在 response 段（mask_time=1）使用。
    """
    values = forward_values_via_actor(model, critic, seqs, device_type, ptdtype=ptdtype, micro_batch_size=micro_batch_size)
    mask_time = action_mask.clone()
    return values, mask_time


# =========================
# 其他小工具
# =========================

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


def apply_entropy_mask(
    logits: torch.Tensor,          # [B, T, V]
    action_mask: torch.Tensor,     # [B, T]
    keep_ratio: float = 0.2
) -> torch.Tensor:
    """
    基于 token 的熵筛选 action_mask，只保留高熵 token（例如 top-20%）。
    返回 new_action_mask: [B, T]，高熵 token 保留，其余置 0。
    """
    assert 0.0 < keep_ratio <= 1.0
    device = logits.device
    B, T, V = logits.shape

    probs = F.softmax(logits, dim=-1)             # [B, T, V]
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)    # [B, T]

    new_mask = torch.zeros_like(action_mask, dtype=torch.long, device=device)

    for i in range(B):
        act_pos = (action_mask[i] > 0).nonzero(as_tuple=False).squeeze(-1)
        if act_pos.numel() == 0:
            continue
        ent_vals = entropy[i, act_pos]
        k = max(1, int(ent_vals.numel() * keep_ratio))
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
