# common.py
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
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: torch.Tensor
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
    num_actions: torch.Tensor
    kl: torch.Tensor

class ExperienceBuffer:
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

def _to_long_tensor(x, device, name: str):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    assert isinstance(x, (list, tuple)), f"{name} must be list/tensor"
    return torch.tensor(list(x), dtype=torch.long, device=device)

def build_samples_from_generations(
    gens: List[dict],
    block_size: int,
    pad_to_multiple_of: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    pad_id: int = 0,
) -> Samples:
    """
    支持两种输入：
      - dict 内部是 Tensor：{"prompt_ids": Tensor[Ti], "full_ids": Tensor[Ti+Tj], "response_ids": Tensor[Tj]?}
      - 或 List[int]
    自动右 pad 到本 batch 的最大实际长度（不超过 block_size）。
    只在 response 段打 action_mask。
    """
    assert len(gens) > 0, "gens 为空"
    # 推断 device
    if device is None:
        if torch.is_tensor(gens[0].get("full_ids", None)):
            device = gens[0]["full_ids"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 先标准化到张量，并收集长度
    norm = []
    total_lens = []
    resp_lens  = []
    for g in gens:
        p = _to_long_tensor(g["prompt_ids"], device, "prompt_ids")
        f = _to_long_tensor(g["full_ids"],   device, "full_ids")
        # 可选 response_ids；若没有则用差值估算
        if "response_ids" in g and g["response_ids"] is not None:
            r = _to_long_tensor(g["response_ids"], device, "response_ids")
            r_len = int(r.numel())
        else:
            r_len = max(int(f.numel()) - int(p.numel()), 0)
        norm.append((p, f, r_len))
        total_lens.append(int(f.numel()))
        resp_lens.append(int(r_len))

    B = len(norm)
    T_max = min(max(total_lens), block_size)
    if pad_to_multiple_of and T_max < block_size:
        T_max = min(_pad_to_multiple(T_max, pad_to_multiple_of), block_size)

    # 承载张量
    seqs = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    action_mask    = torch.zeros((B, T_max), dtype=torch.long, device=device)
    response_length = torch.tensor(resp_lens, dtype=torch.long, device=device)
    total_length    = torch.tensor([min(L, T_max) for L in total_lens], dtype=torch.long, device=device)

    # 填充
    for i, (p_ids, f_ids, r_len) in enumerate(norm):
        T_i = min(int(f_ids.numel()), T_max)
        if T_i > 0:
            seqs[i, :T_i] = f_ids[:T_i]
            attention_mask[i, :T_i] = 1
        T_prompt = min(int(p_ids.numel()), T_i)
        # 仅当确实有 response（T_i > T_prompt）时，才标 action_mask
        if T_i > T_prompt:
            action_mask[i, T_prompt:T_i] = 1

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
    seqs: torch.Tensor,
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
) -> torch.Tensor:
    if ptdtype is None:
        ptdtype = next(model.parameters()).dtype
    B = seqs.size(0)
    chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()
        with ctx:
            logits_chunk, _ = model(seqs[s:e], return_all_logits=True)
        chunks.append(logits_chunk)
        del logits_chunk
    return torch.cat(chunks, dim=0)

# =========================
# token logprob / mask
# =========================

def token_logprobs_from_logits(
    logits: torch.Tensor,  # [B, T, V]
    seqs: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    tgt = seqs[:, 1:].unsqueeze(-1)                  # [B, T-1, 1]
    return torch.gather(logp, dim=-1, index=tgt).squeeze(-1).contiguous()

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom

# =========================
# KL / 熵 / 值函数
# =========================

def approx_kl_from_logprobs(
    actor_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = (actor_logprobs - ref_logprobs)
    kl_mean = masked_mean(diff, mask_target.float())
    return kl_mean, kl_mean  # 保持接口兼容（两者同值）

def entropy_from_logits(logits: torch.Tensor, mask_time: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    return masked_mean(ent, mask_time.float())

def critic_values_from_hidden(
    critic: nn.Module,
    hidden_states: torch.Tensor,
    mask_time: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    v = critic(hidden_states)
    if v.dim() == 3 and v.size(-1) == 1:
        v = v.squeeze(-1)
    if mask_time is not None:
        v = v * mask_time
    return v

def forward_values_via_actor(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
    detach_hidden: bool = False,
) -> torch.Tensor:
    if ptdtype is None:
        ptdtype = next(model.parameters()).dtype
    B = seqs.size(0)
    chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()
        with ctx:
            _, _, h = model(seqs[s:e], return_hidden=True)
        if detach_hidden:
            h = h.detach()
        v = critic(h)
        if v.dim() == 3 and v.size(-1) == 1:
            v = v.squeeze(-1)
        chunks.append(v)
        del h, v
    return torch.cat(chunks, dim=0)

# =========================
# 优势/回报（GAE）
# =========================

def gae_compute(
    values: torch.Tensor,
    rewards: torch.Tensor,
    mask_time: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    use_last_as_terminal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T = values.shape
    device = values.device

    if rewards.dim() == 1:
        r_full = torch.zeros_like(values)
        last_idx = (mask_time * torch.arange(T, device=device).view(1, T)).argmax(dim=1)
        r_full[torch.arange(B, device=device), last_idx] = rewards
        rewards = r_full
    else:
        rewards = rewards * mask_time

    V = values
    V_next = torch.zeros_like(V)
    V_next[:, :-1] = V[:, 1:]
    # 若 use_last_as_terminal=True，末位默认 0，不做 bootstrap

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
# 便捷打包 / 奖励工具
# =========================

@torch.no_grad()
def _ref_logits(ref: nn.Module, seqs: torch.Tensor, device_type: str, micro_batch_size: int):
    return model_all_logits(ref, seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)

@torch.no_grad()
def compute_actor_ref_logprobs(
    actor: nn.Module,
    ref: nn.Module,
    seqs: torch.Tensor,
    action_mask: torch.Tensor,
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_actor = model_all_logits(actor, seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)
    logits_ref   = model_all_logits(ref,   seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)
    lp_actor = token_logprobs_from_logits(logits_actor, seqs)
    lp_ref   = token_logprobs_from_logits(logits_ref,   seqs)
    mask_tgt = action_mask[:, 1:].contiguous().long()
    L = min(lp_actor.size(1), lp_ref.size(1), mask_tgt.size(1))
    if (lp_actor.size(1) != L) or (lp_ref.size(1) != L) or (mask_tgt.size(1) != L):
        lp_actor = lp_actor[:, :L]
        lp_ref   = lp_ref[:, :L]
        mask_tgt = mask_tgt[:, :L]
    return lp_actor, lp_ref, mask_tgt

def compute_values_on_response(
    model: nn.Module,
    critic: nn.Module,
    seqs: torch.Tensor,
    action_mask: torch.Tensor,
    device_type: str,
    ptdtype: Optional[torch.dtype] = None,
    micro_batch_size: int = 0,
    detach_hidden: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = forward_values_via_actor(
        model, critic, seqs, device_type, ptdtype=ptdtype,
        micro_batch_size=micro_batch_size, detach_hidden=detach_hidden
    )
    mask_time = action_mask.clone()
    return values, mask_time

def last_indices_from_mask(mask_time: torch.Tensor) -> torch.Tensor:
    B, T = mask_time.shape
    ar = torch.arange(T, device=mask_time.device).view(1, T)
    return (mask_time * ar).amax(dim=1)

def scatter_last_token_rewards(
    r_seq: torch.Tensor,
    mask_time: torch.Tensor,
    beta_kl: Optional[Tuple[torch.Tensor, float]] = None,
) -> torch.Tensor:
    B, T = mask_time.shape
    rewards_t = torch.zeros((B, T), dtype=r_seq.dtype, device=mask_time.device)
    last_idx = last_indices_from_mask(mask_time)
    shaped = r_seq.clone()
    if beta_kl is not None:
        kl_t, beta = beta_kl
        kl_sum = (kl_t * mask_time).sum(dim=1)
        shaped = shaped - beta * kl_sum
    rewards_t.scatter_(1, last_idx.view(B, 1), shaped.unsqueeze(1))
    return rewards_t

# =========================
# 其他
# =========================

def to_device_rec(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device_rec(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device_rec(v, device) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj

def detach_rec(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    if isinstance(obj, dict):
        return {k: detach_rec(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [detach_rec(v) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj

def clip_by_global_norm(params: Iterable[torch.Tensor], max_norm: float) -> float:
    total_norm = torch.nn.utils.clip_grad_norm_(list(params), max_norm)
    return float(total_norm)

def apply_entropy_mask(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    keep_ratio: float = 0.2
) -> torch.Tensor:
    assert 0.0 < keep_ratio <= 1.0
    device = logits.device
    B, T, V = logits.shape
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)
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
    "last_indices_from_mask",
    "scatter_last_token_rewards",
]
