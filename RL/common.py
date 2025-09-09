# RL/common.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Union

from contextlib import nullcontext
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

# =========================================================
# 数据结构
# =========================================================

@dataclass
class Samples:
    """
    训练主循环用到的一致打包格式（prompt+response 已经拼到 seqs 里）。
    - action_mask: 只在 response 段为 1（含终止 token 前的可学习位）。
    - num_actions: 每条样本的 response token 数（= action_mask.sum(1)）。
    - response_length / total_length: 便于日志与断言。
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
    Trainer.evaluate_experience(...) 产出的最小可训练单元。
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
    """
    简单经验缓存（PPO/GRPO/DAPO 均可复用）。
    """
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

# =========================================================
# 文本归一化（奖励模型友好）
# =========================================================

def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)

def normalize_for_reward(text: str, reward_tokenizer=None) -> str:
    # 归一化 EOS 标记
    if reward_tokenizer is not None and getattr(reward_tokenizer, "eos_token", None):
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    # 换行统一
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 中英标点简易统一
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    # 去掉不可打印字符
    text = "".join(c for c in text if c.isprintable())
    # 去除多余空白（避免奖励侧不必要差异）
    return text.strip()

# =========================================================
# Ragged → Padded（构造 Samples）
# =========================================================

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
    兼容两种输入：
      - dict 内部直接是 Tensor：{"prompt_ids": Tensor[Ti], "full_ids": Tensor[Ti+Tj], "response_ids": Tensor[Tj]?}
      - 或 Python list[int]

    右 pad 到本 batch 的最大实际长度（不超过 block_size）。
    仅在 response 段打 action_mask。
    """
    assert len(gens) > 0, "gens 为空"

    if device is None:
        if torch.is_tensor(gens[0].get("full_ids", None)):
            device = gens[0]["full_ids"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    norm = []
    total_lens, resp_lens = [], []
    for g in gens:
        p = _to_long_tensor(g["prompt_ids"], device, "prompt_ids")
        f = _to_long_tensor(g["full_ids"],   device, "full_ids")
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

    seqs = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    action_mask    = torch.zeros((B, T_max), dtype=torch.long, device=device)
    # 先用“原始”长度占位，最后再根据 action_mask 纠正
    response_length = torch.tensor(resp_lens, dtype=torch.long, device=device)
    total_length    = torch.tensor([min(L, T_max) for L in total_lens], dtype=torch.long, device=device)

    for i, (p_ids, f_ids, r_len) in enumerate(norm):
        T_i = min(int(f_ids.numel()), T_max)
        if T_i > 0:
            seqs[i, :T_i] = f_ids[:T_i]
            attention_mask[i, :T_i] = 1
        T_prompt = min(int(p_ids.numel()), T_i)
        if T_i > T_prompt:
            action_mask[i, T_prompt:T_i] = 1

    # —— 关键修复：按“最终 action_mask”重新计算 response_length/num_actions ——
    num_actions = action_mask.sum(dim=1).to(torch.long)
    response_length = num_actions.clone()

    return Samples(
        seqs=seqs,
        attention_mask=attention_mask,
        action_mask=action_mask,
        num_actions=num_actions,
        response_length=response_length,
        total_length=total_length,
    )

# =========================================================
# 显存友好：micro-batch 前向
# =========================================================

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
    """
    对整段 seqs 做一次“全时间步”logits 推理（用于 token logprob）。
    - device_type: "cuda" | "cpu"
    - ptdtype:   显式推理精度（未给则用 model 参数 dtype）
    """
    if ptdtype is None:
        ptdtype = next(model.parameters()).dtype
    B = seqs.size(0)
    chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type != 'cpu') else nullcontext()
        with ctx:
            logits_chunk, _ = model(seqs[s:e], return_all_logits=True)
        chunks.append(logits_chunk)
        # 及时释放中间张量，避免堆叠
        del logits_chunk
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

# =========================================================
# token logprob / mask
# =========================================================

def token_logprobs_from_logits(
    logits: torch.Tensor,  # [B, T, V]
    seqs: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    # 对应 p(x_t | x_<t)，因此对齐为 [:-1] 与 [1:]
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    tgt = seqs[:, 1:].unsqueeze(-1)                  # [B, T-1, 1]
    return torch.gather(logp, dim=-1, index=tgt).squeeze(-1).contiguous()

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mask.sum().clamp_min(eps)
    return (x * mask).sum() / denom

# =========================================================
# KL / 熵 / 值函数
# =========================================================

def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    返回逐 token 的 log_ratio（不取均值）。k3 近似 KL 用下方工具计算。
    """
    log_ratio = log_probs.float() - ref_log_probs.float()
    return log_ratio if action_mask is None else (log_ratio * action_mask)

def kl_k3_from_logratio(log_ratio: torch.Tensor) -> torch.Tensor:
    """
    PPO 常用的对称 KL 近似：exp(lr) - 1 - lr
    """
    return log_ratio.exp() - 1.0 - log_ratio

def approx_kl_from_logprobs(
    actor_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    项目原有接口：返回 KL 的 masked 均值（两者相同，兼容老代码）。
    这里使用 k3 近似。
    """
    lr = (actor_logprobs - ref_logprobs)
    k3 = kl_k3_from_logratio(lr)
    kl_mean = masked_mean(k3, mask_target.float())
    return kl_mean, kl_mean

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
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

# =========================================================
# 优势/回报（GAE）
# =========================================================

def gae_compute(
    values: torch.Tensor,      # [B, T]
    rewards: torch.Tensor,     # [B, T] 或 [B]
    mask_time: torch.Tensor,   # [B, T]（1 表示有效 token）
    gamma: float = 1.0,
    lam: float = 0.95,
    use_last_as_terminal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    支持句末聚合奖励（scalar）或逐 token 奖励。
    当 use_last_as_terminal=True 时，不做末位 bootstrap（V_next_T = 0）。
    """
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
    V_next[:, :-1] = V[:, 1:]  # 末位 0（若 use_last_as_terminal=True，不做 bootstrap）

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

def get_advantages_and_returns(
    values: torch.Tensor,      # [B, T_resp]
    rewards: torch.Tensor,     # [B, T_resp]（已是 response 段）或 [B, 1] 句末
    action_mask: torch.Tensor, # [B, T_resp]
    gamma: float,
    lambd: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    参考实现接口的别名：按 response 时间轴计算 GAE。
    """
    returns, advantages = gae_compute(values, rewards, action_mask, gamma=gamma, lam=lambd)
    return advantages.detach(), returns.detach()

# =========================================================
# 便捷打包 / 奖励工具
# =========================================================

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
    """
    返回：
      - lp_actor, lp_ref: [B, T-1] 的逐 token logprob（与 seq 对齐）
      - mask_tgt:        [B, T-1] 的 action 区间掩码（= action_mask[:, 1:]）
    """
    forced_dtype = ptdtype if ptdtype is not None else torch.float32
    logits_actor = model_all_logits(actor, seqs, device_type, ptdtype=forced_dtype, micro_batch_size=micro_batch_size)
    logits_ref   = model_all_logits(ref,   seqs, device_type, ptdtype=forced_dtype, micro_batch_size=micro_batch_size)
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
    """
    取 actor 的 hidden 送给 critic，得到 [B, T] 的 values；
    再对齐到 response 时间轴（外层通常会切到 [:, 1:]）。
    """
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
    r_seq: torch.Tensor,             # [B] 句级奖励
    mask_time: torch.Tensor,         # [B, T]（response 时间轴）
    beta_kl: Optional[Tuple[torch.Tensor, float]] = None,  # (per_token_kl, beta)
) -> torch.Tensor:
    """
    将句级奖励散射到“最后一个 action token”，可选叠加 KL shaping。
    用于 PPO/GRPO 的“末位奖励”范式。
    """
    B, T = mask_time.shape
    rewards_t = torch.zeros((B, T), dtype=r_seq.dtype, device=mask_time.device)
    last_idx = last_indices_from_mask(mask_time)
    shaped = r_seq.clone()
    if beta_kl is not None:
        kl_t, beta = beta_kl
        denom = mask_time.sum(dim=1).clamp_min(1e-8).float()
        kl_mean = (kl_t * mask_time).sum(dim=1) / denom
        shaped = shaped - beta * kl_mean
    rewards_t.scatter_(1, last_idx.view(B, 1), shaped.unsqueeze(1))
    return rewards_t

def scatter_uniform_rewards(r_seq, mask_time, beta_kl=None):
    """
    将句级奖励在 response 区间“均匀摊”，并可按 KL 形状惩罚分摊。
    """
    B, T = mask_time.shape
    m = mask_time.float()
    denom = m.sum(dim=1, keepdim=True).clamp_min(1e-8)
    base = (r_seq.view(-1, 1) / denom) * m  # 均匀摊
    if beta_kl is None:
        return base
    k3, beta = beta_kl
    pen = beta * (k3 * m) / denom  # 与 token 级 k3 成比例分摊
    return base - pen

# =========================================================
# 诊断/统计工具（供日志打印）
# =========================================================

def ratio_stats(log_ratio: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float, float]:
    """
    给定逐 token log_ratio 和掩码，返回：
      - rΔ 的 q50 / q90 / q99（其中 rΔ = exp(lr)-1）
      - rΔ 的最大值
    """
    with torch.no_grad():
        m = mask.float()
        lr = log_ratio
        r_delta = lr.exp() - 1.0
        flat = r_delta[m > 0].detach()
        if flat.numel() == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
        return float(q[0].item()), float(q[1].item()), float(q[2].item()), float(flat.max().item())

def adv_abs_mean(advantages: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        a = (advantages.abs() * mask.float())
        denom = mask.sum().clamp_min(1.0)
        return float(a.sum().item() / denom.item())

# =========================================================
# 杂项
# =========================================================

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
    logits: torch.Tensor,      # [B, T, V]
    action_mask: torch.Tensor, # [B, T]
    keep_ratio: float = 0.25,
) -> torch.Tensor:
    """
    在 response 时间轴上选高熵 token 的子集，返回新的 action_mask。
    """
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
    "compute_approx_kl",
    "kl_k3_from_logratio",
    "approx_kl_from_logprobs",
    "entropy_from_logits",
    "critic_values_from_hidden",
    "forward_values_via_actor",
    "gae_compute",
    "get_advantages_and_returns",
    "masked_mean",
    "to_device_rec",
    "detach_rec",
    "clip_by_global_norm",
    "normalize_for_reward",
    "contains_chinese",
    "apply_entropy_mask",
    "last_indices_from_mask",
    "scatter_last_token_rewards",
    "scatter_uniform_rewards",
    "ratio_stats",
    "adv_abs_mean",
]
