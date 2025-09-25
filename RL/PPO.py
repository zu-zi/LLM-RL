# RL/PPO.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Union
from contextlib import nullcontext
import math, random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

# ===================== 常量（稳定，不暴露为超参） =====================
_CAP_LOGR = 0.5      # clamp on Δlogp per token
_CAP_K3   = 1.5      # clamp on k3 per token
_ENT_KEEP = 0.25     # entropy mask keep ratio

# ===================== Data structs =====================
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
        if len(self.buffer) > self.limit: self.buffer = self.buffer[-self.limit:]
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def clear(self):
        self.buffer = []

# ===================== 文本归一 =====================
def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)

def normalize_for_reward(text: str, reward_tokenizer=None) -> str:
    if reward_tokenizer is not None and getattr(reward_tokenizer, "eos_token", None):
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if contains_chinese(text): text = text.replace("，", ",").replace("。", ".")
    text = "".join(c for c in text if c.isprintable()).replace("\ufffd", "")
    return text.strip()

# ===================== Ragged -> Padded =====================
def _pad_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1: return length
    r = length % multiple
    return length if r == 0 else (length + multiple - r)

def _to_long_tensor(x, device, name: str):
    if torch.is_tensor(x): return x.to(device=device, dtype=torch.long)
    assert isinstance(x, (list, tuple)), f"{name} must be list/tensor"
    return torch.tensor(list(x), dtype=torch.long, device=device)

def build_samples_from_generations(
    gens: List[dict],
    block_size: int,
    pad_to_multiple_of: int = 8,
    device: Optional[Union[str, torch.device]] = None,
    pad_id: int = 0,
) -> Samples:
    assert len(gens) > 0
    if device is None:
        device = gens[0]["full_ids"].device if torch.is_tensor(gens[0].get("full_ids", None)) \
                 else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    norm, total_lens, resp_lens = [], [], []
    for g in gens:
        p = _to_long_tensor(g["prompt_ids"], device, "prompt_ids")
        f = _to_long_tensor(g["full_ids"],   device, "full_ids")
        r_len = _to_long_tensor(g["response_ids"], device, "response_ids").numel() \
                if "response_ids" in g and g["response_ids"] is not None else max(f.numel()-p.numel(), 0)
        norm.append((p, f, int(r_len)))
        total_lens.append(int(f.numel())); resp_lens.append(int(r_len))

    B = len(norm)
    T_max = min(max(total_lens), block_size)
    if pad_to_multiple_of and T_max < block_size:
        T_max = min(_pad_to_multiple(T_max, pad_to_multiple_of), block_size)

    seqs = torch.full((B, T_max), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, T_max), dtype=torch.long, device=device)
    action_mask    = torch.zeros((B, T_max), dtype=torch.long, device=device)

    total_length    = torch.tensor([min(L, T_max) for L in total_lens], dtype=torch.long, device=device)
    for i, (p_ids, f_ids, r_len) in enumerate(norm):
        T_i = min(int(f_ids.numel()), T_max)
        if T_i > 0:
            seqs[i, :T_i] = f_ids[:T_i]; attention_mask[i, :T_i] = 1
        T_prompt = min(int(p_ids.numel()), T_i)
        if T_i > T_prompt: action_mask[i, T_prompt:T_i] = 1

    num_actions = action_mask.sum(dim=1).to(torch.long)
    response_length = num_actions.clone()

    return Samples(
        seqs=seqs, attention_mask=attention_mask, action_mask=action_mask,
        num_actions=num_actions, response_length=response_length, total_length=total_length,
    )

# ===================== Forward helpers =====================
def iter_batch_indices(n_items: int, micro_batch_size: int) -> Iterable[Tuple[int, int]]:
    if micro_batch_size is None or micro_batch_size <= 0 or micro_batch_size >= n_items:
        yield 0, n_items; return
    s = 0
    while s < n_items:
        e = min(s + micro_batch_size, n_items); yield s, e; s = e

def model_all_logits(
    model: nn.Module, seqs: torch.Tensor, device_type: str,
    ptdtype: Optional[torch.dtype] = None, micro_batch_size: int = 0,
) -> torch.Tensor:
    if ptdtype is None: ptdtype = next(model.parameters()).dtype
    B = seqs.size(0); chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type != 'cpu') else nullcontext()
        with ctx:
            logits_chunk, _ = model(seqs[s:e], return_all_logits=True)
        chunks.append(logits_chunk); del logits_chunk
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

def token_logprobs_from_logits(logits: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)   # [B, T-1, V]
    tgt  = seqs[:, 1:].unsqueeze(-1)                  # [B, T-1, 1]
    return torch.gather(logp, dim=-1, index=tgt).squeeze(-1).contiguous()

# ===================== Basic math =====================
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    denom = mask.sum().clamp_min(eps); return (x * mask).sum() / denom

# ===================== KL / Entropy / Critic routing =====================
def compute_approx_kl(log_probs: torch.Tensor, ref_log_probs: torch.Tensor, action_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    lr = log_probs.float() - ref_log_probs.float()
    return lr if action_mask is None else (lr * action_mask)

def kl_k3_from_logratio(log_ratio: torch.Tensor) -> torch.Tensor:
    return log_ratio.exp() - 1.0 - log_ratio

def approx_kl_from_logprobs(actor_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, mask_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    lr = (actor_logprobs - ref_logprobs)
    k3 = kl_k3_from_logratio(lr)
    kl_mean = masked_mean(k3, mask_target.float())
    return kl_mean, kl_mean

def entropy_from_logits(logits: torch.Tensor, mask_time: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    return masked_mean(ent, mask_time.float())

def forward_values_via_actor(
    model: nn.Module, critic: nn.Module, seqs: torch.Tensor, device_type: str,
    ptdtype: Optional[torch.dtype] = None, micro_batch_size: int = 0, detach_hidden: bool = False,
) -> torch.Tensor:
    if ptdtype is None: ptdtype = next(model.parameters()).dtype
    B = seqs.size(0); chunks = []
    for s, e in iter_batch_indices(B, micro_batch_size):
        ctx = amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()
        with ctx:
            _, _, h = model(seqs[s:e], return_hidden=True)
        if detach_hidden: h = h.detach()
        v = critic(h)
        if v.dim() == 3 and v.size(-1) == 1: v = v.squeeze(-1)
        chunks.append(v); del h, v
    return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

# ===================== GAE / returns =====================
def gae_compute(
    values: torch.Tensor, rewards: torch.Tensor, mask_time: torch.Tensor,
    gamma: float = 1.0, lam: float = 0.95, use_last_as_terminal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T = values.shape; device = values.device
    if rewards.dim() == 1:
        r_full = torch.zeros_like(values)
        last_idx = (mask_time * torch.arange(T, device=device).view(1, T)).argmax(dim=1)
        r_full[torch.arange(B, device=device), last_idx] = rewards
        rewards = r_full
    else:
        rewards = rewards * mask_time

    V = values; V_next = torch.zeros_like(V); V_next[:, :-1] = V[:, 1:]
    deltas = (rewards + gamma * V_next - V) * mask_time

    advantages = torch.zeros_like(V); gae = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        mask_t = mask_time[:, t]
        gae = deltas[:, t] + gamma * lam * gae * mask_t
        advantages[:, t] = gae

    returns = advantages + V
    advantages = advantages * mask_time; returns = returns * mask_time
    return returns, advantages

def get_advantages_and_returns(values: torch.Tensor, rewards: torch.Tensor, action_mask: torch.Tensor, gamma: float, lambd: float):
    return gae_compute(values, rewards, action_mask, gamma=gamma, lam=lambd)

# ===================== Reward 与杂项 =====================
@torch.no_grad()
def _ref_logits(ref: nn.Module, seqs: torch.Tensor, device_type: str, micro_batch_size: int):
    return model_all_logits(ref, seqs, device_type, ptdtype=None, micro_batch_size=micro_batch_size)

@torch.no_grad()
def compute_actor_ref_logprobs(
    actor: nn.Module, ref: nn.Module, seqs: torch.Tensor, action_mask: torch.Tensor,
    device_type: str, ptdtype: Optional[torch.dtype] = None, micro_batch_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forced_dtype = ptdtype if ptdtype is not None else torch.float32
    logits_actor = model_all_logits(actor, seqs, device_type, ptdtype=forced_dtype, micro_batch_size=micro_batch_size)
    logits_ref   = model_all_logits(ref,   seqs, device_type, ptdtype=forced_dtype, micro_batch_size=micro_batch_size)
    lp_actor = token_logprobs_from_logits(logits_actor, seqs)
    lp_ref   = token_logprobs_from_logits(logits_ref,   seqs)
    mask_tgt = action_mask[:, 1:].contiguous().long()
    L = min(lp_actor.size(1), lp_ref.size(1), mask_tgt.size(1))
    if (lp_actor.size(1) != L) or (lp_ref.size(1) != L) or (mask_tgt.size(1) != L):
        lp_actor = lp_actor[:, :L]; lp_ref = lp_ref[:, :L]; mask_tgt = mask_tgt[:, :L]
    return lp_actor, lp_ref, mask_tgt

def last_indices_from_mask(mask_time: torch.Tensor) -> torch.Tensor:
    B, T = mask_time.shape; ar = torch.arange(T, device=mask_time.device).view(1, T)
    return (mask_time * ar).amax(dim=1)

def scatter_last_token_rewards(r_seq: torch.Tensor, mask_time: torch.Tensor, beta_kl: Optional[Tuple[torch.Tensor, float]] = None) -> torch.Tensor:
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
    B, T = mask_time.shape; m = mask_time.float()
    denom = m.sum(dim=1, keepdim=True).clamp_min(1e-8)
    base = (r_seq.view(-1, 1) / denom) * m
    if beta_kl is None: return base
    k3, beta = beta_kl; pen = beta * (k3 * m) / denom
    return base - pen

# ===================== Diagnostics =====================
def ratio_stats(log_ratio: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float, float, float]:
    with torch.no_grad():
        m = mask.float(); r_delta = (log_ratio).exp() - 1.0
        flat = r_delta[m > 0].detach()
        if flat.numel() == 0: return float("nan"), float("nan"), float("nan"), float("nan")
        q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99], device=flat.device))
        return float(q[0].item()), float(q[1].item()), float(q[2].item()), float(flat.max().item())

def adv_abs_mean(advantages: torch.Tensor, mask: torch.Tensor) -> float:
    with torch.no_grad():
        a = (advantages.abs() * mask.float()); denom = mask.sum().clamp_min(1.0)
        return float(a.sum().item() / denom.item())

# ===================== Utils =====================
def to_device_rec(obj, device):
    if torch.is_tensor(obj): return obj.to(device)
    if isinstance(obj, dict): return {k: to_device_rec(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device_rec(v, device) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj

def detach_rec(obj):
    if torch.is_tensor(obj): return obj.detach()
    if isinstance(obj, dict): return {k: detach_rec(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [detach_rec(v) for v in obj]
        return type(obj)(t) if not isinstance(obj, tuple) else tuple(t)
    return obj

def clip_by_global_norm(params, max_norm: float) -> float:
    total = torch.nn.utils.clip_grad_norm_(list(params), max_norm)
    return float(total)

def apply_entropy_mask(logits: torch.Tensor, action_mask: torch.Tensor, keep_ratio: float = _ENT_KEEP) -> torch.Tensor:
    assert 0.0 < keep_ratio <= 1.0
    device = logits.device; B, T, V = logits.shape
    probs = F.softmax(logits, dim=-1); log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum(dim=-1)
    new_mask = torch.zeros_like(action_mask, dtype=torch.long, device=device)
    for i in range(B):
        act_pos = (action_mask[i] > 0).nonzero(as_tuple=False).squeeze(-1)
        if act_pos.numel() == 0: continue
        ent_vals = entropy[i, act_pos]
        k = max(1, int(ent_vals.numel() * keep_ratio))
        topk_idx = torch.topk(ent_vals, k=k, largest=True).indices
        selected_pos = act_pos[topk_idx]
        new_mask[i, selected_pos] = 1
    return new_mask

# ===================== 数值稳定 =====================
def _clean_logp(x: torch.Tensor, fallback: torch.Tensor = None):
    if fallback is not None and fallback.shape == x.shape:
        return torch.where(torch.isfinite(x), x, fallback)
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))

# ===================== Critic =====================
class Critic(nn.Module):
    def __init__(self, actor_like: nn.Module, width: int = 2, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        h = getattr(getattr(actor_like, "config", None), "n_embd", None) \
            or getattr(getattr(actor_like, "config", None), "hidden_size", None)
        if h is None: raise ValueError("Cannot infer hidden size from actor model config.")
        layers = []
        for _ in range(int(depth)):
            layers += [nn.LayerNorm(h), nn.Linear(h, width*h), nn.GELU(), nn.Dropout(dropout), nn.Linear(width*h, h)]
        self.adapter = nn.Sequential(*layers) if layers else nn.Identity()
        self.value_head = nn.Linear(h, 1)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        z = self.adapter(hidden_states); return self.value_head(z)

# ===================== PPO Trainer =====================
class PPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module, ref_model: nn.Module, reward_model: nn.Module, critic_model: nn.Module,
        actor_tokenizer, reward_tokenizer, optimizer_actor, optimizer_critic,
        device: str = "cuda", mb_size_logits: int = 1, mb_size_values: int = 1,
        kl_ctl: float = 0.3, ppo_clip: float = 0.2, vf_clip: Optional[float] = 0.2,
        entropy_coef: float = 0.0, max_grad_norm: float = 1.0, gae_gamma: float = 1.0, gae_lambda: float = 0.95,
    ):
        self.actor, self.ref, self.critic = actor_model, ref_model, critic_model
        self.reward_model, self.actor_tok, self.reward_tok = reward_model, actor_tokenizer, reward_tokenizer
        if getattr(self.reward_tok, "pad_token_id", None) is None and getattr(self.reward_tok, "eos_token", None) is not None:
            self.reward_tok.pad_token = self.reward_tok.eos_token
        try: self.reward_tok.padding_side = "right"
        except Exception: pass

        self.opt_actor, self.opt_critic = optimizer_actor, optimizer_critic
        self.device = device; self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.mb_logits, self.mb_values = max(1, int(mb_size_logits)), max(1, int(mb_size_values))

        self.kl_ctl, self.ppo_clip, self.vf_clip = float(kl_ctl), float(ppo_clip), vf_clip
        self.entropy_coef, self.max_grad_norm = float(entropy_coef), float(max_grad_norm)
        self.gae_gamma, self.gae_lambda = float(gae_gamma), float(gae_lambda)

        self.last_stats = {}

        self.kl_target = 0.20     
        self.kl_adapt_rate = 0.20  
        self.kl_ctl_min, self.kl_ctl_max = 0.05, 10.0  
        
    # ----- Reward scoring -----
    @torch.no_grad()
    def _decode_dialogue_and_score(self, seqs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = seqs.size(); texts = []
        for i in range(B):
            L_i = int(attention_mask[i].sum().item())
            ids = seqs[i, :L_i].detach().cpu().tolist()
            raw = self.actor_tok.decode(ids) if hasattr(self.actor_tok, "decode") else self.actor_tok.batch_decode([ids])[0]
            txt = normalize_for_reward(raw, reward_tokenizer=self.reward_tok).replace("\ufffd", "")
            if "Assistant:" not in txt and "Human:" in txt: txt = txt.rstrip() + "\n\nAssistant:"
            texts.append(txt)

        toks = self.reward_tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        outs = self.reward_model(**{k: v for k, v in toks.items()})
        logits = getattr(outs, "logits", None)
        if logits is None: return torch.zeros(B, dtype=torch.float32, device=self.device)
        if logits.dim() == 2 and logits.size(-1) == 1: logits = logits.squeeze(-1)
        return logits.detach().to(self.device).float()

    # ----- Step1: rollouts -> Experience -----
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples):
        seqs = samples.seqs.to(self.device)
        attn = samples.attention_mask.to(self.device)
        amsk = samples.action_mask.to(self.device)
        mask_tgt = amsk[:, 1:]; assert int(mask_tgt.sum().item()) == int(samples.num_actions.sum().item())

        lp_actor_full, lp_ref_full, mask_target = compute_actor_ref_logprobs(
            self.actor, self.ref, seqs, amsk, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits
        )
        lp_actor_full = _clean_logp(lp_actor_full, fallback=lp_ref_full)
        lp_ref_full   = _clean_logp(lp_ref_full,   fallback=lp_actor_full)

        log_ratio_rep = (lp_actor_full - lp_ref_full).clamp_(-8.0, 8.0)
        k3_report = torch.expm1(log_ratio_rep) - log_ratio_rep
        k3_report = torch.clamp(k3_report, 0.0, 50.0)
        report_kl = float(masked_mean(k3_report * mask_target, mask_target.float()).detach().item())

        log_ratio = (lp_actor_full - lp_ref_full).clamp(-_CAP_LOGR, _CAP_LOGR)
        k3 = torch.expm1(log_ratio) - log_ratio
        k3 = torch.clamp(k3, 0.0, _CAP_K3) * mask_target

        denom = mask_target.sum(dim=1).clamp_min(1e-8).float()
        safe_kl_seq = k3.sum(dim=1) / denom
        safe_kl = float(safe_kl_seq.mean().detach().item())

        values_full = forward_values_via_actor(self.actor, self.critic, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_values, detach_hidden=False)
        values = values_full[:, 1:]
        action_mask = mask_target

        r_seq = self._decode_dialogue_and_score(seqs, attn)
        r_raw_mean = float(r_seq.mean().detach().item())

        rewards_t = scatter_uniform_rewards(r_seq, action_mask, beta_kl=None)

        denom2 = action_mask.sum(dim=1).clamp_min(1e-8).float()
        kl_mean_per_seq = (k3_report * action_mask).sum(dim=1) / denom2
        r_shaped_seq = r_seq - self.kl_ctl * kl_mean_per_seq
        r_shaped_mean = float(r_shaped_seq.mean().detach().item())
        r_centered_mean = float((r_seq - r_seq.mean()).mean().detach().item())

        returns, advantages = gae_compute(values=values, rewards=rewards_t, mask_time=action_mask, gamma=self.gae_gamma, lam=self.gae_lambda, use_last_as_terminal=True)

        experiences = []
        B = seqs.size(0)
        for i in range(B):
            experiences.append(Experience(
                seqs=seqs[i:i+1],
                action_log_probs=lp_actor_full[i:i+1].detach(),
                values=values[i:i+1].detach(),
                returns=returns[i:i+1].detach(),
                advantages=advantages[i:i+1].detach(),
                attention_mask=attn[i:i+1],
                action_mask=action_mask[i:i+1],
                reward=r_seq[i:i+1].detach(),
                num_actions=action_mask[i:i+1].sum(dim=1).to(torch.long),
                kl=k3_report[i:i+1].detach(),
            ))

        with torch.no_grad():
            approx_kl_pi = float(masked_mean(k3_report * action_mask, action_mask.float()).detach().item())
            logits_actor = model_all_logits(self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
            entropy_tok = float(entropy_from_logits(logits_actor[:, 1:, :], action_mask).detach().item())
            v_mae = float(((values - returns).abs() * action_mask).sum() / action_mask.sum().clamp_min(1e-8))

            m = action_mask.float(); y = returns
            y_mean = (y * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            var_y = (((y - y_mean) ** 2) * m).sum() / m.sum().clamp_min(1e-8)
            err = (returns - values)
            e_mean = (err * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            var_err = (((err - e_mean) ** 2) * m).sum() / m.sum().clamp_min(1e-8)
            explained_var = float(1.0 - (var_err / var_y.clamp_min(1e-8)))

        self.last_stats.update({
            "approx_kl_pi": float(approx_kl_pi),
            "entropy": float(entropy_tok),
            "v_mae": float(v_mae),
            "explained_var": float(explained_var),
        })

        return experiences, report_kl, r_raw_mean, r_shaped_mean, r_centered_mean, safe_kl

    # ----- Step2: train (actor + critic) -----
    def train_on_experience(self, exp: Experience, use_token_entropy: bool = False):
        seqs, attn, action_mask = exp.seqs, exp.attention_mask, exp.action_mask
        old_logp, returns, old_values, adv = exp.action_log_probs, exp.returns, exp.values, exp.advantages

        with torch.no_grad():
            m = action_mask.float()
            denom = m.sum(dim=1, keepdim=True).clamp_min(1e-8)
            mean = (adv * m).sum(dim=1, keepdim=True) / denom
            var  = (((adv - mean) ** 2) * m).sum(dim=1, keepdim=True) / denom
            std  = var.sqrt().clamp_min(1e-6)
            adv.copy_(((adv - mean) / std).clamp_(-5.0, 5.0))

        # ----- Actor -----
        self.actor.train(); self.opt_actor.zero_grad(set_to_none=True)

        logits = model_all_logits(self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)  # [B, T, V]
        logp_all = token_logprobs_from_logits(logits, seqs)  # [B, T-1]
        logp_all = _clean_logp(logp_all)

        L = old_logp.size(1)
        logp = logp_all[:, :L]; action_mask = action_mask[:, :L]; attn = attn[:, :L]

        if use_token_entropy:
            sel_mask = apply_entropy_mask(logits[:, 1:, :].detach(), action_mask, keep_ratio=_ENT_KEEP)
        else:
            sel_mask = action_mask

        raw_delta = logp - old_logp
        ratio = torch.exp(raw_delta)  # 不再引入额外 ratio_min/max

        surr1 = ratio * (adv * sel_mask)
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * (adv * sel_mask)
        policy_loss_tok = -torch.min(surr1, surr2)

        if self.entropy_coef != 0.0:
            ent_tok = -(F.softmax(logits[:, 1:, :], dim=-1).clamp_min(1e-12) * F.log_softmax(logits[:, 1:, :], dim=-1)).sum(-1)
            policy_loss_tok = policy_loss_tok - self.entropy_coef * ent_tok * sel_mask

        policy_loss = policy_loss_tok.sum() / sel_mask.sum().clamp_min(1e-8)

        with torch.no_grad():
            logits_ref = model_all_logits(self.ref, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
            lp_ref_full = token_logprobs_from_logits(logits_ref, seqs)
            lp_ref_full = _clean_logp(lp_ref_full, fallback=logp_all)

        Lr = min(logp.size(1), lp_ref_full.size(1))
        lp_ref = lp_ref_full[:, :Lr]; logp_cut = logp[:, :Lr]; sel = sel_mask[:, :Lr].float()

        log_ratio = (logp_cut - lp_ref).clamp(-_CAP_LOGR, _CAP_LOGR)
        k3 = torch.expm1(log_ratio) - log_ratio
        k3 = torch.clamp(k3, 0.0, _CAP_K3)
        kl_mean = (k3 * sel).sum() / sel.sum().clamp_min(1e-8)
        policy_loss = policy_loss + self.kl_ctl * kl_mean
        # KL控制
        with torch.no_grad():
            err = kl_mean.detach() / max(self.kl_target, 1e-8) - 1.0
            step = float(torch.clamp(err, -self.kl_adapt_rate, self.kl_adapt_rate).item())
            self.kl_ctl = float(self.kl_ctl * math.exp(step))
            self.kl_ctl = float(max(self.kl_ctl_min, min(self.kl_ctl, self.kl_ctl_max)))
            self.last_stats["kl_ctl_now"] = float(self.kl_ctl)
            self.last_stats["kl_mean_train"] = float(kl_mean.detach().item())

        policy_loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0: clip_by_global_norm(self.actor.parameters(), self.max_grad_norm)
        self.opt_actor.step()

        with torch.no_grad():
            Lm = ratio.size(1); sel = sel_mask[:, :Lm].to(ratio.dtype)
            over = (torch.abs(ratio - 1.0) > self.ppo_clip).to(ratio.dtype)
            clipped = over * sel; denom = sel.sum().clamp_min(1e-8)
            self.last_stats["clip_frac"] = float((clipped.sum() / denom).item())

        # ----- Critic -----
        self.critic.train(); self.opt_critic.zero_grad(set_to_none=True)

        values_full = forward_values_via_actor(self.actor, self.critic, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_values, detach_hidden=True)
        values_new = values_full[:, 1:]

        if self.vf_clip is not None:
            values_clipped = old_values + (values_new - old_values).clamp(-self.vf_clip, self.vf_clip)
            vloss1 = (values_new - returns) ** 2
            vloss2 = (values_clipped - returns) ** 2
            v_loss_tok = torch.max(vloss1, vloss2)
        else:
            v_loss_tok = F.smooth_l1_loss(values_new, returns, beta=1.0, reduction="none")

        v_loss = (v_loss_tok * action_mask).sum() / action_mask.sum().clamp_min(1e-8)
        v_loss.backward()
        clip_by_global_norm(self.critic.parameters(), 0.5)
        self.opt_critic.step()
        return policy_loss.detach(), v_loss.detach()
