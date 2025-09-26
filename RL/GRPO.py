# RL/GRPO.py
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 PPO 工具
from .PPO import (
    Samples,
    build_samples_from_generations,
    model_all_logits,
    token_logprobs_from_logits,
    masked_mean,
    entropy_from_logits,
    normalize_for_reward,
    _clean_logp,
)

# utils
def _flatten_groups(groups: List[List[dict]]) -> Tuple[List[dict], List[Tuple[int, int]]]:
    flat, meta = [], []
    s = 0
    for g in groups:
        flat.extend(g)
        e = s + len(g)
        meta.append((s, e))
        s = e
    return flat, meta

def _group_slices(n: int, meta: List[Tuple[int, int]]):
    for s, e in meta:
        if 0 <= s < e <= n and (e - s) >= 2:
            yield s, e

# GRPO
class GRPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        device: str = "cuda",
        mb_size_logits: int = 1,
        tau: float = 0.5,
        kl_ctl: float = 0.3,
        length_norm: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.actor = actor_model
        self.ref = ref_model
        self.rm = reward_model
        self.actor_tok = actor_tokenizer
        self.rm_tok = reward_tokenizer
        if getattr(self.rm_tok, "pad_token_id", None) is None and getattr(self.rm_tok, "eos_token", None):
            self.rm_tok.pad_token = self.rm_tok.eos_token
        try: self.rm_tok.padding_side = "right"
        except Exception: pass

        self.opt = optimizer_actor
        self.device = torch.device(device)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.mb_logits = max(1, int(mb_size_logits))

        self.tau = float(tau)
        self.kl_ctl = float(kl_ctl)
        self.length_norm = bool(length_norm)
        self.max_grad_norm = float(max_grad_norm)

        self.last_stats: Dict[str, float] = {}

    # RM
    @torch.no_grad()
    def _rm_score(self, seqs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = seqs.size()
        texts = []
        for i in range(B):
            L = int(attention_mask[i].sum().item())
            ids = seqs[i, :L].detach().cpu().tolist()
            raw = self.actor_tok.decode(ids) if hasattr(self.actor_tok, "decode") else self.actor_tok.batch_decode([ids])[0]
            txt = normalize_for_reward(raw, reward_tokenizer=self.rm_tok).replace("\ufffd", "")
            if "Assistant:" not in txt and "Human:" in txt:
                txt = txt.rstrip() + "\n\nAssistant:"
            texts.append(txt)

        rm_dev = next(self.rm.parameters()).device
        toks = self.rm_tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        toks = {k: v.to(rm_dev) for k, v in toks.items()}

        outs = self.rm(**toks)
        logits = getattr(outs, "logits", None)
        if logits is None:
            return torch.zeros(B, device=self.device, dtype=torch.float32)
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        return logits.detach().to(self.device).float()

    # actor/ref 的 token logp
    def _seq_logp(self, seqs: torch.Tensor):
        logits_a = model_all_logits(self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
        lp_a = token_logprobs_from_logits(logits_a, seqs)  # [B, T-1]

        with torch.no_grad():
            logits_r = model_all_logits(self.ref, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
            lp_r = token_logprobs_from_logits(logits_r, seqs)

        lp_a = _clean_logp(lp_a, fallback=lp_r)
        lp_r = _clean_logp(lp_r, fallback=lp_a)
        return logits_a, lp_a, lp_r

    def _pack(self, groups: List[List[dict]], block_size: int) -> Tuple[Samples, List[Tuple[int, int]]]:
        flat, meta = _flatten_groups(groups)
        smp = build_samples_from_generations(flat, block_size=block_size, device=self.device)
        return smp, meta

    # 
    def step_on_groups(self, groups: List[List[dict]], block_size: int) -> Dict[str, float]:
        smp, meta = self._pack(groups, block_size)
        if int(smp.action_mask.sum().item()) == 0:
            return {"skip": True}
    
        seqs = smp.seqs.to(self.device)
        attn = smp.attention_mask.to(self.device)
        actm = smp.action_mask.to(self.device)
    
        logits_a, lp_a_full, lp_r_full = self._seq_logp(seqs)
    
        mask_t = actm[:, 1:].contiguous().long()
        L = min(lp_a_full.size(1), lp_r_full.size(1), mask_t.size(1))
        lp_a = lp_a_full[:, :L]
        lp_r = lp_r_full[:, :L]
        mask_t = mask_t[:, :L]
        denom = mask_t.sum(dim=1).clamp_min(1e-8).float()
    
        # 长度归一
        alpha = 1.0 if self.length_norm else 0.0
        lp_resp = (lp_a * mask_t).sum(dim=1) / (denom ** alpha)
    
        # KL 近似放宽 + 记录 forward KL 近似
        with torch.no_grad():
            lr = (lp_a - lp_r).clamp(-4.0, 4.0)
            k3 = torch.expm1(lr) - lr
            k3 = torch.clamp(k3, 0.0, 10.0)
            kl_seq = (k3 * mask_t).sum(dim=1) / denom
    
            kl_forward = ((lp_a - lp_r).exp() - 1 - (lp_a - lp_r))
            kl_forward = (kl_forward * mask_t).sum(dim=1) / denom
    
        with torch.no_grad():
            r = self._rm_score(seqs, attn)
    
        group_losses = []
        groups_cnt = 0
        total_items = 0
        pstar_ent_list = []
        r_std_acc, r_iqr_acc, r_grp_cnt = 0.0, 0.0, 0
    
        for s, e in _group_slices(seqs.size(0), meta):
            if e - s < 2:
                continue
            r_g  = r[s:e]
            lp_g = lp_resp[s:e]
    
            # 组内标准化
            mu = r_g.mean()
            std = r_g.std()
            if float(std.item()) < 1e-6:
                p_star = torch.full_like(r_g, 1.0 / r_g.numel())
            else:
                r_scale = (r_g - mu) / (std + 1e-6)
                p_star  = F.softmax(r_scale / max(self.tau, 1e-6), dim=0)
    
            log_p_theta = F.log_softmax(lp_g, dim=0)
            ce  = -(p_star * log_p_theta).sum()
            kl_g = kl_seq[s:e].mean()
    
            group_losses.append(ce + self.kl_ctl * kl_g)
            groups_cnt += 1
            total_items += (e - s)
    
            # 统计
            ps = p_star
            pstar_ent_list.append(float(-(ps * ps.clamp_min(1e-12).log()).sum().item()))
            r_std_acc += float(std.item())
            # IQR
            q25, q75 = torch.quantile(r_g, 0.25), torch.quantile(r_g, 0.75)
            r_iqr_acc += float((q75 - q25).item())
            r_grp_cnt += 1
    
        if not group_losses:
            return {"skip": True}
    
        loss = torch.stack(group_losses).mean()
    
        self.actor.train()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt.step()
    
        with torch.no_grad():
            ent_tok = float(entropy_from_logits(logits_a[:, 1:, :], mask_t).item())
            kl_mean = float(kl_seq.mean().item())
            kl_fwd_mean = float(kl_forward.mean().item())
            r_mean = float(r.mean().item())
            pstar_entropy = sum(pstar_ent_list) / max(len(pstar_ent_list), 1)
            r_std_mean = r_std_acc / max(r_grp_cnt, 1)
            r_iqr_mean = r_iqr_acc / max(r_grp_cnt, 1)
    
        self.last_stats.update({
            "loss": float(loss.item()),
            "kl_mean": kl_mean,
            "kl_forward_mean": kl_fwd_mean,
            "rm_mean": r_mean,
            "entropy_tok": ent_tok,
            "tau": float(self.tau),
            "groups": int(groups_cnt),
            "items": int(total_items),
            "pstar_entropy": float(pstar_entropy),
            "r_std": float(r_std_mean),    
            "r_iqr": float(r_iqr_mean),     
            "skip": False,
        })
        return dict(self.last_stats)