# RL/DAPO.py
from typing import List, Tuple, Iterable, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 PPO 工具
from .PPO import (
    Samples,
    build_samples_from_generations,
    model_all_logits,
    token_logprobs_from_logits,
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

# DAPO
class DAPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        reward_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        ref_model: Optional[nn.Module] = None,   # 可选
        device: str = "cuda",
        mb_size_logits: int = 1,
        tau: float = 0.3,            # 软目标温度（小 -> 更尖锐）
        kl_ctl: float = 0.0,         # 可选 KL 系数（默认 0：关闭）
        length_norm: bool = True,    # 长度归一（按 token 数）
        max_grad_norm: float = 1.0,
        he_on: bool = False, 
        he_frac: float = 0.2, 
        he_temp: float = 1.0,
    ):
        # 模型 / 设备
        self.actor = actor_model
        self.ref = ref_model
        self.rm = reward_model
        self.actor_tok = actor_tokenizer
        self.rm_tok = reward_tokenizer
        if getattr(self.rm_tok, "pad_token_id", None) is None and getattr(self.rm_tok, "eos_token", None):
            self.rm_tok.pad_token = self.rm_tok.eos_token
        try:
            self.rm_tok.padding_side = "right"
        except Exception:
            pass

        self.opt = optimizer_actor
        self.device = torch.device(device)
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.mb_logits = max(1, int(mb_size_logits))

        self.tau = float(tau)
        self.kl_ctl = float(kl_ctl)
        self.length_norm = bool(length_norm)
        self.max_grad_norm = float(max_grad_norm)

        self.last_stats: Dict[str, float] = {}
        self.he_on   = bool(he_on)
        self.he_frac = float(he_frac)
        self.he_temp = float(he_temp)

    # token entropy
    @torch.no_grad()
    def _forking_mask(self, logits_next: torch.Tensor, mask_t: torch.Tensor):
        logits = logits_next / max(self.he_temp, 1e-6)
        probs  = torch.softmax(logits, dim=-1)                       # [B, L, V]
        H      = -(probs * (probs.clamp_min(1e-12).log())).sum(-1)   # [B, L]
        H_resp = H * mask_t.float()
    
        he_mask = torch.zeros_like(mask_t)
        total_sel = 0
        total_resp = int(mask_t.sum().item())
    
        for i in range(H_resp.size(0)):
            m = mask_t[i].bool()
            n = int(m.sum().item())
            if n <= 0:
                continue
            k = max(1, int(round(n * self.he_frac)))
            topk_vals = torch.topk(H_resp[i][m], k=k, largest=True).values
            thr = topk_vals.min()
            sel = (H_resp[i] >= thr) & m
            he_mask[i] = sel.long()
            total_sel += int(sel.sum().item())
    
        sel_ratio = (total_sel / max(total_resp, 1)) if total_resp > 0 else 0.0
        return he_mask, H_resp, float(sel_ratio)
    

    # RM
    @torch.no_grad()
    def _rm_score(self, seqs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, _ = seqs.size()
        texts = []
        for i in range(B):
            L = int(attention_mask[i].sum().item())
            ids = seqs[i, :L].detach().cpu().tolist()
            raw = (
                self.actor_tok.decode(ids)
                if hasattr(self.actor_tok, "decode")
                else self.actor_tok.batch_decode([ids])[0]
            )
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

    # token logp
    def _seq_logp(self, seqs: torch.Tensor):
        # actor logits: 保留梯度
        logits_a = model_all_logits(self.actor, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
        lp_a = token_logprobs_from_logits(logits_a, seqs)  # [B, T-1]

        # 参考策略: 若存在；无梯度
        if self.ref is not None:
            with torch.no_grad():
                logits_r = model_all_logits(self.ref, seqs, self.device_type, ptdtype=torch.float32, micro_batch_size=self.mb_logits)
                lp_r = token_logprobs_from_logits(logits_r, seqs)
        else:
            lp_r = None

        if lp_r is not None:
            lp_a = _clean_logp(lp_a, fallback=lp_r)
            lp_r = _clean_logp(lp_r, fallback=lp_a)
        else:
            lp_a = _clean_logp(lp_a, fallback=None)

        return logits_a, lp_a, lp_r

    # 
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
        actm = smp.action_mask.to(self.device)      # [B, T]
    
        logits_a, lp_a_full, lp_r_full = self._seq_logp(seqs)
    
        # 对齐到响应段
        mask_t = actm[:, 1:].contiguous().long()
        L = min(lp_a_full.size(1), mask_t.size(1))
        if lp_r_full is not None:
            L = min(L, lp_r_full.size(1))
        lp_a = lp_a_full[:, :L]
        mask_t = mask_t[:, :L]
    
        # 新增：高熵掩码
        if self.he_on and self.he_frac > 0.0:
            he_mask, H_resp, he_ratio = self._forking_mask(logits_a[:, 1:, :][:, :L, :], mask_t)
            mask_use = he_mask
        else:
            H_resp = torch.zeros_like(mask_t, dtype=torch.float)
            he_ratio = 1.0
            mask_use = mask_t
    
        # 从这往下，聚合与统计一律用 mask_use
        denom = mask_use.sum(dim=1).clamp_min(1e-8).float()
    
        # 序列响应 logp（可导），α-长度归一
        alpha = 1.0 if self.length_norm else 0.0
        lp_resp = (lp_a * mask_use).sum(dim=1) / (denom ** alpha)  # [B]
    
        # KL：对参考策略（若存在）
        if (self.ref is not None) and (self.kl_ctl > 0.0) and (lp_r_full is not None):
            with torch.no_grad():
                lp_r = lp_r_full[:, :L]
                d = (lp_a - lp_r).clamp(-4.0, 4.0)      
                k3 = torch.expm1(d) - d                  
                k3 = torch.clamp(k3, 0.0, 10.0)      
                kl_seq = (k3 * mask_use).sum(dim=1) / denom
        else:
            kl_seq = None
    
        with torch.no_grad():
            r = self._rm_score(seqs, attn)  # [B]
    
        # 组内构造 p* 与损失
        group_losses = []
        groups_cnt, total_items = 0, 0
        pstar_ent_list = []
    
        for s, e in _group_slices(seqs.size(0), meta):
            r_g  = r[s:e]
            lp_g = lp_resp[s:e]
    
            # 组内标准化 -> softmax/τ
            mu, std = r_g.mean(), r_g.std()
            if float(std.item()) < 1e-6:
                p_star = torch.full_like(r_g, 1.0 / r_g.numel())
            else:
                z = (r_g - mu) / (std + 1e-6)
                p_star = F.softmax(z / max(self.tau, 1e-6), dim=0)
    
            log_p_theta = F.log_softmax(lp_g, dim=0)
            ce = -(p_star * log_p_theta).sum()
    
            if kl_seq is not None:
                kl_g = kl_seq[s:e].mean()
                loss_g = ce + self.kl_ctl * kl_g
            else:
                loss_g = ce
    
            group_losses.append(loss_g)
            groups_cnt += 1
            total_items += (e - s)
    
            # 监控 p* 的熵
            pstar_ent_list.append(float(-(p_star * p_star.clamp_min(1e-12).log()).sum().item()))
    
        if not group_losses:
            return {"skip": True}
    
        loss = torch.stack(group_losses).mean()

        self.actor.train()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt.step()
    
        # 统计
        with torch.no_grad():
            ent_tok = float(entropy_from_logits(logits_a[:, 1:, :], mask_use).item())
            r_mean = float(r.mean().item())
            pstar_entropy = sum(pstar_ent_list) / max(len(pstar_ent_list), 1)
            kl_mean = float(kl_seq.mean().item()) if kl_seq is not None else 0.0
    
            # 高熵监控
            H_mean     = float((H_resp.sum() / mask_t.sum().clamp_min(1)).item()) if mask_t.sum() > 0 else 0.0
            H_sel_mean = float(((H_resp * mask_use).sum() / mask_use.sum().clamp_min(1)).item()) if mask_use.sum() > 0 else 0.0
    
        self.last_stats.update({
            "loss": float(loss.item()),
            "rm_mean": r_mean,
            "entropy_tok": ent_tok,
            "pstar_entropy": float(pstar_entropy),
            "kl_mean": float(kl_mean),
            "tau": float(self.tau),
            "groups": int(groups_cnt),
            "items": int(total_items),
            "he_ratio": float(he_ratio),
            "H_mean": H_mean,
            "H_sel_mean": H_sel_mean,
            "skip": False,
        })
        return dict(self.last_stats)
