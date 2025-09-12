# RL/DAPO.py
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL.common import (
    Samples, Experience,
    compute_actor_ref_logprobs, approx_kl_from_logprobs, kl_k3_from_logratio,
    masked_mean, model_all_logits, token_logprobs_from_logits,
    scatter_last_token_rewards, scatter_uniform_rewards,
    ratio_stats, adv_abs_mean,
    normalize_for_reward,
)

@dataclass
class DAPOArgs:
    group_size: int = 4                 # 每个 prompt 组内生成条数（含离线一条 + 在线补样）
    kl_coef: float = 0.01               # 句级 KL 正则（shaping）
    beta: float = 1.0                   # token 级 KL shaping 系数（散到奖励里时使用）
    adv_norm: str = "zscore"            # ["zscore" | "center" | "none"]
    adv_clip: float = 5.0               # 对优势做 abs 裁剪
    mb_size_logits: int = 1             # logits 计算用的 micro-batch
    max_new_tokens: int = 48            # 在线补样生成上限（与主程保持一致）
    use_uniform_scatter: bool = False   # 奖励散到 response：True=均匀，False=末 token
    min_resp_tokens: int = 8            # 生成时的最短 response 约束（和主程一致）

class DAPOTrainer:
    """
    Decoupled Advantage Policy Optimization（无 critic；句级优势，token 级更新）。
    - 组内相对：对同一 prompt 采多条（离线池 + 在线采样），用 RM 得句级 reward → 组内标准化成优势。
    - policy-only：无 value 网络。
    - KL：两路
        (1) 句级 KL（per-seq 平均 KL）乘以 kl_coef，参与 reward shaping；
        (2) token 级近似 KL（k3）乘以 beta，直接作为 per-token 正则加到 loss。
    """
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor: torch.optim.Optimizer,
        device: str = "cuda",
        group_size: int = 4,
        kl_coef: float = 0.01,
        beta: float = 1.0,
        adv_norm: str = "zscore",
        adv_clip: float = 5.0,
        mb_size_logits: int = 1,
        max_new_tokens: int = 48,
        use_uniform_scatter: bool = False,
        min_resp_tokens: int = 8,
        block_size: Optional[int] = None,
        # use_token_entropy: bool = False,
        # ent_keep_ratio: float = 0.2,
    ):
        self.actor = actor_model
        self.ref = ref_model.eval()
        for p in self.ref.parameters(): p.requires_grad = False

        self.reward_model = reward_model.eval()
        for p in self.reward_model.parameters(): p.requires_grad = False

        self.tok = actor_tokenizer
        self.rtok = reward_tokenizer

        self.opt_actor = optimizer_actor
        self.device = device

        self.args = DAPOArgs(
            group_size=group_size, kl_coef=kl_coef, beta=beta,
            adv_norm=adv_norm, adv_clip=adv_clip, mb_size_logits=mb_size_logits,
            max_new_tokens=max_new_tokens, use_uniform_scatter=use_uniform_scatter,
            min_resp_tokens=min_resp_tokens,
        )
        self.block_size = block_size
        # self.use_token_entropy = bool(use_token_entropy)
        # self.ent_keep_ratio = float(ent_keep_ratio)

        # 运行时统计（日志打印对齐 PPO）
        self.last_stats = {}

        # 采样超参与主循环保持一致（从 env 里拿）
        self.sample_temperature  = float(os.getenv("SAMPLE_TEMPERATURE", "0.8"))
        self.sample_top_p        = float(os.getenv("SAMPLE_TOP_P", "0.9"))
        self.sample_top_k        = int(os.getenv("SAMPLE_TOP_K", "0"))
        self.sample_rep_penalty  = float(os.getenv("SAMPLE_REP_PENALTY", "1.1"))
        self.sample_stops        = ["\nHuman:", "\n\nHuman:"]  # 与 rollout 保持一致
        if "\nAssistant:" not in self.sample_stops:
            self.sample_stops.append("\nAssistant:")

        self.ptdtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                        else torch.float16)

    # ============ 核心流程：评估经验（不更新参数） ============
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples):
        """
        输入：打包好的样本（每条 1 个 response）
        输出：组装后的 Experience 列表（每组同 prompt 若干条）
        """
        B = samples.seqs.size(0)
        assert B > 0

        # 1) 先拿当前 batch 的“基底样本”（来自离线池或在线实时一条）
        base_list = []
        for i in range(B):
            seq_i = samples.seqs[i:i+1]
            am_i  = samples.action_mask[i:i+1]
            base_list.append({
                "seqs": seq_i, "action_mask": am_i,
                "prompt_len": int((am_i[0]==0).sum().item()),
            })

        # 2) 对每个 prompt 在线补齐 group_size-1 条（用 actor 采样）
        groups = []
        for base in base_list:
            prompt_ids = base["seqs"][0, :base["prompt_len"]].detach().clone()
            prompt_ids = prompt_ids.to(self.device, dtype=torch.long).unsqueeze(0)
            seqs_g = [base["seqs"]]
            am_g   = [base["action_mask"]]

            # 采样补齐
            need = max(0, int(self.args.group_size) - 1)
            tries = 0
            while len(seqs_g) < self.args.group_size and tries < need*3:
                tries += 1
                out = self._decode_with_sampling(
                    self.actor, prompt_ids,
                    max_new_tokens=self.args.max_new_tokens,
                    eos_id=self.tok.eos_id,
                    block_size=self.block_size or base["seqs"].size(1),
                    min_resp=self.args.min_resp_tokens,
                )
                full = out[0].to(self.device, dtype=torch.long).unsqueeze(0)
                # 动态 action_mask
                am = torch.zeros_like(full, dtype=torch.long, device=self.device)
                plen = prompt_ids.size(1)
                am[:, :full.size(1)] = 0
                am[:, plen:full.size(1)] = 1
                seqs_g.append(full)
                am_g.append(am)

            # 对齐到同长度（右 pad 到本组最大）
            T = max(int(x.size(1)) for x in seqs_g)
            T = min(T, self.block_size or T)
            seqs_pad = []
            am_pad   = []
            for s, m in zip(seqs_g, am_g):
                S = s.size(1)
                if S >= T:
                    seqs_pad.append(s[:, :T])
                    am_pad.append(m[:, :T])
                else:
                    pad = torch.full((1, T-S), self.tok.pad_token_id, dtype=torch.long, device=self.device)
                    seqs_pad.append(torch.cat([s, pad], dim=1))
                    mpad = torch.zeros((1, T-S), dtype=torch.long, device=self.device)
                    am_pad.append(torch.cat([m, mpad], dim=1))

            groups.append({
                "seqs": torch.cat(seqs_pad, dim=0),        # [G, T]
                "am":   torch.cat(am_pad,   dim=0),        # [G, T]
            })

        # 3) 对每组：算 actor/ref token logprob、KL，奖励与优势
        experiences: List[Experience] = []
        total_rewards_raw = []
        total_rewards_shaped = []
        total_kl_safe = []
        total_ctr = 0.0

        for g in groups:
            seqs = g["seqs"]            # [G, T]
            am   = g["am"]              # [G, T]
            G, T = seqs.size()

            # token logprob
            lp_actor, lp_ref, mask_tgt = compute_actor_ref_logprobs(
                self.actor, self.ref, seqs, am, device_type=("cuda" if "cuda" in self.device else "cpu"),
                ptdtype=self.ptdtype, micro_batch_size=self.args.mb_size_logits,
            )  # [G, T-1], [G, T-1], [G, T-1]（已对齐 action 轴）

            # === Token-entropy 掩码（2/8 高熵保留）===
            if self.use_token_entropy:
                logits_actor_full = model_all_logits(
                    self.actor, seqs, device_type=self.device_type, ptdtype=None, micro_batch_size=self.mb_logits
                )
                new_mask_tgt = apply_entropy_mask(
                    logits_actor_full[:, 1:, :].detach(),  # 与 mask_tgt 对齐
                    mask_tgt,
                    keep_ratio=float(ent_keep_ratio)
                )
                if new_mask_tgt.sum() > 0:  # 防极端回退
                    mask_tgt = new_mask_tgt
                

            # per-token k3 与句级 KL
            log_ratio = (lp_actor - lp_ref)
            k3 = kl_k3_from_logratio(log_ratio)
            kl_mean = masked_mean(k3, mask_tgt.float())         # 句级近似 KL

            # 奖励：把文本发到 RM（句级）
            texts = self._decode_texts(seqs, am)
            texts_norm = [normalize_for_reward(t, self.rtok) for t in texts]
            toks = self.rtok(texts_norm, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            rm_out = self.reward_model(**{k: v.to(self.reward_model.device) for k, v in toks.items()})
            logits = getattr(rm_out, "logits", None)
            if logits is None:
                r_seq = torch.zeros((G,), device=self.device, dtype=torch.float32)
            else:
                if logits.dim() == 2 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                r_seq = logits.detach().to(self.device, dtype=torch.float32)

            # 句级 KL shaping（减罚）
            r_shaped = r_seq - float(self.args.kl_coef) * kl_mean

            # 组内优势（DAPO/GRPO 风格）
            if self.args.adv_norm == "zscore":
                mu = r_shaped.mean()
                sd = r_shaped.std().clamp_min(1e-8)
                adv = (r_shaped - mu) / sd
            elif self.args.adv_norm == "center":
                mu = r_shaped.mean()
                adv = (r_shaped - mu)
            else:
                adv = r_shaped

            if self.args.adv_clip and self.args.adv_clip > 0:
                adv = adv.clamp(min=-float(self.args.adv_clip), max=float(self.args.adv_clip))

            # 把句级奖励（可选含 KL）散到 response token（仅用于日志/可视化）
            if self.args.use_uniform_scatter:
                r_tok = scatter_uniform_rewards(r_shaped, am[:, 1:], beta_kl=(k3, float(self.args.beta)))
            else:
                r_tok = scatter_last_token_rewards(r_shaped, am[:, 1:], beta_kl=(k3, float(self.args.beta)))

            # 经验条：这里不需要 critic/value；returns/advantages 都用句级（在 loss 里按 seq 广播）
            # 为了与现有训练循环兼容，我们把 advantage/return 铺成与 action 轴同形状，全部等于句级值
            Gmask = am[:, 1:]
            A_tok = adv.view(G, 1).expand_as(Gmask).contiguous()
            Ret_tok = A_tok  # policy-only，不用 returns 概念

            exp = Experience(
                seqs=seqs,
                action_log_probs=lp_actor.detach(),
                values=torch.zeros_like(A_tok),          # 占位
                returns=Ret_tok.detach(),
                advantages=A_tok.detach(),
                attention_mask=(seqs.ne(self.tok.pad_token_id)).long(),
                action_mask=Gmask.long(),
                reward=r_seq.view(-1, 1).detach(),
                num_actions=(Gmask.sum(dim=1).to(torch.long)),
                kl=k3.detach(),
            )
            experiences.append(exp)

            total_rewards_raw.append(float(r_seq.mean().item()))
            total_rewards_shaped.append(float(r_shaped.mean().item()))
            total_kl_safe.append(float(kl_mean.item()))
            total_ctr += 1.0

        # ====== 汇总日志（与 PPO 打印字段风格一致）======
        if total_ctr > 0:
            avg_raw = float(np.mean(total_rewards_raw))
            avg_shp = float(np.mean(total_rewards_shaped))
            safe_kl = float(np.mean(total_kl_safe))
        else:
            avg_raw = float("nan"); avg_shp = float("nan"); safe_kl = float("nan")

        # 近似 KL（再算一次 token 级 mask 下的均值，仅用于日志）
        if len(experiences) > 0:
            all_lp_a = torch.cat([e.action_log_probs for e in experiences], dim=0)
            all_lp_r = torch.cat([e.action_log_probs - e.kl for e in experiences], dim=0)  # 粗略还原
            all_mask = torch.cat([e.action_mask for e in experiences], dim=0)
            akl_mean, _ = approx_kl_from_logprobs(all_lp_a, all_lp_r, all_mask)
            approx_kl_pi = float(akl_mean.detach().item())
            r50, r90, r99, rmax = ratio_stats(all_lp_a - all_lp_r, all_mask)
            advabs = adv_abs_mean(torch.cat([e.advantages for e in experiences], dim=0), all_mask)
            H = float("nan")  # DAPO 这里不做 entropy 正则
            v_mae = float("nan"); ev = float("nan")
            sel_tokens = int(all_mask.sum().item())
        else:
            approx_kl_pi = float("nan")
            r50=r90=r99=rmax=float("nan")
            advabs = float("nan")
            H=v_mae=ev=float("nan"); sel_tokens = 0

        self.last_stats = {
            "clip_frac": float("nan"),       # DAPO 的 clip_frac 在训练步里统计
            "approx_kl_pi": approx_kl_pi,
            "entropy": H,
            "v_mae": v_mae,
            "explained_var": ev,
            "ratio_q50_q90_q99": (r50, r90, r99),
            "ratio_max": rmax,
            "adv_abs_mean": advabs,
            "sel_tokens": sel_tokens,
            "ppo_clip": float(os.getenv("PPO_CLIP", "0.2")),  # 仅展示
            "kl_ctl_now": float(self.args.kl_coef),
        }
        # 返回：经验列表 + 日志字段
        report_kl = approx_kl_pi
        r_raw = avg_raw
        r_shaped = avg_shp
        r_ctr = 0.0  # DAPO 无 “r_ctr”（PPO 的对比项）
        return experiences, report_kl, r_raw, r_shaped, r_ctr, safe_kl

    # ============ 单条经验上训练一步（只更 policy） ============
    def train_on_experience(self, exp: Experience):
        """
        DAPO 的损失：
          - 令 R_t = 序列优势 A_seq（按 seq 广播到 token）
          - 重要性比 r = exp(logπ - logπ_old)，做对称 clipping（上限/下限可不同，读取 env）
          - token 级正则：beta * k3
          - 对 action 轴做 masked 平均，然后对 batch 平均
        """
        self.actor.train()
        self.opt_actor.zero_grad(set_to_none=True)

        # 重新计算当前策略的 logπ
        lp_actor_new, _, mask_tgt = compute_actor_ref_logprobs(
            self.actor, self.actor, exp.seqs, exp.action_mask,
            device_type=("cuda" if "cuda" in self.device else "cpu"),
            ptdtype=self.ptdtype, micro_batch_size=self.args.mb_size_logits,
        )

        # 重要性比 + 裁剪
        with torch.no_grad():
            lp_old = exp.action_log_probs
        ratio = torch.exp(lp_actor_new - lp_old)  # [B, T-1]

        rmin = float(os.getenv("PPO_RATIO_MIN", "0.66"))
        rmax = float(os.getenv("PPO_RATIO_MAX", "1.5"))
        ratio_clamped = torch.clamp(ratio, rmin, rmax)

        # 句级优势按 token 广播
        A = exp.advantages  # [B, T-1]（evaluate 已经广播到 action 轴）
        per_token_loss1 = - ratio * A
        per_token_loss2 = - ratio_clamped * A
        per_token_loss = torch.minimum(per_token_loss1, per_token_loss2)

        # token 级 KL 正则（k3）
        with torch.no_grad():
            # 用 evaluate_experience 的 k3 更“准”，但为了减少带宽，在此快速重算一遍
            lp_ref_now, _, _ = compute_actor_ref_logprobs(
                self.ref, self.ref, exp.seqs, exp.action_mask,
                device_type=("cuda" if "cuda" in self.device else "cpu"),
                ptdtype=self.ptdtype, micro_batch_size=self.args.mb_size_logits,
            )
        k3 = kl_k3_from_logratio(lp_actor_new - lp_ref_now)

        # 只在 action 轴上平均
        mask = mask_tgt.float()
        loss_tok = (per_token_loss + float(self.args.beta) * k3) * mask
        denom = mask.sum().clamp_min(1e-8)
        loss = loss_tok.sum() / denom

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # 训练时的 clip 统计
        with torch.no_grad():
            clipped = (ratio_clamped != ratio).float() * mask
            clip_frac = clipped.sum() / denom
        self.last_stats["clip_frac"] = float(clip_frac.item())

        # 返回两路 loss（为了和 PPO 兼容打印）
        return loss, torch.tensor(float("nan"), device=loss.device)

    # ============ 采样（和主程保持口径一致） ============
    @torch.no_grad()
    def _decode_with_sampling(self, model, idx, max_new_tokens, eos_id, block_size, min_resp=8):
        device = idx.device
        out = idx
        start_len = out.size(1)
        for _ in range(int(max_new_tokens)):
            idx_cond = out[:, -int(block_size):]
            logits = model(idx_cond)
            if isinstance(logits, tuple): logits = logits[0]
            last = logits[:, -1, :]

            # 轻度去重复
            if self.sample_rep_penalty and out.numel() > 0:
                uniq = torch.unique(out)
                last[:, uniq] = last[:, uniq] / float(self.sample_rep_penalty)
            last = last / max(float(self.sample_temperature), 1e-6)

            if self.sample_top_k and self.sample_top_k > 0:
                kth = torch.topk(last, k=min(int(self.sample_top_k), last.size(-1)), dim=-1).values[..., -1:]
                last = torch.where(last < kth, torch.full_like(last, -1e10), last)

            probs = torch.softmax(last, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > float(self.sample_top_p)).float().argmax(dim=-1, keepdim=True)
            mask = torch.arange(probs.size(-1), device=probs.device).view(1, -1) <= cutoff
            kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(kept, num_samples=1)
            next_id = sorted_idx.gather(1, next_sorted)

            # 未达最短长度时禁止 EOS
            if (out.size(1) - start_len) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                alt = sorted_idx[:, 1:2] if sorted_idx.size(1) > 1 else next_id
                next_id = alt

            out = torch.cat([out, next_id.to(device)], dim=1)

            # 达到最短后允许停词
            if (out.size(1) - start_len) >= int(min_resp):
                if eos_id is not None and int(next_id.item()) == int(eos_id):
                    break
                tail_ids = out[0, -int(block_size):].tolist()
                tail = self.tok.decode(tail_ids)
                if any(s in tail for s in self.sample_stops):
                    break
        return out

    # ============ 文本解码辅助（RM 输入） ============
    @torch.no_grad()
    def _decode_texts(self, seqs: torch.Tensor, action_mask: torch.Tensor) -> List[str]:
        G, T = seqs.size()
        texts = []
        for i in range(G):
            L = int((seqs[i] != self.tok.pad_token_id).sum().item())
            L = max(1, min(L, T))
            texts.append(self.tok.decode(seqs[i, :L].tolist()))
        return texts
