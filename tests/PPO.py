# PPO.py
import math, os, sys, random, time
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast, GradScaler

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ====================== 工具 ======================
def shift_right(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    # 用于构造 teacher-forcing targets（不在此处使用，留作通用工具）
    y = x.new_full(x.size(), pad_id)
    y[:, 1:] = x[:, :-1]
    return y

def gather_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], tokens: [B, T]
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)  # [B, T]

# =================== KL 控制器 ====================
class AdaptiveKLController:
    def __init__(self, init_kl_coef=0.01, target=0.03, horizon=1000):
        self.kl_coef = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, measured_kl: float, n_steps: int = 1):
        # OpenAI summarize 类似：根据偏差调整 beta
        proportional_error = (measured_kl / self.target) - 1
        multiplier = 1 + proportional_error * min(n_steps / self.horizon, 1.0)
        self.kl_coef *= max(0.1, min(10.0, multiplier))

# ====================== RM（CPU） ======================
class RewardModelCPU:
    def __init__(self, model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"):
        self.device = torch.device("cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        # 将 prompt + response 拼接；RM 自身会处理 attention_mask（右填充亦可）
        texts = [p.strip() + "\n" + r.strip() for p, r in zip(prompts, responses)]
        batch = self.tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        # OpenAssistant RM：logits[..., 0] 越大越好（或取 sigmoid）；这里直接取标量
        logits = out.logits.squeeze(-1)
        return logits.detach().cpu().tolist()

# ====================== 经验&配置 ======================
@dataclass
class PPOConfig:
    # rollout / 算法
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    max_grad_norm: float = 1.0

    # 采样生成
    min_new_tokens: int = 16
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.9

    # 优化
    lr: float = 2e-6
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.1
    use_bnb8bit: bool = True

    # 批设置
    batch_size: int = 64     # 每次更新的样本数（可由 grad_accum 聚合）
    minibatch_size: int = 16
    ppo_epochs: int = 2

    # 设备&AMP
    amp_dtype = torch.float16

@dataclass
class RolloutBatch:
    input_ids: torch.Tensor      # [B, T]
    response_mask: torch.Tensor  # [B, T]  (action_mask=1 表示response区间)
    logprobs: torch.Tensor       # [B, T]  (策略采样得到的对数概率，仅response位有意义)
    ref_logprobs: torch.Tensor   # [B, T]
    values: torch.Tensor         # [B, T]
    rewards: torch.Tensor        # [B, T]  (仅response位非零；末位含RM+KL)

# ====================== PPO Trainer ======================
class PPOTrainer:
    def __init__(self, policy, ref_policy, tokenizer_gpt2, rm: RewardModelCPU, cfg: PPOConfig, device: str):
        self.policy = policy
        self.ref = ref_policy
        for p in self.ref.parameters(): p.requires_grad = False
        self.rm = rm
        self.cfg = cfg
        self.device = device
        self.tok = tokenizer_gpt2
        self.kl_ctl = AdaptiveKLController(init_kl_coef=0.01, target=0.03, horizon=1000)

        self.optimizer = self.policy.configure_optimizers(
            weight_decay=cfg.weight_decay, learning_rate=cfg.lr, betas=cfg.betas, use_bnb8bit=cfg.use_bnb8bit
        )
        self.scaler = GradScaler(enabled=(self.cfg.amp_dtype is not None and self.device == "cuda"))

    # --------- 文本 <-> token ----------
    def encode_gpt2(self, texts: List[str], add_eos: bool=False) -> List[List[int]]:
        EOS_ID = 50256
        res = []
        for t in texts:
            ids = self.tok.encode_ordinary(t)
            if add_eos:
                ids.append(EOS_ID)
            res.append(ids)
        return res

    def decode_gpt2(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

    # --------- 生成 ----------
    @torch.no_grad()
    def _generate_batch(self, prompt_ids_batched: List[List[int]], block_size: int):
        B = len(prompt_ids_batched)
        prompt_lens = [len(x) for x in prompt_ids_batched]
        max_prompt = min(max(prompt_lens), max(8, block_size - self.cfg.max_new_tokens))

        inputs = []
        for ids in prompt_ids_batched:
            ids = ids[-max_prompt:]
            inputs.append(ids)
        max_prompt = max(len(x) for x in inputs)

        EOS_ID = 50256
        batch = torch.full((B, max_prompt), EOS_ID, dtype=torch.long)  # 左填充用 EOS
        for i, ids in enumerate(inputs):
            batch[i, -len(ids):] = torch.tensor(ids, dtype=torch.long)
        batch = batch.to(self.device)

        out = self.policy.generate(
            batch,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p
        )
        return out, batch, prompt_lens

        # 采样响应（长度在 [min,max] 间；这里直接取 max_new_tokens，上层可改不同长度策略）
        max_new = self.cfg.max_new_tokens
        out = self.policy.generate(batch, max_new_tokens=max_new,
                                   temperature=self.cfg.temperature, top_p=self.cfg.top_p)
        # 记录每条样本的 prompt_len
        return out, batch, prompt_lens

    @torch.no_grad()
    def _compute_logprobs_values(self, model, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _, values = model(tokens)
        logprobs = F.log_softmax(logits, dim=-1)
        # 对齐到每个位生成的 token（下一个token），shift gather
        next_tokens = tokens[:, 1:]
        lp = logprobs[:, :-1, :].gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        # pad 回 [B, T]，最后一位没有 next_token
        lp = torch.cat([lp, lp.new_zeros(lp.size(0), 1)], dim=1)
        return lp, values

    @torch.no_grad()
    def build_rollout(self, prompts: List[str], block_size: int) -> Tuple[RolloutBatch, List[str], List[str]]:
        # 1) encode prompts
        enc_prompts = self.encode_gpt2(prompts, add_eos=False)
        # 2) 生成
        gen_tokens, prompt_tokens, prompt_lens = self._generate_batch(enc_prompts, block_size)
        # 构造 response_mask
        B, T = gen_tokens.size()
        response_mask = torch.zeros((B, T), dtype=torch.bool, device=self.device)
        for i, Lp in enumerate(prompt_lens):
            Lp = min(Lp, T-1)
            response_mask[i, Lp:] = True

        # 3) 策略与参考 logprob/value
        pol_logp, values = self._compute_logprobs_values(self.policy, gen_tokens)
        ref_logp, _      = self._compute_logprobs_values(self.ref,    gen_tokens)

        # 4) 文本化以喂 RM
        prompts_txt, responses_txt = [], []
        for i in range(B):
            Lp = min(prompt_lens[i], T-1)
            full_ids = gen_tokens[i].tolist()
            resp_ids = full_ids[Lp:]
            prom_ids = full_ids[:Lp]
            responses_txt.append(self.decode_gpt2(resp_ids))
            prompts_txt.append(self.decode_gpt2(prom_ids))

        # 5) RM 打分（CPU），得到标量 reward，分配在 **最后一个 response token**（end-only）
        rm_scores = self.rm.score(prompts_txt, responses_txt)
        rewards = torch.zeros_like(pol_logp)
        for i in range(B):
            idxs = torch.where(response_mask[i])[0]
            if len(idxs) > 0:
                last = idxs[-1].item()
                rewards[i, last] = rm_scores[i]

        # 6) KL 惩罚：对 response 位置逐位 KL (π||π_ref) 的和（用 logprob 差近似）
        kl_per_token = (pol_logp - ref_logp)  # ≈ -KL 的梯度方向（logπ - logπ_ref）
        # 注意：准确 KL 需对分布，简化实现用 on-policy token 的对数概率差作为近似项
        kl_coeff = self.kl_ctl.kl_coef
        rewards = rewards - kl_coeff * kl_per_token

        batch = RolloutBatch(
            input_ids=gen_tokens, response_mask=response_mask,
            logprobs=pol_logp, ref_logprobs=ref_logp,
            values=values, rewards=rewards
        )
        # 更新 KL 控制器（用 batch 均值）
        measured_kl = float((-(kl_per_token * response_mask).sum() / (response_mask.sum().clamp_min(1))).item())
        self.kl_ctl.update(measured_kl, n_steps=1)
        return batch, prompts_txt, responses_txt

    # --------- GAE 计算（仅 response 段） ----------
    @torch.no_grad()
    def compute_advantages(self, values: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = rewards.size()
        adv = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            next_values = values[:, t+1] if t < T-1 else torch.zeros_like(values[:, t])
            delta = rewards[:, t] + self.cfg.gamma * next_values - values[:, t]
            delta = delta * mask[:, t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.lam * lastgaelam
            adv[:, t] = lastgaelam
        returns = adv + values
        # 仅对 response 位做标准化
        m = (adv * mask).sum() / mask.sum().clamp_min(1)
        v = ((adv - m)**2 * mask).sum() / mask.sum().clamp_min(1)
        std = torch.sqrt(v.clamp_min(1e-6))
        adv = torch.where(mask, (adv - m) / std, torch.zeros_like(adv))
        return adv, returns

    # --------- PPO 更新 ----------
    def ppo_update(self, batch: RolloutBatch) -> Dict[str, float]:
        B, T = batch.input_ids.size()
        mask = batch.response_mask

        with torch.no_grad():
            advantages, returns = self.compute_advantages(batch.values, batch.rewards, mask)

        # 展平到序列维度（仅保留 mask==1 的位置）
        def flat_mask(x):  # x: [B, T]
            return x[mask]

        old_logp = flat_mask(batch.logprobs).detach()
        old_v    = flat_mask(batch.values).detach()
        adv_flat = flat_mask(advantages).detach()
        ret_flat = flat_mask(returns).detach()

        idx = torch.nonzero(mask, as_tuple=False)
        # 将 (b,t) 索引划分成小批
        perm = torch.randperm(idx.size(0), device=idx.device)
        minibatches = []
        for i in range(0, perm.size(0), self.cfg.minibatch_size * 64):  # 近似按 token 数组批次划分
            sel = perm[i:i + self.cfg.minibatch_size * 64]
            minibatches.append(idx[sel])

        stats = {"loss_pi": 0.0, "loss_v": 0.0, "kl": 0.0, "clip_frac": 0.0, "entropy": 0.0}
        total_mb = 0

        for _ in range(self.cfg.ppo_epochs):
            for mb in minibatches:
                if mb.numel() == 0: continue
                # 取到对应序列片段的 token 上下文。为保证因果，我们以“完整序列”前向，再在对应位置 gather。
                tokens = batch.input_ids
                with autocast(device_type="cuda", dtype=self.cfg.amp_dtype, enabled=(self.device=="cuda" and self.cfg.amp_dtype is not None)):
                    logits, _, values = self.policy(tokens)
                    logp_all = F.log_softmax(logits, dim=-1)
                    next_tokens = tokens[:, 1:]
                    logp_step = logp_all[:, :-1, :].gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
                    # pad 回 T
                    logp_step = torch.cat([logp_step, logp_step.new_zeros(logp_step.size(0), 1)], dim=1)

                    # 仅取 mask 索引位置
                    new_logp = logp_step[mb[:,0], mb[:,1]]
                    new_v    = values[mb[:,0], mb[:,1]]
                    # policy loss（截断）
                    ratio = torch.exp(new_logp - old_logp[:new_logp.size(0)])
                    unclipped = ratio * adv_flat[:new_logp.size(0)]
                    clipped = torch.clamp(ratio, 1.0 - self.cfg.cliprange, 1.0 + self.cfg.cliprange) * adv_flat[:new_logp.size(0)]
                    loss_pi = -torch.mean(torch.min(unclipped, clipped))
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > self.cfg.cliprange).float())

                    # value loss（clip）
                    v_clipped = old_v[:new_v.size(0)] + (new_v - old_v[:new_v.size(0)]).clamp(-self.cfg.cliprange, self.cfg.cliprange)
                    v_loss1 = (new_v - ret_flat[:new_v.size(0)])**2
                    v_loss2 = (v_clipped - ret_flat[:new_v.size(0)])**2
                    loss_v = 0.5 * torch.mean(torch.max(v_loss1, v_loss2))

                    # entropy（近似用当前步 logits 的 entropy 均值）
                    p_all = logp_all.exp()
                    ent = -(p_all * logp_all).sum(dim=-1)  # [B,T]
                    entropy = torch.mean(ent[mb[:,0], mb[:,1]])

                    loss = loss_pi + self.cfg.vf_coef * loss_v - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                stats["loss_pi"] += float(loss_pi.detach().cpu())
                stats["loss_v"]  += float(loss_v.detach().cpu())
                stats["entropy"] += float(entropy.detach().cpu())
                stats["clip_frac"] += float(clip_frac.detach().cpu())
                total_mb += 1

        for k in stats:
            stats[k] = stats[k] / max(1, total_mb)
        return stats
