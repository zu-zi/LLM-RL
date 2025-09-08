# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import amp
# from typing import List, Tuple

# from .common import (
#     Samples, Experience,
#     normalize_for_reward,
#     build_samples_from_generations,
#     model_all_logits, token_logprobs_from_logits,
#     compute_actor_ref_logprobs,
#     gae_compute, masked_mean, clip_by_global_norm,
#     forward_values_via_actor,
# )

# # -------------------------- Critic --------------------------
# class Critic(nn.Module):
#     """
#     简单 value 头：基于 actor 的隐藏态输出每个 token 的 V(s_t)。
#     注意不要只对 response 片段单独前向 actor，否则会丢上下文。
#     """
#     def __init__(self, base_model: nn.Module):
#         super().__init__()
#         self.n_embd = base_model.config.n_embd
#         self.value_head = nn.Linear(self.n_embd, 1)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         # hidden_states: [B, T, C]  ->  values: [B, T]
#         return self.value_head(hidden_states).squeeze(-1)


# # ------------------------ PPO Trainer ------------------------
# class PPOTrainer:
#     def __init__(
#         self,
#         actor_model: nn.Module,
#         ref_model: nn.Module,
#         reward_model: nn.Module,
#         critic_model: nn.Module,
#         actor_tokenizer,              # GPT2Tok（自写）或任意 tokenizer
#         reward_tokenizer,             # HF tokenizer（奖励模型）
#         optimizer_actor,
#         optimizer_critic,
#         kl_ctl: float = 0.02,         # KL 系数（β）
#         clip_reward: float = 5.0,     # 奖励裁剪（对序列级奖励裁剪）
#         gamma: float = 1.0,           # RLHF 常用 1.0
#         lambd: float = 0.95,          # GAE 参数
#         mb_size_logits: int = 0,      # actor/ref logits 前向的 micro-batch（0=不切）
#         mb_size_values: int = 0,      # critic values 前向的 micro-batch（0=不切）
#         device: str = "cuda",
#     ):
#         self.actor = actor_model.to(device)
#         self.ref = ref_model.to(device)
#         # 奖励模型保持 CPU（外部包装器已 to('cpu')）
#         self.reward_model = reward_model
#         self.reward_model.eval()
#         self.critic = critic_model.to(device)

#         self.actor_tokenizer = actor_tokenizer
#         self.reward_tokenizer = reward_tokenizer

#         self.optimizer_actor = optimizer_actor
#         self.optimizer_critic = optimizer_critic

#         self.kl_ctl = kl_ctl
#         self.clip_reward = clip_reward
#         self.gamma = gamma
#         self.lambd = lambd

#         self.mb_size_logits = mb_size_logits
#         self.mb_size_values = mb_size_values

#         self.device = torch.device(device)
#         self.use_amp = (self.device.type == "cuda")
#         self.scaler_actor  = amp.GradScaler(enabled=self.use_amp)
#         self.scaler_critic = amp.GradScaler(enabled=self.use_amp)

#         # pad/eos（兼容自定义 GPT2Tok 与 HF tokenizer）
#         self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
#         self.eos_token_id = getattr(actor_tokenizer, "eos_token_id",
#                              getattr(actor_tokenizer, "eos_id", 50256))

#         # 奖励 EMA 基线
#         self.r_baseline = 0.0

#         # 最近一次训练统计
#         self.last_stats = {}

#     # --------- 小工具：安全解码（兼容自定义/HF tokenizer） ---------
#     @staticmethod
#     def _safe_decode(tok, ids_tensor) -> str:
#         ids = ids_tensor.detach().cpu().tolist()
#         # 优先 decode(skip_special_tokens=False)
#         if hasattr(tok, "decode"):
#             try:
#                 return tok.decode(ids, skip_special_tokens=False)
#             except TypeError:
#                 try:
#                     return tok.decode(ids)
#                 except Exception:
#                     pass
#         # 退路 batch_decode
#         if hasattr(tok, "batch_decode"):
#             try:
#                 return tok.batch_decode([ids], skip_special_tokens=False)[0]
#             except TypeError:
#                 try:
#                     return tok.batch_decode([ids])[0]
#                 except Exception:
#                     pass
#         # 退路 tokens->join
#         if hasattr(tok, "convert_ids_to_tokens"):
#             try:
#                 return " ".join(tok.convert_ids_to_tokens(ids))
#             except Exception:
#                 pass
#         # 最后兜底：id 串
#         return " ".join(str(x) for x in ids)

#     # ------------------- 逐样本生成（避免 pad 干扰） -------------------
#     @torch.no_grad()
#     def generate_samples(
#         self,
#         inputs: Tuple[torch.Tensor, torch.Tensor],
#         max_length: int,
#         max_new_tokens: int
#     ) -> List[Samples]:
#         """
#         inputs: (input_ids, attention_mask)（来自固定 prompts bin）
#         - 精准按各自 prompt_len 生成；不把 pad 当上下文
#         - 返回 [Samples]（与上游调用对齐）
#         """
#         input_ids, attention_mask = inputs
#         input_ids = input_ids.to(self.device)
#         attention_mask = attention_mask.to(self.device)

#         B = input_ids.size(0)
#         block_size = self.actor.config.block_size

#         gens = []
#         for i in range(B):
#             p_len = int(attention_mask[i].sum().item())
#             p_len = min(p_len, block_size - 1)  # 预留 EOS
#             if p_len <= 0:
#                 continue
#             prompt = input_ids[i:i+1, :p_len]   # [1, T_prompt]
#             room = max(1, min(max_new_tokens, block_size - p_len - 1))

#             out = self.actor.generate(
#                 idx=prompt,
#                 max_new_tokens=room,
#                 temperature=1.0,
#                 top_k=None,
#                 eos_token_id=self.eos_token_id
#             )  # [1, T_prompt + T_resp]

#             full = out[0]
#             resp = full[p_len:]
#             gens.append({
#                 "prompt_ids": prompt[0],
#                 "full_ids": full,
#                 "response_ids": resp,
#             })

#         samples = build_samples_from_generations(
#             gens=gens,
#             block_size=block_size,
#             pad_to_multiple_of=8,
#             device=self.device,
#         )
#         return [samples]

#     # ------------------- 经验评估 -------------------
#     @torch.no_grad()
#     def evaluate_experience(self, samples: Samples, debug: bool = False):
#         """
#         - 计算 actor/ref 的 token logprob（target=seqs[:,1:]）
#         - 奖励模型得到样本标量奖励（序列级），把奖励均分到 response 段
#         - critic 计算 values，以 GAE 得到 returns/advantages
#         - 返回的 avg_kl：**非负**，便于自适应 KL 控制与日志直观
#         """
#         seqs        = samples.seqs.to(self.device)           # [B, T]
#         attn_mask   = samples.attention_mask.to(self.device) # [B, T]
#         act_mask    = samples.action_mask.to(self.device)    # [B, T]
#         num_actions = samples.num_actions.to(self.device)    # [B]

#         B, T = seqs.size()
#         if num_actions.sum().item() == 0:
#             empty = Experience(
#                 seqs=seqs,
#                 action_log_probs=torch.zeros((B, 0), device=self.device),
#                 values=torch.zeros((B, 0), device=self.device),
#                 returns=torch.zeros((B, 0), device=self.device),
#                 advantages=torch.zeros((B, 0), device=self.device),
#                 attention_mask=attn_mask[:, :0],
#                 action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
#                 reward=torch.zeros((B, 1), device=self.device),
#                 num_actions=num_actions,
#                 kl=torch.zeros((B, 0), device=self.device),
#             )
#             return [empty], 0.0, 0.0, 0.0, 0.0

#         # ---- 1) actor/ref token logprob（target=seqs[:,1:]）----
#         actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
#             actor=self.actor, ref=self.ref,
#             seqs=seqs, action_mask=act_mask,
#             device_type=self.device.type,
#             ptdtype=None,
#             micro_batch_size=self.mb_size_logits,
#         )  # [B, T-1], [B, T-1], [B, T-1]

#         # 统一裁到最短长度，防止任何 off-by-one
#         L = min(actor_lp.size(1), ref_lp.size(1), mask_tgt.size(1))
#         actor_lp = actor_lp[:, :L]
#         ref_lp   = ref_lp[:, :L]
#         mask_tgt = mask_tgt[:, :L]  # 这是 response 段在 target 对齐维度上的掩码（B, L）

#         # ---- 2) 奖励模型（标量），只用有效 token（去除右侧 pad）----
#         valid_lens = attn_mask.sum(dim=1).tolist()  # [B]
#         texts = []
#         for i in range(B):
#             Li = int(valid_lens[i])
#             toks = seqs[i, :Li]
#             texts.append(self._safe_decode(self.actor_tokenizer, toks))

#         texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

#         # 安全夹紧 reward_tokenizer 的 max_length
#         rm_max_len = getattr(self.reward_tokenizer, "model_max_length", 4096)
#         try:
#             rm_max_len = int(rm_max_len)
#         except Exception:
#             rm_max_len = 4096
#         if rm_max_len <= 0 or rm_max_len > 8192:
#             rm_max_len = 4096

#         r_inputs = self.reward_tokenizer(
#             texts,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=rm_max_len,
#         )
#         # 奖励模型在 CPU，结果搬回 GPU
#         r_out = self.reward_model(**r_inputs)
#         logits = r_out.logits
#         if logits.dim() == 2 and logits.size(1) == 1:
#             reward_scores = logits.squeeze(1).to(self.device)
#         else:
#             reward_scores = logits[..., 0].to(self.device)

#         # ---- 2.1) EMA 基线（中心化） + KL（仅日志）----
#         self.r_baseline = 0.95 * self.r_baseline + 0.05 * reward_scores.mean().item()
#         r_center = (reward_scores - self.r_baseline).clamp(-self.clip_reward, self.clip_reward)  # [B]

#         # 抽样动作的 KL 仅用于报告
#         na = mask_tgt.sum(dim=1).clamp_min(1)                               # [B]
#         kl_raw_per_seq = ((actor_lp - ref_lp) * mask_tgt).sum(dim=1) / na   # [B]
#         kl_report = float((kl_raw_per_seq.abs()).mean().item())

#         # 奖励塑形：只用中心化奖励（不扣 KL）
#         shaped = r_center.clamp(-self.clip_reward, self.clip_reward)  # [B]
        
#         # 只在每个样本的“最后一个 action token”打奖励
#         B, L = mask_tgt.size()
#         rewards_t = torch.zeros_like(mask_tgt, dtype=torch.float)     # [B, L]
        
#         # 计算每行最后一个为1的位置（action的最后位置）
#         idx = (mask_tgt.long() * torch.arange(L, device=mask_tgt.device).unsqueeze(0)).amax(dim=1)  # [B]
        
#         # scatter 写入奖励（不做长度归一，直接给 shaped）
#         rewards_t.scatter_(1, idx.unsqueeze(1), shaped.unsqueeze(1))
        
#         # 保险：再与掩码对齐
#         rewards_t = rewards_t * mask_tgt
        
#         # 逐 token KL（仅调试）
#         kl_t = (actor_lp - ref_lp) * mask_tgt

#         # ---- 3) critic values & GAE（只在 response 段）----
#         with torch.no_grad():
#             values_full = forward_values_via_actor(
#                 model=self.actor, critic=self.critic,
#                 seqs=seqs, device_type=self.device.type, ptdtype=None,
#                 micro_batch_size=self.mb_size_values,
#                 detach_hidden=True,
#             )  # [B, T]
#         values_t = values_full[:, 1:][:, :L]  # [B, L] 与 target 维对齐

#         returns_t, adv_t = gae_compute(
#             values=values_t, rewards=rewards_t, mask_time=mask_tgt,
#             gamma=self.gamma, lam=self.lambd, use_last_as_terminal=True
#         )

#         # 记录标准化前优势统计
#         denom = mask_tgt.sum().clamp_min(1)
#         adv_mean_raw = ((adv_t * mask_tgt).sum() / denom).item()
#         adv_var_raw  = (((adv_t - adv_mean_raw)**2 * mask_tgt).sum() / denom).item()
#         adv_std_raw  = float((adv_var_raw + 1e-8) ** 0.5)

#         # 对 response 段做优势标准化（用于训练）
#         mean  = (adv_t * mask_tgt).sum() / denom
#         var   = ((adv_t - mean)**2 * mask_tgt).sum() / denom
#         adv_t = (adv_t - mean) / torch.sqrt(var + 1e-8)

#         avg_reward = reward_scores.mean().item()  # 原始 RM reward 的平均值

#         exp = Experience(
#             seqs=seqs,                              # [B, T]
#             action_log_probs=actor_lp,              # [B, L]（old logp）
#             values=values_t,                        # [B, L]
#             returns=returns_t,                      # [B, L]
#             advantages=adv_t,                       # [B, L]
#             attention_mask=attn_mask[:, 1:][:, :L], # [B, L]
#             action_mask=mask_tgt,                   # [B, L]
#             reward=reward_scores.unsqueeze(1),      # [B, 1]
#             num_actions=num_actions,                # [B]
#             kl=kl_t,                                # [B, L]
#         )

#     # 保存若干评估期指标
#         self.last_stats.update({
#             "rm_reward_mean": float(avg_reward),
#             "rm_reward_center_mean": float(r_center.mean().item()),
#             "rm_reward_shaped_mean": float(shaped.mean().item()),
#             "kl_report": float(kl_report),
#             "adv_mean_raw": float(adv_mean_raw),
#             "adv_std_raw": float(adv_std_raw),
#         })

#         # KL 返回非负，便于外层自适应 KL 控制
#         return [exp], float(kl_report), float(avg_reward), float(shaped.mean().item()), float(r_center.mean().item())


#     # ------------------- Loss -------------------
#     @staticmethod
#     def _policy_loss(new_logp, old_logp, advantages, mask, clip_eps=0.2):
#         ratio = (new_logp - old_logp).exp()
#         surr1 = ratio * advantages
#         surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
#         loss = -torch.min(surr1, surr2)
#         denom = mask.sum(dim=1).clamp_min(1)
#         return ((loss * mask).sum(dim=1) / denom).mean(), ratio

#     @staticmethod
#     def _value_loss(values, old_values, returns, mask, clip_eps=0.2):
#         old_values = old_values.detach()
#         clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
#         surr1 = (returns - values).pow(2)
#         surr2 = (returns - clipped).pow(2)
#         loss = torch.max(surr1, surr2)
#         denom = mask.sum(dim=1).clamp_min(1)
#         return ((loss * mask).sum(dim=1) / denom).mean()

#     # ------------------- 训练一步 -------------------
#     def train_on_experience(self, experience: Experience, use_token_entropy: bool = False):
#         self.actor.train()
#         self.critic.train()

#         # ------- Actor update -------
#         with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
#             # 前向拿新策略 logp（支持 micro-batch）
#             logits_all = model_all_logits(
#                 model=self.actor, seqs=experience.seqs,
#                 device_type=self.device.type, ptdtype=None,
#                 micro_batch_size=self.mb_size_logits,
#             )  # [B, T, V]

#             new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B, T-1]

#             # 与 old_logp / advantages / action_mask 对齐到相同 L
#             L = min(
#                 new_lp_all.size(1),
#                 experience.action_log_probs.size(1),
#                 experience.advantages.size(1),
#                 experience.action_mask.size(1),
#             )
#             new_lp = new_lp_all[:, :L]                       # [B, L]
#             old_lp = experience.action_log_probs[:, :L]      # [B, L]
#             adv    = experience.advantages[:, :L]            # [B, L]
#             mask   = experience.action_mask[:, :L].float()   # [B, L]

#             # 可选：token 熵掩码（默认关）
#             if use_token_entropy:
#                 from .common import apply_entropy_mask
#                 mask = apply_entropy_mask(logits_all[:, :-1][:, :L], mask, keep_ratio=0.2).float()

#             # PPO policy clip loss
#             ratio = (new_lp - old_lp).exp()
#             surr1 = ratio * adv
#             surr2 = ratio.clamp(1 - 0.2, 1 + 0.2) * adv
#             p_loss_tok = -torch.min(surr1, surr2)                        # [B, L]
#             denom = mask.sum(dim=1).clamp_min(1)                         # [B]
#             p_loss = ((p_loss_tok * mask).sum(dim=1) / denom).mean()     # 标量

#             # === KL-to-ref 正则（稳定、可导）===
#             # evaluate_experience 里存了 experience.kl = (old_actor_lp - ref_lp_old)
#             # => ref_lp_old = old_actor_lp - experience.kl
#             ref_lp_old = (old_lp - experience.kl[:, :L]).detach()        # [B, L]（不反传到旧量）

#             # 稳定 KL proxy：让 new_lp 靠近 ref_lp_old（按采样动作的 masked 均值）
#             kl_proxy_tok = (new_lp - ref_lp_old) ** 2                    # [B, L]
#             kl_loss = ((kl_proxy_tok * mask).sum(dim=1) / denom).mean()  # 标量

#             # 最终 actor loss = policy loss + β * KL
#             actor_loss = p_loss + self.kl_ctl * kl_loss

#             # 诊断
#             with torch.no_grad():
#                 # 被裁剪比例
#                 clip_mask = (ratio.clamp(1 - 0.2, 1 + 0.2) != ratio).float()
#                 clip_frac = (clip_mask * mask).sum() / mask.sum().clamp_min(1)
#                 # 经典 approx_kl（old→new），仅作监控
#                 approx_kl_pi = masked_mean((old_lp - new_lp), mask)
#                 # 轻量“熵” proxy：-mean(logp of taken actions)
#                 entropy_tok = -masked_mean(new_lp, mask)

#         # 反传：actor
#         self.optimizer_actor.zero_grad(set_to_none=True)
#         self.scaler_actor.scale(actor_loss).backward()
#         self.scaler_actor.unscale_(self.optimizer_actor)
#         clip_by_global_norm(self.actor.parameters(), 1.0)
#         self.scaler_actor.step(self.optimizer_actor)
#         self.scaler_actor.update()

#         # ------- Critic update -------
#         with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
#             values_full = forward_values_via_actor(
#                 model=self.actor, critic=self.critic,
#                 seqs=experience.seqs, device_type=self.device.type, ptdtype=None,
#                 micro_batch_size=self.mb_size_values,
#                 detach_hidden=True,
#             )  # [B, T]
#             values_t_all = values_full[:, 1:]  # [B, T-1]

#             Lv = min(
#                 values_t_all.size(1),
#                 experience.values.size(1),
#                 experience.returns.size(1),
#                 experience.action_mask.size(1),
#             )
#             values_t = values_t_all[:, :Lv]
#             mask_v   = experience.action_mask[:, :Lv].float()

#             # value clip loss（PPO 风格）
#             old_v = experience.values[:, :Lv].detach()
#             clipped_v = old_v + (values_t - old_v).clamp(-0.2, 0.2)
#             surr1 = (experience.returns[:, :Lv] - values_t).pow(2)
#             surr2 = (experience.returns[:, :Lv] - clipped_v).pow(2)
#             v_loss_tok = torch.max(surr1, surr2)
#             denom_v = mask_v.sum(dim=1).clamp_min(1)
#             v_loss = ((v_loss_tok * mask_v).sum(dim=1) / denom_v).mean()

#             # 诊断
#             with torch.no_grad():
#                 v_mae = masked_mean(torch.abs(values_t - experience.returns[:, :Lv]), mask_v)
#                 # explained variance（按 token）
#                 y   = experience.returns[:, :Lv]
#                 yhat= values_t
#                 y_mean = masked_mean(y, mask_v)
#                 var_y  = masked_mean((y - y_mean)**2, mask_v)
#                 var_e  = masked_mean((y - yhat)**2,  mask_v)
#                 explained_var = 1.0 - (var_e / (var_y + 1e-8))

#         # 反传：critic
#         self.optimizer_critic.zero_grad(set_to_none=True)
#         self.scaler_critic.scale(v_loss).backward()
#         self.scaler_critic.unscale_(self.optimizer_critic)
#         clip_by_global_norm(self.critic.parameters(), 1.0)
#         self.scaler_critic.step(self.optimizer_critic)
#         self.scaler_critic.update()

#         # 供外层日志使用
#         self.last_stats = {
#             "clip_frac": float(clip_frac.detach().item()),
#             "approx_kl_pi": float(approx_kl_pi.detach().item()),
#             "entropy": float(entropy_tok.detach().item()) if torch.is_tensor(entropy_tok) else float(entropy_tok),
#             "v_mae": float(v_mae.detach().item()),
#             "explained_var": float(explained_var.detach().item()),
#             "actor_loss": float(actor_loss.detach().item()),
#             "p_loss": float(p_loss.detach().item()),
#             "kl_loss": float(kl_loss.detach().item()),
#         }
#         # 外层把第一个返回当作 "p" 打印，这里返回 actor_loss 更符合真实目标
#         return actor_loss, v_loss

# 根据参考实现改的
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from typing import List, Tuple

from .common import (
    Samples, Experience,
    normalize_for_reward,
    build_samples_from_generations,
    model_all_logits, token_logprobs_from_logits,
    compute_actor_ref_logprobs,
    gae_compute, masked_mean, clip_by_global_norm,
    forward_values_via_actor,
)

# -------------------------- Critic --------------------------
class Critic(nn.Module):
    """
    简单 value 头：基于 actor 的隐藏态输出每个 token 的 V(s_t)。
    注意不要只对 response 片段单独前向 actor，否则会丢上下文。
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.n_embd = base_model.config.n_embd
        self.value_head = nn.Linear(self.n_embd, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, C]  ->  values: [B, T]
        return self.value_head(hidden_states).squeeze(-1)


# ------------------------ PPO Trainer ------------------------
class PPOTrainer:
    def __init__(
        self,
        actor_model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        critic_model: nn.Module,
        actor_tokenizer,
        reward_tokenizer,
        optimizer_actor,
        optimizer_critic,
        kl_ctl: float = 0.35,         # KL 系数（β）
        clip_reward: float = 5.0,     # 奖励裁剪（对序列级奖励裁剪）
        gamma: float = 1.0,           # RLHF 常用 1.0
        lambd: float = 0.95,          # GAE 参数
        mb_size_logits: int = 0,      # actor/ref logits 前向的 micro-batch（0=不切）
        mb_size_values: int = 0,      # critic values 前向的 micro-batch（0=不切）
        device: str = "cuda",
    ):
        self.actor = actor_model.to(device)
        self.ref = ref_model.to(device)
        # 奖励模型保持 CPU（外部包装器已 to('cpu')）
        self.reward_model = reward_model
        self.reward_model.eval()
        self.critic = critic_model.to(device)

        self.actor_tokenizer = actor_tokenizer
        self.reward_tokenizer = reward_tokenizer

        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic

        self.kl_ctl = kl_ctl
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.lambd = lambd

        self.mb_size_logits = mb_size_logits
        self.mb_size_values = mb_size_values

        self.device = torch.device(device)
        self.use_amp = (self.device.type == "cuda")
        self.scaler_actor  = amp.GradScaler(enabled=self.use_amp)
        self.scaler_critic = amp.GradScaler(enabled=self.use_amp)

        # pad/eos（兼容自定义 GPT2Tok 与 HF tokenizer）
        self.pad_token_id = getattr(actor_tokenizer, "pad_token_id", 0)
        self.eos_token_id = getattr(actor_tokenizer, "eos_token_id",
                             getattr(actor_tokenizer, "eos_id", 50256))

        # 奖励 EMA 基线
        self.r_baseline = 0.0

        # 最近一次训练统计
        self.last_stats = {}

    # --------- 小工具：安全解码（兼容自定义/HF tokenizer） ---------
    @staticmethod
    def _safe_decode(tok, ids_tensor) -> str:
        ids = ids_tensor.detach().cpu().tolist()
        # 优先 decode(skip_special_tokens=False)
        if hasattr(tok, "decode"):
            try:
                return tok.decode(ids, skip_special_tokens=False)
            except TypeError:
                try:
                    return tok.decode(ids)
                except Exception:
                    pass
        # 退路 batch_decode
        if hasattr(tok, "batch_decode"):
            try:
                return tok.batch_decode([ids], skip_special_tokens=False)[0]
            except TypeError:
                try:
                    return tok.batch_decode([ids])[0]
                except Exception:
                    pass
        # 退路 tokens->join
        if hasattr(tok, "convert_ids_to_tokens"):
            try:
                return " ".join(tok.convert_ids_to_tokens(ids))
            except Exception:
                pass
        # 最后兜底：id 串
        return " ".join(str(x) for x in ids)

    # ------------------- 逐样本生成（避免 pad 干扰） -------------------
    @torch.no_grad()
    def generate_samples(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        max_length: int,
        max_new_tokens: int
    ) -> List[Samples]:
        input_ids, attention_mask = inputs
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        B = input_ids.size(0)
        block_size = self.actor.config.block_size

        gens = []
        for i in range(B):
            p_len = int(attention_mask[i].sum().item())
            p_len = min(p_len, block_size - 1)  # 预留 EOS
            if p_len <= 0:
                continue
            prompt = input_ids[i:i+1, :p_len]   # [1, T_prompt]
            room = max(1, min(max_new_tokens, block_size - p_len - 1))

            out = self.actor.generate(
                idx=prompt,
                max_new_tokens=room,
                temperature=1.0,
                top_k=None,
                eos_token_id=self.eos_token_id
            )  # [1, T_prompt + T_resp]

            full = out[0]
            resp = full[p_len:]
            gens.append({
                "prompt_ids": prompt[0],
                "full_ids": full,
                "response_ids": resp,
            })

        samples = build_samples_from_generations(
            gens=gens,
            block_size=block_size,
            pad_to_multiple_of=8,
            device=self.device,
        )
        return [samples]

    # ------------------- 经验评估 -------------------
    @torch.no_grad()
    def evaluate_experience(self, samples: Samples, debug: bool = False):
        """
        - 计算 actor/ref 的 token logprob（target=seqs[:,1:]）
        - 奖励模型得到样本标量奖励（序列级）并只打到“最后一个 action token”
        - KL 作为逐 token 负奖励（-β * KL）加入 reward（不进 loss）
        - critic 计算 values，以 GAE 得到 returns/advantages
        """
        seqs        = samples.seqs.to(self.device)           # [B, T]
        attn_mask   = samples.attention_mask.to(self.device) # [B, T]
        act_mask    = samples.action_mask.to(self.device)    # [B, T]
        num_actions = samples.num_actions.to(self.device)    # [B]

        B, T = seqs.size()
        if num_actions.sum().item() == 0:
            empty = Experience(
                seqs=seqs,
                action_log_probs=torch.zeros((B, 0), device=self.device),
                values=torch.zeros((B, 0), device=self.device),
                returns=torch.zeros((B, 0), device=self.device),
                advantages=torch.zeros((B, 0), device=self.device),
                attention_mask=attn_mask[:, :0],
                action_mask=torch.zeros((B, 0), device=self.device, dtype=torch.long),
                reward=torch.zeros((B, 1), device=self.device),
                num_actions=num_actions,
                kl=torch.zeros((B, 0), device=self.device),
            )
            return [empty], 0.0, 0.0, 0.0, 0.0

        # ---- 1) actor/ref token logprob（target=seqs[:,1:]）----
        actor_lp, ref_lp, mask_tgt = compute_actor_ref_logprobs(
            actor=self.actor, ref=self.ref,
            seqs=seqs, action_mask=act_mask,
            device_type=self.device.type,
            ptdtype=None,
            micro_batch_size=self.mb_size_logits,
        )  # [B, T-1], [B, T-1], [B, T-1]

        # 对齐长度
        L = min(actor_lp.size(1), ref_lp.size(1), mask_tgt.size(1))
        actor_lp = actor_lp[:, :L]
        ref_lp   = ref_lp[:, :L]
        mask_tgt = mask_tgt[:, :L].to(actor_lp.dtype)  # [B, L] 用作权重

        # ---- 2) 奖励模型（标量），只用有效 token（去除右侧 pad）----
        valid_lens = attn_mask.sum(dim=1).tolist()  # [B]
        texts = []
        for i in range(B):
            Li = int(valid_lens[i])
            toks = seqs[i, :Li]
            texts.append(self._safe_decode(self.actor_tokenizer, toks))

        texts = [normalize_for_reward(t, reward_tokenizer=self.reward_tokenizer) for t in texts]

        # 安全夹紧 RM tokenizer 的 max_length
        rm_max_len = getattr(self.reward_tokenizer, "model_max_length", 4096)
        try:
            rm_max_len = int(rm_max_len)
        except Exception:
            rm_max_len = 4096
        if rm_max_len <= 0 or rm_max_len > 8192:
            rm_max_len = 4096

        r_inputs = self.reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=rm_max_len,
        )
        # 奖励模型在 CPU，结果搬回 GPU
        r_out = self.reward_model(**r_inputs)
        logits = r_out.logits
        if logits.dim() == 2 and logits.size(1) == 1:
            reward_scores = logits.squeeze(1).to(self.device)  # [B]
        else:
            reward_scores = logits[..., 0].to(self.device)     # [B]

        # ---- 2.1) EMA 基线、中心化 ----
        self.r_baseline = 0.95 * self.r_baseline + 0.05 * reward_scores.mean().item()
        r_center = (reward_scores - self.r_baseline).clamp(-self.clip_reward, self.clip_reward)  # [B]

        # ---- KL（仅用于 shaping & 监控，不进 loss）----
        kl_t = (actor_lp - ref_lp) * mask_tgt                  # [B, L]
        na = mask_tgt.sum(dim=1).clamp_min(1.0)                # [B]
        kl_seq = (kl_t.sum(dim=1) / na)                        # [B]
        kl_report = float(kl_seq.abs().mean().item())

        # ---- 合成逐 token 奖励 ----
        # 1) KL 负奖励：-β * KL_t
        rewards_t = (-self.kl_ctl) * kl_t.float()              # [B, L]

        # 2) RM 标量只打到“最后一个 action token”
        Bsz, LL = mask_tgt.size()
        pos = torch.arange(LL, device=mask_tgt.device).unsqueeze(0)  # [1, L]
        last_idx = (mask_tgt * pos).amax(dim=1).long()               # [B]
        rewards_t.scatter_add_(1, last_idx.unsqueeze(1), r_center.unsqueeze(1))

        # 3) 保守处理：确保 mask 外为 0
        rewards_t = rewards_t * mask_tgt

        # ---- 3) critic values & GAE（只在 response 段）----
        with torch.no_grad():
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=seqs, device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_values,
                detach_hidden=True,
            )  # [B, T]
        values_t = values_full[:, 1:][:, :L]  # [B, L]

        returns_t, adv_t = gae_compute(
            values=values_t, rewards=rewards_t, mask_time=mask_tgt,
            gamma=self.gamma, lam=self.lambd, use_last_as_terminal=True
        )

        # 记录标准化前优势统计
        denom = mask_tgt.sum().clamp_min(1.0)
        adv_mean_raw = ((adv_t * mask_tgt).sum() / denom).item()
        adv_var_raw  = (((adv_t - adv_mean_raw)**2 * mask_tgt).sum() / denom).item()
        adv_std_raw  = float((adv_var_raw + 1e-8) ** 0.5)

        # 优势标准化
        mean  = (adv_t * mask_tgt).sum() / denom
        var   = ((adv_t - mean)**2 * mask_tgt).sum() / denom
        adv_t = (adv_t - mean) / torch.sqrt(var + 1e-8)

        avg_reward = reward_scores.mean().item()

        exp = Experience(
            seqs=seqs,                              # [B, T]
            action_log_probs=actor_lp,              # [B, L]（old logp）
            values=values_t,                        # [B, L]
            returns=returns_t,                      # [B, L]
            advantages=adv_t,                       # [B, L]
            attention_mask=attn_mask[:, 1:][:, :L], # [B, L]
            action_mask=mask_tgt,                   # [B, L]
            reward=reward_scores.unsqueeze(1),      # [B, 1]
            num_actions=num_actions,                # [B]
            kl=kl_t,                                # [B, L]
        )

        # 保存若干评估期指标
        self.last_stats.update({
            "rm_reward_mean": float(avg_reward),
            "rm_reward_center_mean": float(r_center.mean().item()),
            "kl_report": float(kl_report),
            "adv_mean_raw": float(adv_mean_raw),
            "adv_std_raw": float(adv_std_raw),
        })

        return [exp], float(kl_report), float(avg_reward), float(r_center.mean().item()), float(r_center.mean().item())

    # ------------------- Loss -------------------
    @staticmethod
    def _policy_loss(new_logp, old_logp, advantages, mask, clip_eps=0.2):
        ratio = (new_logp - old_logp).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        denom = mask.sum(dim=1).clamp_min(1)
        return ((loss * mask).sum(dim=1) / denom).mean(), ratio

    @staticmethod
    def _value_loss(values, old_values, returns, mask, clip_eps=0.2):
        old_values = old_values.detach()
        clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        surr1 = (returns - values).pow(2)
        surr2 = (returns - clipped).pow(2)
        loss = torch.max(surr1, surr2)
        denom = mask.sum(dim=1).clamp_min(1)
        return ((loss * mask).sum(dim=1) / denom).mean()

    # ------------------- 训练一步 -------------------
    def train_on_experience(self, experience: Experience, use_token_entropy: bool = False):
        self.actor.train()
        self.critic.train()

        # ------- Actor update -------
        with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits_all = model_all_logits(
                model=self.actor, seqs=experience.seqs,
                device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_logits,
            )  # [B, T, V]

            new_lp_all = token_logprobs_from_logits(logits_all, experience.seqs)  # [B, T-1]

            # 与 old_logp / advantages / action_mask 对齐到相同 L
            L = min(
                new_lp_all.size(1),
                experience.action_log_probs.size(1),
                experience.advantages.size(1),
                experience.action_mask.size(1),
            )
            new_lp = new_lp_all[:, :L]                       # [B, L]
            old_lp = experience.action_log_probs[:, :L]      # [B, L]
            adv    = experience.advantages[:, :L]            # [B, L]
            mask   = experience.action_mask[:, :L].float()   # [B, L]

            # 可选：token 熵掩码（默认关）
            if use_token_entropy:
                from .common import apply_entropy_mask
                mask = apply_entropy_mask(logits_all[:, :-1][:, :L], mask, keep_ratio=0.2).float()

            # PPO policy clip loss（KL 不在 loss 中）
            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - 0.2, 1 + 0.2) * adv
            p_loss_tok = -torch.min(surr1, surr2)                        # [B, L]
            denom = mask.sum(dim=1).clamp_min(1)                         # [B]
            p_loss = ((p_loss_tok * mask).sum(dim=1) / denom).mean()     # 标量
            actor_loss = p_loss

            # 诊断
            with torch.no_grad():
                clip_mask = (ratio.clamp(1 - 0.2, 1 + 0.2) != ratio).float()
                clip_frac = (clip_mask * mask).sum() / mask.sum().clamp_min(1)
                approx_kl_pi = masked_mean((old_lp - new_lp), mask)
                entropy_tok  = -masked_mean(new_lp, mask)

        # 反传：actor
        self.optimizer_actor.zero_grad(set_to_none=True)
        self.scaler_actor.scale(actor_loss).backward()
        self.scaler_actor.unscale_(self.optimizer_actor)
        clip_by_global_norm(self.actor.parameters(), 1.0)
        self.scaler_actor.step(self.optimizer_actor)
        self.scaler_actor.update()

        # ------- Critic update -------
        with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            values_full = forward_values_via_actor(
                model=self.actor, critic=self.critic,
                seqs=experience.seqs, device_type=self.device.type, ptdtype=None,
                micro_batch_size=self.mb_size_values,
                detach_hidden=True,
            )  # [B, T]
            values_t_all = values_full[:, 1:]  # [B, T-1]

            Lv = min(
                values_t_all.size(1),
                experience.values.size(1),
                experience.returns.size(1),
                experience.action_mask.size(1),
            )
            values_t = values_t_all[:, :Lv]
            mask_v   = experience.action_mask[:, :Lv].float()

            # value clip loss（PPO 风格）
            old_v = experience.values[:, :Lv].detach()
            clipped_v = old_v + (values_t - old_v).clamp(-0.2, 0.2)
            surr1 = (experience.returns[:, :Lv] - values_t).pow(2)
            surr2 = (experience.returns[:, :Lv] - clipped_v).pow(2)
            v_loss_tok = torch.max(surr1, surr2)
            denom_v = mask_v.sum(dim=1).clamp_min(1)
            v_loss = ((v_loss_tok * mask_v).sum(dim=1) / denom_v).mean()

            with torch.no_grad():
                v_mae = masked_mean(torch.abs(values_t - experience.returns[:, :Lv]), mask_v)
                y   = experience.returns[:, :Lv]
                yhat= values_t
                y_mean = masked_mean(y, mask_v)
                var_y  = masked_mean((y - y_mean)**2, mask_v)
                var_e  = masked_mean((y - yhat)**2,  mask_v)
                explained_var = 1.0 - (var_e / (var_y + 1e-8))

        # 反传：critic
        self.optimizer_critic.zero_grad(set_to_none=True)
        self.scaler_critic.scale(v_loss).backward()
        self.scaler_critic.unscale_(self.optimizer_critic)
        clip_by_global_norm(self.critic.parameters(), 1.0)
        self.scaler_critic.step(self.optimizer_critic)
        self.scaler_critic.update()

        # 供外层日志使用
        self.last_stats = {
            "clip_frac": float(clip_frac.detach().item()),
            "approx_kl_pi": float(approx_kl_pi.detach().item()),
            "entropy": float(entropy_tok.detach().item()) if torch.is_tensor(entropy_tok) else float(entropy_tok),
            "v_mae": float(v_mae.detach().item()),
            "explained_var": float(explained_var.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "p_loss": float(p_loss.detach().item()),
            "kl_loss": 0.0,  # KL 不进 loss，这里置 0 以免误解
        }
        return actor_loss, v_loss

