"""
Full definition of a GPT Language Model in a single file,
hardened for RL (PPO/GRPO/DAPO) and robust GPT-2 weight import.

References:
1) OpenAI GPT-2 TF: https://github.com/openai/gpt-2/blob/master/src/model.py
2) HF GPT-2 PyTorch: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


# -------- LayerNorm (optional bias) --------
class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn't support bias=False)."""

    def __init__(self, ndim, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# -------- Causal Self-Attention --------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            # fallback causal mask
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# -------- MLP --------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# -------- Transformer Block --------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# -------- GPT Config --------
@dataclass
class GPTConfig:
    block_size: int = 1024
    # Note: default to 50304 (padded to multiple of 64, NanoGPT style)
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True matches GPT-2


# -------- GPT Model --------
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # ----- utils -----
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ----- forward variants (RL-friendly) -----
    def forward(
        self,
        idx: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
        return_all_logits: bool = False,
        return_hidden: bool = False,
    ):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        pos_emb = self.transformer.wpe(pos)  # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (B, T, C)

        if return_all_logits:
            logits_all = self.lm_head(x)  # (B, T, V)
            if return_hidden:
                return logits_all, None, x
            return logits_all, None

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            if return_hidden:
                return logits, loss, x
            return logits, loss
        else:
            logits_last = self.lm_head(x[:, [-1], :])  # (B, 1, V)
            if return_hidden:
                return logits_last, None, x
            return logits_last, None

    @torch.no_grad()
    def forward_logprobs(
        self, idx: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Return per-token logprobs for the *next* token at each position.
        Shape: (B, T, V) -> returns (B, T, V) log-softmax over vocab.
        RL often needs logprobs; computing once is cheaper than softmax many times.
        """
        logits, _ = self(idx, return_all_logits=True)  # (B, T, V)
        return F.log_softmax(logits, dim=-1)

    # ----- block size surgery -----
    def crop_block_size(self, block_size: int):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # ----- import GPT-2 weights from HF, with robust vocab padding -----
    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        """
        Load GPT-2 weights from HF and copy into this NanoGPT-style model.

        Differences vs. vanilla NanoGPT:
        - Defaults to vocab_size=50304 (padded to multiple of 64), then safely copies
          the first 50257 rows of wte/lm_head and zero-fills the tail.
        - You can force 50257 by passing override_args={'vocab_size': 50257}.
        - 'dropout' can still be overridden.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}

        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # base arch
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]

        # vocab & block
        # default: padded 50304 (NanoGPT style). Allow override to pure 50257 if needed.
        target_vocab = int(override_args.get("vocab_size", 50304))
        if target_vocab <= 0:
            target_vocab = 50304
        config_args["vocab_size"] = target_vocab
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate -> {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # create fresh model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # remove non-param buffers from our list
        own_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # init HF model
        hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf.state_dict()
        # filter out HF buffers we don't use
        hf_keys = [
            k
            for k in sd_hf.keys()
            if (not k.endswith(".attn.masked_bias")) and (not k.endswith(".attn.bias"))
        ]

        # helper to copy (with conv1d transpose), and handle vocab pad
        transposed_suffixes = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        def _is_transposed_key(k: str) -> bool:
            return any(k.endswith(suf) for suf in transposed_suffixes)

        hf_vocab = 50257  # GPT-2 official vocab size
        copied, transposed, padded = 0, 0, 0

        # iterate over our param keys, copy from HF by matching name
        for k in own_keys:
            if k not in sd:
                continue
            if k not in sd_hf:
                # allow missing only for our padded vocab rows (handled below)
                if k in ("transformer.wte.weight", "lm_head.weight"):
                    # we'll fill manually from HF weights below
                    pass
                else:
                    raise RuntimeError(f"[import] missing HF key: {k}")
            w = sd[k]
            if k in ("transformer.wte.weight", "lm_head.weight"):
                # vocab may be larger on our side (50304 vs 50257)
                w_hf = sd_hf["transformer.wte.weight" if k.startswith("transformer.wte") else "lm_head.weight"]
                if w.shape[0] < w_hf.shape[0]:
                    raise RuntimeError(
                        f"[import] our {k} rows {w.shape[0]} < HF rows {w_hf.shape[0]}"
                    )
                with torch.no_grad():
                    # copy first 50257 rows, zero-init tail
                    w[:hf_vocab, :].copy_(w_hf)
                    if w.shape[0] > hf_vocab:
                        w[hf_vocab:, :].zero_()
                        padded += 1
                copied += 1
                continue

            # all other weights: shapes must match (except conv1d transpose)
            w_hf = sd_hf[k]
            if _is_transposed_key(k):
                assert (
                    w_hf.shape[::-1] == w.shape
                ), f"[import] shape mismatch (T) {k}: HF {w_hf.shape} vs ours {w.shape}"
                with torch.no_grad():
                    w.copy_(w_hf.t())
                transposed += 1
                copied += 1
            else:
                assert (
                    w_hf.shape == w.shape
                ), f"[import] shape mismatch {k}: HF {w_hf.shape} vs ours {w.shape}"
                with torch.no_grad():
                    w.copy_(w_hf)
                copied += 1

        print(
            f"[import] copy: matched={copied} transposed={transposed} "
            f"padded_vocab={'yes' if padded else 'no'} (target_vocab={target_vocab})"
        )
        return model

    # ----- AdamW config -----
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters"
        )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    # ----- MFU estimate -----
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bf16 peak
        return flops_achieved / flops_promised

    # ----- generate (sampling) -----
    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        eos_id: Optional[int] = None,  # alias
    ):
        if eos_token_id is None and eos_id is not None:
            eos_token_id = eos_id

        B = idx.size(0)
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )
            logits_last, _ = self(idx_cond)  # (B, 1, V)
            logits = logits_last[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            if eos_token_id is not None:
                idx_next = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(idx_next, eos_token_id),
                    idx_next,
                )

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None:
                just_finished = idx_next.squeeze(1).eq(eos_token_id)
                finished = finished | just_finished
                if finished.all():
                    break
        return idx
