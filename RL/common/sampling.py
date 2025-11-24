# RL/common/sampling.py
import torch
from RL.PPO import Samples

@torch.no_grad()
def decode_with_sampling(model, idx, max_new, eos_id, block_size,
                         temperature, top_p, top_k, rep_penalty,
                         stop_strs, tokenizer_decode, min_resp):
    device = idx.device
    was_train = model.training
    model.eval()
    try:
        out = idx
        start = out.size(1)
        for _ in range(int(max_new)):
            x = out[:, -int(block_size):]
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            last = logits[:, -1, :]

            if rep_penalty and out.numel() > 0:
                uniq = torch.unique(out)
                last[:, uniq] = last[:, uniq] / float(rep_penalty)

            last = last / max(float(temperature), 1e-6)
            if top_k and top_k > 0:
                kth = torch.topk(last, k=min(int(top_k), last.size(-1)), dim=-1).values[..., -1:]
                last = torch.where(last < kth, torch.full_like(last, -1e10), last)

            probs = torch.softmax(last, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > float(top_p)).float().argmax(dim=-1, keepdim=True)
            mask = torch.arange(probs.size(-1), device=probs.device).view(1, -1) <= cutoff
            kept = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            kept = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            next_sorted = torch.multinomial(kept, num_samples=1)
            next_id = sorted_idx.gather(1, next_sorted)

            if (out.size(1) - start) < int(min_resp) and eos_id is not None and int(next_id.item()) == int(eos_id):
                alt = sorted_idx[:, 1:2] if sorted_idx.size(1) > 1 else next_id
                next_id = alt

            out = torch.cat((out, next_id.to(device)), dim=1)

            if (out.size(1) - start) >= int(min_resp):
                if eos_id is not None and int(next_id.item()) == int(eos_id):
                    break
                if stop_strs and tokenizer_decode is not None:
                    tail = tokenizer_decode(out[0][-min(out.size(1), block_size):].tolist())
                    if any(s in tail for s in stop_strs):
                        break
        return out
    finally:
        if was_train:
            model.train()


def pack_samples(gen_list, pad_id, block_size, device):
    if not gen_list:
        L = 1
        z = torch.zeros((0, L), dtype=torch.long, device=device)
        return Samples(
            seqs=z, attention_mask=z, action_mask=z,
            num_actions=torch.zeros(0, dtype=torch.long, device=device),
            response_length=torch.zeros(0, dtype=torch.long, device=device),
            total_length=torch.zeros(0, dtype=torch.long, device=device),
        )

    L = min(block_size, max(len(x["full_ids"]) for x in gen_list))
    B = len(gen_list)
    seqs = torch.full((B, L), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    amsk = torch.zeros((B, L), dtype=torch.long, device=device)
    num_actions = torch.zeros(B, dtype=torch.long, device=device)
    resp_len = torch.zeros(B, dtype=torch.long, device=device)
    total_len = torch.zeros(B, dtype=torch.long, device=device)

    for i, it in enumerate(gen_list):
        full = torch.tensor(it["full_ids"][:L], dtype=torch.long, device=device)
        p_len = min(len(it["prompt_ids"]), L)
        t = full.numel()
        seqs[i, :t] = full
        attn[i, :t] = 1
        total_len[i] = t
        if p_len < t:
            amsk[i, p_len:t] = 1
            na = int((amsk[i] == 1).sum().item())
            num_actions[i] = na
            resp_len[i] = na

    return Samples(
        seqs=seqs, attention_mask=attn, action_mask=amsk,
        num_actions=num_actions, response_length=resp_len, total_length=total_len,
    )
