# RL/common/train_utils.py
import random, numpy as np, torch

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

@torch.no_grad()
def greedy_eval_reward(actor_model, gpt2_tok, eval_prompt_ids, reward_tokenizer, reward_model, block_size, max_new_eval, stop_strs, min_resp_tok, decode_fn):
    dev = next(actor_model.parameters()).device
    eos_id = gpt2_tok.eos_id
    texts = []
    was_train = actor_model.training
    actor_model.eval()
    try:
        for ids in eval_prompt_ids:
            idx = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
            room = block_size - idx.size(1) - 1
            if room <= 0:
                texts.append(gpt2_tok.decode(ids[:block_size]))
                continue
            gen_len = max(8, min(int(max_new_eval), int(room)))
            out = decode_fn(
                actor_model, idx, gen_len, eos_id, block_size,
                temperature=1e-6, top_p=1.0, top_k=0, rep_penalty=1.0,
                stop_strs=stop_strs, tokenizer_decode=gpt2_tok.decode,
                min_resp=min_resp_tok,
            )
            full = out[0].tolist()[:block_size]
            texts.append(gpt2_tok.decode(full))
    finally:
        if was_train:
            actor_model.train()

    from RL.PPO import normalize_for_reward  # 你原来定义的位置
    texts = [normalize_for_reward(t, reward_tokenizer) for t in texts]
    toks = reward_tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    outs = reward_model(**toks)
    logits = getattr(outs, "logits", None)
    if logits is None:
        return float("nan")
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    return float(logits.mean().item()) if len(texts) > 0 else float("nan")
