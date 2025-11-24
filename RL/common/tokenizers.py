# RL/common/tokenizers.py
import tiktoken
import torch

class GPT2Tok:
    def __init__(self):
        enc = tiktoken.get_encoding("gpt2")
        self.enc = enc
        self.eos_token = "<|endoftext|>"
        self.eos_id = enc.encode(self.eos_token, allowed_special={self.eos_token})[0]
        self.pad_token_id = 0
        self.eos_token_id = self.eos_id

    def encode(self, s):
        return self.enc.encode(s, allowed_special="all")

    def decode(self, ids):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.enc.decode(ids)
