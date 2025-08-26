from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import random

# Samples = 一次完整文本生成的“样本”
# Experience = 在此基础上，结合强化学习信号封装的训练用经历

@dataclass
class Samples:
    seqs: torch.Tensor  # (B, L)
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
        if len(self.buffer) > self.limit:
            self.buffer = self.buffer[-self.limit:]

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = []

def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def normalize_for_reward(text: str, reward_tokenizer=None) -> str:
    if reward_tokenizer is not None and hasattr(reward_tokenizer, "eos_token"):
        text = text.replace("<|endoftext|>", reward_tokenizer.eos_token)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if contains_chinese(text):
        text = text.replace("，", ",").replace("。", ".")
    return "".join(c for c in text if c.isprintable())