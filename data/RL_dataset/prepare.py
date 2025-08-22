import os
import torch
from datasets import load_dataset
import tiktoken
import random

DATASET_NAME = "Anthropic/hh-rlhf"   # 替换为实际数据集
BLOCK_SIZE = 1024      # 原策略模型的 block_size
MAX_PROMPT_TOKENS = 256
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "prompt.bin")

# 现在的分词逻辑还是有问题，再想想
# 需要测试无损转换
class TokenizerWrapper:
    def __init__(self, enc, max_length):
        self.enc = enc
        self.max_length = max_length
        self.pad_token_id = 0
        self.eos_token_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.enc.decode(token_ids)

    def batch_decode(self, batch_tokens):
        return [self.decode(tokens) for tokens in batch_tokens]

    def __call__(self, texts):
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = self.enc.encode(text, allowed_special="all")
            ids = ids[:MAX_PROMPT_TOKENS]  # 严格截断
            mask = [1] * len(ids)
            pad_len = self.max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

enc = tiktoken.get_encoding("gpt2")
enc_wrapper = TokenizerWrapper(enc, max_length=BLOCK_SIZE)

def load_rl_dataset(split="train"):
    dataset = load_dataset(DATASET_NAME, split=split)
    if "text" not in dataset.column_names:
        for col in dataset.column_names:
            if dataset.features[col].dtype == "string":
                dataset = dataset.rename_column(col, "text")
                break
    return dataset

def collect_prompts(dataset, max_prompt_tokens=MAX_PROMPT_TOKENS, min_prompt_tokens=5):
    prompts = []
    for sample in dataset["text"]:
        tokens = enc.encode(sample, allowed_special="all")
        if len(tokens) < min_prompt_tokens:
            continue
        tokens = tokens[:max_prompt_tokens]  # 强制截断
        prompts.append(enc.decode(tokens))
    return prompts

if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_rl_dataset(split="train")
    print(f"Dataset size: {len(dataset)}")

    print("Collecting prompts...")
    prompts = collect_prompts(dataset, BLOCK_SIZE)

    print("Tokenizing...")
    tokenizer = TokenizerWrapper(enc, max_length=BLOCK_SIZE)
    tokenized = tokenizer(prompts)

    print(f"Saving to {OUTPUT_FILE} ...")
    torch.save(tokenized, OUTPUT_FILE)

    print(f"Saved {len(prompts)} prompts. Done.")