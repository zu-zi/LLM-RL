import os
import torch
from datasets import load_dataset
import tiktoken
import random

DATASET_NAME = "Anthropic/hh-rlhf"   # 替换为实际数据集
BLOCK_SIZE = 1024      # 原策略模型的 block_size
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "prompt.bin")

# 现在的分词逻辑还是有问题，再想想
# 需要测试无损转换
class TokenizerWrapper:
    def __init__(self, enc, max_length):
        self.enc = enc
        self.max_length = max_length
        self.pad_token_id = 0
        # 新版本 tiktoken 必须允许特殊 token 才能编码
        self.eos_token_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    def decode(self, token_ids):
        # token_ids是列表或张量
        # 转成 Python list 再调用 tiktoken 解码
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.enc.decode(token_ids)  
        
    def batch_decode(self, batch_tokens):
        return [self.decode(tokens) for tokens in batch_tokens]
    
    def __call__(self, texts):
        input_ids = []
        attention_mask = []
        for text in texts:
            # allowed_special="all" 允许所有特殊 token，不然 '<|endoftext|>' 会报错
            ids = self.enc.encode(text, allowed_special="all")
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            mask = [1] * len(ids)

            while len(ids) < self.max_length:
                ids.append(self.pad_token_id)
                mask.append(0)

            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
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

def collect_prompts(dataset, block_size, min_prompt_tokens=5):
    prompts = []
    max_prompt_len = block_size // 2
    for sample in dataset["text"]:
        tokens = enc.encode(sample, allowed_special="all")
        if len(tokens) >= min_prompt_tokens:
            tokens = tokens[:max_prompt_len]
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