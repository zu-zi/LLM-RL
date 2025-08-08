import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_reward_model():
    device = "cuda:0"
    model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"

    # 加载tokenizer和模型（不指定device_map）
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 用半精度节省显存
        # num_labels=1,
    )
    model.to(device)  # 手动搬到GPU
    model.eval()

    candidates = [
        "我喜欢这款产品，质量非常好，值得购买！",
        "这款产品一般般，质量还可以，不是特别满意。",
        "产品非常差，使用后很失望，建议不要买。"
    ]

    inputs = tokenizer(candidates, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits.squeeze(-1)

    for text, score in zip(candidates, scores):
        print(f"文本: {text}\n得分: {score.item():.4f}\n")

if __name__ == "__main__":
    test_reward_model()
