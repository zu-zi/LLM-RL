"""
Quick test: verify SGLang offline engine with GPT2 small
- 不依赖 train.py / RL 部分
- 仅测试加载 & 单次生成
"""

import torch

def main():
    try:
        import sglang as sgl
    except ImportError:
        print("SGLang 未安装，请先: pip install sglang")
        return

    # 用 GPT-2 small 测试即可，避免太大
    from model import GPTConfig, GPT

    block_size = 128
    vocab_size = 50304
    cfg = GPTConfig(
        n_layer=2, n_head=2, n_embd=128,
        block_size=block_size, vocab_size=vocab_size
    )
    model = GPT(cfg).to("cuda")

    try:
        # 初始化引擎（offline 模式）
        engine = sgl.Engine(model=model, mode="offline")
        print("✅ SGLang engine loaded (offline mode).")

        # 构造一个 prompt: "Hello"
        from tiktoken import get_encoding
        enc = get_encoding("gpt2")
        prompt_ids = enc.encode("Hello")
        ids_t = torch.tensor(prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)

        # 生成
        out = engine.generate(ids_t, max_new_tokens=10, eos_token_id=enc.eot_token)
        print("Generated ids:", out)
        print("Decoded:", enc.decode(out[0].tolist()))

    except Exception as e:
        print(f"SGLang 引擎测试失败: {e}")


if __name__ == "__main__":
    main()
