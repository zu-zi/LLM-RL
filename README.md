## finetue:
+ pip install torch numpy transformers datasets tiktoken wandb tqdm
+ python data/shakespeare_char/prepare.py
+ python train.py config/finetune_yourdata.py

## autodl 
```
!apt-get update
!apt-get install -y git
!git clone https://github.com/zu-zi/LLM-RL.git
!pip install torch numpy transformers datasets tiktoken wandb tqdm
cd LLM-RL
***
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
***
!python data/shakespeare_char/prepare.py
!python train.py config/finetune_shakespeare.py
```
//注：如果减小（gradient_accumulation_steps = 32；block_size =128）显存仍不够，可以换成更小的gpt2-medium/gpt2-large先验证跑通

最终在4090D上设置的可行结果：32/1024/large

## Process
### Model
+ input:idx (B, T)
+ Embedding（token + position）:(B, T, C)
+ Transformer Block：
   + LayerNorm:(B, T, C)
   + Causal Self-Attention:x -> (B, T, C);y -> (B, T, C)
   + MLP:(B, T, C);(B, T, 4*C);(B, T, C)
+ LayerNorm:(B, T, C)
+ lm_head:logits(B, T, vocab_size)
+ loss(train)/sampling(eval):(B, T+max_new_tokens)

### Superparameter
#### 现在配置下(large)，各数值：
+ B = batch_size*gradient_accumulation_steps = 1 * 32 = 32
+ T = block_size = 1024
+ C = n_embd = 1280
+ vocab_size = 50304
+ n_layer = 36
+ n_head = 20
#### 用于PPO设计:
+ log_probs:
```
probs = F.softmax(logits, dim=-1)         # (B, T, vocab_size)
dist = Categorical(probs)
log_probs = dist.log_prob(actions)        # (B, T)
```
+ old_log_probs:(B, T)
+ actions.shape == (B, T)
+ rewards:
  + sentence-level:rewards.shape == (B,) 
  + token-level:(B, T)
+ value estimate
```
x = self.transformer.ln_f(x)             # (B, T, C)
value = value_head(hidden_states)        # (B, T)
```
+ advantages:(B, T)
+ ratio:(B, T):exp(log_prob - old_log_prob)


### Train & config
```
# 根据3种情况初始化模型和优化器
model = GPT(config)
optimizer = optim.AdamW(...)

# 加载训练集
train_data = ...

for iter in range(max_iters):

    x, y = get_batch(...)
    
    logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    ...
```

### Tokenizer
```
collect_prompts()              # prompt 是字符串
    ↓ enc_wrapper(prompt)      # tiktoken → token id → Actor
Actor.generate()
    ↓ enc.decode(token_id)     # tiktoken 解码
Reward_tokenizer.encode(text)  # reward 模型自己的分词器
    ↓ Reward Model 

```

### 代做
+ 小规模测试：先用很小的模型(如1层transformer)和少量数据测试流程能否跑通；检查各环节的输入输出形状是否匹配
+ 指标监控：初始阶段应看到：奖励值缓慢上升；KL散度保持在一定范围内(建议0.5-5之间)；价值损失逐渐下降