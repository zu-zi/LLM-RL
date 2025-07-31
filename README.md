## finetue:
+ pip install torch numpy transformers datasets tiktoken wandb tqdm
+ python data/shakespeare_char/prepare.py
+ python train.py config/finetune_yourdata.py

## 
+ from google.colab import drive
+ drive.mount('/content/drive')
+ cd /content/drive/MyDrive/LLM+RL
+ git clone https://github.com/zu-zi/LLM-RL.git
+ finetue:
+ git add .
+ git commit -m "...."
+ git push origin main

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