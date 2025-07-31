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
!apt-get update
!apt-get install -y git
!git clone https://github.com/zu-zi/LLM-RL.git
!pip install torch numpy transformers datasets tiktoken wandb tqdm
cd LLM-RL
!python data/shakespeare_char/prepare.py
!python train.py config/finetune_shakespeare.py