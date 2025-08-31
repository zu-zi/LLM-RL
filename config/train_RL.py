import time

out_dir = 'Results'        
eval_interval = 20  
eval_iters = 40 
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'      # 不影响
init_from = 'gpt2-large'

always_save_checkpoint = False

batch_size = 4               # 根据显存支持来调
gradient_accumulation_steps = 4  # RL逻辑没用这个
max_iters = 200

learning_rate = 3e-5         # RL不用
RL_learning_rate = 1e-5      # RL用
decay_lr = False

# RL
use_grpo = False          
group_size = 4            
block_size = 512            
use_ppo = True
use_dapo = False
use_token_entropy = False
