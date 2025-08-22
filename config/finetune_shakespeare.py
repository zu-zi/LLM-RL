import time

out_dir = 'Results'
eval_interval = 20 #up
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-medium' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
# gradient_accumulation_steps = 32
gradient_accumulation_steps = 4 # RL初期测试需要调小一点
max_iters = 200 #up

# finetune at constant LR
learning_rate = 3e-5
RL_learning_rate=1e-5
decay_lr = False

#RL
use_ppo = True