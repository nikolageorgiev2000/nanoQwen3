# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-small-normed-kq'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 80 # NEW: changed from 12 to 16 since it fits in 12 GB of VRAM
block_size = 512 # NEW: changed from 1024 to 512 to fit in 12 GB of VRAM
gradient_accumulation_steps = 6 # NEW: changed from 5 * 8 to 25
# RESULT: The effective batch size is 16 * 25 = 400 while before it was 12 * 5 * 8 = 480. Should be fine.

max_iters = 100_000
lr_decay_iters = 100_000

# eval stuff
eval_interval = 250
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1