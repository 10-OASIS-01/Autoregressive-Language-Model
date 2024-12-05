import os
import time
from contextlib import nullcontext
import torch
import yaml
from tqdm import tqdm
import wandb

from data.TextDataset import TextDataset, DataLoader
from model.transformer import Transformer, ModelConfig
from utils.helpers import estimate_loss, get_lr, configure_optimizers, estimate_mfu

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# 设置环境变量
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# -----------------------------------------------------------------------------
# Load config values
# -----------------------------------------------------------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# I/O settings
out_dir = config['out_dir']
model_name = config['model_name']
eval_interval, log_interval, eval_iters = config['eval_interval'], config['log_interval'], config['eval_iters']
eval_only, always_save_checkpoint = config['eval_only'], config['always_save_checkpoint']
init_from = config['init_from']

# Wandb logging
wandb_log, wandb_project, wandb_run_name = config['wandb_log'], config['wandb_project'], config['wandb_run_name']

# Data settings
dataset, gradient_accumulation_steps, batch_size, block_size, vocab_size = config['dataset'], config[
    'gradient_accumulation_steps'], config['batch_size'], config['block_size'], config['vocab_size']

# Model architecture settings
n_layer, n_head, n_embd, dropout, bias = config['n_layer'], config['n_head'], config['n_embd'], config['dropout'], \
config['bias']

# AdamW optimizer settings
learning_rate, max_iters, weight_decay, beta1, beta2, grad_clip = float(config['learning_rate']), config[
    'max_iters'], float(config['weight_decay']), config['beta1'], config['beta2'], config['grad_clip']

# Learning rate decay settings
decay_lr, warmup_iters, lr_decay_iters, min_lr = config['decay_lr'], config['warmup_iters'], config[
    'lr_decay_iters'], float(config['min_lr'])

# System settings
device, dtype, compile = config['device'], config['dtype'], config['compile']

# -----------------------------------------------------------------------------
# System setup
# -----------------------------------------------------------------------------
master_process = True
seed_offset = 0

# Various inits, derived attributes, I/O setup
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn

device_type = 'cuda' if 'cuda' in device else 'cpu'  # For later use in torch.autocast
device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
print(f"Using device: {device_type} ({device_name})")

# Data type (automatic gradient scaling for float16)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if wandb_log and master_process:
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

# Create the dataset object for training
train_dataset = TextDataset(data_dir, split='train', block_size=block_size)
val_dataset = TextDataset(data_dir, split='val', block_size=block_size)

# Create the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout
)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = train_dataset.vocab_size if train_dataset.vocab_size is not None else 32768
    gptconf = ModelConfig(**model_args)
    model = Transformer(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'tiny_shakespeare.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = ModelConfig(**model_args)
    model = Transformer(gptconf)
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# -----------------------------------------------------------------------------
# Optimizer setup
# -----------------------------------------------------------------------------
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # Free up memory

# Compile the model if needed
if compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # Requires PyTorch 2.0

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

X, Y = next(iter(train_dataloader))  # Fetch the very first batch
t0 = time.time()
local_iter_num = 0  # Number of iterations in the lifetime of this process
raw_model = model  # Single GPU
running_mfu = -1.0

# Create tqdm progress bar for the entire training loop
with tqdm(total=max_iters, desc="Training Progress", unit="iter") as pbar:
    while iter_num < max_iters:
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, train_dataloader, val_dataloader, eval_iters)
            pbar.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            pbar.set_postfix({
                "train_loss": losses['train'],
                "val_loss": losses['val'],
                "lr": lr,
                "mfu": f"{running_mfu * 100:.2f}%"
            })
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # Convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    pbar.write(f"Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, f'{model_name}.pt'))

        # Exit after the first eval if eval_only is True
        if iter_num == 0 and eval_only:
            break

        # Training step with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # Scale the loss for gradient accumulation
            X, Y = next(iter(train_dataloader))
            scaler.scale(loss).backward()
            pbar.set_postfix({"loss": loss.item()})

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Log training progress
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = estimate_mfu(raw_model, batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            pbar.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        pbar.update(1)
        iter_num += 1
        local_iter_num += 1
