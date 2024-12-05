# utils/helpers.py
import inspect
import math

import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from utils.sampling import generate


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter output_directory those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def estimate_mfu(model, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = model.get_num_params()
    cfg = model.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0 / dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, eval_iters):
    """Estimate the loss over multiple batches for both training and validation."""
    out = {}

    model.eval()  # Set the model to evaluation mode

    # Evaluate loss on the training split
    train_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = next(iter(train_dataloader))  # Fetch the next batch from the train_dataloader
        logits, loss = model(X, Y)  # Forward pass
        train_losses[k] = loss.item()  # Store the loss value

    # Evaluate loss on the validation split
    val_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = next(iter(val_dataloader))  # Fetch the next batch from the val_dataloader
        logits, loss = model(X, Y)  # Forward pass
        val_losses[k] = loss.item()  # Store the loss value

    # Store the average losses for train and val
    out['train'] = train_losses.mean()
    out['val'] = val_losses.mean()

    model.train()  # Set the model back to training mode

    return out


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Learning rate decay scheduler (cosine with warmup)."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def calculate_ppl(model, dataloader, device):
    """
    Calculate perplexity (PPL) on a dataset.

    Args:
        model (torch.nn.Module): Trained language model.
        device (str): Device for computation (e.g., "cuda" or "cpu").


    Returns:
        float: The perplexity of the model on the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            total_loss += loss.item() * X.size(0)
            total_tokens += X.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_bleu_rouge(model, tokenizer, device, start_string, target_text, num_tokens=500, temperature=1.0,
                        top_k=None):
    """
    Generate text and calculate BLEU and ROUGE scores against the target text.

    Args:
        model: The language model.
        device: Device for model computation.
        start_string: The starting text for generation.
        target_text: The reference text to compare against.
        num_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.

    Returns:
        dict: A dictionary with BLEU and ROUGE scores.
    """
    idx = torch.tensor(tokenizer.encode(start_string)).to(device)

    generated_indices = generate(
        model=model,
        idx=idx,
        max_new_tokens=num_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k
    )
    generated_text = tokenizer.decode(generated_indices.tolist())

    # Tokenize the generated and target text
    generated_tokens = nltk.word_tokenize(generated_text)
    target_tokens = nltk.word_tokenize(target_text)

    # Calculate BLEU score
    bleu_score = sentence_bleu([target_tokens], generated_tokens, weights=(0.5, 0.5),
                               smoothing_function=SmoothingFunction().method1)

    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(' '.join(target_tokens), ' '.join(generated_tokens))

    scores = {
        'BLEU': bleu_score,
        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
        'ROUGE-L': rouge_scores['rougeL'].fmeasure
    }

    return scores, generated_text
