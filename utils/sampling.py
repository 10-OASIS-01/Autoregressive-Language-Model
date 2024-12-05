# utils/sampling.py

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, idx, max_new_tokens, device, temperature=1.0, do_sample=True, top_k=None):
    """
    Generate text given a model and a starting sequence of indices (idx).

    Args:
        model: The language model used for generation.
        idx: A tensor of shape (T,) containing the starting indices.
        max_new_tokens: Number of tokens to generate.
        device: The device to run the model on.
        temperature: Sampling temperature.
        do_sample: If True, sample probabilistically; if False, use argmax.
        top_k: If provided, restrict sampling to top_k tokens.

    Returns:
        A tensor containing the generated indices.
    """
    block_size = model.get_block_size()
    model.eval()
    idx = idx.to(device)
    generated = idx.clone()
    for _ in range(max_new_tokens):
        idx_cond = generated[-block_size:]
        logits, _ = model(idx_cond.unsqueeze(0))
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        generated = torch.cat((generated, idx_next.squeeze(-1)))
    return generated


def print_samples(model, tokenizer, args, num=500, start_string="ROMEO: "):
    """
    Generate and print samples from the model.

    Args:
        model: The trained language model.
        args: Arguments object containing settings like 'device' and 'top_k'.
        num: Number of characters to generate.
    """
    model.eval()
    idx = torch.tensor(tokenizer.encode(start_string), dtype=torch.long)
    idx = idx.to(args.device)
    generated = generate(
        model=model,
        idx=idx,
        max_new_tokens=num,
        device=args.device,
        temperature=1.0,
        top_k=args.top_k
    )
    generated_text = tokenizer.decode(generated.tolist())
    print('-' * 80)
    print(generated_text)
    print('-' * 80)
