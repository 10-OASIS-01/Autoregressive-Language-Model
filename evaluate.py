import json
import os
import sys
import nltk
import torch

# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.transformer import Transformer, ModelConfig
from data.TextDataset import TextDataset, DataLoader
from utils.helpers import calculate_ppl, evaluate_bleu_rouge
from tokenizer import load_tokenizer

nltk.download('punkt')
nltk.download("punkt_tab")


def main(model_path, input_file, tokenizer_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Main function to evaluate the trained model.

    Args:
        model_path: Path to the saved model checkpoint.
        input_file: Path to the dataset file.
        device: Device to use for computation.
    """
    block_size = 128
    batch_size = 16

    # Load dataset
    tokenizer = load_tokenizer(tokenizer_path)
    data_dir = os.path.join('data', f"{input_file}")

    # Create the dataset object for training
    val_dataset = TextDataset(data_dir, split='val', block_size=block_size)
    # Create the DataLoader
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size
    print(f"Loaded dataset with vocab_size={vocab_size}")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    print("Checkpoint keys:", checkpoint.keys())  # Print the keys of the checkpoint dictionary

    gptconf = ModelConfig(**checkpoint['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    model.eval()
    model.to(args.device)
    print("Model loaded successfully.")

    # Load evaluation data
    eval_file = f"data/{input_file}/evaluate_string_{input_file}.json"
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)

    total_bleu = 0
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_L = 0
    total_ppl = 0  # Initialize total perplexity
    num_examples = len(eval_data)

    for example in eval_data:
        start_string = example['start_string']
        target_text = example['target_text']
        target_tokens = target_text.split()
        num_tokens = len(target_tokens)

        scores, generated_text = evaluate_bleu_rouge(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_string=start_string,
            target_text=target_text,
            num_tokens=num_tokens,
            temperature=1.0,
            top_k=None
        )

        # Calculate Perplexity for this example
        example_ppl = calculate_ppl(model, dataloader=val_dataloader, device=device)
        total_ppl += example_ppl
        total_bleu += scores['BLEU']
        total_rouge_1 += scores['ROUGE-1']
        total_rouge_2 += scores['ROUGE-2']
        total_rouge_L += scores['ROUGE-L']

    # Calculate average scores
    avg_bleu = total_bleu / num_examples
    avg_rouge_1 = total_rouge_1 / num_examples
    avg_rouge_2 = total_rouge_2 / num_examples
    avg_rouge_L = total_rouge_L / num_examples
    avg_ppl = total_ppl / num_examples

    print("\nAverage Evaluation Scores:")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-1 F1: {avg_rouge_1:.4f}")
    print(f"Average ROUGE-2 F1: {avg_rouge_2:.4f}")
    print(f"Average ROUGE-L F1: {avg_rouge_L:.4f}")
    print(f"Average Perplexity (PPL): {avg_ppl:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Trained Language Model")
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the input text file for dataset creation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model checkpoint.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the pre-trained tokenizer.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the evaluation on.')
    args = parser.parse_args()

    main(args.model_path, args.input_file, args.tokenizer_path, args.device)

"""
python evaluate.py ^
  --input_file tiny_shakespeare_data ^
  --model_path output_directory/tiny_shakespeare.pt ^
  --tokenizer_path "tokenizer/wikitext_tokenizer.model" ^
  --device "cuda"
"""
