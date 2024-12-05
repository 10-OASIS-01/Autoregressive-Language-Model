import os
import sys
import torch
import argparse

# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.transformer import Transformer, ModelConfig
from utils.sampling import print_samples
from tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate Samples from Trained Model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the pre-trained tokenizer.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on.')
    parser.add_argument('--num_chars', type=int, default=500, help='Number of characters to generate.')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling parameter.')
    parser.add_argument('--start_string', type=str, default="", help='String to prime the generation.')

    args = parser.parse_args()

    # Load dataset to get vocab and mappings
    tokenizer = load_tokenizer(args.tokenizer_path)

    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=True)
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

    # Generate and print samples with the start string
    print_samples(model, tokenizer, args, num=args.num_chars, start_string=args.start_string)


if __name__ == '__main__':
    main()

"""
python inference.py ^
    --model_path "output_directory/tiny_shakespeare.pt" ^
    --tokenizer_path "tokenizer/wikitext_tokenizer.model" ^
    --device "cuda" ^
    --num_chars 256 ^
    --top_k 40 ^
    --start_string "ROMEO: " 
"""
