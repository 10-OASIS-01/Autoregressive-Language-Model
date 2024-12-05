import argparse

from DatasetProcessor import DatasetProcessor


def main():
    # 初始化命令行参数解析器
    parser = argparse.ArgumentParser(description="Download and process datasets for tokenization.")

    # 数据集参数
    parser.add_argument('--dataset_name', type=str, help="Hugging Face dataset name (e.g., 'Salesforce/wikitext').",
                        default=None)
    parser.add_argument('--config_name', type=str,
                        help="Configuration name for the dataset (e.g., 'wikitext-2-raw-v1').", default=None)
    parser.add_argument('--data_url', type=str,
                        help="URL for small text datasets (e.g., 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').",
                        default=None)

    # 处理参数
    parser.add_argument('--block_size', type=int, help="The block size for tokenization.", default=128)
    parser.add_argument('--num_proc', type=int, help="Number of processes to use for tokenization.", default=8)
    parser.add_argument('--num_proc_load_dataset', type=int, help="Number of processes to load dataset (Hugging Face).",
                        default=8)

    # 输出目录
    parser.add_argument('--output_dir', type=str, help="Directory to save processed data.", required=True)

    # 设备类型（默认为自动选择）
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                        help="Device to run the script on ('cpu' or 'cuda').")

    # 解析命令行参数
    args = parser.parse_args()

    # 根据命令行参数初始化 DatasetProcessor
    processor = DatasetProcessor(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        block_size=args.block_size,
        num_proc=args.num_proc,
        num_proc_load_dataset=args.num_proc_load_dataset,
        device=args.device,
        data_url=args.data_url
    )

    # 调用处理和保存数据的方法
    processor.process_and_save(args.output_dir)


if __name__ == '__main__':
    main()

""""
处理 Hugging Face 上的数据集：
python downloaddata.py \
    --dataset_name "Salesforce/wikitext" \
    --config_name "wikitext-2-raw-v1" \
    --block_size 128 \
    --num_proc 8 \
    --output_dir "processed_wikitext_data"
"""

"""
处理小型文本数据集：
python downloaddata.py \
    --data_url "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
    --block_size 128 \
    --output_dir "processed_tiny_shakespeare_data"
"""
