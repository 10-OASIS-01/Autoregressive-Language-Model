import math
import os
import sys
import numpy as np
import requests
import tiktoken
import torch
from datasets import Features, Sequence, Value
from datasets import load_dataset
from tqdm import tqdm
# Append the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer.regex import load_tokenizer

class DatasetProcessor:
    def __init__(self, dataset_name=None, config_name=None, block_size=128, num_proc=8, num_proc_load_dataset=8,
                 device=None,
                 data_url=None):
        """
        初始化数据集处理器，支持多种数据集，包括 Salesforce/wikitext。
        :param dataset_name: 数据集名称 (Salesforce/wikitext)
        :param config_name: 配置名称 (例如 "wikitext-2-raw-v1")
        :param block_size: 每个样本的大小（用于训练时的批处理）
        :param num_proc: 用于数据处理的进程数
        :param num_proc_load_dataset: 用于加载数据集的进程数
        :param device: 设备类型，'cuda' 或 'cpu'
        :param data_url: 用于下载 .txt 数据集的 URL（如果提供）
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.block_size = block_size
        self.num_proc = num_proc
        self.num_proc_load_dataset = num_proc_load_dataset
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = load_tokenizer("../tokenizer/wikitext_tokenizer.model")  # 使用 wikitext_tokenizer 编码器
        self.data_url = data_url

        if self.dataset_name:
            # 加载 HuggingFace 数据集（例如 Salesforce/wikitext）
            self.dataset = load_dataset(self.dataset_name, self.config_name, num_proc=self.num_proc_load_dataset)
            self.split_dataset = self.dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
            self.split_dataset['val'] = self.split_dataset.pop('test')  # 重命名验证集为 val
        elif self.data_url:
            # 对于给定的 URL 处理文本数据集
            self.split_dataset = None  # 仅适用于文本数据集，后续会单独处理
        else:
            # 默认 Tiny Shakespeare 数据集处理
            self.split_dataset = None

        # 初始化总token计数
        self.total_tokens = 0

    def process_example(self, example):
        """
        对每个样本进行分词处理。
        :param example: 单个数据样本
        :return: 处理后的 token IDs 和长度
        """
        ids = self.tokenizer.encode_ordinary(example['text'])  # wikitext_tokenizer 编码
        # ids.append(self.tokenizer.eot_token())  # 添加文本结束符
        return {'ids': ids, 'len': len(ids)}

    def tokenize_largedata_via_huggingface(self):
        """对大型数据集进行分词处理并保存。"""
        if self.split_dataset:
            features = Features({
                'ids': Sequence(feature=Value(dtype='int64'), length=-1),
                'len': Value(dtype='int64')
            })
            tokenized = self.split_dataset.map(
                self.process_example,
                remove_columns=['text'],  # 移除原始文本列
                desc="Tokenizing the splits",
                num_proc=self.num_proc,
                features=features  # 明确指定特征
            )

            # 计算总token数
            for split in tokenized.keys():
                split_total_tokens = np.sum(tokenized[split]['len'])
                print(f"{split} 集共有 {split_total_tokens:,} 个 tokens")
                self.total_tokens += split_total_tokens

            return tokenized
        else:
            print("No dataset loaded.")
            return None

    def process_smalldata_via_url(self, output_dir):
        """
        下载并处理小型文本数据集（如果未下载），然后分词并保存为二进制文件。
        :param output_dir: 输出目录
        """
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 下载数据集（如果不存在）
        input_file_path = os.path.join(output_dir, 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = self.data_url or 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            print(f"Downloading dataset from {data_url}...")
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

        # 读取数据并拆分为训练集和验证集
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]

        # 对文本进行分词
        train_ids = self.tokenizer.encode_ordinary(train_data)
        val_ids = self.tokenizer.encode_ordinary(val_data)

        print(f"train 集共有 {len(train_ids):,} 个 tokens")
        print(f"val 集共有 {len(val_ids):,} 个 tokens")

        # 更新总token数
        self.total_tokens = len(train_ids) + len(val_ids)
        print(f"总共处理了 {self.total_tokens:,} 个 tokens")

        # 保存为二进制文件
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)

        train_file = os.path.join(output_dir, 'train.bin')
        val_file = os.path.join(output_dir, 'val.bin')

        train_ids.tofile(train_file)
        val_ids.tofile(val_file)

        print(f"数据已保存到 {output_dir}")

    def save_to_binary(self, tokenized_data, output_dir):
        """
        将分词后的数据保存为二进制文件（.bin）。
        :param tokenized_data: 分词后的数据
        :param output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        for split, dset in tokenized_data.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(output_dir, f'{split}.bin')
            dtype = np.uint16  # GPT2 最大 token 为 50256，足以存储在 uint16 中
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            idx = 0
            desired_batch_size = 1000  # 根据需要调整批次大小
            total_batches = math.ceil(len(dset) / desired_batch_size)

            for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
                start = batch_idx * desired_batch_size
                end = min(start + desired_batch_size, len(dset))
                batch = dset.select(range(start, end)).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx: idx + len(arr_batch)] = arr_batch  # 写入内存映射文件
                idx += len(arr_batch)

            arr.flush()  # 确保所有数据写入文件

    def process_and_save(self, output_dir):
        """整体处理函数：分词并保存数据，并打印总token数。"""
        if self.dataset_name:
            # 处理大型数据集
            tokenized_data = self.tokenize_largedata_via_huggingface()
            self.save_to_binary(tokenized_data, output_dir)
            print(f"总共处理了 {self.total_tokens:,} 个 tokens")
        elif self.data_url:
            # 处理小型数据集
            self.process_smalldata_via_url(output_dir)
        else:
            print("No valid dataset provided. Please specify a dataset or URL.")


"""
processed_wikitext-103-raw-v1
train 集共有 121,016,060 个 tokens
val 集共有 61,744 个 tokens
"""

"""
processed_wikitext-2-raw-v1
train 集共有 2,389,901 个 tokens
val 集共有 1,380 个 tokens
"""
