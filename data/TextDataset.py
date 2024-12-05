import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data_dir, split, block_size, default_vocab_size=32768):
        """
        Dataset that loads text data in chunks of `block_size` and attempts to derive `vocab_size` from meta.pkl.

        :param data_dir: The directory where the dataset is stored
        :param split: 'train' or 'val' to select the appropriate split
        :param block_size: The size of each input/output block
        :param default_vocab_size: Default vocab size if 'meta.pkl' is not found
        """
        self.split = split
        self.block_size = block_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        self.vocab_size = default_vocab_size  # Default value if no meta.pkl is found

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta.get('vocab_size', default_vocab_size)
            print(f"Found vocab_size = {self.vocab_size} (inside {meta_path})")
        else:
            print(f"Using default vocab_size = {self.vocab_size} (no meta.pkl found)")

        # Load the data into memory with memory-mapped arrays
        data_file = 'train.bin' if self.split == 'train' else 'val.bin'
        self.data = np.memmap(os.path.join(data_dir, data_file), dtype=np.uint16, mode='r')

        # Compute the number of batches available
        self.num_batches = len(self.data) // self.block_size

    def __len__(self):
        """Returns the number of batches in the dataset."""
        return self.num_batches

    def __getitem__(self, idx):
        """
        Fetch a batch of data (x, y) for a given index `idx`.
        :param idx: Index of the batch to fetch.
        :return: A tuple (x, y) where x is the input and y is the target.
        """
        # Calculate the start index of the batch
        start = idx * self.block_size
        end = start + self.block_size

        # Get input and target sequences
        x = torch.from_numpy(self.data[start:end].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1:start + 1 + self.block_size].astype(np.int64))

        # If on CUDA, move the data to the GPU (if available)
        if self.device.type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y


"""
# Example usage:
data_dir = os.path.join('data', 'tiny_shakespeare_data')  # Replace 'dataset' with your actual dataset name
block_size = 128  # Set your block size
batch_size = 32  # Set your batch size

# Create the dataset object for training
train_dataset = TextDataset(data_dir, split='train', block_size=block_size)

# Access vocab_size from the dataset
print(f"Vocab size: {train_dataset.vocab_size}")

# Create the DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Example of iterating over the DataLoader
for x_batch, y_batch in train_dataloader:
    print(x_batch.shape, y_batch.shape)  # Each batch is of shape (batch_size, block_size)
"""
