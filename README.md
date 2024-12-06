# Autoregressive Language Model

This project provides a **hands-on implementation** of an autoregressive Transformer-based language model, designed with **beginners in mind**. It offers a comprehensive pipeline that covers data processing, model training, evaluation, and inference. What distinguishes this project is that it starts from the **ground up**, beginning with **manual tokenization**. This approach provides full transparency into the entire process of building a language model, offering users complete control and insight into each step of the workflow.

Inspired by the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy, this implementation has been **reworked** to be more accessible and easier to understand, especially for those new to deep learning and natural language processing (NLP). Several parts of the code, which can be difficult to grasp in nanoGPT, have been **completely rewritten** to improve clarity. Key improvements include:

- **Hands-on Data Processing**: The data loading and processing pipeline has been extended to support a wider variety of datasets, with improved control over preprocessing steps.
- **Manual Tokenizer Implementation**: The tokenizer has been built from scratch, providing insight into the tokenization process itself. This avoids the need for pre-trained tokenizers and allows for a deeper understanding of how text is converted into tokens.
- **Removed Multi-GPU DDP Training**: To make the project more approachable for beginners, the complex multi-GPU Distributed Data Parallel (DDP) training setup has been removed. The project now focuses on single-GPU or CPU training, making it easier to get started without the need for specialized hardware.

By implementing a **custom-built Autoregressive Language Model** based on principles similar to those in GPT models, this project offers an excellent opportunity for **Deep Learning** and **NLP** beginners to learn how core components of a Transformer model are constructed, trained, and fine-tuned. The project is designed to be **beginner-friendly**, featuring clear documentation, flexible configurations, and simple-to-follow steps for training and evaluating your own models.

Whether you're new to deep learning or exploring the mechanics behind language models, this project serves as a practical entry point, helping you understand the intricacies of tokenization, model architecture, and training without relying on high-level abstractions.

## Key Features:
- **Manual Tokenization**: Tokenization is implemented from scratch using regular expressions and Byte Pair Encoding (BPE), following the approach used by GPT models.
- **Beginner-Friendly**: Designed to be approachable for newcomers to deep learning and NLP, with extensive documentation, flexible configurations, and clear, simple scripts.
- **Customizable and Transparent**: Every component of the pipeline, from data processing to model training, is fully customizable, allowing you to experiment and adjust the system to your needs.
## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
  - [Data Loading and Processing](#data-loading-and-processing)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Weights & Biases (Wandb) Integration](#weights--biases-wandb-integration)
- [Requirements](#requirements)
- [License](#license)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/10-OASIS-01/Autoregressive-Language-Model.git
   cd Autoregressive-Language-Model
   ```

2. **Set up a Conda environment**:

   ```bash
   conda create --name your_env_name python=3.6  # or a higher Python version as needed
   conda activate your_env_name
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
   
## Project Structure

```
autoregressive_language_model/
├── data/
│   ├── __init__.py
│   ├── TextDataset.py
│   ├── downloaddata.py
│   ├── DatasetProcessor.py
│   ├── processed_wikitext-103-raw-v1/
│   ├── processed_wikitext-2-raw-v1/
│   └── tiny_shakespeare_data/
├── model/
│   ├── __init__.py
│   ├── model_unit_tests.py
│   └── transformer.py
├── output_directory/
├── tokenizer/
│   ├── __init__.py
│   ├── regex.py
│   ├── wikitext_tokenizer.model
│   └── wikitext_tokenizer.vocab
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── sampling.py
├── config.yaml
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
└── inference.py
```

## File Descriptions

The code for **evaluation**, **inference**, and **sampling** was developed and refined in my other foundational project [Character-Level Language Modeling](https://github.com/10-OASIS-01/Character_Level_Language_Modeling/blob/main/README.md). The implementation in that project served as the base for the corresponding functionalities in this project, including the generation of text samples and model evaluation procedures.

### Root Directory

- `config.yaml`: Configuration file for training and evaluation settings.
- `train.py`: Script to train the Transformer model.
- `evaluate.py`: Script to evaluate the trained model.
- `inference.py`: Script to generate text using the trained model.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.

### `data` Directory

- `__init__.py`: Initialization file for the data module.
- `TextDataset.py`: Defines the `TextDataset` class for loading and processing text data in chunks of a specified block size. It supports memory-mapped arrays for efficient data loading.
- `downloaddata.py`: Script to download and process datasets for tokenization. It supports downloading datasets from the Hugging Face Datasets Hub or from a direct URL.
- `DatasetProcessor.py`: Contains the `DatasetProcessor` class for handling dataset loading, processing, and tokenization. It supports both large datasets from Hugging Face and small text datasets from URLs, and saves the processed data in binary format for efficient loading during training.

### `model` Directory

- `model_unit_tests.py`: Unit tests for the Transformer model, verifying various aspects such as forward pass, loss computation, sequence length handling, parameter initialization, and gradient computation.
- `transformer.py`: Implementation of the Transformer model, including the model architecture, forward pass, and weight initialization.

### `tokenizer` Directory

The tokenizer used in this project is a custom-trained BPE (Byte Pair Encoding) Tokenizer, similar to the GPT-4 Tokenizer. It supports tokenization using customizable regular expression patterns, including GPT-4 regex patterns. The training code for this tokenizer can be found in another open-source project: [BPEtokenizer](https://github.com/10-OASIS-01/BPEtokenizer).

- `regex.py`: Implementation of a regex-based tokenizer. This file contains the `RegexTokenizer` class, which handles tokenization using regular expressions. It supports special tokens and can encode and decode text. The tokenizer uses Byte Pair Encoding (BPE) and can load a pre-trained tokenizer model from a file. The file also includes utility functions for loading the tokenizer and managing token statistics.
- `wikitext_tokenizer.model`: Pre-trained tokenizer model.
- `wikitext_tokenizer.vocab`: Vocabulary file for the tokenizer.

### `utils` Directory

- `helpers.py`: Utility functions for training and evaluation, including optimizer configuration, loss estimation, learning rate scheduling, and evaluation metrics such as BLEU, ROUGE, and perplexity.
- `sampling.py`: Functions for generating text samples from the trained model, including text generation with temperature and top-k sampling, and printing generated samples.

## Usage

### Data Loading and Processing

To download and process datasets for tokenization, you can run `data/downloaddata.py`. This script supports downloading datasets from the Hugging Face Datasets Hub or from a direct URL.

For example, to process the `wikitext-2-raw-v1` dataset from Hugging Face:
```sh
python data/downloaddata.py \
  --dataset_name "Salesforce/wikitext" \
  --config_name "wikitext-2-raw-v1" \
  --block_size 128 \
  --num_proc 8 \
  --output_dir "processed_wikitext_data"
```

Or to process a small text dataset from a URL:
```sh
python data/downloaddata.py \
  --data_url "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" \
  --block_size 128 \
  --output_dir "processed_tiny_shakespeare_data"
```

This command will:
- Download the specified dataset or text file.
- Tokenize the text data using a pre-trained tokenizer.
- Split the data into training and validation sets.
- Save the processed data in binary format for efficient loading during training.

### **Training**

To train the model, follow these steps:

1. **Configure the `config.yaml` file**:
   - Set the `out_dir` to specify the output directory for saving model checkpoints.
   - Adjust `eval_interval`, `log_interval`, and `eval_iters` for evaluation and logging frequency.
   - Set `init_from` to "scratch" to start training from scratch or "resume" to continue from a checkpoint.
   - Enable or disable Weights & Biases (Wandb) logging by setting `wandb_log` to `true` or `false`.
   - Specify the dataset name, batch size, block size, and other data settings.
   - Configure model architecture settings such as the number of layers (`n_layer`), number of heads (`n_head`), embedding size (`n_embd`), and dropout rate (`dropout`).
   - Set optimizer settings including learning rate, weight decay, and gradient clipping.
   - Adjust learning rate decay settings if needed.
   - Specify the device (`cuda` or `cpu`) and data type (`float32`, `bfloat16`, or `float16`).

2. **Run the training script**:
   ```sh
   python train.py --dataset <dataset_name>
   ```

   Replace `<dataset_name>` with the name of the dataset you want to use, either `wikitext-103-raw-v1` or `tiny_shakespeare_data`.
   
### Inference

To generate text using the trained model, run:
```sh
python inference.py \
    --model_path "output_directory/tiny_shakespeare.pt" \
    --tokenizer_path "tokenizer/wikitext_tokenizer.model" \
    --device "cuda" \
    --num_chars 256 \
    --top_k 40 \
    --start_string "ROMEO: 
```

### Evaluation

To evaluate the trained model, use the following command:
```sh
python evaluate.py \
  --input_file tiny_shakespeare_data \
  --model_path output_directory/tiny_shakespeare.pt \
  --tokenizer_path "tokenizer/wikitext_tokenizer.model" \
  --device "cuda"
```

#### Evaluation Output

The evaluation script calculates and displays the following metrics:
- **BLEU Score**: Measures similarity between generated and target text based on overlapping n-grams.
- **ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)**: Measures text overlap by evaluating precision, recall, and F1 scores for unigrams, bigrams, and longest common subsequences.
- **Perplexity (PPL)**: Evaluates the language model’s performance by analyzing token likelihood.

### Weights & Biases (Wandb) Integration

This project integrates with **Weights & Biases** (Wandb) for visualizing the training process and tracking experiments. To enable Wandb logging, follow these steps:

1. **Install the Wandb package**:
   
   If you haven't installed Wandb, do so by running:
   ```sh
   pip install wandb
   ```
   
   Before running training, log in to Wandb by executing:
   ```sh
   wandb login
   ```

3. **Configure Wandb in your training script**:
   
   In the `train.py` script, Wandb integration is already configured. It logs key metrics like loss, accuracy, and others during the training process. When running the training, Wandb will automatically track the experiment and create an interactive dashboard.

4. **Tracking training with Wandb**:
   
   After training, you can view your experiment on the Wandb dashboard:
   - Go to [Wandb Dashboard](https://wandb.ai) and find your project.
   - Explore real-time training curves, metrics, and logs.

You can control the Wandb logging behavior by modifying the settings in the `config.yaml` file. The key options to adjust are:
   - **wandb_project**: Your Wandb project name.
   - **wandb_run_name**: The name for the specific run.
   - **wandb_log**: Set to `True` to enable logging, or `False` to disable.

This integration allows you to compare multiple runs, track hyperparameter changes, and share visual results across team members.

## Limitations

While the current implementation of the tokenizer in this project supports special tokens (e.g., `<|endoftext|>`, `<|fim_prefix|>`, etc.), it does not yet incorporate them into the pre-training data pipeline. As a result, special tokens are handled correctly during inference and evaluation, but they are not part of the training data by default. In future updates, the training data processing pipeline should be adjusted to include appropriate handling of special tokens, ensuring they are properly integrated during training. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
