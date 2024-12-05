# Autoregressive Language Model

This project is a comprehensive implementation of a Transformer-based language model, including data processing, training, evaluation, and inference functionalities.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Requirements](#requirements)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required packages:
    ```sh
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
│   └── tiny_shakespeare.pt
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

### Root Directory

- `config.yaml`: Configuration file for training and evaluation settings.
- `train.py`: Script to train the Transformer model.
- `evaluate.py`: Script to evaluate the trained model.
- `inference.py`: Script to generate text using the trained model.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.

### `data` Directory

- `__init__.py`: Initialization file for the data module.
- `TextDataset.py`: Defines the `TextDataset` class for loading and processing text data.
- `downloaddata.py`: Script to download and process datasets for tokenization.
- `DatasetProcessor.py`: Contains the `DatasetProcessor` class for handling dataset loading, processing, and tokenization.

### `model` Directory

- `model_unit_tests.py`: Unit tests for the Transformer model.
- `transformer.py`: Implementation of the Transformer model.
- `__init__.py`: Initialization file for the model module.

### `output_directory` Directory

- `tiny_shakespeare.pt`: Example of a trained model checkpoint.

### `tokenizer` Directory

- `regex.py`: Implementation of a regex-based tokenizer.
- `wikitext_tokenizer.model`: Pre-trained tokenizer model.
- `wikitext_tokenizer.vocab`: Vocabulary file for the tokenizer.
- `__init__.py`: Initialization file for the tokenizer module.

### `utils` Directory

- `helpers.py`: Utility functions for training and evaluation.
- `sampling.py`: Functions for generating text samples.
- `__init__.py`: Initialization file for the utils module.

## Usage

### Training

To train the model, run the following command:
```sh
python train.py
```
Make sure to configure the `config.yaml` file according to your requirements before running the training script.

### Evaluation

To evaluate the trained model, use the following command:
```sh
python evaluate.py --input_file <input_file> --model_path <model_path> --tokenizer_path <tokenizer_path> --device <device>
```

### Inference

To generate text using the trained model, run:
```sh
python inference.py --model_path <model_path> --tokenizer_path <tokenizer_path> --device <device> --num_chars <num_chars> --top_k <top_k> --start_string <start_string>
```

This README provides an overview of the project, its structure, and usage instructions. Make sure to update the `<repository_url>` and `<repository_directory>` placeholders with the actual values.