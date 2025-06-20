# DistilT5

A model distillation pipeline for automatically generating assertions from test methods. This project is part of a
Bachelor Thesis at Delft University of Technology.

## Overview

This project implements a knowledge distillation approach for a CodeT5-based model to generate assertions for test
methods. The pipeline includes training, evaluation, inference speed measurement components and auxiliary code for logit
decompression and plotting of results.

## Requirements

- Python 3.13.2
  The rest of the requirements are specified in the `requirements.txt` file together with their version.
- (Optional) CUDA GPU with CUDA 12.2 support for faster training and evaluation.

## Installation

To install the required packages, run:

```bash
git clone https://github.com/AndreyVLD/DistilT5
cd DistilT5
pip install -r requirements.txt
```

## Project Structure

- `src/` - Source code directory.
    - `pipeline/` - Contains the main pipeline for training and evaluation.
        - `dataset.py` - Dataset handling.
        - `model.py` - Student model and loss function implementation.
        - `train.py` - Training and evaluation logic for distillation together with the configuration class for this.
    - `plots/` - Contains the code for plotting results.
        - `metrics.py` - Runnable file for generating plots.
    - `utils/` - Contains utility functions.
        - `decompression.py` - Logit decompression utilities.
        - `evaluation.py` - Class for keeping track of evaluation metrics and teacher evaluation utilities.
        - `split_json.py` - Utilities for splitting JSON files in training and validation.
    - `__main__.py` - Entry point with training and evaluation functions.

## Usage

### Instructions

Before running the training and evaluation process, ensure:

1. You have the required datasets in the `data/` directory:
    - Training data: `DistilT5/data/distillation_data_training.jsonl`
    - Validation data: `DistilT5/data/distillation_data_validation.jsonl`
2. You can customize the distillation process by modifying fields in the `DistillationConfig` class.
3. In the evaluation functions from `__main__.py`, you can specify the path to the model you want to evaluate by
   changing `model_path` variable. This path needs to point to the directory where the model is saved. If the model
   is trained by our script then it needs to point to
   `DistilT5/distillation_output/epoch_<placeholder_for_epoch_number>` directory.

The output will be saved to `DistilT5/distillation_output` by default, but this can be changed in the configuration.

### Training

To train the model, edit the `__main__.py` file and uncomment the `train()` function call in the `main()` function:

```python
def main() -> None:
    set_seed(42)
    train()
    # evaluate()
    # evaluate_with_time()
```

Then run the script:

```bash
python -m src
```

### Evaluation

To evaluate a trained model, uncomment the `evaluate()` function call:

```python
def main() -> None:
    set_seed(42)
    # train()
    evaluate()
    # evaluate_with_time()
```

### Measuring Generation Speed

To measure the inference speed of the model, uncomment the `evaluate_with_time()` function call:

```python
def main() -> None:
    set_seed(42)
    # train()
    # evaluate()
    evaluate_with_time()
```

### Configuring Distillation

The model and training settings can be configured by modifying the `DistillationConfig` class in `pipeline/train.py`.
