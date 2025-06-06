import os
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from pipeline.dataset import MapAssertGenDataset
from pipeline.model import StudentModel, ModelType
from pipeline.train import DistillationConfig, DistillationTrainer


def set_seed(seed: int = 42):
    # 1. Python built-in RNG (if you use `random.random()`, `random.randint()`, etc.)
    random.seed(seed)

    # 2. NumPy RNG
    np.random.seed(seed)

    # 3. PyTorch CPU RNG
    torch.manual_seed(seed)

    # 4. PyTorch GPU RNG (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. Enforce deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. (Optional) Ensure hash-based operations are reproducible across runs
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate() -> None:
    model_path = Path(__file__).resolve().parents[1] / "small_distill" / "epoch_62"
    config = DistillationConfig()
    student_model = StudentModel.load_model(str(model_path), ModelType.CODET5PLUS).to(config.device)
    trainer = DistillationTrainer(config, student_model)

    validation_dataset = MapAssertGenDataset(
        tokenizer=trainer.tokenizer,
        file_path=config.eval_dataset_path,
        max_src_length=config.max_src_length,
        max_trg_length=config.max_trg_length
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    avg_loss, eval_results = trainer.evaluate(validation_loader, False)

    print(f"Average evaluation loss: {avg_loss:.6f}")
    print(f"Evaluation Results: {eval_results}")


def train() -> None:
    config = DistillationConfig()
    trainer = DistillationTrainer(config, None)

    # Load the dataset
    train_dataset = MapAssertGenDataset(
        tokenizer=trainer.tokenizer,
        file_path=config.train_dataset_path,
        max_src_length=config.max_src_length,
        max_trg_length=config.max_trg_length
    )

    validation_dataset = MapAssertGenDataset(
        tokenizer=trainer.tokenizer,
        file_path=config.eval_dataset_path,
        max_src_length=config.max_src_length,
        max_trg_length=config.max_trg_length
    )

    # Create DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    trainer.train(train_loader, validation_loader)


def main() -> None:
    # Set random seed for reproducibility
    set_seed(42)
    # Uncomment the function you want to run
    # train()
    evaluate()


if __name__ == '__main__':
    main()
