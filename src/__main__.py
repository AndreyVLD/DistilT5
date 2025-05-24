from pathlib import Path

from torch.utils.data import DataLoader
from pipeline.dataset import MapAssertGenDataset
from pipeline.model import StudentModel, ModelType
from pipeline.train import DistillationConfig, DistillationTrainer


def evaluate() -> None:
    model_path = Path(__file__).parents[1] / "distillation_output" / "epoch_4"
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
        num_workers=config.num_workers,
        pin_memory=True,
    )

    avg_loss, eval_results = trainer.evaluate(validation_loader, False)

    print(f"Average evaluation: {avg_loss}")
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
    # Uncomment the function you want to run
    # train()
    evaluate()


if __name__ == '__main__':
    main()
