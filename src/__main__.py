from torch.utils.data import DataLoader
from pipeline.dataset import MapAssertGenDataset
from pipeline.train import DistillationConfig, DistillationTrainer


def main() -> None:
    config = DistillationConfig()
    trainer = DistillationTrainer(config)

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

    # Train the model
    trainer.train(train_loader, validation_loader)


if __name__ == '__main__':
    main()
