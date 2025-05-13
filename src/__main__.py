from torch.utils.data import DataLoader
from pipeline.dataset import MapAssertGenDataset
from pipeline.train import DistillationConfig, DistillationTrainer


def main() -> None:
    config = DistillationConfig()
    trainer = DistillationTrainer(config)

    # Load the dataset
    train_dataset = MapAssertGenDataset(
        tokenizer=trainer.tokenizer,
        file_path=config.dataset_path,
        max_src_length=64,
        max_trg_len=config.max_trg_len
    )

    print("Decoded input")
    print(trainer.tokenizer.decode(train_dataset[0]['input_ids'], skip_special_tokens=False))
    print("Decoded ground labels")
    print(
        trainer.tokenizer.decode(train_dataset[0]['labels'][train_dataset[0]['labels'] > 0], skip_special_tokens=False))
    print("Decoded teacher labels")
    print(trainer.tokenizer.decode(train_dataset[0]['teacher_labels'][train_dataset[0]['teacher_labels'] > 0],
                                   skip_special_tokens=False))
    print("Decoded ground truth")
    print(train_dataset[0]['ground_truth'])
    print("teacher logits")
    print(train_dataset[0]['teacher_logits'].shape)

    # Create DataLoader
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=config.train_batch_size,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    # )
    #
    # # Train the model
    # trainer.train(train_loader)
    #
    # tokenizer = trainer.tokenizer
    # model = trainer.student_model
    #
    # for sample in train_dataset:
    #     tokenized_input = tokenizer(sample['original_text'], return_tensors="pt", padding=True, truncation=True,
    #                                 max_length=config.max_src_length)
    #     original_target = sample['ground_truth']
    #
    #     input_ids = tokenized_input['input_ids']
    #     attention_mask = tokenized_input['attention_mask']
    #
    #     generated_ids = model.model.generate(input_ids=input_ids, attention_mask=attention_mask,
    #                                          max_length=512, num_beams=4,
    #                                          early_stopping=True)
    #     generated_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    #
    #     with open("output.txt", "a", encoding="utf-8") as output_file:
    #         output_file.write(f"Input: {sample['original_text']}\n")
    #         output_file.write(f"Original target: {original_target}\n")
    #         output_file.write(f"Generated output: \n {generated_output}\n")
    #         output_file.write("------\n")
    #
    #     print("------")
    #
    #     print("Original target: ", original_target)
    #     print("---")
    #     print("Generated output: ", generated_output)
    #     print("------")


if __name__ == '__main__':
    main()
