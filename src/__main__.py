import torch

from transformers import AutoTokenizer
from torch.optim import AdamW

from pipeline.model import StudentModel
from pipeline.model import DistillationLoss
from pipeline.train import train
from pipeline.dataset import TestGenDataset


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
    #
    # # Load the dataset
    # dataset = MockTestGenDataset(tokenizer)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    #
    # # Initialize the model
    # student = StudentModel(tokenizer).to(device)
    # student.config.use_cache = False
    #
    # # Initialize the optimizer and loss function
    # criterion = DistillationLoss()
    # optimizer = AdamW(student.parameters(), lr=1e-4)
    #
    # # Train the model
    # train(student, dataloader, optimizer, criterion, device, num_epochs=5)
    dataset = TestGenDataset(tokenizer)
    sample = next(iter(dataset))
    print(sample)
    print(sample["teacher_logits"].shape)
    print(tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
    print(tokenizer.decode(list(filter(lambda x: x != -100.0, sample['labels'].tolist())), skip_special_tokens=True))


if __name__ == '__main__':
    main()
