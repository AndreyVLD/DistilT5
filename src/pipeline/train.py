from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model import StudentModel


def train(model: StudentModel, train_loader: DataLoader, optimizer: Optimizer, criterion: nn.Module, device: any,
          num_epochs: int = 10) -> None:
    """
    Train the model using the provided DataLoader.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (nn.Module): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (str): Device to use for training ('cpu' or 'cuda').
        num_epochs (int): Number of epochs to train for.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, teacher_logits, labels=labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
