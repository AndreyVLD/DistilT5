import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer

from .model import StudentModel, DistillationLoss


class DistillationConfig:
    def __init__(self) -> None:
        # Models
        self.student_model_name = "Salesforce/codet5-small"

        # Dataset
        self.dataset_path = "../data/val_split.jsonl"
        self.max_src_length = 1024
        self.max_trg_len = 128

        # Training
        self.train_batch_size = 8
        self.eval_batch_size = 64
        self.learning_rate = 1e-4
        self.num_train_epochs = 10
        self.warmup_steps = 50
        self.weight_decay = 0.01
        self.temperature = 2.0  # Temperature for softening probability distributions
        self.alpha = 0.7  # Weight for  task-specific loss vs distillation loss

        # Output
        self.output_dir = "../distillation_output"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainer:
    def __init__(self, config: DistillationConfig) -> None:
        """
        Initialize the DistillationTrainer with the provided configuration.
        Args:
            config (DistillationConfig): Configuration object containing training parameters.
        """
        self.config = config

        # Initialize the tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(config.student_model_name)

        # Initialize the model
        self.student_model = StudentModel(self.tokenizer, config.student_model_name).to(config.device)
        self.student_model.config.use_cache = False

        # Initialize loss function
        self.criterion = DistillationLoss(temperature=config.temperature, alpha=config.alpha)

    def train(self, train_loader: DataLoader) -> None:
        """
        Train the model using the provided DataLoader.

        Args:
            train_loader (DataLoader): DataLoader for training data.
        """
        self.student_model.train()

        total_steps = len(train_loader) * self.config.num_train_epochs

        # Initialize optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.student_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )

        # TODO: Enhance tqdm progress bar
        for epoch in range(self.config.num_train_epochs):
            epoch_loss = 0.0
            batch_losses = []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)
                teacher_logits = batch['teacher_logits'].to(self.config.device)

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Compute distillation loss
                loss = self.criterion(outputs.logits, teacher_logits, labels=labels)

                # Backward pass and optimization
                loss.backward()

                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)

                # Update parameters
                optimizer.step()
                scheduler.step()

                # Update tracking
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                epoch_loss += batch_loss

                progress_bar.set_postfix({
                    'avg_loss': sum(batch_losses) / len(batch_losses),
                })

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}, Loss: {avg_loss:.4f}")
            self.student_model.save_model(self.config.output_dir + f"/epoch_{epoch + 1}")

    def evaluate(self, eval_loader: DataLoader) -> None:
        """
        Evaluate the model using the provided DataLoader.
        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
        """
        # TODO implement evaluation logic
        pass
