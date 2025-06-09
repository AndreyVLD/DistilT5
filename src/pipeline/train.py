import random
import time

import numpy as np
import torch

from pathlib import Path
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from typing import Optional
from utils.evaluation import MetricsEvaluator, ComputeAllResult
from .model import StudentModel, DistillationLoss


class DistillationConfig:
    def __init__(self) -> None:
        # Models
        self.student_model_name = "Salesforce/codet5-small"
        self.pretrained_model = True

        # Dataset
        self.train_dataset_path = Path(__file__).resolve().parents[2] / "data/distillation_data_training.jsonl"
        self.eval_dataset_path = Path(__file__).resolve().parents[2] / "data/distillation_data_validation.jsonl"
        self.max_src_length = 1024
        self.max_trg_length = 512

        # Training
        self.train_batch_size = 16
        self.eval_batch_size = 12
        self.learning_rate = 1e-4
        self.num_train_epochs = 15
        self.warmup_steps_ratio = 0.1  # Percentage of total steps for warmup
        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.temperature = 2.0  # Temperature for softening probability distributions
        self.alpha = 0.7  # Weight for distillation loss vs  task-specific loss
        self.eval_steps = 0  # Steps between evaluations (in a single epoch)
        self.eval_epochs = 1  # Number of epochs between evaluations
        self.num_workers = 8  # Number of workers for DataLoader

        # Output
        self.output_dir = Path(__file__).resolve().parents[2] / "output_distillation_v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistillationTrainer:
    def __init__(self, config: DistillationConfig, model: Optional[StudentModel]) -> None:
        """
        Initialize the DistillationTrainer with the provided configuration.
        Args:
            config (DistillationConfig): Configuration object containing training parameters.
            model (nn.Module): Pre-trained model to be used for distillation. If None, a new model will be created.
                               Useful for loading an already trained model for evaluation
        """
        self.config = config

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)

        # Initialize the model
        if model is not None:
            self.student_model = model.to(config.device)
        else:
            self.student_model = StudentModel(self.tokenizer, config.student_model_name, config.pretrained_model).to(
                config.device)
        self.student_model.config.use_cache = False

        # Initialize loss function
        self.criterion = DistillationLoss(temperature=config.temperature, alpha=config.alpha)

    def train(self, train_loader: DataLoader, eval_loader: DataLoader) -> None:
        """
        Train the model using the provided DataLoader.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            eval_loader (DataLoader): DataLoader for evaluation data.
        """
        self.student_model.train()

        total_steps = len(train_loader) * self.config.num_train_epochs
        self.config.warmup_steps = int(total_steps * self.config.warmup_steps_ratio)  # 10% of total steps for warmup
        global_step = 0

        # Log some info
        print(f"    Total training steps: {total_steps}")
        print(f"    Warmup steps: {self.config.warmup_steps}")
        print(f"    Memory usage: {self.student_model.get_memory()} MB")

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

        # Create metrics file
        metrics_file = self.config.output_dir / "metrics.csv"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            f.write(
                "epoch,global_step,train_loss,eval_loss,accuracy,similarity,f1,precision,recall,codeblue_avg,codebert_avg,rouge_l\n")

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

                global_step += 1

                # Evaluate the model
                if self.config.eval_steps > 0 and global_step % self.config.eval_steps == 0 and global_step > 0:
                    print(f"\nEvaluation at step {global_step}:")
                    val_loss, eval_results = self.evaluate(eval_loader)

                    print(f"  Validation loss: {val_loss:.4f}")
                    print(f"  Evaluation results:\n{eval_results}")

            # Log end of epoch
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.config.num_train_epochs}, Loss: {avg_loss:.4f}")

            # Save Model
            path = self.config.output_dir / f"epoch_{epoch + 1}"
            self.student_model.save_model(str(path))

            # Evaluate the model
            if (((epoch + 1) % self.config.eval_epochs == 0 or epoch == self.config.num_train_epochs - 1)
                    and self.config.eval_epochs > 0):
                print(f"  Evaluating epoch {epoch + 1}...")
                val_loss, eval_results = self.evaluate(eval_loader)

                with open(metrics_file, "a") as f:
                    f.write(f"{epoch + 1},{global_step},{avg_loss:.6f},{val_loss:.6f},"
                            f"{eval_results['accuracy']:.6f},{eval_results['similarity_score_avg']:.6f},"
                            f"{eval_results['f1']:.6f},{eval_results['precision']:.6f},{eval_results['recall']:.6f},"
                            f"{eval_results['codeblue_avg']:.6f},{eval_results['codebert_avg']:.6f},"
                            f"{eval_results['rougeL_avg']:.6f}\n")

                print(f"  Validation loss: {val_loss:.4f}")
                print(f"  Evaluation results:\n{eval_results}")

    def evaluate(self, eval_loader: DataLoader, use_teacher_pred: bool = False) -> tuple[float, ComputeAllResult]:
        """
        Evaluate the model using the provided DataLoader.
        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
            use_teacher_pred (bool): Whether to use teacher predictions for evaluation.

        Returns:
            tuple: A tuple containing the average loss and evaluation results.
        """
        self.student_model.eval()
        eval_loss = 0.0
        evaluator = MetricsEvaluator()

        progress_bar = tqdm(eval_loader, desc="Evaluating")
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.config.device)
                attention_mask = batch["attention_mask"].to(self.config.device)
                labels = batch["labels"].to(self.config.device)
                teacher_logits = batch['teacher_logits'].to(self.config.device)

                # Forward pass
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # Track loss
                loss = self.criterion(outputs.logits, teacher_logits, labels=labels)
                eval_loss += loss.item()

                # Generate predictions for a subset of examples to save time
                subset_size = min(self.config.eval_batch_size, input_ids.size(0))
                subset_indices = np.random.choice(input_ids.size(0), subset_size, replace=False)

                # Generate for subset
                for idx in subset_indices:
                    # Generate
                    generated_ids = self.student_model.model.generate(
                        input_ids=input_ids[idx:idx + 1],
                        attention_mask=attention_mask[idx:idx + 1],
                        max_length=self.config.max_trg_length,
                        num_beams=4,
                        early_stopping=True
                    )

                    # Decode prediction and reference
                    generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                    if use_teacher_pred:
                        reference_text = batch["predicted_assertions"][idx]
                    else:
                        reference_text = batch["original_target"][idx]

                    # Evaluate
                    metrics = evaluator.evaluate_assertions(generated_text, reference_text)

                    # Update metrics
                    evaluator.update(metrics)

                    # Display sample predictions occasionally
                    if np.random.random() < 0.01:  # Show ~1% of predictions
                        print("\nExample evaluation:")
                        print(f"Reference: {reference_text[:100]}...")
                        print(f"Generated: {generated_text[:100]}...")
                        print(f"Metrics: Exact matches={metrics['exact_matches']}, "
                              f"Accuracy={metrics['accuracy']:.4f}, "
                              f"Similarity={metrics['similarity_score_avg']:.4f}")

        # Calculate overall metrics
        avg_loss = eval_loss / len(eval_loader)

        eval_results = evaluator.compute_all()

        self.student_model.train()  # Reset to training mode after evaluation

        return avg_loss, eval_results

    def evaluate_generation_speed(
            self,
            dataset: Dataset,
            output_file: Optional[Path] = None,
            num_samples: int = 100,
            max_length: Optional[int] = None,
            num_beams: int = 4,
    ) -> dict:
        """
        Measure generation latency on a random subset of eval data.

        Args:
            dataset (Dataset): Evaluation dataset.
            output_file (Optional[Path]): If provided, will save generated samples to this file.
            num_samples (int): Number of examples to sample for timing.
            max_length (int, optional): Max generation length (defaults to self.config.max_trg_length).
            num_beams (int): Number of beams for generation.

        Returns:
            dict: {
                "times": List[float],    # per-sample generation times (seconds)
                "mean": float,           # mean latency
                "std": float,            # sample standard deviation
                "ci95": float            # 95% CI margin (± around the mean)
            }
        """
        # Put model in eval mode and disable gradients
        self.student_model.eval()
        device = self.config.device
        max_len = max_length or self.config.max_trg_length

        # Sample indices without replacement
        n_total = len(dataset)
        k = min(num_samples, n_total)
        indices: list[int] = random.sample(range(n_total), k)

        times = []
        progress_bar = tqdm(indices, desc="Evaluating")
        with torch.no_grad():
            for idx in progress_bar:
                item = dataset[idx]
                # assume item dict has these keys as in your train/eval loops
                input_ids = item["input_ids"].unsqueeze(0).to(device)
                attention_mask = item["attention_mask"].unsqueeze(0).to(device)

                t0 = time.perf_counter()
                generated_ids = self.student_model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_len,
                    num_beams=num_beams,
                    early_stopping=True,
                )
                t1 = time.perf_counter()
                times.append(t1 - t0)

                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                if output_file:
                    with open(output_file, "a", encoding='utf-8') as f:
                        f.write(f"Sample {idx}:\nInput text:\n{item['original_input']}\nOriginal assertions:\n"
                                f"{item['original_target']}\nGenerated assertion:\n{generated_text}\n\n")

        times_arr = np.array(times)
        mean = float(times_arr.mean())
        std = float(times_arr.std(ddof=1))
        ci95 = 1.96 * std / np.sqrt(len(times))

        # Restore train mode
        self.student_model.train()

        return {
            "times": times,
            "mean": mean,
            "std": std,
            "ci95": ci95,
        }
