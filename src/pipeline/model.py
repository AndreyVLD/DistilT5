import json
import os
from enum import Enum

import torch
import torch.nn.functional as F
from typing import Optional
from torch import nn, Tensor
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput


class ModelType(Enum):
    CODET5PLUS = "codet5plus"
    CODET5 = "codet5"


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute the distillation loss with safe handling of tensor shapes.
        Args:
            student_logits (Tensor): Logits from the student model.
            teacher_logits (Tensor): Logits from the teacher model.
            labels (Optional[Tensor]): True labels for the task.
        Returns:
            Tensor: Computed loss.
        """
        # Handle potential shape differences between teacher and student logits
        min_seq_len = min(student_logits.shape[1], teacher_logits.shape[1])

        # Safely truncate both to matching sequence lengths
        truncated_student_logits = student_logits[:, :min_seq_len, :]
        truncated_teacher_logits = teacher_logits[:, :min_seq_len, :]

        # Check if we have identical shapes after truncation
        if truncated_student_logits.shape != truncated_teacher_logits.shape:
            raise ValueError(
                f"Shape mismatch after truncation: "
                f"student={truncated_student_logits.shape}, teacher={truncated_teacher_logits.shape}"
            )

        # Compute softmax probabilities
        student_probs = F.log_softmax(truncated_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(truncated_teacher_logits / self.temperature, dim=-1).detach()

        # Compute distillation loss
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # Compute cross-entropy loss if labels are provided
        if labels is not None:
            # Make sure we're using the student logits directly (full sequence length) for CE loss
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            return (1 - self.alpha) * ce_loss + self.alpha * distillation_loss

        return distillation_loss


# TODO investigate other pretrained models
class StudentModel(nn.Module):
    model: T5ForConditionalGeneration

    def __init__(self, tokenizer: RobertaTokenizer, name: Optional[str], use_pretrained: bool = False) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=256,
            num_layers=4,
            num_decoder_layers=4,
            num_heads=4,
            d_ff=1024,
            dropout_rate=0.1,
            decoder_start_token_id=tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if name is None:
            # Initialize a new model if no name is provided
            self.model = T5ForConditionalGeneration(config=self.config)
        elif not use_pretrained:
            # If not using a pretrained model, initialize a new one and inherit weights from the provided name
            self.model = T5ForConditionalGeneration.from_pretrained(name, config=self.config,
                                                                    ignore_mismatched_sizes=True)
        else:
            # Load a pre-trained model without changing the config
            self.model = T5ForConditionalGeneration.from_pretrained(name)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Optional[Tensor]) -> Seq2SeqLMOutput:
        outputs: Seq2SeqLMOutput = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def get_memory(self) -> float:
        """
        Get the memory usage of the model in MB.
        Returns:
            float: Memory usage in MB.
        """
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        mem = mem_params + mem_bufs
        return mem / (1024 ** 2)  # Convert to MB

    def save_model(self, path: str) -> None:
        """
        Save the model, tokenizer and config to the specified path.
        Args:
            path (str): Path to save the model.
        """
        os.makedirs(path, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), f"{path}/model.pt")

        # Save tokenizer and config
        self.tokenizer.save_pretrained(path)
        self.model.config.save_pretrained(path)

        # Save extra metadata
        metadata = {
            "memory_usage_mb": self.get_memory(),
        }

        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Model, tokenizer and config saved to {path}")

    @classmethod
    def load_model(cls, path: str, model_type: ModelType) -> 'StudentModel':
        """
        Load a saved model from disk

        Args:
            path: Directory path where the model and tokenizer are saved
            model_type: The type of model to load (e.g., CODET5, CODET5PLUS)

        Returns:
            The loaded StudentModel instance
        """
        # Create model instance using the path
        if model_type == ModelType.CODET5:
            tokenizer = RobertaTokenizer.from_pretrained(path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path)

        student_model = cls(tokenizer, None)

        # Load saved weights
        try:
            state_dict = torch.load(f"{path}/model.pt")
            config = T5Config.from_pretrained(path)
            student_model.model = T5ForConditionalGeneration(config=config)
            student_model.model.load_state_dict(state_dict)
            print(f"Model weights loaded from {path}/model.pt")
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {str(e)}")

        return student_model
