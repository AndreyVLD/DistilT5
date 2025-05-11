from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """
        Compute the distillation loss.
        Args:
            student_logits (Tensor): Logits from the student model.
            teacher_logits (Tensor): Logits from the teacher model.
            labels (Optional[Tensor]): True labels for the task.
        Returns:
            Tensor: Computed loss.
        """

        # Compute softmax probabilities
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Compute distillation loss
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # Compute cross-entropy loss if labels are provided
        if labels is not None:
            ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1),
                                      ignore_index=-100)
            return self.alpha * ce_loss + (1 - self.alpha) * distillation_loss

        return distillation_loss


class StudentModel(nn.Module):
    def __init__(self, tokenizer: T5Tokenizer) -> None:
        super().__init__()
        self.config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=768,
            num_layers=6,
            num_heads=4,
            d_ff=2048,
            dropout_rate=0.1,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        self.model = T5ForConditionalGeneration(self.config)

    def forward(self, input_ids: Tensor, attention_mask: Tensor, labels: Optional[Tensor] = None) -> Seq2SeqLMOutput:
        outputs: Seq2SeqLMOutput = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
