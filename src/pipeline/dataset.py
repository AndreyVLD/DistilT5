import json
import torch
from pathlib import Path
from transformers import T5Tokenizer
from typing import Optional, Any, TypedDict, Iterator
from torch.utils.data import IterableDataset

from utils.decompression import decompress_tensor_optimized


class Sample(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor]


class RawEntry(TypedDict):
    repository: str
    focal_file: str
    test_method_masked: str
    assertions: list[str]
    method_under_test: str
    teacher_prediction: str
    teacher_parsed_assertions: list[str]
    teacher_metrics: dict[str, Any]
    teacher_logits: Any


class TestGenDataset(IterableDataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    """

    def __init__(self, tokenizer: T5Tokenizer, max_len=64):
        self.file_path = Path(__file__).resolve().parents[2] / "data" / "dataset_with_predictions.jsonl"

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self) -> Iterator[Sample]:
        with open(self.file_path, 'r') as f:
            for line in f:
                raw: RawEntry = json.loads(line)

                # TODO: Extend to increase the context length
                src = raw['test_method_masked']
                trg = raw['teacher_prediction']

                enc = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_len,
                                     return_tensors='pt')
                dec = self.tokenizer(trg, padding='max_length', truncation=True, max_length=self.max_len,
                                     return_tensors='pt')

                sample: Sample = {
                    'input_ids': enc.input_ids.squeeze(0),
                    'attention_mask': enc.attention_mask.squeeze(0),
                    'labels': dec.input_ids.squeeze(0),
                    'teacher_logits': None
                }

                if raw.get('teacher_logits') is not None:
                    sample['teacher_logits'] = decompress_tensor_optimized(raw['teacher_logits'])

                yield sample
