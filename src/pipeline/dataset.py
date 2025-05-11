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
    teacher_metrics: dict[str, float]
    teacher_logits: Any


class TestGenDataset(IterableDataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    """

    def __init__(self, tokenizer: T5Tokenizer, file_name: str = "dataset_with_predictions.jsonl", max_src_length=64,
                 max_trg_len=64):
        self.file_path = Path(__file__).resolve().parents[2] / "data" / file_name

        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_trg_len = max_trg_len

    def __iter__(self) -> Iterator[Sample]:
        with open(self.file_path, 'r') as f:
            for line in f:
                raw: RawEntry = json.loads(line)

                # TODO: Extend src with more context and information
                #       input_text = f"FOCAL CODE:\n{item['focal_file']}\n\nTEST METHOD:\n{item['test_method_masked']}"
                src = raw['test_method_masked']
                trg = raw['teacher_prediction']

                source_encoding = self.tokenizer(src, padding='max_length', truncation=True,
                                                 max_length=self.max_src_length,
                                                 return_tensors='pt')
                target_encoding = self.tokenizer(trg, padding='max_length', truncation=True,
                                                 max_length=self.max_trg_len,
                                                 return_tensors='pt')

                input_ids = source_encoding['input_ids'].squeeze()
                attention_mask = source_encoding['attention_mask'].squeeze()
                labels = target_encoding['input_ids'].squeeze()

                # Replace padding token id with -100 so it's ignored in loss computation
                labels[labels == self.tokenizer.pad_token_id] = -100

                sample: Sample = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'teacher_logits': None
                }

                if raw.get('teacher_logits') is not None:
                    sample['teacher_logits'] = decompress_tensor_optimized(raw['teacher_logits'])

                yield sample
