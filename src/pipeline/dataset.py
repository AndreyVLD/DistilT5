import orjson
import torch
from transformers import RobertaTokenizer
from typing import Optional, Any, TypedDict, Iterator
from torch.utils.data import IterableDataset, Dataset

from utils.decompression import decompress_tensor_optimized


class Sample(TypedDict):
    original_text: str
    ground_truth: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor]
    teacher_labels: torch.Tensor


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


class AssertGenMixin:
    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_len: int = 64) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_trg_len = max_trg_len
        self.len = None

    def _process_raw(self, raw: RawEntry) -> Sample:
        # TODO: Extend src with more context and information
        #       input_text = f"FOCAL CODE:\n{item['focal_file']}\n\nTEST METHOD:\n{item['test_method_masked']}"
        # src = raw['test_method_masked']
        src = f"FOCAL CODE:\n{raw['focal_file']}\n\nTEST METHOD:\n{raw['test_method_masked']}"
        trg = raw['teacher_prediction']

        src_enc = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_src_length,
                                 return_tensors='pt')
        trg_enc = self.tokenizer(trg, padding='max_length', truncation=True, max_length=self.max_trg_len,
                                 return_tensors='pt')
        gt = '\n'.join(raw['assertions'])
        gt_enc = self.tokenizer(gt, padding='max_length', truncation=True,
                                max_length=self.max_trg_len,
                                return_tensors='pt')

        input_ids = src_enc['input_ids'].squeeze()
        attention_mask = src_enc['attention_mask'].squeeze()
        labels = gt_enc['input_ids'].squeeze()
        teacher_labels = trg_enc['input_ids'].squeeze()

        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        teacher_labels[teacher_labels == self.tokenizer.pad_token_id] = -100

        sample: Sample = {
            'original_text': src,
            'ground_truth': gt,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'teacher_labels': teacher_labels,
            'labels': labels,
            'teacher_logits': None
        }

        if raw.get('teacher_logits') is not None:
            sample['teacher_logits'] = decompress_tensor_optimized(raw['teacher_logits'])
        return sample

    def _iter_raws(self) -> Iterator[RawEntry]:
        with open(self.file_path, "r") as f:
            for line in f:
                yield orjson.loads(line)


class IterableAssertGenDataset(AssertGenMixin, IterableDataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    It loads the data from a JSONL lazily, processes it, and returns it in a format suitable for training.
    """

    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_len: int = 64) -> None:
        super().__init__(tokenizer, file_path, max_src_length, max_trg_len)
        self._len = None

    def __len__(self) -> int:
        if self.len is None:
            with open(self.file_path, 'r') as f:
                self.len = sum(1 for _ in f)
        return self.len

    def __iter__(self) -> Iterator[Sample]:
        for raw in self._iter_raws():
            yield self._process_raw(raw)


class MapAssertGenDataset(AssertGenMixin, Dataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    It loads the data from a JSONL eagerly, processes it, and returns it in a format suitable for training.
    """

    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_len: int = 64) -> None:
        super().__init__(tokenizer, file_path, max_src_length, max_trg_len)
        # eagerly load everything into memory
        self._data = list(self._iter_raws())

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Sample:
        raw = self._data[idx]
        return self._process_raw(raw)
