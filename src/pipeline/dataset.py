import orjson
import torch
import javalang
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
        method_name = raw["method_under_test"]
        full_java = raw["focal_file"]
        method_code = extract_method_via_ast(full_java, method_name)

        src = f"METHOD UNDER TEST:\n{method_code}\n\nTEST METHOD:\n{raw['test_method_masked']}"
        trg = raw['teacher_prediction']

        # Temporarily switch to left truncation to avoid cutting the test method
        old_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "left"

        src_enc = self.tokenizer(
            src,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_length,
            return_tensors="pt",
        )

        self.tokenizer.truncation_side = old_side

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


def extract_method_via_ast(java_src: str, method_name: str) -> str:
    """
    Parse the Java source, find the MethodDeclaration whose .name == method_name,
    then return its full text (from its start line through matching braces).
    Handles potential JavaSyntaxError.
    """
    try:
        # Parse into AST
        tree = javalang.parse.parse(java_src)
        # Read lines once
        lines = java_src.splitlines(keepends=True)

        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.name == method_name:
                # node.position gives (line, col) of the signature
                start_line = node.position.line - 1
                # Now walk forward to extract until braces balance
                brace_count = 0
                snippet_lines = []
                for i, line in enumerate(lines[start_line:], start=start_line):
                    snippet_lines.append(line)
                    brace_count += line.count("{") - line.count("}")
                    if brace_count == 0:
                        break
                return "".join(snippet_lines)

    except javalang.parser.JavaSyntaxError as e:
        # Return method name string to indicate parsing failed
        return java_src
    except Exception as e:
        # Catch other potential exceptions during parsing
        return java_src
    
    return method_name
