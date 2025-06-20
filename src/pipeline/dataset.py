import orjson
import torch
from transformers import RobertaTokenizer
from typing import Optional, Any, TypedDict, Iterator
from torch.utils.data import IterableDataset, Dataset

from utils.decompression import decompress_logits


class Sample(TypedDict):
    """Represents a single sample in the assert generation dataset."""
    original_input: str
    original_target: str
    predicted_assertions: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor]


class RawEntry(TypedDict):
    """Represents a raw entry in the assert generation dataset jsonl file."""
    focal_file: str
    test_method_masked: str
    original_target: list[str] | str
    predicted_assertions: str
    compressed_logits: dict[str, Any]


def validate_raw(raw: dict[str, Any]) -> bool:
    """
    Validate a raw entry to ensure it contains all required fields.
    Args:
        raw: A dictionary representing a raw entry from the dataset. A line from the JSONL file.

    Returns:
        bool: True if the raw entry is valid, False otherwise.
    """
    if not isinstance(raw, dict):
        print(f"Invalid raw entry: {raw}")
        return False
    required = set(RawEntry.__annotations__.keys())
    missing = required - raw.keys()

    if missing:
        print(f"Missing keys in raw entry: {missing}")
        return False
    else:
        return True


class AssertGenMixin:
    """
    Mixin class for assert generation datasets.
    This class provides common functionality for processing raw entries and iterating over the dataset.
    """

    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_length: int = 64) -> None:

        """
        Initialize the AssertGenMixin with tokenizer and dataset parameters.
        Args:
            tokenizer (RobertaTokenizer): Tokenizer to use for encoding the inputs and targets.
            file_path (str): Path to the JSONL file containing the dataset.
            max_src_length (int): Maximum length of the source input (test method + focal file).
            max_trg_length (int): Maximum length of the target output (assertions).
        """

        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.max_trg_length = max_trg_length
        self.len = None

    def _process_raw(self, raw: RawEntry) -> Optional[Sample]:
        """
        Process a raw entry from the dataset and convert it into a Sample.
        Args:
            raw: A dictionary representing a raw entry from the dataset. A line from the JSONL file.

        Returns:
            Sample: A dictionary containing the processed input and target tensors, along with metadata.
            Returns None if the raw entry is invalid.
        """

        # Extract fields from the raw entry
        focal_file = raw["focal_file"]
        test_method = raw["test_method_masked"]
        assertions = raw["original_target"]
        predicted_assertions = raw["predicted_assertions"]

        # Tokenize the test method to check its length
        test_method_tokens = self.tokenizer(
            f"TEST METHOD:\n{test_method}",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_src_length,
            return_tensors="pt"
        )
        test_method_length = test_method_tokens.input_ids.size(1)

        # If test method already exceeds limit (rare but possible), we must truncate it
        if test_method_length >= self.max_src_length - 10:  # Leave room for special tokens
            # Just keep the test method, already truncated
            input_text = f"{self.tokenizer.decode(test_method_tokens.input_ids[0], skip_special_tokens=True)}"
        else:
            # Determine how much space we have left for the focal file
            space_for_focal = self.max_src_length - test_method_length - 20  # Reserve tokens for prefix and special tokens

            # Format input text based on available space
            if space_for_focal <= 0:
                # Not enough space - use only test method
                input_text = f"TEST METHOD:\n{test_method}"
            else:
                # Tokenize focal file to check its length, with explicit truncation
                focal_tokens = self.tokenizer(
                    focal_file,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=space_for_focal,
                    return_tensors="pt"
                )

                # Create combined input with truncated focal file if needed
                truncated_focal = self.tokenizer.decode(focal_tokens.input_ids[0], skip_special_tokens=True)
                input_text = f"FOCAL CODE:\n{truncated_focal}\n\nTEST METHOD:\n{test_method}"

        # Target text
        target_text = "\n".join(assertions) if isinstance(assertions, list) else assertions

        # Tokenize the input and target texts
        source_encoding = self.tokenizer(
            input_text,
            max_length=self.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_trg_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Double-check lengths and force truncation if needed (safety check)
        if source_encoding["input_ids"].size(1) > self.max_src_length:
            source_encoding["input_ids"] = source_encoding["input_ids"][:, :self.max_src_length]
            source_encoding["attention_mask"] = source_encoding["attention_mask"][:, :self.max_src_length]

        if target_encoding["input_ids"].size(1) > self.max_trg_length:
            target_encoding["input_ids"] = target_encoding["input_ids"][:, :self.max_trg_length]

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 so it's ignored in loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100

        sample: Sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "original_input": input_text,
            "original_target": target_text,
            "predicted_assertions": predicted_assertions,
            "teacher_logits": None,
        }

        if raw.get('compressed_logits') is not None:
            sample['teacher_logits'] = decompress_logits(raw['compressed_logits']).squeeze()
        return sample

    def _iter_raws(self) -> Iterator[RawEntry]:
        """
        Iterate over the JSONL file lazily, yielding raw entries.
        Returns:
            Iterator[RawEntry]: An iterator that yields raw entries from the dataset.
        """
        with open(self.file_path, "r") as f:
            for line in f:
                raw_json = orjson.loads(line)
                if validate_raw(raw_json):
                    yield raw_json


class IterableAssertGenDataset(AssertGenMixin, IterableDataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    It loads the data from a JSONL lazily, processes it, and returns it in a format suitable for training.
    """

    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_length: int = 64) -> None:
        """
        Initialize the IterableAssertGenDataset with tokenizer and dataset parameters.
        Args:
            tokenizer (RobertaTokenizer): Tokenizer to use for encoding the inputs and targets.
            file_path (str): Path to the JSONL file containing the dataset.
            max_src_length (int): Maximum length of the source input (test method + focal file).
            max_trg_length (int): Maximum length of the target output (assertions).
        """
        super().__init__(tokenizer, file_path, max_src_length, max_trg_length)
        self._len = None

    def __len__(self) -> int:
        """
        Get the length of the dataset by counting the number of lines in the JSONL file.
        Returns:
            int: The number of samples in the dataset.
        """
        if self.len is None:
            with open(self.file_path, 'r') as f:
                self.len = sum(1 for _ in f)
        return self.len

    def __iter__(self) -> Iterator[Sample]:
        """
        Iterate over the dataset, processing each raw entry into a Sample.
        Returns:
            Iterator[Sample]: An iterator that yields processed samples from the dataset.
        """
        for raw in self._iter_raws():
            yield self._process_raw(raw)


class MapAssertGenDataset(AssertGenMixin, Dataset):
    """
    Dataset for assert generation.
    This dataset is used to load and preprocess the data for training and evaluation.
    It loads the data from a JSONL eagerly, processes it, and returns it in a format suitable for training.
    """

    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, max_src_length: int = 64,
                 max_trg_length: int = 64) -> None:
        """
        Initialize the MapAssertGenDataset with tokenizer and dataset parameters.
        Args:
            tokenizer (RobertaTokenizer): Tokenizer to use for encoding the inputs and targets.
            file_path (str): Path to the JSONL file containing the dataset.
            max_src_length (int): Maximum length of the source input (test method + focal file).
            max_trg_length (int): Maximum length of the target output (assertions).
        """

        super().__init__(tokenizer, file_path, max_src_length, max_trg_length)
        # Eagerly load everything into memory
        self._data = list(self._iter_raws())

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        Returns:
            Returns the number of samples in the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx: int) -> Sample:
        """
        Get a single item from the dataset by index.
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Sample: A dictionary containing the processed input and target tensors, along with metadata.
        """
        raw = self._data[idx]
        return self._process_raw(raw)
