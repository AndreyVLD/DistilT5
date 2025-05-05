import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class MockTestGenDataset(Dataset):
    """
    A mock dataset that uses hard-coded method-test pairs.
    Tokenizes inputs/tests and creates random teacher logits.
    """

    def __init__(self, tokenizer: T5Tokenizer, max_len=64):
        data_path = Path(__file__).resolve().parents[2] / "data" / "sample_data.json"
        with data_path.open("r") as f:
            self.samples = json.load(f)

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.vocab_size = 32128

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        src = sample['method']
        trg = sample['test']

        # Tokenize source and target together for encoder-decoder
        enc = self.tokenizer(src, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        dec = self.tokenizer(trg, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        input_ids = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        labels = dec.input_ids.squeeze(0)

        # Create random teacher logits matching [seq_len, vocab_size]
        # TODO: Replace with actual teacher model logits
        seq_len = labels.size(0)
        teacher_logits = torch.randn(seq_len, self.vocab_size)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'teacher_logits': teacher_logits
        }
