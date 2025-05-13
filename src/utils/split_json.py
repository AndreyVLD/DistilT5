from pathlib import Path


def split_jsonl(input_path: str, train_output: str, val_output: str, val_frac: float = 0.2) -> None:
    """
    Splits a JSONL file into train/validation by taking the last val_frac lines as validation.

    Args:
        input_path: Path to the source .jsonl file.
        train_output: Path where training split is saved.
        val_output: Path where validation split is saved.
        val_frac: Fraction of lines to reserve for validation (e.g. 0.2 for last 20%).
    """
    input_path = Path(input_path)
    train_output = Path(train_output)
    val_output = Path(val_output)

    # Count total lines
    with input_path.open('r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    split_idx = int(total_lines * (1 - val_frac))
    
    print(f"Splitting {total_lines} lines into {split_idx} for training and {total_lines - split_idx} for validation.")

    # Write splits
    with input_path.open('r', encoding='utf-8') as src_f, \
            train_output.open('w', encoding='utf-8') as train_f, \
            val_output.open('w', encoding='utf-8') as val_f:
        for idx, line in enumerate(src_f):
            if idx < split_idx:
                train_f.write(line)
            else:
                val_f.write(line)
