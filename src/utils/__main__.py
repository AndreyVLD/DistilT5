from utils.split_json import split_jsonl


def main() -> None:
    split_jsonl(
        input_path="../../data/dataset_with_predictions.jsonl",
        train_output="../../data/train_split.jsonl",
        val_output="../../data/val_split.jsonl",
        val_frac=0.2
    )


if __name__ == '__main__':
    main()
