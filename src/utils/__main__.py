from utils.evaluation import evaluate_teacher


def main() -> None:
    output = evaluate_teacher('../../data/distillation_data_validation_base.jsonl')
    print(output)


if __name__ == '__main__':
    main()
