import matplotlib
import matplotlib.pyplot as plt


def retained_metrics() -> None:
    student_metrics = {
        'precision': 0.3062518006338231,
        'recall': 0.33104951728433507,
        'f1': 0.3181682131098474,
        'accuracy': 0.3803534573312824,
        'similarity_score_avg': 0.7980514364493576,
        'codeblue_avg': 0.495959670545932,
        'codebert_avg': 0.9073392686247825,
        'rougeL_avg': 0.6676108429225623
    }
    teacher_metrics = {
        'precision': 0.37719578409453847,
        'recall': 0.36779819370912487,
        'f1': 0.3724377168085777,
        'accuracy': 0.46889468144830426,
        'similarity_score_avg': 0.8393990736288519,
        'codeblue_avg': 0.5277710792057665,
        'codebert_avg': 0.9222477597671075,
        'rougeL_avg': 0.7255797936704341
    }

    # Compute retention percentages
    retention = {metric: (student_metrics[metric] / teacher_metrics[metric]) * 100
                 for metric in student_metrics}

    # Abbreviated labels (avoid writing "precision" or "recall")
    labels = ['F1', 'Accuracy', 'Similarity', 'CodeBLEU', 'CodeBERT', 'Rouge-L']
    retention_values = [
        retention['f1'],
        retention['accuracy'],
        retention['similarity_score_avg'],
        retention['codeblue_avg'],
        retention['codebert_avg'],
        retention['rougeL_avg']
    ]

    # Plotting
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, retention_values, color='orange')
    plt.ylim(0, 105)
    plt.ylabel('Retention (%)')
    plt.title('Student Retention of Teacher Performance')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    retained_metrics()


if __name__ == '__main__':
    matplotlib.use('qtagg')
    main()
