import matplotlib.pyplot as plt


def main() -> None:
    metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'Similarity']

    # Teacher vs GT (baseline)
    teacher_vs_gt = [0.38985841290747447,  # Precision
                     0.36873248209280596,  # Recall
                     0.37900128040973113,  # F1
                     0.45747815508422546,  # Accuracy
                     0.8428544205452826]  # Similarity

    # Student vs GT
    student_vs_gt = [
        0.184103,  # Precision
        0.223606,  # Recall
        0.201941,  # F1
        0.243899,  # Accuracy
        0.701946  # Similarity
    ]

    # Compute retention percentages: student vs GT relative to teacher vs GT
    retention = [(s / t) * 100 for s, t in zip(student_vs_gt, teacher_vs_gt)]

    #  Student retention of teacher performance on Ground Truth
    plt.figure()
    plt.bar(metrics, retention)
    plt.title("Student Retention of Teacher's Performance on Ground Truth")
    plt.ylabel('Retention (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
