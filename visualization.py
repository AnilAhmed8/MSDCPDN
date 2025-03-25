import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes=["0", "1"], title="Confusion Matrix", cmap="Blues"):
    """
    Plots the confusion matrix.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def plot_metrics(precision, recall, f1, accuracy):
    """
    Plots a bar chart of evaluation metrics.
    """
    metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1, "Accuracy": accuracy}
    plt.figure()
    keys = list(metrics.keys())
    values = list(metrics.values())
    plt.bar(keys, values)
    plt.ylim(0, 1)
    plt.title("Evaluation Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.show()