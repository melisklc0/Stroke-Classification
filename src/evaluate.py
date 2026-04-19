"""
Plotting and metric logging helpers (curves, confusion matrix, text summaries).
"""
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def plot_metrics(
    train_losses,
    test_losses,
    train_accuracies,
    test_accuracies,
    fold_name,
    save_dir,
    prefix="",
):
    """
    Plot and save training/testing loss and accuracy curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{prefix}{fold_name}.png")

    plt.figure(figsize=(10, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{fold_name} Training and Testing Accuracy")
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{fold_name} Training and Testing Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"[{fold_name}] Saved plots to {file_path}")


def save_metrics_to_file_txt(
    fold_name, epoch, train_loss, test_loss, train_acc, test_acc, save_dir, prefix=""
):
    """
    Append per-epoch loss and accuracy lines to a text log.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{prefix}losses_{fold_name}.txt")

    with open(file_path, "a") as file:
        file.write(f"Epoch {epoch + 1}:\n")
        file.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n")
        file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
        file.write("=" * 50 + "\n")


def save_final_metrics_to_file(fold_name, labels, preds, save_dir, prefix=""):
    """
    Write aggregate test metrics to a text file and print them.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{prefix}metrics_{fold_name}.txt")

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    with open(file_path, "w") as file:
        file.write(f"Final Metrics for {fold_name}:\n")
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
    print(f"[{fold_name}] Metrics saved to text file")


def plot_confusion_matrix(
    labels, preds, fold_name, save_dir, prefix="", class_names=None
):
    """
    Plot a confusion matrix heatmap and save it as PNG.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = ["No-Stroke", "Stroke"]

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 15},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {fold_name}")

    os.makedirs(save_dir, exist_ok=True)
    cm_img_path = os.path.join(save_dir, f"{prefix}confusion_matrix_{fold_name}.png")
    plt.savefig(cm_img_path, bbox_inches="tight")
    plt.close()
    print(f"[{fold_name}] Confusion matrix saved to {cm_img_path}")
