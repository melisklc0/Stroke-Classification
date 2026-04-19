"""
Knowledge-distillation training: student learns from hard labels and teacher soft targets.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import os

from .models import get_model
from .train_cnn import StrokeDataset, evaluate_model
from .evaluate import plot_metrics, save_metrics_to_file_txt, save_final_metrics_to_file, plot_confusion_matrix


def knowledge_distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha):
    """
    Combined KL distillation loss and hard-label cross-entropy.
    """
    kd_loss = nn.KLDivLoss(reduction="batchmean")(
        nn.functional.log_softmax(student_outputs / temperature, dim=1),
        nn.functional.softmax(teacher_outputs / temperature, dim=1),
    )
    ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss


def train_kd(config):
    """
    Train a student network with logits supervision from a frozen teacher.
    """
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    student_name = config["kd_training"]["student_model"]
    base_path = config["data"]["base_path"]
    result_path = os.path.join(
        config.get("data", {}).get("result_path", "results"), "kd", student_name
    )
    checkpoint_path = os.path.join(
        config.get("data", {}).get("checkpoint_path", "checkpoints"), "kd", student_name
    )
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    img_size = config["data"]["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    folds = config["training"]["folds"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]

    teacher_name = config["kd_training"]["teacher_model"]
    teacher_weights_path = config["kd_training"]["teacher_weights"]
    temperature = config["kd_training"]["temperature"]
    alpha = config["kd_training"]["alpha"]
    prefix = f"kd_{student_name}_student_"
    class_to_idx = config["data"]["classes"]
    class_names = [n for n, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    print("Loading teacher model...")
    teacher_model = get_model(teacher_name, pretrained=False, num_classes=config["model"]["num_classes"])
    # Strict false in case of aux logits differences between teacher and student
    teacher_model.load_state_dict(
        torch.load(teacher_weights_path, map_location=device), strict=False
    )
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    for fold in folds:
        print(f"\n--- [KD] Training {student_name} on {fold} ---")
        train_dir = os.path.join(base_path, fold, "train")
        test_dir = os.path.join(base_path, fold, "test")

        train_dataset = StrokeDataset(train_dir, transform=transform, class_to_idx=class_to_idx)
        test_dataset = StrokeDataset(test_dir, transform=transform, class_to_idx=class_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        student_model = get_model(
            student_name, pretrained=True, num_classes=config["model"]["num_classes"]
        )
        student_model = student_model.to(device)

        optimizer = optim.Adam(student_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_model_wts = copy.deepcopy(student_model.state_dict())
        best_acc = 0.0
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

        for epoch in range(epochs):
            student_model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                student_outputs = student_model(images)

                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                    # Some torchvision models return (logits, aux); KD uses the primary logits only.
                    if isinstance(teacher_outputs, tuple):
                        teacher_outputs = teacher_outputs[0]

                loss = knowledge_distillation_loss(
                    student_outputs, teacher_outputs, labels, temperature, alpha
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(student_outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            student_model.eval()
            correct_test, total_test, test_loss = 0, 0, 0.0
            criterion = nn.CrossEntropyLoss()

            # Student-only cross-entropy on the validation fold (teacher not used here).
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student_model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_test += (preds == labels).sum().item()
                    total_test += labels.size(0)

            epoch_test_loss = test_loss / len(test_loader)
            epoch_test_acc = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_acc)

            scheduler.step(epoch_test_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if epoch_test_acc > best_acc:
                best_acc = epoch_test_acc
                best_model_wts = copy.deepcopy(student_model.state_dict())
                torch.save(
                    student_model.state_dict(),
                    os.path.join(checkpoint_path, f"{prefix}best_model_{fold}.pth"),
                )

            print(f"Epoch {epoch + 1} done, learning rate: {current_lr:.8f}")
            print(
                f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}"
            )
            save_metrics_to_file_txt(
                fold,
                epoch,
                epoch_train_loss,
                epoch_test_loss,
                epoch_train_acc,
                epoch_test_acc,
                result_path,
                prefix,
            )

        student_model.load_state_dict(best_model_wts)
        print(f"Best student weights for {fold} saved")

        print(f"Evaluating {fold}")
        student_model.eval()
        labels, preds = evaluate_model(student_model, test_loader, device)
        save_final_metrics_to_file(fold, labels, preds, result_path, prefix)
        plot_confusion_matrix(labels, preds, fold, result_path, prefix, class_names=class_names)
        plot_metrics(
            train_losses, test_losses, train_accuracies, test_accuracies, fold, result_path, prefix
        )
        print(f"Finished KD training for {fold}")
