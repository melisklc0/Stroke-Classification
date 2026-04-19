"""
Baseline transfer-learning CNN training per fold, checkpointing, and metric plots.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import os

from .data_prep import create_folds
from .models import get_model
from .evaluate import plot_metrics, save_metrics_to_file_txt, save_final_metrics_to_file, plot_confusion_matrix


class StrokeDataset(torch.utils.data.Dataset):
    """
    Image folder dataset for binary stroke classification.
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        from PIL import Image

        self.root_dir = root_dir
        self.transform = transform
        if class_to_idx is None:
            class_to_idx = {"No-Stroke": 0, "Stroke": 1}
        self.classes = class_to_idx
        self.image_paths = []
        self.labels = []
        valid_extensions = [".jpg", ".jpeg", ".png"]

        for class_name, class_idx in self.classes.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for img_name in os.listdir(class_folder):
                if os.path.splitext(img_name)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(class_folder, img_name))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def evaluate_model(model, test_loader, device):
    """
    Run the model on the test loader and return label and prediction lists.
    """
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_labels, all_preds


def train_plain_cnn(config):
    """
    Train a single transfer-learning CNN per fold and save metrics / checkpoints.
    """
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Backbone name from config (must be defined before paths and checkpoints use it).
    model_name = config["model"]["name"]
    # Prefix for metric filenames under results/ (e.g. "inceptionv3_").
    prefix = f"{model_name}_"

    # Subfolder layout: results/cnn/<model_name>/ and checkpoints/cnn/<model_name>/
    result_path = os.path.join(
        config.get("data", {}).get("result_path", "results"), "cnn", model_name
    )
    checkpoint_path = os.path.join(
        config.get("data", {}).get("checkpoint_path", "checkpoints"), "cnn", model_name
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
    class_to_idx = config["data"]["classes"]

    for fold in folds:
        print(f"Training {fold}")
        train_dir = os.path.join(config["data"]["base_path"], fold, "train")
        test_dir = os.path.join(config["data"]["base_path"], fold, "test")

        train_dataset = StrokeDataset(train_dir, transform=transform, class_to_idx=class_to_idx)
        test_dataset = StrokeDataset(test_dir, transform=transform, class_to_idx=class_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = get_model(
            model_name,
            pretrained=config["model"]["pretrained"],
            num_classes=config["model"]["num_classes"],
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop entry for this fold
        print(f"Starting {fold} training for {epochs} epochs")

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

        for epoch in range(epochs):
            model.train()
            running_loss, correct_train, total_train = 0.0, 0, 0

            # One full pass over the training fold.
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

            epoch_train_loss = running_loss / len(train_loader)
            epoch_train_acc = correct_train / total_train
            train_losses.append(epoch_train_loss)
            train_accuracies.append(epoch_train_acc)

            model.eval()
            correct_test, total_test, test_loss = 0, 0, 0.0
            # Validation pass for epoch-level test loss/accuracy and best-model selection.
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct_test += (preds == labels).sum().item()
                    total_test += labels.size(0)

            epoch_test_loss = test_loss / len(test_loader)
            epoch_test_acc = correct_test / total_test
            test_losses.append(epoch_test_loss)
            test_accuracies.append(epoch_test_acc)

            if epoch_test_acc > best_acc:
                best_acc = epoch_test_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Persist best weights for this fold under checkpoints/cnn/<model_name>/
                torch.save(
                    best_model_wts,
                    os.path.join(checkpoint_path, f"{model_name}_best_model_{fold}.pth"),
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
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, "
                f"Train Acc: {epoch_train_acc:.4f}, Test Loss: {epoch_test_loss:.4f}, "
                f"Test Acc: {epoch_test_acc:.4f}"
            )

        model.load_state_dict(best_model_wts)
        print(f"Best weights for {fold} saved")

        print(f"Evaluating {fold}")
        # Reuse trained weights already in memory; eval mode for inference-only metrics.
        model.eval()
        labels, preds = evaluate_model(model, test_loader, device)
        save_final_metrics_to_file(fold, labels, preds, result_path, prefix)
        class_names = [n for n, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        plot_confusion_matrix(labels, preds, fold, result_path, prefix, class_names=class_names)
        plot_metrics(
            train_losses, test_losses, train_accuracies, test_accuracies, fold, result_path, prefix
        )
        print(f"Finished training {fold}")
