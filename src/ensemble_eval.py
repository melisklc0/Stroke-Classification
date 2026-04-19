"""
Ensemble evaluation: load multiple checkpoints and average logits (soft voting).
"""
from typing import Any


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from .models import get_model
from .train_cnn import StrokeDataset
from .evaluate import save_final_metrics_to_file, plot_confusion_matrix


def run_ensemble(config, data_path: str, model_weights: dict, output_prefix: str = "ensemble"):
    """
    Average logits (soft voting) from multiple checkpoints and evaluate on a folder dataset.
    """
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result_path = os.path.join(config.get("data", {}).get("result_path", "results"), "ensemble")
    os.makedirs(result_path, exist_ok=True)

    img_size = config["data"]["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    class_to_idx = config["data"]["classes"]
    class_names = [n for n, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    test_dataset = StrokeDataset(data_path, transform=transform, class_to_idx=class_to_idx)
    test_loader = DataLoader[Any](
        test_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    loaded_models = []
    print("Loading models...")
    for model_name, weight_path in model_weights.items():
        if not os.path.exists(weight_path):
            print(f"WARNING: {weight_path} not found, skipping {model_name}.")
            continue

        model = get_model(model_name, pretrained=False, num_classes=config["model"]["num_classes"])
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        loaded_models.append(model)

    if not loaded_models:
        print("ERROR: no weights could be loaded. Ensemble aborted.")
        return

    true_labels = []
    predicted_labels = []

    print(f"Running ensemble with {len(loaded_models)} models...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            true_labels.extend(labels.numpy())

            logits = []
            for model in loaded_models:
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
                logits.append(out)

            # Soft voting: same as averaging class logits before argmax.
            logits_avg = torch.mean(torch.stack(logits), dim=0)
            predicted_labels.extend(torch.argmax(logits_avg, dim=1).cpu().numpy())

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    save_final_metrics_to_file(
        "ensemble", true_labels, predicted_labels, result_path, prefix=output_prefix + "_"
    )
    plot_confusion_matrix(
        true_labels,
        predicted_labels,
        "ensemble",
        result_path,
        prefix=output_prefix + "_",
        class_names=class_names,
    )
