"""
Model factory: torchvision backbones with a shared two-layer 256-d classifier head.
"""
import torch.nn as nn
from torchvision import models


class StrokeHead(nn.Module):
    """Two-layer 256-unit classifier head for binary (or N-class) stroke logits."""

    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


def get_model(model_name: str, pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Build a torchvision backbone with the custom StrokeHead (256-256) classifier.
    """
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = StrokeHead(model.fc.in_features, num_classes)
        return model

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = StrokeHead(model.fc.in_features, num_classes)
        return model

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = StrokeHead(model.classifier.in_features, num_classes)
        return model

    elif model_name == "densenet201":
        model = models.densenet201(pretrained=pretrained)
        model.classifier = StrokeHead(model.classifier.in_features, num_classes)
        return model

    elif model_name == "efficientnetb0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = StrokeHead(model.classifier[1].in_features, num_classes)
        return model

    elif model_name == "efficientnetb3":
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier[1] = StrokeHead(model.classifier[1].in_features, num_classes)
        return model

    elif model_name == "inceptionv3":
        model = models.inception_v3(pretrained=pretrained, aux_logits=False)
        model.fc = StrokeHead(model.fc.in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")
