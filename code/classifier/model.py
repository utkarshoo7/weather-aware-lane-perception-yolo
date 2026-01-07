"""
WeatherNet Architecture Definition
----------------------------------
Defines the CNN architecture used for weather classification.

This module contains ONLY the model definition.
Weights are loaded separately during inference or training.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class WeatherNet(nn.Module):
    """
    Weather classification network based on ResNet-18.

    Output classes (fixed order, must match training):
        [clear, rainy, snowy, overcast, night]
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()

        # Backbone: ResNet-18 without ImageNet weights
        self.backbone = models.resnet18(weights=None)

        # Replace final classification layer
