"""CNN model for character recognition."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True, frozen=True)
class CNNConfig:
    """Configuration for the CNN model."""

    conv1_channels: int = 32
    conv2_channels: int = 64
    dropout1: float = 0.25
    dropout2: float = 0.5
    linear: int = 128
    input_size: int = 28


class Net(nn.Module):
    """
    CNN model for MNIST character recognition.

    Input is a 28x28 image, output is a log-probability for each of the 10 classes.
    """

    def __init__(self, config: CNNConfig | None = None):
        """Initialize the model with the given configuration."""
        config = config or CNNConfig()
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.conv1_channels, 3, 1)
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, 3, 1)
        self.dropout1 = nn.Dropout(config.dropout1)
        self.dropout2 = nn.Dropout(config.dropout2)

        # Input size, minus 2 layers of padding from Conv layers, divided by 2 from max pooling.
        linear_input_size = pow((config.input_size - 4) // 2, 2) * config.conv2_channels

        self.fc1 = nn.Linear(linear_input_size, config.linear)
        self.fc2 = nn.Linear(config.linear, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
