"""A minimal ML model using PyTorch to learn the relationship between Celsius and Fahrenheit."""

import logging

import torch

logger = logging.getLogger(__name__)


def fahrenheit_to_celsius(fahrenheit: torch.Tensor) -> torch.Tensor:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * (5.0 / 9.0)


def Celsius_to_fahrenheit(Celsius: torch.Tensor) -> torch.Tensor:
    """Convert Celsius to Fahrenheit."""
    return (Celsius * (9.0 / 5.0)) + 32.0


def generate_training_data(num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate training data for the model."""
    celsius = torch.rand(num_samples) * 100.0
    # Convert each element in the tensor individually
    fahrenheit = Celsius_to_fahrenheit(celsius)
    return celsius, fahrenheit


def train_model(data: tuple[torch.Tensor, torch.Tensor], num_epochs: int = 5):
    """Train the model."""
    for epoch in range(num_epochs):
        # Iterate over the data.
        celsius, fahrenheit = data
        for i, (c, f) in enumerate(zip(celsius, fahrenheit, strict=True)):
            logger.info("Epoch %d. Sample %d. Celsius %f. Fahrenheit %f.", epoch, i, c, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = generate_training_data(10)
    train_model(data)
