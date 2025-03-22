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


def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the loss between the predicted and true values."""
    return torch.mean((y_pred - y_true) ** 2)


def train_model(model: torch.nn.Module, data: tuple[torch.Tensor, torch.Tensor], num_epochs: int = 5):
    """Train the model."""
    optimizer = torch.optim.AdamW(model.parameters())
    for epoch in range(num_epochs):
        celsius, fahrenheit = data
        for i, (c, f) in enumerate(zip(celsius, fahrenheit, strict=True)):
            logger.info("Epoch %d. Sample %d. Celsius %f. Fahrenheit %f.", epoch, i, c, f)
            logger.info("Model Fahrenheit prediction: %f.", model(torch.tensor([c])))
            loss = loss_fn(model(torch.tensor([c])), torch.tensor([f]))
            logger.info("Loss: %f.", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class LinearRegression(torch.nn.Module):
    """
    A simple linear regression model implemented using PyTorch.

    This class implements a single-input, single-output linear regression model
    of the form y = wx + b, where w and b are learnable parameters.
    """

    def __init__(self):
        """
        Initialize the linear regression model.

        Creates a single linear layer with input and output dimensions of 1.
        """
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)

        """
        return self.linear(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = generate_training_data(1000)
    model = LinearRegression()
    train_model(model, data, num_epochs=10)
