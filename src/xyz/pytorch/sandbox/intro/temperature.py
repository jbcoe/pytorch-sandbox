"""A minimal ML model using PyTorch to learn the relationship between Celsius and Fahrenheit."""

import argparse
import logging
import typing

import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fahrenheit_to_celsius(fahrenheit: torch.Tensor) -> torch.Tensor:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32.0) * (5.0 / 9.0)


def Celsius_to_fahrenheit(Celsius: torch.Tensor) -> torch.Tensor:
    """Convert Celsius to Fahrenheit."""
    return (Celsius * (9.0 / 5.0)) + 32.0


def generate_training_data(num_samples: int, noise_std: float = 0.0) -> torch.Tensor:
    """
    Generate training data for the model.

    Args:
        num_samples: Number of samples to generate
        noise_std: Standard deviation of Gaussian noise to add to fahrenheit values

    """
    celsius = torch.rand(num_samples) * 100.0
    fahrenheit = Celsius_to_fahrenheit(celsius)
    if noise_std > 0:
        noise = torch.randn_like(fahrenheit) * noise_std
        fahrenheit = fahrenheit + noise
    return torch.stack([celsius, fahrenheit], dim=1)


def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute the loss between the predicted and true values."""
    return torch.mean(torch.square(y_pred - y_true))


def train_model(
    model: torch.nn.Module,
    data: torch.Tensor,
    num_epochs: int = 5,
    learning_rate: float = 0.01,
) -> None:
    """Train the model."""
    assert data.ndim == 2, "Data must be a 2D tensor"
    assert data.shape[1] == 2, "Data must have two columns: Celsius and Fahrenheit"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (c, f) in enumerate(data):
            prediction = model(c.unsqueeze(0))
            logger.info("Epoch %d. Sample %d. Celsius %f. Fahrenheit %f.", epoch, i, c, f)
            logger.info("Model Fahrenheit prediction: %f.", prediction)
            loss = loss_fn(prediction, f.unsqueeze(0))
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

    def __init__(self) -> None:
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
        return typing.cast(torch.Tensor, self.linear(x))


def main() -> None:
    """Main entry point for the Celsius to Fahrenheit conversion model."""
    parser = argparse.ArgumentParser(
        description="Train a model to convert Celsius to Fahrenheit",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="number of training samples (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="learning rate for optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="standard deviation of noise to add (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)  # Set the random seed
    data = generate_training_data(args.samples, args.noise)
    model = LinearRegression()
    train_model(model, data, num_epochs=args.epochs, learning_rate=args.learning_rate)


if __name__ == "__main__":
    main()
