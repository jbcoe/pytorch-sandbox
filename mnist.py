"""
MNIST character recognition following https://github.com/pytorch/examples/blob/main/mnist/main.py.

Uses MNIST dataset to train a simple CNN model for character recognition.

Command line arguments are parsed using the wonderful tyro package.

For usage, run `python mnist.py --help`.
"""

import dataclasses
import datetime
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms  # type: ignore
import logging
from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import tyro
from typing import Literal
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

device = torch.device("mps")


@dataclasses.dataclass
class CNNConfig:
    """Configuration for the CNN model."""

    conv1_channels: int = 32
    conv2_channels: int = 64
    dropout1: float = 0.25
    dropout2: float = 0.5
    linear: int = 128
    input_size: int = 28


class Net(nn.Module):
    """CNN model for MNIST character recognition.

    Input is a 28x28 image, output is a log-probability for each of the 10 classes.
    """

    def __init__(self, config: CNNConfig | None = None):
        """Initialize the model with the given configuration."""
        config = config or CNNConfig()
        super().__init__()
        self.conv1 = nn.Conv2d(1, config.conv1_channels, 3, 1)
        self.conv2 = nn.Conv2d(32, config.conv2_channels, 3, 1)
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


def train(
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    training_data_len: int,
):
    """Train the model for one epoch."""
    model.train()
    model.to(device)

    _LOGGER.info(f"Train Epoch: {epoch}")
    unlogged_data_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        unlogged_data_count += data.size(0)
        if unlogged_data_count >= training_data_len / 10:
            _LOGGER.info(
                "Train Epoch: {} [{:>5}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    training_data_len,
                    100.0 * batch_idx * len(data) / training_data_len,
                    loss.item(),
                )
            )
            unlogged_data_count = 0


def test(model, device, test_loader) -> float:
    """Test the model on the test data."""
    model.eval()
    model.to(device)

    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    _LOGGER.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy


class LogLevel(enum.IntEnum):
    """Log levels for the logger as an enum."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclasses.dataclass
class CompileConfig:
    """Configuration for model compilation."""

    fullgraph: bool = False
    mode: str = "reduce-overhead"
    backend: str = "inductor"


@dataclasses.dataclass
class Config:
    """Configuration for training and evaluating the model."""

    learning_rate: float = 0.01
    seed: int = 42
    epochs: int = 5
    gamma: float = 0.7
    batch_size: int = 8
    data_step: int = 1
    num_workers: int = 0
    device: Literal["cpu", "mps"] = "cpu"
    ckpt: str = ""
    data_dir: str = "data"
    training_data_fraction: float = 1.0
    log_level: LogLevel = LogLevel.INFO
    compile: CompileConfig | None = None
    cnn_config: CNNConfig = dataclasses.field(default_factory=CNNConfig)


def create_data_loaders(config: Config):
    """Load MNIST data and return training and test data loaders."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(config.data_dir, train=False, transform=transform)
    training_data_len = int(config.training_data_fraction * len(dataset1))
    train_loader = StatefulDataLoader(
        dataset1,
        sampler=torch.utils.data.SubsetRandomSampler(range(0, training_data_len)),
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    test_loader = StatefulDataLoader(dataset2, batch_size=config.batch_size)
    return train_loader, training_data_len, test_loader


def create_model_and_optimizer(config: Config):
    """Create the model and optimizer using the given config."""
    model = Net(config=config.cnn_config)
    optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)
    if config.compile:
        model = torch.compile(
            model,
            mode=config.compile.mode,
            fullgraph=config.compile.fullgraph,
        )  # type: ignore
    return model, optimizer


def main(config: Config):
    """Training and evaluation loop."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, training_data_len, test_loader = create_data_loaders(config)

    model, optimizer = create_model_and_optimizer(config)

    now = int(datetime.datetime.now(datetime.UTC).timestamp())

    for epoch in range(1, config.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, training_data_len)
        test(model, device, test_loader)
        if config.ckpt:
            torch.save(model.state_dict(), Path(config.ckpt) / f"mnist_{now}_e{epoch}.pt")


if __name__ == "__main__":
    config = tyro.cli(Config)
    logging.basicConfig(level=config.log_level)
    raise SystemExit(main(config))
