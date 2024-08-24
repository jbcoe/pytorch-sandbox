"""MNIST data set."""

from torchvision import datasets, transforms  # type: ignore
from typing import Protocol


class MNISTConfig(Protocol):
    """Configuration for the MNIST data set."""

    data_dir: str


def load_mnist(config: MNISTConfig) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST data set."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(config.data_dir, train=False, transform=transform)

    return train, test
