"""MNIST data set."""

from typing import Protocol

import torch
from torchvision import datasets, transforms  # type: ignore


class MNISTConfig(Protocol):
    """Configuration for the MNIST data set."""

    @property
    def data_dir(self) -> str:
        """Data directory for downloaded MNIST data"""
        pass


def load_mnist(config: MNISTConfig) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST data set."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    train = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(config.data_dir, train=False, transform=transform)

    return train, test


def to_ascii_art(data: torch.Tensor, skip_blank_lines: bool = True) -> str:
    """Convert a 2-D tensor to ASCII art."""
    assert data.dim() == 2, "Data must be 2-D"
    assert data.size(0) == 28 and data.size(1) == 28, "Data must be <= 28x28"

    GREYSCALE = " .:-=+*#%@"

    ascii_data = data.clone()
    ascii_data = ascii_data - ascii_data.min()
    ascii_data = ascii_data / ascii_data.max()
    ascii_data = ascii_data * (len(GREYSCALE) - 1)
    ascii_data = ascii_data.int()

    ascii_art_lines = []
    for row in ascii_data:
        ascii_row_chars = []
        for col in row:
            ascii_row_chars.append(GREYSCALE[col.item()])
        if not skip_blank_lines or any(c != " " for c in ascii_row_chars):
            ascii_art_lines.append("".join(ascii_row_chars))
    return "\n".join(ascii_art_lines)
