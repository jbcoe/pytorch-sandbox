"""
Play with DataLoaders and DataPipes from pytorch.

Written using https://pytorch.org/data/beta/torchdata.datapipes.iter.html#
"""

import argparse
import enum
import logging
import os
from dataclasses import dataclass
from subprocess import call

import torch
from torch.utils.data.datapipes.iter import IterableWrapper
from torchvision import datasets, transforms

from xyz.pytorch.sandbox.mnist.data import to_ascii_art

_LOGGER = logging.getLogger(__name__)


class LogLevel(enum.IntEnum):
    """Log levels for the logger as an enum."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class Config:
    """Configuration for main."""

    log_level: LogLevel = LogLevel.INFO
    data_dir: str = ".DATA"


class IsValue:
    """Filter data by a specific value. Avoids lambda functions."""

    def __init__(self, value: int) -> None:
        """Initialize the filter."""
        self.value = value

    def __call__(self, x: tuple[torch.Tensor, int]) -> bool:
        """Return True if the value is the same as the filter value."""
        return x[1] == self.value


class IsPrime:
    """Filter data. Avoids lambda functions."""

    def __call__(self, x: tuple[torch.Tensor, int]) -> bool:
        """Return True if the value is prime."""
        assert x[1] >= 0 and x[1] < 10, f"Value {x[1]} must be in [0, 10)."
        return x[1] in (2, 3, 5, 7)


def main() -> None:
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="Data loading demo.")
    parser.add_argument("--data_dir", type=str, default=".DATA", help="Directory to store data")
    args = parser.parse_args()

    config = Config(log_level=LogLevel.INFO, data_dir=args.data_dir)
    logging.basicConfig(level=config.log_level)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train = datasets.MNIST(
        config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    datapipes = IterableWrapper(train).fork(2)

    for data, _ in datapipes[0].filter(IsPrime()):
        call("clear" if os.name == "posix" else "cls")
        print(to_ascii_art(data[0]), "\n")
        input()


if __name__ == "__main__":
    main()
