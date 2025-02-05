"""
Play with DataLoaders and DataPipes from pytorch.

Written using https://pytorch.org/data/beta/torchdata.datapipes.iter.html#
"""

import enum
import logging
import os
from dataclasses import dataclass
from subprocess import call

import tyro
from torchdata.datapipes.iter import IterableWrapper
from torchvision import datasets, transforms  # type: ignore

from pytorch_sandbox.data.mnist_data import to_ascii_art

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

    def __init__(self, value):
        """Initialize the filter."""
        self.value = value

    def __call__(self, x):
        """Return True if the value is the same as the filter value."""
        return x[1] == self.value


class IsPrime:
    """Filter data. Avoids lambda functions."""

    def __call__(self, x):
        """Return True if the value is prime."""
        assert x[1] >= 0 and x[1] < 10, f"Value {x[1]} must be between in [0, 10)."
        return x[1] in (2, 3, 5, 7)


def main():
    """Main function for the demo."""
    config = tyro.cli(Config)
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

    for data, _ in datapipes[0].filter(IsPrime()).repeat(2):
        call("clear" if os.name == "posix" else "cls")
        print(to_ascii_art(data[0]), "\n")
        input()


if __name__ == "__main__":
    raise SystemExit(main())
