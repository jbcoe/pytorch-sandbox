"""A command line demo for a trained CNN model with ASCII art."""

import enum
import itertools
import logging
import os
from dataclasses import dataclass
from subprocess import call

import torch
import tyro

import data.mnist as mnist_data
import model.cnn as cnn

_LOGGER = logging.getLogger(__name__)


class LogLevel(enum.IntEnum):
    """Log levels for the logger as an enum."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class DemoConfig:
    """Configuration for the demo."""

    ckpt: str
    log_level: LogLevel = LogLevel.WARNING
    data_dir: str = ".DATA"
    only_errors: bool = False


def main(config: DemoConfig):
    """Main function for the demo."""
    config = tyro.cli(DemoConfig)
    logging.basicConfig(level=config.log_level)
    model = cnn.Net()
    state_dict = torch.load(config.ckpt, weights_only=True)
    model.load_state_dict(state_dict)

    _LOGGER.info("Model loaded from %s", config.ckpt)

    # Load the MNIST data set
    _, mnist_test = mnist_data.load_mnist(config)

    for data, target in itertools.chain(mnist_test):

        output = model(data.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True)

        if pred.item() == target and config.only_errors:
            continue

        print(f"{mnist_data.to_ascii_art(data.squeeze(0))}")
        print()

        input()
        print(f"Predicted: {pred.item()}")
        print()

        input()
        print(f"Actual: {target}")
        print()

        input()
        call("clear" if os.name == "posix" else "cls")


if __name__ == "__main__":
    config = tyro.cli(DemoConfig)
    logging.basicConfig(level=config.log_level)
    main(config)
