"""A command line demo for a trained CNN model with ASCII art."""

import enum
import itertools
import logging
import os
from dataclasses import dataclass
from subprocess import call

import torch
import torch.distributed.checkpoint as dcp
import tyro

import data.mnist_data as mnist_data
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
    if not os.path.exists(config.ckpt):
        _LOGGER.error("Model checkpoint not found: %s", config.ckpt)
        return -1

    if os.path.isfile(config.ckpt):
        state_dict = torch.load(config.ckpt, weights_only=True)
        model.load_state_dict(state_dict)
    elif os.path.isdir(config.ckpt):
        state_dict = {"model": model.state_dict()}
        fs_reader = dcp.FileSystemReader(config.ckpt)
        dcp.load(state_dict, storage_reader=fs_reader)
        state_dict = state_dict["model"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        _LOGGER.error("Error loading model: %s", e)
        _LOGGER.error("Model state dict keys: %s", [k for k in state_dict.keys()])
        _LOGGER.error("Model keys: %s", [k for k in model.state_dict().keys()])
        return -1

    _LOGGER.info("Model loaded from %s", config.ckpt)

    # Load the MNIST data set
    _, mnist_test = mnist_data.load_mnist(config)

    num_images = len(mnist_test)
    for idx, (data, target) in enumerate(itertools.chain(mnist_test)):
        output = model(data.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True)

        if pred.item() == target and config.only_errors:
            continue

        print(f"{mnist_data.to_ascii_art(data.squeeze(0))}\n")

        print(f"Image: {idx+1}/{num_images}")
        input()
        print(f"Predicted: {pred.item()}")

        input()
        print(f"Actual: {target}")

        input()
        call("clear" if os.name == "posix" else "cls")


if __name__ == "__main__":
    config = tyro.cli(DemoConfig)
    logging.basicConfig(level=config.log_level)
    main(config)
