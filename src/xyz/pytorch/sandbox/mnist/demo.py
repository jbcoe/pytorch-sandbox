"""A command line demo for a trained CNN model with ASCII art."""

import argparse
import enum
import itertools
import logging
import os
from dataclasses import dataclass
from subprocess import call

import torch
import torch.distributed.checkpoint as dcp

import xyz.pytorch.sandbox.mnist.data as mnist_data
import xyz.pytorch.sandbox.mnist.model.cnn as cnn

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


def main(args=None):
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="A command line demo for a trained CNN model with ASCII art.")
    parser.add_argument("ckpt", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--log-level",
        type=lambda x: LogLevel[x.upper()],
        default=LogLevel.WARNING,
        choices=list(LogLevel),
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument("--data-dir", type=str, default=".DATA", help="Directory containing the data")
    parser.add_argument("--only-errors", action="store_true", help="Only show incorrect predictions")

    parsed_args = parser.parse_args(args)
    config = DemoConfig(
        ckpt=parsed_args.ckpt,
        log_level=parsed_args.log_level,
        data_dir=parsed_args.data_dir,
        only_errors=parsed_args.only_errors,
    )

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

        print(f"Image: {idx + 1}/{num_images}")
        input()
        print(f"Predicted: {pred.item()}")

        input()
        print(f"Actual: {target}")

        input()
        call("clear" if os.name == "posix" else "cls")


if __name__ == "__main__":
    raise SystemExit(main())
