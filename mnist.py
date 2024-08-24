"""
MNIST character recognition following https://github.com/pytorch/examples/blob/main/mnist/main.py.

Uses MNIST dataset to train a simple CNN model for character recognition.

Command line arguments are parsed using the wonderful tyro package.

For usage, run `python mnist.py --help`.
"""

import dataclasses
import datetime
import enum
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore

import data.mnist as mnist_data
import model.cnn as cnn

_LOGGER = logging.getLogger(__name__)

device = torch.device("mps")


def train(
    rank: int,
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

    _LOGGER.info(f"{rank}: Train Epoch: {epoch}")
    unlogged_data_count = 0
    for batch_idx, (data, target) in zip(itertools.count(start=1), train_loader):
        data, target = data.to(device), target.to(device)

        # _LOGGER.debug(f"{rank}: Train Epoch: {epoch}, target: {target}")

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        unlogged_data_count += data.size(0)
        if unlogged_data_count >= training_data_len / 10:
            _LOGGER.info(
                "{}: Train Epoch: {} [{:>5}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    rank,
                    epoch,
                    batch_idx * len(data),
                    training_data_len,
                    100.0 * batch_idx * len(data) / training_data_len,
                    loss.item(),
                )
            )
            unlogged_data_count = 0


def test(rank: int, model, device, test_loader) -> float:
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
        "{}: Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            rank, test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy


@dataclasses.dataclass
class DDPConfig:
    """Configuration for DDP parallelism."""

    world_size: int = 4
    hostname: str = "localhost"
    port: str = "12345"


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
    ckpt: str = ".CKPT"
    data_dir: str = ".DATA"
    training_data_fraction: float = 1.0
    shuffle: bool = True
    log_level: LogLevel = LogLevel.INFO
    cnn_config: cnn.CNNConfig = dataclasses.field(default_factory=cnn.CNNConfig)
    parallel: DDPConfig | None = None
    compile: CompileConfig | None = None


def create_data_loaders(rank: int, config: Config) -> tuple[DataLoader, int, DataLoader]:
    """Load MNIST data and return training and test data loaders."""
    mnist_train, mnist_test = mnist_data.load_mnist(config)

    assert config.training_data_fraction == 1.0
    training_data_len = int(len(mnist_train) * config.training_data_fraction)

    match config.parallel:
        case None:
            train_sampler, test_sampler = None, None
        case DDPConfig():
            train_sampler = DistributedSampler(
                mnist_train,
                num_replicas=config.parallel.world_size,
                rank=rank,
                seed=config.seed,
                shuffle=config.shuffle,
            )
            test_sampler = DistributedSampler(
                mnist_test,
                num_replicas=config.parallel.world_size,
                rank=rank,
                seed=config.seed,
            )
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")

    train_loader = StatefulDataLoader(
        mnist_train,
        sampler=train_sampler,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    test_loader = StatefulDataLoader(mnist_test, sampler=test_sampler, batch_size=config.batch_size)
    return train_loader, training_data_len, test_loader


def create_model_and_optimizer(config: Config):
    """Create the model and optimizer using the given config."""
    model: Any = cnn.Net(config=config.cnn_config)
    optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)

    match config.parallel:
        case None:
            pass
        case DDPConfig():
            model = torch.nn.parallel.DistributedDataParallel(model)
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")

    if config.compile:
        model = torch.compile(
            model,
            mode=config.compile.mode,
            fullgraph=config.compile.fullgraph,
        )  # type: ignore
    return model, optimizer


def _ddp_main(rank: int, config: Config):
    """Entry point for DDP parallelism. Manages process group initialization and cleanup."""
    assert isinstance(config.parallel, DDPConfig)

    logging.basicConfig(level=config.log_level)

    os.environ["MASTER_ADDR"] = config.parallel.hostname
    os.environ["MASTER_PORT"] = config.parallel.port

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=config.parallel.world_size)
        _main(rank, config)
    finally:
        dist.destroy_process_group()


def main(config: Config):
    """Main entry point."""
    match config.parallel:
        case None:
            _main(0, config)
        case DDPConfig():
            torch.multiprocessing.spawn(_ddp_main, args=(config,), nprocs=config.parallel.world_size)
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")


def _main(rank: int, config: Config):
    """Single process training and evaluation loop."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, training_data_len, test_loader = create_data_loaders(rank, config)

    model, optimizer = create_model_and_optimizer(config)

    now = int(datetime.datetime.now(datetime.UTC).timestamp())

    for epoch in range(1, config.epochs + 1):
        train(rank, model, device, train_loader, optimizer, epoch, training_data_len)
        test(rank, model, device, test_loader)
        if rank == 0 and config.ckpt:
            # All processes should see the same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(model.state_dict(), Path(config.ckpt) / f"mnist_{now}_e{epoch}.pt")


if __name__ == "__main__":
    config = tyro.cli(Config)
    logging.basicConfig(level=config.log_level)
    raise SystemExit(main(config))
