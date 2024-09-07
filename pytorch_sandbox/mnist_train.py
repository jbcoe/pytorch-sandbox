"""
MNIST character recognition following https://github.com/pytorch/examples/blob/main/mnist/main.py.

Uses MNIST dataset to train a simple CNN model for character recognition.

Command line arguments are parsed using the wonderful tyro package.

For usage, run `python mnist.py --help`.
"""

import dataclasses
import datetime
import enum
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.distributed
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict
import torch.distributed.fsdp
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore

import pytorch_sandbox.data.mnist_data as mnist_data
import pytorch_sandbox.model.cnn as cnn

_LOGGER = logging.getLogger(__name__)


def train(
    rank: int,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    verbose: bool = False,
):
    """Train the model for one epoch."""
    model.train()
    model.to(device)

    data_len = len(train_loader.sampler) if train_loader.sampler else len(train_loader.dataset)  # type: ignore
    batch_size = train_loader.batch_size or 1

    _LOGGER.info(f"Train Epoch: {epoch}")
    unlogged_steps = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        unlogged_steps += 1
        if verbose or unlogged_steps >= len(train_loader) / 10:
            unlogged_steps = 0
            _LOGGER.info(
                "Train Epoch: {} [{:>5}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * batch_size,
                    data_len,  # type: ignore
                    100.0 * batch_idx * batch_size / data_len,
                    loss.item(),
                )
            )


@torch.no_grad()
def test(rank: int, model, device, test_loader, aggregate_test_results=False) -> float:
    """Test the model on the test data."""
    model.eval()
    model.to(device)

    data_len = len(test_loader.sampler) if test_loader.sampler else len(test_loader.dataset)  # type: ignore

    test_loss = 0.0
    correct = 0.0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= data_len

    _LOGGER.debug(f"Data Length: {data_len} Correct: {correct}")

    if aggregate_test_results:
        all_test_loss = torch.tensor([test_loss], device=device)
        dist.all_reduce(all_test_loss, op=dist.ReduceOp.SUM)
        test_loss = all_test_loss.item()

        all_data_len = torch.tensor([data_len], device=device, dtype=torch.int)
        dist.all_reduce(all_data_len, op=dist.ReduceOp.SUM)
        data_len = all_data_len.item()  # type: ignore

        all_correct = torch.tensor([correct], device=device)
        dist.all_reduce(all_correct, op=dist.ReduceOp.SUM)
        correct = all_correct.item()

    accuracy = 100.0 * correct / data_len
    if rank == 0 or not aggregate_test_results:
        _LOGGER.info("Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(test_loss, correct, data_len, accuracy))
    return accuracy


@dataclass(slots=True, frozen=True)
class LocalParallelConfig:
    """Configuration for local parallelism."""

    world_size: int = 4
    hostname: str = "localhost"
    port: str = "12345"
    aggregate_test_results: bool = True


@dataclass(slots=True, frozen=True)
class DDPConfig(LocalParallelConfig):
    """Configuration for DDP parallelism."""

    pass


@dataclass(slots=True, frozen=True)
class FSDPConfig(LocalParallelConfig):
    """Configuration for FSDP parallelism."""

    pass


class LogLevel(enum.IntEnum):
    """Log levels for the logger as an enum."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass(slots=True, frozen=True)
class CompileConfig:
    """Configuration for model compilation."""

    fullgraph: bool = False
    mode: str = "reduce-overhead"
    # backend "inductor" is torch's default but does not work on MPS.
    # https://discuss.pytorch.org/t/torch-compile-seems-to-hang/177089/4
    backend: str = "aot_eager"


@dataclass(slots=True, frozen=True)
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
    ckpt: str = ".CKPT"  # TODO: Avoid hardcoding local paths.
    data_dir: str = ".DATA"  # TODO: Avoid hardcoding local paths.
    training_data_fraction: float = 1.0  # TODO: Implement fractional training data.
    shuffle: bool = True
    log_level: LogLevel = LogLevel.INFO
    cnn_config: cnn.CNNConfig = field(default_factory=cnn.CNNConfig)
    parallel: DDPConfig | FSDPConfig | None = None
    compile: CompileConfig | None = None
    verbose: bool = False


def create_data_loaders(rank: int, config: Config) -> tuple[DataLoader, DataLoader]:
    """Load MNIST data and return training and test data loaders."""
    mnist_train, mnist_test = mnist_data.load_mnist(config)

    assert config.training_data_fraction == 1.0, "Fractional training data not implemented"

    match config.parallel:
        case None:
            train_sampler, test_sampler = None, None
        case DDPConfig() | FSDPConfig():
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
    return train_loader, test_loader


def create_model_and_optimizer(config: Config):
    """Create the model and optimizer using the given config."""
    model: torch.nn.Module | FSDP | DDP = cnn.Net(config=config.cnn_config)

    match config.parallel:
        case None:
            pass
        case DDPConfig():
            model = DDP(model)
        case FSDPConfig():
            model = FSDP(
                model,
                device_id=torch.device(config.device),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=CustomPolicy(lambda _: True),
            )
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")

    optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)

    if config.compile:
        model = torch.compile(
            model,
            # mode=config.compile.mode,
            backend=config.compile.backend,
            fullgraph=config.compile.fullgraph,
        )  # type: ignore
    return model, optimizer


def maybe_save_model_state(
    model: torch.nn.Module,
    config: Config,
    rank: int,
    now: int,
    epoch: int,
):
    """Save the model state if a checkpoint directory is provided."""
    if not config.ckpt:
        return

    checkpoint_filepath = Path(config.ckpt) / f"mnist_{now}_e{epoch}.pt"
    match config.parallel:
        case None:
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, checkpoint_filepath)
        case DDPConfig():
            # All processes should see the same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            # DDP has model state dict in model.module.
            if rank == 0:
                model_state_dict = model.module.state_dict()
                torch.save(model_state_dict, checkpoint_filepath)
        case FSDPConfig():
            model_state_dict = dcp.state_dict.get_model_state_dict(model)
            dcp.save(
                state_dict={"model": model_state_dict},
                storage_writer=dcp.FileSystemWriter(checkpoint_filepath),
            )
            dist.barrier()
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")
    _LOGGER.info("Saved model state at epoch %d to %s", epoch, checkpoint_filepath)


def _configure_logging(log_level: LogLevel, rank: int | None = None):
    """Configure logging for the application."""
    if rank is not None:
        format = f"%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | {rank} | %(message)s"
    else:
        format = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"
    logging.basicConfig(level=log_level, format=format)


def _multiprocess_main(rank: int, config: Config):
    """Entry point for DDP and FSDP parallelism. Manages process group initialization and cleanup."""
    assert isinstance(config.parallel, (DDPConfig, FSDPConfig)), f"Invalid parallel config {config.parallel}"

    _configure_logging(config.log_level, rank)

    os.environ["MASTER_ADDR"] = config.parallel.hostname
    os.environ["MASTER_PORT"] = config.parallel.port

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=config.parallel.world_size)
        _main(rank, config)
    finally:
        dist.destroy_process_group()


def main(args=None):
    """Main entry point."""
    config = tyro.cli(Config, args=args)

    match config.parallel:
        case None:
            _configure_logging(config.log_level)
            _main(0, config)
        case DDPConfig() | FSDPConfig():
            torch.multiprocessing.spawn(_multiprocess_main, args=(config,), nprocs=config.parallel.world_size)
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")


def _main(rank: int, config: Config):
    """Single process training and evaluation loop."""

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, test_loader = create_data_loaders(rank, config)

    model, optimizer = create_model_and_optimizer(config)

    now = int(datetime.datetime.now(datetime.UTC).timestamp())

    if rank == 0 and config.ckpt:
        with open(Path(config.ckpt) / f"mnist_{now}_config.txt", "w") as f:
            f.write(json.dumps(dataclasses.asdict(config)))

    maybe_save_model_state(model, config, rank, now, epoch=0)

    for epoch in range(1, config.epochs + 1):
        train(
            rank,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            verbose=config.verbose,
        )
        if config.parallel:
            dist.barrier()
        test(
            rank,
            model,
            device,
            test_loader,
            aggregate_test_results=config.parallel and config.parallel.aggregate_test_results,
        )
        maybe_save_model_state(model, config, rank, now, epoch)


if __name__ == "__main__":
    raise SystemExit(main())
