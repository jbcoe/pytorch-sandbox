"""
MNIST character recognition following https://github.com/pytorch/examples/blob/main/mnist/main.py.

Uses MNIST dataset to train a simple CNN model for character recognition.

For usage, run `python mnist.py --help`.
"""

import argparse
import contextlib
import dataclasses
import datetime
import enum
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlflow
import torch
import torch.distributed
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict
import torch.distributed.fsdp
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader  # type: ignore

import xyz.pytorch.sandbox.data.mnist_data as mnist_data
import xyz.pytorch.sandbox.model.cnn as cnn

_LOGGER = logging.getLogger(__name__)


def train(
    *,
    rank: int,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: Optimizer,
    epoch: int,
    global_step: int,
    verbose: bool = False,
    mlflow_run: mlflow.ActiveRun | None = None,
) -> int:
    """Train the model for one epoch."""
    model.train()
    model.to(device)

    data_len: int = (
        len(train_loader.sampler)  # type: ignore[arg-type]
        if train_loader.sampler
        else len(train_loader.dataset)  # type: ignore[arg-type]
    )
    batch_size: int = train_loader.batch_size if train_loader.batch_size is not None else 1

    _LOGGER.info(f"Train Epoch: {epoch}")
    unlogged_steps = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if mlflow_run and rank == 0:
            mlflow.log_metric("train_loss", loss.item(), step=batch_idx)

        global_step += 1

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
    return global_step


@torch.no_grad()
def test(*, rank: int, model, device, test_loader, aggregate_test_results=False) -> float:
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


@dataclass(frozen=True)
class LocalParallelConfig:
    """Configuration for local parallelism."""

    world_size: int = 4
    hostname: str = "localhost"
    port: str = "12345"
    aggregate_test_results: bool = True


@dataclass(frozen=True)
class DDPConfig(LocalParallelConfig):
    """Configuration for DDP parallelism."""

    pass


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CompileConfig:
    """Configuration for model compilation."""

    fullgraph: bool = False
    mode: str = "reduce-overhead"
    # backend "inductor" is torch's default but does not work on MPS.
    # https://discuss.pytorch.org/t/torch-compile-seems-to-hang/177089/4
    backend: str = "aot_eager"


@dataclass(frozen=True)
class MLFlowConfig:
    """Configuration for MLFlow logging."""

    experiment_name: str = "mnist"
    tracking_uri: str = ".MLFLOW"
    run_name: str | None = None
    log_system_metrics: bool = True


@dataclass(frozen=True)
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
    mlflow: MLFlowConfig | None = None


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
    model: torch.nn.Module | FSDP | DDP = cnn.Net()  # config=config.cnn_config)

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
    *,
    model: torch.nn.Module,
    config: Config,
    rank: int,
    now: int,
    epoch: int,
    mlflow_run: mlflow.ActiveRun | None = None,
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
                assert isinstance(model, DDP)
                model_state_dict = model.module.state_dict()
                torch.save(model_state_dict, checkpoint_filepath)
        case FSDPConfig():
            assert isinstance(model, FSDP)
            model_state_dict = dcp.state_dict.get_model_state_dict(model)
            dcp.save(
                state_dict={"model": model_state_dict},
                storage_writer=dcp.FileSystemWriter(checkpoint_filepath),
            )
            dist.barrier()
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")
    _LOGGER.info("Saved model state at epoch %d to %s", epoch, checkpoint_filepath)
    if mlflow_run and not config.parallel:
        mlflow.log_artifact(str(checkpoint_filepath))


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


def create_arg_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="MNIST character recognition training")

    # Basic training arguments
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate decay factor")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--data-step", type=int, default=1, help="Step size for data loading")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu", help="Device to use for training")
    parser.add_argument("--ckpt", default=".CKPT", help="Checkpoint directory")
    parser.add_argument("--data-dir", default=".DATA", help="Data directory")
    parser.add_argument("--training-data-fraction", type=float, default=1.0, help="Fraction of training data to use")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Disable data shuffling")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # CNN Config arguments
    cnn_group = parser.add_argument_group("CNN configuration")
    cnn_group.add_argument("--cnn-channels1", type=int, default=32, help="Number of channels in first conv layer")
    cnn_group.add_argument("--cnn-channels2", type=int, default=64, help="Number of channels in second conv layer")
    cnn_group.add_argument("--cnn-fc1-size", type=int, default=1024, help="Size of first fully connected layer")
    cnn_group.add_argument("--cnn-dropout", type=float, default=0.25, help="Dropout probability")

    # Parallel processing arguments
    parallel_group = parser.add_argument_group("Parallel processing")
    parallel_group.add_argument(
        "--parallel-type", choices=["ddp", "fsdp", "none"], default="none", help="Type of parallelism to use"
    )
    parallel_group.add_argument("--world-size", type=int, default=4, help="Number of parallel processes")
    parallel_group.add_argument("--hostname", default="localhost", help="Host for parallel processing")
    parallel_group.add_argument("--port", default="12345", help="Port for parallel processing")
    parallel_group.add_argument(
        "--no-aggregate-test-results",
        action="store_false",
        dest="aggregate_test_results",
        help="Disable test results aggregation",
    )

    # Compile arguments
    compile_group = parser.add_argument_group("Model compilation")
    compile_group.add_argument("--compile", action="store_true", help="Enable model compilation")
    compile_group.add_argument("--compile-fullgraph", action="store_true", help="Enable full graph compilation")
    compile_group.add_argument("--compile-mode", default="reduce-overhead", help="Compilation mode")
    compile_group.add_argument("--compile-backend", default="aot_eager", help="Compilation backend")

    # MLFlow arguments
    mlflow_group = parser.add_argument_group("MLFlow configuration")
    mlflow_group.add_argument("--mlflow-tracking-uri", default=None, help="MLFlow tracking URI")
    mlflow_group.add_argument("--mlflow-experiment", default="mnist", help="MLFlow experiment name")
    mlflow_group.add_argument("--mlflow-run-name", help="MLFlow run name (default: timestamp)")
    mlflow_group.add_argument(
        "--mlflow-log-system-metrics", action="store_true", default=True, help="Log system metrics"
    )
    return parser


def args_to_config(args):
    """Convert parsed arguments to Config object."""
    # Create CNN Config
    cnn_config = cnn.CNNConfig(
        conv1_channels=args.cnn_channels1,
        conv2_channels=args.cnn_channels2,
        dropout1=args.cnn_dropout,
        dropout2=args.cnn_dropout,
        linear=args.cnn_fc1_size,
        input_size=28,  # MNIST image size
    )

    # Create Parallel Config
    parallel_config: DDPConfig | FSDPConfig | None = None
    if args.parallel_type != "none":
        parallel_base = {
            "world_size": args.world_size,
            "hostname": args.hostname,
            "port": args.port,
            "aggregate_test_results": args.aggregate_test_results,
        }
        if args.parallel_type == "ddp":
            parallel_config = DDPConfig(**parallel_base)
        elif args.parallel_type == "fsdp":
            parallel_config = FSDPConfig(**parallel_base)

    # Create Compile Config
    compile_config = None
    if args.compile:
        compile_config = CompileConfig(
            fullgraph=args.compile_fullgraph, mode=args.compile_mode, backend=args.compile_backend
        )

    # Create MLFlow Config
    mlflow_config = None
    if args.mlflow_tracking_uri:
        mlflow_config = MLFlowConfig(
            experiment_name=args.mlflow_experiment,
            tracking_uri=args.mlflow_tracking_uri,
            run_name=args.mlflow_run_name,
            log_system_metrics=args.mlflow_log_system_metrics,
        )

    return Config(
        learning_rate=args.learning_rate,
        seed=args.seed,
        epochs=args.epochs,
        gamma=args.gamma,
        batch_size=args.batch_size,
        data_step=args.data_step,
        num_workers=args.num_workers,
        device=args.device,
        ckpt=args.ckpt,
        data_dir=args.data_dir,
        training_data_fraction=args.training_data_fraction,
        shuffle=args.shuffle,
        log_level=LogLevel[args.log_level],
        cnn_config=cnn_config,
        parallel=parallel_config,
        compile=compile_config,
        verbose=args.verbose,
        mlflow=mlflow_config,
    )


def main(args=None):
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args(args)
    config = args_to_config(args)

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

    with contextlib.ExitStack() as stack:
        # Initialize MLFlow run if enabled.
        mlflow_run = None
        if config.mlflow and rank == 0:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
            mlflow.set_experiment(config.mlflow.experiment_name)
            mlflow_run = stack.enter_context(
                mlflow.start_run(
                    run_name=config.mlflow.run_name or f"mnist_{now}",
                    log_system_metrics=config.mlflow.log_system_metrics,
                )
            )
            # Log hyperparameters
            mlflow.log_params(dataclasses.asdict(config))

        # Save config.
        if rank == 0 and config.ckpt:
            config_path = Path(config.ckpt) / f"mnist_{now}_config.txt"
            with open(config_path, "w") as f:
                f.write(json.dumps(dataclasses.asdict(config)))
            if mlflow_run:
                mlflow.log_artifact(str(config_path))

        # Save initial model state.
        maybe_save_model_state(model=model, config=config, rank=rank, now=now, epoch=0)

        global_step = 0

        for epoch in range(1, config.epochs + 1):
            global_step = train(
                rank=rank,
                model=model,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                verbose=config.verbose,
                mlflow_run=mlflow_run,
            )
            if config.parallel:
                dist.barrier()
            test(
                rank=rank,
                model=model,
                device=device,
                test_loader=test_loader,
                aggregate_test_results=config.parallel and config.parallel.aggregate_test_results,
            )
            maybe_save_model_state(
                model=model,
                config=config,
                rank=rank,
                now=now,
                epoch=epoch,
                mlflow_run=mlflow_run,
            )


if __name__ == "__main__":
    raise SystemExit(main())
