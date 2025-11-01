"""
MNIST character recognition following https://github.com/pytorch/examples/blob/main/mnist/main.py.

Uses MNIST dataset to train a simple CNN model for character recognition.

For usage, run `python mnist.py --help`.
"""

import contextlib
import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import xyz.pytorch.sandbox.mnist.model.cnn as cnn
from xyz.pytorch.sandbox.mnist.train.checkpoint import maybe_save_model_state
from xyz.pytorch.sandbox.mnist.train.config import (
    Config,
    DDPConfig,
    FSDPConfig,
    LogLevel,
    args_to_config,
    create_arg_parser,
)
from xyz.pytorch.sandbox.mnist.train.data import create_data_loaders

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
    reference_model: torch.nn.Module | None = None,
) -> int:
    """Train the model for one epoch."""
    model.train()
    model.to(device)

    if reference_model:
        reference_model.eval()
        reference_model.to(device)

    data_len: int = (
        len(train_loader.sampler)  # type: ignore[arg-type]
        if train_loader.sampler
        else len(train_loader.dataset)  # type: ignore[arg-type]
    )
    batch_size: int = train_loader.batch_size or 1

    _LOGGER.info(f"Train Epoch: {epoch}")
    unlogged_steps = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if reference_model:
            # We need to convert model output (Negative Log Likelyhood) into probabilities.
            probabilities = torch.exp(output)
            with torch.no_grad():
                reference_output = reference_model(data)
                reference_probabilities = torch.exp(reference_output)
            loss = F.cross_entropy(probabilities, reference_probabilities)
        else:
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

    data_len = len(test_loader.sampler) if test_loader.sampler else len(test_loader.dataset)

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


def _maybe_load_reference_model(config: Config) -> torch.nn.Module | None:
    """Load a reference model from the specified checkpoint if requested."""
    if not config.reference_model_ckpt:
        return None

    reference_model: torch.nn.Module = cnn.Net()
    state_dict = torch.load(config.reference_model_ckpt, weights_only=True)
    reference_model.load_state_dict(state_dict)

    match config.parallel:
        case None:
            pass
        case DDPConfig():
            reference_model = DDP(reference_model)
        case FSDPConfig():
            reference_model = FSDP(
                reference_model,
                device_id=torch.device(config.device),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=CustomPolicy(lambda _: True),
            )
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")

    if config.compile:
        reference_model = torch.compile(
            reference_model,
            # mode=config.compile.mode,
            backend=config.compile.backend,
            fullgraph=config.compile.fullgraph,
        )  # type: ignore
    return reference_model


def _create_model_and_optimizer(config: Config) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Create the model and optimizer using the given config."""
    model: torch.nn.Module = cnn.Net()  # config=config.cnn_config)

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


def _configure_logging(log_level: LogLevel, rank: int | None = None) -> None:
    """Configure logging for the application."""
    if rank is not None:
        format = f"%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | {rank} | %(message)s"
    else:
        format = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"
    logging.basicConfig(level=log_level, format=format)


def _multiprocess_main(rank: int, config: Config) -> None:
    """Entry point for DDP and FSDP parallelism. Manages process group initialization and cleanup."""
    assert isinstance(config.parallel, (DDPConfig, FSDPConfig)), f"Invalid parallel config {config.parallel}"

    _configure_logging(config.log_level, rank)

    os.environ["MASTER_ADDR"] = config.parallel.hostname
    os.environ["MASTER_PORT"] = config.parallel.port

    try:
        dist.init_process_group(backend="gloo", rank=rank, world_size=config.parallel.world_size)
        _single_process_main(rank, config)
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args(argv)
    config = args_to_config(args)

    match config.parallel:
        case None:
            _configure_logging(config.log_level)
            _single_process_main(0, config)
        case DDPConfig() | FSDPConfig():
            torch.multiprocessing.spawn(_multiprocess_main, args=(config,), nprocs=config.parallel.world_size)
        case _:
            raise NotImplementedError(f"Parallelism kind {config.parallel} not implemented")


def _single_process_main(rank: int, config: Config) -> None:
    """Single process training and evaluation loop."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_loader, test_loader = create_data_loaders(rank, config)

    model, optimizer = _create_model_and_optimizer(config)
    reference_model = _maybe_load_reference_model(config)

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
            os.makedirs(config.ckpt, exist_ok=True)
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
                reference_model=reference_model,
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
    main()
