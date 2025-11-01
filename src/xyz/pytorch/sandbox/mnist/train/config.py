"""Config for the MNIST trainer."""

import argparse
import enum
import logging
import os
from dataclasses import dataclass, field
from typing import Literal

from xyz.pytorch.sandbox.mnist.model import cnn


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
    reference_model_ckpt: str | None = None
    data_dir: str = ".DATA"  # TODO: Avoid hardcoding local paths.
    training_data_fraction: float = 1.0  # TODO: Implement fractional training data.
    shuffle: bool = True
    log_level: LogLevel = LogLevel.INFO
    cnn_config: cnn.CNNConfig = field(default_factory=cnn.CNNConfig)
    parallel: DDPConfig | FSDPConfig | None = None
    compile: CompileConfig | None = None
    verbose: bool = False
    mlflow: MLFlowConfig | None = None


def create_arg_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="MNIST character recognition training",
    )

    # Basic training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Learning rate decay factor",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--data-step",
        type=int,
        default=1,
        help="Step size for data loading",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--ckpt",
        default=".CKPT",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--reference_model_ckpt",
        default=None,
        help="Optional checkpoint for a reference model to distill.",
    )
    parser.add_argument(
        "--data-dir",
        default=".DATA",
        help="Data directory",
    )
    parser.add_argument(
        "--training-data-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to use",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_false",
        dest="shuffle",
        help="Disable data shuffling",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # CNN Config arguments
    cnn_group = parser.add_argument_group(
        "CNN configuration",
    )
    cnn_group.add_argument(
        "--cnn-channels1",
        type=int,
        default=32,
        help="Number of channels in first conv layer",
    )
    cnn_group.add_argument(
        "--cnn-channels2",
        type=int,
        default=64,
        help="Number of channels in second conv layer",
    )
    cnn_group.add_argument(
        "--cnn-fc1-size",
        type=int,
        default=1024,
        help="Size of first fully connected layer",
    )
    cnn_group.add_argument(
        "--cnn-dropout",
        type=float,
        default=0.25,
        help="Dropout probability",
    )

    # Parallel processing arguments
    parallel_group = parser.add_argument_group(
        "Parallel processing",
    )
    parallel_group.add_argument(
        "--parallel-type",
        choices=["ddp", "fsdp", "none"],
        default="none",
        help="Type of parallelism to use",
    )
    parallel_group.add_argument(
        "--world-size",
        type=int,
        default=4,
        help="Number of parallel processes",
    )
    parallel_group.add_argument(
        "--hostname",
        default="localhost",
        help="Host for parallel processing",
    )
    parallel_group.add_argument(
        "--port",
        default="12345",
        help="Port for parallel processing",
    )
    parallel_group.add_argument(
        "--no-aggregate-test-results",
        action="store_false",
        dest="aggregate_test_results",
        help="Disable test results aggregation",
    )

    # Compile arguments
    compile_group = parser.add_argument_group(
        "Model compilation",
    )
    compile_group.add_argument(
        "--compile",
        action="store_true",
        help="Enable model compilation",
    )
    compile_group.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Enable full graph compilation",
    )
    compile_group.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="Compilation mode",
    )
    compile_group.add_argument(
        "--compile-backend",
        default="aot_eager",
        help="Compilation backend",
    )

    # MLFlow arguments
    mlflow_group = parser.add_argument_group(
        "MLFlow configuration",
    )
    mlflow_group.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLFlow tracking URI",
    )
    mlflow_group.add_argument(
        "--mlflow-experiment",
        default="mnist",
        help="MLFlow experiment name",
    )
    mlflow_group.add_argument(
        "--mlflow-run-name",
        help="MLFlow run name (default: timestamp)",
    )
    mlflow_group.add_argument(
        "--mlflow-log-system-metrics",
        action="store_true",
        default=True,
        help="Log system metrics",
    )
    return parser


def args_to_config(args: argparse.Namespace) -> Config:
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
            fullgraph=args.compile_fullgraph,
            mode=args.compile_mode,
            backend=args.compile_backend,
        )

    # Create MLFlow Config
    mlflow_config = None
    if mlflow_tracking_uri := args.mlflow_tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
    ):
        mlflow_config = MLFlowConfig(
            experiment_name=args.mlflow_experiment,
            tracking_uri=mlflow_tracking_uri,
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
        reference_model_ckpt=args.reference_model_ckpt,
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
