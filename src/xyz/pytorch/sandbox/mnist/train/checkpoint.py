"""Checkpointing for MNIST training."""


import logging
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP

from xyz.pytorch.sandbox.mnist.train.config import (
    Config,
    DDPConfig,
    FSDPConfig,
)

_LOGGER = logging.getLogger(__name__)

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

