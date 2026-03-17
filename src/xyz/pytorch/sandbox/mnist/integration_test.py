"""Integration tests for the MNIST training pipeline using fake data."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset

from xyz.pytorch.sandbox.mnist.train import Config, LogLevel, _main, cnn


class FakeMNIST(Dataset):
    """A fake dataset that mimics MNIST for testing."""

    def __init__(self, size: int = 10):
        self.size = size
        self.data = torch.randn(size, 1, 28, 28)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_training_pipeline_with_fake_data(tmp_path: Path, batch_size: int) -> None:
    """
    Runs a full training loop for 1 epoch using fake data.

    Verifies that the pipeline (train, test, checkpointing) works without
    downloading MNIST.
    """
    ckpt_dir = tmp_path / f"ckpt_{batch_size}"
    data_dir = tmp_path / f"data_{batch_size}"
    ckpt_dir.mkdir()
    data_dir.mkdir()

    # Minimal config for testing
    config = Config(
        epochs=1,
        batch_size=batch_size,
        num_workers=0,
        device="cpu",
        ckpt=str(ckpt_dir),
        data_dir=str(data_dir),
        log_level=LogLevel.DEBUG,
    )

    # Mock load_mnist to return our FakeMNIST dataset
    fake_train = FakeMNIST(size=4)
    fake_test = FakeMNIST(size=2)

    with patch("xyz.pytorch.sandbox.mnist.data.load_mnist", return_value=(fake_train, fake_test)):
        # Run the main training loop
        _main(rank=0, config=config)

    # Verify that checkpoints were created
    ckpt_files = sorted(list(ckpt_dir.glob("*.pt")))
    assert len(ckpt_files) == 2, f"Expected 2 model checkpoints, found {len(ckpt_files)}"

    # 1. Meaningful check: Ensure the saved model can be loaded back
    model = cnn.Net()
    state_dict = torch.load(ckpt_files[-1], weights_only=True)
    model.load_state_dict(state_dict)

    config_files = list(ckpt_dir.glob("*_config.txt"))
    assert len(config_files) == 1, "Expected 1 config file"

    # Verify config file content
    with open(config_files[0], "r") as f:
        saved_config = json.loads(f.read())
        assert saved_config["batch_size"] == batch_size


@pytest.mark.parametrize("batch_size", [1, 2])
def test_parameters_actually_update(tmp_path: Path, batch_size: int) -> None:
    """
    Verifies that model parameters change after a training step.

    This ensures the loss.backward() and optimizer.step() are correctly hooked up.
    """
    from torch.optim import Adadelta

    from xyz.pytorch.sandbox.mnist.train import train

    model = cnn.Net()
    # Capture initial weights of one layer
    initial_weight = model.conv1.weight.clone().detach()

    optimizer = Adadelta(model.parameters(), lr=0.1)
    dataset = FakeMNIST(size=batch_size * 2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Run one training step
    train(
        rank=0,
        model=model,
        device=torch.device("cpu"),
        train_loader=loader,
        optimizer=optimizer,
        epoch=1,
        global_step=0,
    )

    # Verify weights have changed
    updated_weight = model.conv1.weight
    assert not torch.equal(initial_weight, updated_weight), "Weights did not change after training step"
    assert model.conv1.weight.grad is not None, "Gradients were not computed"


def test_config_serialization():
    """Verify that the Config dataclass remains JSON serializable."""
    config = Config()
    config_dict = dataclasses.asdict(config)
    try:
        json.dumps(config_dict)
    except TypeError as e:
        pytest.fail(f"Config is not JSON serializable: {e}")
