"""Tests for the CNN model."""

import torch

from .cnn import CNNConfig, Net


def test_cnn_output_shape() -> None:
    """
    Test that the CNN produces expected output shapes.

    Verifies:
    - Output batch dimension matches input
    - Output has correct number of classes (10 for MNIST)
    - Output is a valid probability distribution (sums to 1)
    """
    batch_size = 4
    model = Net()
    x = torch.randn(batch_size, 1, 28, 28)  # (batch, channels, height, width)

    output = model(x)

    assert output.shape == (batch_size, 10), (
        f"Expected output shape (batch_size, n_classes) = ({batch_size}, 10), but got {output.shape}"
    )

    # Check if output sums to 1 (within numerical precision)
    probs = torch.exp(output)
    sums = probs.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), msg="Output probabilities should sum to 1 for each sample")


def test_cnn_config() -> None:
    """
    Test that the CNN respects different configurations.

    Verifies that changing the configuration parameters
    results in the expected model architecture changes.
    """
    config = CNNConfig(conv1_channels=16, conv2_channels=32, linear=64, input_size=28)
    model = Net(config)

    assert model.conv1.out_channels == 16, (
        f"Expected conv1 to have 16 output channels, but got {model.conv1.out_channels}"
    )
    assert model.conv2.out_channels == 32, (
        f"Expected conv2 to have 32 output channels, but got {model.conv2.out_channels}"
    )
    assert model.fc1.out_features == 64, f"Expected fc1 to have 64 output features, but got {model.fc1.out_features}"


def test_cnn_invalid_input() -> None:
    """
    Test that the CNN handles invalid input sizes appropriately.

    Verifies that the model raises an error with meaningful message
    when given incorrectly sized input.
    """
    model = Net()
    wrong_size = torch.randn(1, 1, 32, 32)  # Wrong input size (32x32 instead of 28x28)

    try:
        model(wrong_size)
        raise AssertionError("Expected RuntimeError for invalid input size")
    except RuntimeError as e:
        assert "shapes cannot be multiplied" in str(e).lower(), (
            f"Expected error message about size mismatch, but got: {str(e)}"
        )
