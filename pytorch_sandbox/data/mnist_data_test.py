"""Tests for MNIST ASCII art conversion functionality."""

import pytest
import torch

from .mnist_data import to_ascii_art


def test_to_ascii_art_basic() -> None:
    """
    Test basic ASCII art conversion functionality.

    Failure of this test would indicate that:
    1. The conversion function is not returning a string
    2. The conversion is producing empty output
    3. The function is failing to process a simple test pattern
    """
    # Create a simple 28x28 test pattern
    data = torch.zeros(28, 28)
    # Add some patterns to make it visually distinctive
    data[5:20, 5:20] = 0.5  # Create a square
    data[10:15, 10:15] = 1.0  # Create a smaller, brighter square inside

    # Convert to ASCII art
    ascii_result = to_ascii_art(data)

    # Basic checks
    assert isinstance(ascii_result, str)
    assert len(ascii_result.split("\n")) > 0


def test_to_ascii_art_invalid_dimensions() -> None:
    """
    Test input validation for tensor dimensions.

    Failure of this test would indicate that:
    1. The function is accepting 3D tensors when it should only accept 2D
    2. The function is accepting tensors larger than 28x28
    3. The dimension validation assertions are not working correctly
    """
    # Test wrong dimensions
    with pytest.raises(AssertionError, match="Data must be 2-D"):
        to_ascii_art(torch.zeros(3, 28, 28))

    with pytest.raises(AssertionError, match="Data must be <= 28x28"):
        to_ascii_art(torch.zeros(29, 29))


def test_to_ascii_art_invert() -> None:
    """
    Test the inversion functionality of ASCII art conversion.

    Failure of this test would indicate that:
    1. The invert parameter is not having any effect
    2. The normal and inverted outputs are identical when they should differ
    3. The brightness inversion logic is not working correctly
    """
    data = torch.zeros(28, 28)
    data[5:20, 5:20] = 1.0

    # Get normal and inverted versions
    normal = to_ascii_art(data)
    inverted = to_ascii_art(data, invert=True)

    # Check that they're different
    assert normal != inverted


def test_to_ascii_art_skip_blank() -> None:
    """
    Test the blank line skipping functionality.

    Failure of this test would indicate that:
    1. The skip_blank_lines parameter is not having any effect
    2. Empty lines are not being properly identified
    3. The line filtering logic is not working correctly
    """
    data = torch.zeros(28, 28)
    data[14, 14] = 1.0  # Single bright point in middle

    # With and without blank line skipping
    with_skip = to_ascii_art(data, skip_blank_lines=True)
    without_skip = to_ascii_art(data, skip_blank_lines=False)

    # Should have fewer lines when skipping blanks
    assert len(with_skip.split("\n")) < len(without_skip.split("\n"))
