"""Setup script for the package."""

from setuptools import find_packages, setup

setup(
    name="pytorch_sandbox",
    packages=find_packages(),
    py_modules=["pytorch_sandbox"],
    # entry_points={
    #     "console_scripts": ["mnist-train=pytorch_sandbox.mnist_train:main"],
    # },
)
