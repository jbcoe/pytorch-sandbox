# Use Python:3.12-slim image as the base image as the PyTorch image
# https://hub.docker.com/r/pytorch/pytorch is too big (>3GB).
FROM python:3.12-slim

# Install essential packages and create non-root user
RUN apt-get update && apt-get install -y git curl sudo bash-completion vim \
    && useradd -m -s /bin/bash vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace \
    && chown vscode:vscode /workspace

WORKDIR /workspace

# Switch to non-root user
USER vscode

# Set up shell completions
RUN echo 'source /usr/share/bash-completion/completions/git' >> ~/.bashrc \
    && echo 'source /etc/bash_completion' >> ~/.bashrc

# Set up uv and Python environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up environment variables
ENV PATH="/home/vscode/.local/bin:${PATH}"
