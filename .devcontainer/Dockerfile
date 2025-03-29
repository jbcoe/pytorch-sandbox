FROM python:3.12-slim

# Install essential packages and create non-root user
RUN apt-get update && apt-get install -y git curl \
    && useradd -m -s /bin/bash vscode \
    && mkdir -p /workspace \
    && chown vscode:vscode /workspace

RUN mkdir -p /workspace
RUN chmod -R 755 /workspace
WORKDIR /workspace

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="~/.local/bin:${PATH}"
