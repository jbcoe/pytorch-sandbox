#! /bin/bash
# A script to check if the devcontainer docker image is usable.

set -eu -o pipefail

# Build the docker image
echo "Building the docker image..."
CMD="docker build -t pytorch-sandbox -f .devcontainer/Dockerfile ."
echo "Running: $CMD"
eval "$CMD"

# Run the docker image
printf "\e[1;32mChecking if the docker image is usable. Exit the container with ^d.\e[0m\n"

docker run -it pytorch-sandbox /bin/bash
