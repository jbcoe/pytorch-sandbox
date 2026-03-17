# Training with Docker

## Setup

Start the database, MLflow server, and training container:

```bash
docker compose up -d
```

MLflow UI: <http://localhost:5002>

## Train

Run the training command in the container:

```bash
docker compose exec mnist-train mnist-train --epochs 1
```

Metrics are logged directly to the Postgres backend.

## Results

Open <http://localhost:5002> to view runs, metrics (loss, accuracy, gradient
norms), and parameters.

## Teardown

Stop and remove containers:

```bash
docker compose down
```
