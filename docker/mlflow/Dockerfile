FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow>=2.21.2 \
    psycopg2-binary \
    python-dotenv

# Create a non-root user
RUN useradd -m -u 1000 mlflow \
    && mkdir -p /app \
    && chown mlflow:mlflow /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER mlflow

# Expose MLflow port
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflow"]
