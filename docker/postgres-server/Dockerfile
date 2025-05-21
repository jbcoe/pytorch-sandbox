FROM postgres:15

# Set environment variables
ENV POSTGRES_DB=mlflow
ENV POSTGRES_USER=mlflow
ENV POSTGRES_PASSWORD=mlflow

# Create directory for initialization scripts
RUN mkdir -p /docker-entrypoint-initdb.d

# Copy initialization script
COPY init.sql /docker-entrypoint-initdb.d/

# Expose PostgreSQL port
EXPOSE 5432
