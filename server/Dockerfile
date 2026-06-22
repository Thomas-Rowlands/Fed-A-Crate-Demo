# ─────────────────────────────────────────────────────────────────────────
# Dockerfile for the central Flower aggregation server.
# Listens on port 8080. Receives only model weights from each TRE — never
# patient data.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server_main.py strategy.py tre_logger.py ./

# Default env vars (override via docker-compose or `docker run -e ...`)
ENV FLWR_HOST=0.0.0.0
ENV FLWR_PORT=8080
ENV ROUNDS=10
ENV NUM_CLIENTS=3
ENV N_FEATURES=313
ENV RESULTS_DIR=/results
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

# /results is where the final model + history JSON are written.
# Mount as a volume from the host to persist across container restarts.
VOLUME ["/results"]

CMD ["python", "server_main.py"]
