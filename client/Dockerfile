# ─────────────────────────────────────────────────────────────────────────
# Dockerfile for a TRE client.
# Each TRE instance mounts its private cohort CSV under /data/ — the
# specific path is set per-TRE in docker-compose.yml so the real filename
# (e.g. USA_young.csv) stays visible.
# Patient data stays inside the container; only model weights leave.
# ─────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy client code
COPY client_main.py data_utils.py model.py model_io.py tre_logger.py ./

# Default env vars — overridden per-TRE in docker-compose.yml
ENV FLWR_SERVER=fed-server:8080
ENV COHORT_CSV=/data/cohort.csv
ENV COHORT_LABEL=cohort
ENV TRE_NUM=0
ENV PYTHONUNBUFFERED=1

CMD ["python", "client_main.py"]
