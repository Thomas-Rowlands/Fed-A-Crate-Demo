FROM python:3.12-slim-bookworm

# 1. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Set working directory
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# 3. Copy dependency files FROM the local app folder
# Note the source path: app/uv.lock
COPY app/pyproject.toml app/uv.lock ./

# 4. Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# 5. Copy the rest of the application code
# This copies everything inside your local 'app' folder into the container's '/app'
COPY app/ .
COPY data/ ./data

# 6. Final sync
RUN uv sync --frozen --no-dev

CMD ["python", "main.py", "--data-dir", "data"]