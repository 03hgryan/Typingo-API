FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only (no dev group)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ src/

# Install the project itself
RUN uv sync --frozen --no-dev

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080

CMD uv run fastapi run src/main.py --host 0.0.0.0 --port $PORT
