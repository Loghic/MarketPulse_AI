# Containerfile – MarketPulse AI
# -------------------------------------------------------
# Build:   podman build -t marketpulse .
# Run:     podman run --rm -v ./data:/app/data:z marketpulse
# -------------------------------------------------------

FROM python:3.12-slim

# Install uv (much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifest first (Docker layer caching —
# dependencies are re-installed only when pyproject.toml changes)
COPY pyproject.toml .

# Install dependencies into the system Python (no venv needed inside a container)
RUN uv pip install --system .

# Copy source code
COPY engine/ engine/
COPY interface/ interface/
COPY main.py .

# Data directory — mounted as a volume so the DB persists between runs
VOLUME ["/app/data"]

CMD ["python", "main.py"]
