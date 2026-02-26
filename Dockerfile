FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[dashboard]"

# Create data directory for SQLite persistence
RUN mkdir -p /app/data

CMD ["python", "-m", "trading_agent", "--scan", "--schedule"]
