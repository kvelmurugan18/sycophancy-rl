FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system syco \
    && useradd --system --gid syco --create-home syco

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy/env-server/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir --requirement requirements.lock \
    && pip install --no-cache-dir --no-deps .
COPY --chown=syco:syco data/processed/training_pool.jsonl ./data/processed/training_pool.jsonl

USER syco
EXPOSE 8000
CMD ["uvicorn", "sycophancy_rl.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
