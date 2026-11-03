FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY pyproject.toml README.md ./
COPY config ./config
COPY evaluators ./evaluators
COPY ab_testing ./ab_testing
COPY human_review ./human_review
COPY golden_dataset ./golden_dataset
COPY dashboard ./dashboard
COPY workers ./workers
COPY api ./api

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 8088 8089

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8088"]
