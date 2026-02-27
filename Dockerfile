FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Project currently depends only on numpy.
RUN pip install --no-cache-dir numpy==2.4.2

COPY test-plaintext ./test-plaintext

ENTRYPOINT ["python", "test-plaintext/plaintext_bench.py"]
CMD []
