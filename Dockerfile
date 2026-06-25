FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir --upgrade pip && \
    grep -v "github.com/acidify04/hemac.git" requirements.txt > /tmp/requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm -f /tmp/requirements.txt

COPY . .

RUN python -m pip install --no-cache-dir --no-deps .
