FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FlashAttention (separate step due to build time)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Copy project
COPY . .
RUN pip install -e .

# Default: run training
ENTRYPOINT ["python"]
CMD ["scripts/train.py", "--help"]
