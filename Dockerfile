FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's/archive.ubuntu.com/mirror.arvancloud.ir/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirror.arvancloud.ir/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    gnupg \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    unixodbc \
    unixodbc-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir lap

COPY . .
RUN mkdir -p /app/outputs

ENV PYTHONPATH=/app/src