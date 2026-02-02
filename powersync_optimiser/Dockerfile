ARG BUILD_FROM=ghcr.io/hassio-addons/base:16.3.2
FROM ${BUILD_FROM}

# Remove musl version lock and upgrade all packages
RUN sed -i '/^musl=/d' /etc/apk/world && \
    apk update && \
    apk upgrade --no-cache --available

# Install build dependencies for cvxpy
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    build-base \
    python3-dev \
    cmake \
    git \
    gfortran \
    openblas-dev

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY optimiser/ ./optimiser/

# Copy s6-overlay service configuration
COPY rootfs /
