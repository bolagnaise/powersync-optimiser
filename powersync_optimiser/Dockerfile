ARG BUILD_FROM=ghcr.io/hassio-addons/base:15.0.8
FROM ${BUILD_FROM}

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
COPY run.sh .

# Make run script executable
RUN chmod a+x run.sh

# Run the optimiser server
CMD [ "/app/run.sh" ]
