# Use Python 3.11 slim for better performance and security
FROM python:3.11-slim

# Set build arguments for better cache utilization
ARG DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Create user with ID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 -s /bin/bash user

# Install system dependencies in one layer for smaller image
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libmagic1 \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set HF Spaces specific environment variables
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers
ENV HF_HOME=/tmp/huggingface
ENV CHROMADB_PERSIST_DIRECTORY=/tmp/chromadb
ENV TMPDIR=/tmp

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/transformers_cache \
    /tmp/sentence_transformers \
    /tmp/huggingface \
    /tmp/chromadb \
    /tmp/uploads \
    /app/static \
    /app/data \
    && chown -R user:user /tmp \
    && chmod -R 755 /tmp

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure static directory exists
RUN mkdir -p /app/static

# Set proper ownership
RUN chown -R user:user /app

# Switch to non-root user (required for HF Spaces)
USER user

# Create data directory in /tmp for user
RUN mkdir -p /tmp/studyflow_data

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Expose port 7860 (required for HF Spaces)
EXPOSE 7860

# Start command - production ready with proper workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--loop", "uvloop"]