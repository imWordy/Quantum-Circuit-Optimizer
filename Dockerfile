# syntax=docker/dockerfile:1
# Multi-stage production Dockerfile for Quantum Circuit Optimizer (QCO)
# Optimized for Microsoft Azure App Service & Azure Container Apps

# ==========================================
# Stage 1: Build & Dependencies Pre-compiler
# ==========================================
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /install

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU wheels
COPY requirements.txt pyproject.toml ./
RUN pip install --prefix=/install/wheels torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --prefix=/install/wheels -r requirements.txt

# ==========================================
# Stage 2: Minimal Production Runtime
# ==========================================
FROM python:3.12-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install curl for container health check probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder stage
COPY --from=builder /install/wheels /usr/local

# Create non-root user for security best practices (Azure CIS benchmark)
RUN groupadd -r qcouser && useradd -r -g qcouser -d /app -s /sbin/nologin qcouser

# Copy application source code and configuration
COPY config/ /app/config/
COPY src/ /app/src/
COPY models/ /app/models/
COPY .streamlit/ /app/.streamlit/
COPY pyproject.toml README.md /app/

# Install the qco package in editable/local mode
RUN pip install --no-deps -e /app

# Ensure proper permissions
RUN chown -R qcouser:qcouser /app

# Switch to non-root execution
USER qcouser

# Expose standard Streamlit port
EXPOSE 8501

# Health check probe for Azure App Service container health monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "src/qco/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
