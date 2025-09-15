FROM python:3.11-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with no cache
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf ~/.cache/pip

# Copy application code
COPY app.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /tmp/models && \
    chown -R appuser:appuser /tmp/models

USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT:-5000}/ || exit 1

# Expose port
EXPOSE ${PORT:-5000}

# Run with gunicorn for production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120"]