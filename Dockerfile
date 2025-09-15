# Use Python slim image (smaller than regular)
FROM python:3.9-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with size optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]