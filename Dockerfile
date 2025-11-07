# Production Dockerfile for Hugging Face Spaces
# Civic Issue Urgency Classifier - Text-only NLP System

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p $APP_HOME/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port for Hugging Face Spaces
EXPOSE ${PORT}

# Run FastAPI with Uvicorn on port 7860
CMD ["python", "-m", "uvicorn", "src.demo_api_browser:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2"]
