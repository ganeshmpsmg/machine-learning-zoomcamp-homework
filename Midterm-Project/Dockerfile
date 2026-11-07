# Machine Learning Zoomcamp 2026 Midterm Project
# Bike Rental Demand Prediction API
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=4

# Set working directory
WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application artifacts and code
COPY pipeline_utils.py .
COPY predict.py .
COPY app.py .
COPY models/ ./models/
COPY data/ ./data/

# Expose default API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
