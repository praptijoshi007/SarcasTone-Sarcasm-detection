# ==============================================================================
# SarcasTone Multi-Stage Production Dockerfile
# Serves FastAPI Backend on 8000 and Streamlit Dashboard on 8501
# ==============================================================================
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib_sarcastone

WORKDIR /app

# Install system audio and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt requirements.txt
COPY member4/requirements.txt member4_requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r member4_requirements.txt

# Copy repository sources
COPY . .

# Set PYTHONPATH to application root
ENV PYTHONPATH="/app:/app/SarcasTone"

# Create fixture directory
RUN mkdir -p /tmp/matplotlib_sarcastone && chmod 777 /tmp/matplotlib_sarcastone

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Entrypoint script
COPY member4/docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
