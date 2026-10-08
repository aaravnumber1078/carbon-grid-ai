# CarbonGrid-AI Dockerfile for Hugging Face Spaces (GPU)
# Base: CUDA 11.8 runtime compatible with torch 2.7.1+cu118
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY carbongrid/ ./carbongrid/
COPY frontend/ ./frontend/

# Copy classifier artifacts (required at runtime)
COPY models/quantization_safety/ ./models/quantization_safety/

# Copy carbon replay data (for offline/replay fallback)
COPY data/replay/ ./data/replay/

# Create non-root user for HF Spaces
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port (HF Spaces expects 7860 by default)
EXPOSE 7860

# Environment variables with defaults
ENV HOST=0.0.0.0
ENV PORT=7860
ENV MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
ENV DEVICE_INDEX=0
ENV MAX_CONCURRENT=1
ENV CARBON_MODE=live
ENV CARBON_CONNECT_TIMEOUT=3.0
ENV CARBON_READ_TIMEOUT=7.0
ENV FP16_VRAM_MB=3200
ENV INT4_VRAM_MB=1400
ENV KMP_DUPLICATE_LIB_OK=TRUE

# HF Spaces requires this for proper routing
ENV GRADIO_SERVER_NAME=0.0.0.0

# Start the FastAPI application
CMD ["uvicorn", "carbongrid.api.main:app", "--host", "0.0.0.0", "--port", "7860"]