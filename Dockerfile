# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., git and other tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or install packages directly
# Based on the project dependencies
RUN pip install --no-cache-dir \
    transformers \
    fastapi \
    uvicorn \
    pillow \
    pandas \
    tqdm \
    tensorboard \
    python-multipart

# Pre-cache the Bio_ClinicalBERT model weights
# This prevents downloading model weights at startup during deployment
COPY scripts/setup_hf.py /app/scripts/setup_hf.py
RUN python /app/scripts/setup_hf.py

# Copy the rest of the application files
COPY scripts/ /app/scripts/
COPY models/ /app/models/
COPY data/ /app/data/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "scripts.app:app", "--host", "0.0.0.0", "--port", "8000"]
