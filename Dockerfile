FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for lxml/readability
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache layer)
COPY requirements.txt .

# Install Python dependencies
# Use CPU-only PyTorch to save ~1.5GB
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code (ignore frontend via .dockerignore)
COPY app/ ./app/

# HF Spaces expects port 7860
ENV PORT=7860

# Expose port
EXPOSE 7860

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
