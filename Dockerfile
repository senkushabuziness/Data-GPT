# Use a slim python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install essential system dependencies for pip packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (do NOT copy .env)
COPY . .

# ✅ Copy GCS credentials explicitly (optional, if .dockerignore excludes it)
COPY gcs_service_account.json /app/gcs_service_account.json

# Expose port for Chainlit
EXPOSE 8000

# Run your Chainlit app (no need to source .env here)
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "-w"]
