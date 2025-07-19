# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of your app
COPY . .

# Set environment variables for Google Application Credentials
#ENV GOOGLE_APPLICATION_CREDENTIALS="/app/gcs_service_account.json"

# Expose port
EXPOSE 8080

# Command to run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
