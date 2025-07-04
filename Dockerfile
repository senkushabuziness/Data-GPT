# Use an official Python image
FROM python:3.12

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port if Chainlit runs on 8000
EXPOSE 8000

# Command to run your app
CMD ["chainlit", "run", "main.py", "-w"]
