# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for venv
RUN apt-get update && apt-get install -y python3-venv && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside the container
RUN python3 -m venv /opt/venv

# Activate venv and install dependencies
COPY requirements.txt .
RUN /bin/bash -c "source /opt/venv/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Copy all app files
COPY . .

# Set environment variables to use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"
ENV PORT=8080

# Expose the port
EXPOSE 8080

# Start the Flask app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
