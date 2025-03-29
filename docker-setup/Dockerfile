# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
# ENV HTTP_PROXY=http://squid-proxy:3128
# ENV HTTPS_PROXY=http://squid-proxy:3128

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
RUN mkdir -p /app/scripts && mkdir /app/models

# Install Python dependencies
COPY scripts/requirements.txt /app/scripts/requirements.txt

RUN pip install --no-cache-dir -r /app/scripts/requirements.txt

COPY models/mxbai-rerank-large-v1 /app/models/mxbai-rerank-large-v1/
# Copy the rest of the application code
COPY scripts /app/scripts

# Expose the port that the application will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "/app/scripts/main.py", "--server.port=8501", "--server.address=0.0.0.0"]