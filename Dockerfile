# Use a minimal base OS image with Python pre-installed
FROM python:3.12-slim

# Set a working directory in the container
WORKDIR /app

# Update the package manager and install pip (if not already installed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip install \
    langchain==0.3.25 \
    click==8.1.8 \
    pandas==2.2.3 \
    scikit-learn==1.6.1 \
    langchain_community==0.3.23 \
    langchain-ollama==0.3.2 \
    boto3==1.38.12
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y pciutils lshw
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

# Default command (optional, adjusts based on your use case)
CMD ["python3"]