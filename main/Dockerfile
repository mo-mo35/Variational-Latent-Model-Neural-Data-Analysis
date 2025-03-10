# Use a lightweight Python image
FROM python:3.12-slim

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including Git)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Set the working directory to the script directory
WORKDIR /app/eda

# Run the default script
CMD ["bash"]
