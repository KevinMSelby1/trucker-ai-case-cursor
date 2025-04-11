# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set matplotlib to use non-interactive backend
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to cache dependencies
COPY requirements.txt /app/

# Install the required dependencies
RUN pip install -r requirements.txt

# Create data directory
RUN mkdir -p /app/data

# Set matplotlib to use non-interactive backend
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"

# Copy the rest of the content into the container
COPY . /app/

# Run the script that ties everything together
CMD ["python", "run_all.py"]