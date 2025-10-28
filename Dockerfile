FROM python:3.11-slim

# Install system dependencies (for lxml, numpy, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Install as editable package (allows direct file edits without rebuild)
RUN pip install -e .

# Create data directory for DB
RUN mkdir -p backend/data

# Default command: show help
CMD ["python", "-m", "core.mediator", "--help"]
