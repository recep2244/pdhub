FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy build configuration
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
# We install with [dev] to include testing tools, remove [dev] for production size optimization if needed
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Create output directory
RUN mkdir -p outputs

# Environment variables
ENV PDHUB_OUTPUT_BASE_DIR=/app/outputs
ENV PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Default command: run the web interface
CMD ["pdhub", "web", "--host", "0.0.0.0"]
