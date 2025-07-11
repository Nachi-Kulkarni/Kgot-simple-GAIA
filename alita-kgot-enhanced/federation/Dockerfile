# Simple Federation Server Dockerfile
FROM python:3.11-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache curl

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy federation source code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

# Expose the federation server port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the federation server
CMD ["python", "start_simple_server.py", "--host", "0.0.0.0", "--port", "8080", "--demo"]