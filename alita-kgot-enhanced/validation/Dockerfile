# Validation Service Dockerfile
# Provides testing, benchmarking, and quality assurance capabilities

# === Build Stage ===
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# === Production Stage ===
FROM node:18-alpine AS production

# Install Python for validation scripts
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    ca-certificates \
    dumb-init \
    gcc \
    musl-dev

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S validation -u 1001

# Set working directory
WORKDIR /app

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source
COPY . .

# Copy configuration files
COPY ../config ./config

# Install Python packages for validation and testing
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p logs/validation results/benchmarks results/qa results/tests && \
    chown -R validation:nodejs logs results && \
    chmod -R 755 logs results

# Create Python symlink for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Switch to non-root user
USER validation

# Expose port for validation API
EXPOSE 3007

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3007/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3007
ENV LOG_LEVEL=info
ENV PYTHONPATH=/app

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the validation service
CMD ["node", "index.js"] 