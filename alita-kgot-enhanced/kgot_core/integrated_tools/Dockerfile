# KGoT Integrated Tools Dockerfile
# Multi-language support for Python and Node.js tool execution

# === Build Stage ===
FROM node:20-alpine AS node-builder

# Set working directory
WORKDIR /app

# Install build dependencies for Node.js
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm install --omit=dev --legacy-peer-deps

# === Python Environment Stage ===
FROM python:3.11-alpine AS python-base

# Install Python dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    curl

# Copy Python requirements
COPY requirements.txt ./

# Install Python packages
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# === Production Stage ===
FROM node:20-alpine AS production

# Install Python runtime
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
    adduser -S tools -u 1001

# Set working directory
WORKDIR /app

# Copy Node.js modules from builder stage
COPY --from=node-builder /app/node_modules ./node_modules

# Copy Python packages from python-base stage
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application source
COPY . .

# Copy configuration files
COPY config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/kgot logs/integrated_tools tools workspace && \
    chown -R tools:nodejs logs tools workspace && \
    chmod -R 755 logs tools workspace

# Create Python symlink for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Switch to non-root user
USER tools

# Expose port for tools API
EXPOSE 3005

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3005/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3005
ENV LOG_LEVEL=info
ENV PYTHONPATH=/app

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the integrated tools service
CMD ["node", "kgot_core/integrated_tools/integrated_tools_manager.js"] 