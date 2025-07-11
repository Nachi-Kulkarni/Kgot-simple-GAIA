# Multimodal Processing Service Dockerfile
# Supports audio, vision, and text processing capabilities

# === Build Stage ===
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git

# Copy package files
COPY multimodal/package*.json ./

# Install dependencies
RUN npm install --omit=dev --legacy-peer-deps

# === Production Stage ===
FROM node:20-alpine AS production

# Install Python and multimedia processing libraries
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    ca-certificates \
    dumb-init \
    gcc \
    musl-dev \
    ffmpeg \
    opencv-dev \
    py3-numpy \
    py3-pillow

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S multimodal -u 1001

# Set working directory
WORKDIR /app

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source
COPY multimodal/ .

# Copy configuration files
COPY config ./config

# Install Python packages for multimodal processing
COPY multimodal/requirements.txt ./
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p logs/multimodal storage/audio storage/video storage/images storage/text && \
    chown -R multimodal:nodejs logs storage && \
    chmod -R 755 logs storage

# Create Python symlink for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Switch to non-root user
USER multimodal

# Expose port for multimodal processing API
EXPOSE 3006

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3006/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3006
ENV LOG_LEVEL=info
ENV PYTHONPATH=/app

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the multimodal processing service
CMD ["node", "multimodal/index.js"]