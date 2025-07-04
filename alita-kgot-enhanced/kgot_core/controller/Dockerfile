# KGoT Controller Dockerfile
# Optimized for containerized deployment with health monitoring

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

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S kgot -u 1001

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    dumb-init

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source
COPY . .

# Copy configuration files
COPY ../../config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/kgot logs/controller && \
    chown -R kgot:nodejs logs && \
    chmod -R 755 logs

# Switch to non-root user
USER kgot

# Expose port for health checks and API
EXPOSE 3003

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3003/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3003
ENV LOG_LEVEL=info

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the KGoT controller
CMD ["node", "kgot_controller.js"] 