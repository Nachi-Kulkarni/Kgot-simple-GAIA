# Alita Manager Agent Dockerfile
# Multi-stage build for optimized production image

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

# Copy package files from project root
COPY ../../package*.json ./

# Copy config directory
COPY ../../config ./config

# Install dependencies
RUN npm install --omit=dev --legacy-peer-deps

# === Production Stage ===
FROM node:20-alpine AS production

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S alita -u 1001

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source from project root
COPY ../../ .

# Copy configuration files
COPY --from=builder /app/config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/alita logs/system && \
    chown -R alita:nodejs logs && \
    chmod -R 755 logs

# Switch to non-root user
USER alita

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8000

# Start the application
CMD ["node", "alita_core/manager_agent/index.js"]