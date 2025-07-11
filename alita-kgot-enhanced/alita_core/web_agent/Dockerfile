# Alita Web Agent Dockerfile
# Includes Playwright for browser automation and web scraping

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

# Copy package files from web_agent directory
COPY alita_core/web_agent/package*.json ./

# Install dependencies
RUN npm install --omit=dev

# === Production Stage ===
FROM node:20-alpine AS production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S alita -u 1001

# Set working directory
WORKDIR /app

# Install minimal extra deps
RUN apk add --no-cache \
    curl \
    ca-certificates \
    dumb-init

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source from web_agent directory
COPY alita_core/web_agent/ .

# Copy configuration files
COPY config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/alita logs/web_agent downloads && \
    chown -R alita:nodejs logs downloads && \
    chmod -R 755 logs downloads

# Switch to non-root user
USER alita

# Expose port for web agent API
EXPOSE 3001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3001
ENV LOG_LEVEL=info

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the web agent service
CMD ["node", "index.js"]