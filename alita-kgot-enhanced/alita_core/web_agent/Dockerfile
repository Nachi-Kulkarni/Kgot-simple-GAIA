# Alita Web Agent Dockerfile
# Includes Playwright for browser automation and web scraping

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
FROM mcr.microsoft.com/playwright:v1.40.0-focal AS production

# Install Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Create non-root user for security
RUN groupadd -r alita && useradd -r -g alita alita

# Set working directory
WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source
COPY . .

# Copy configuration files
COPY ../../config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/alita logs/web_agent downloads && \
    chown -R alita:alita logs downloads && \
    chmod -R 755 logs downloads

# Install Playwright browsers
RUN npx playwright install --with-deps chromium firefox webkit

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
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the web agent service
CMD ["node", "index.js"] 