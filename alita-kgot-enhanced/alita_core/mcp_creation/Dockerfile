# Alita MCP Creation Dockerfile  
# Supports LangChain-based AI agent development and MCP generation

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

# Copy package files from mcp_creation directory
COPY alita_core/mcp_creation/package*.json ./

# Install dependencies
RUN npm install --omit=dev

# === Production Stage ===
FROM node:20-alpine AS production

# Install Python for LangChain support
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
    adduser -S mcp -u 1001

# Set working directory
WORKDIR /app

# Copy node_modules from builder stage
COPY --from=builder /app/node_modules ./node_modules

# Copy application source from mcp_creation directory
COPY alita_core/mcp_creation/ .

# Copy configuration files
COPY config ./config

# Install LangChain and related Python packages
COPY requirements.txt ./
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p logs/alita logs/mcp_creation mcp_storage generated_mcps && \
    chown -R mcp:nodejs logs mcp_storage generated_mcps && \
    chmod -R 755 logs mcp_storage generated_mcps

# Create Python symlink for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Switch to non-root user
USER mcp

# Expose port for MCP creation API
EXPOSE 3002

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3002/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3002
ENV LOG_LEVEL=info
ENV PYTHONPATH=/app

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the MCP creation service
CMD ["node", "index.js"]