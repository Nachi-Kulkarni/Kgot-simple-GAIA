# KGoT Graph Store Dockerfile
# Handles graph data persistence with Neo4j and RDF4j connectivity

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
COPY package*.json ./

# Install dependencies
RUN npm install --omit=dev --legacy-peer-deps

# === Production Stage ===
FROM node:20-alpine AS production

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S graphstore -u 1001

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
COPY config ./config

# Create necessary directories with proper permissions
RUN mkdir -p logs/kgot logs/graph_store data snapshots && \
    chown -R graphstore:nodejs logs data snapshots && \
    chmod -R 755 logs data snapshots

# Switch to non-root user
USER graphstore

# Expose port for graph store API
EXPOSE 3004

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:3004/health || exit 1

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3004
ENV LOG_LEVEL=info

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the graph store service
CMD ["node", "kgot_core/graph_store/index.js"] 