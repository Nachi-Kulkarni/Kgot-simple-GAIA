#!/bin/bash
# KGoT Production Deployment - Build Script
# =========================================
#
# This script handles building Docker images for all KGoT services
# with integrated security scanning and registry management.
#
# Usage: ./build.sh [OPTIONS]
#   -v, --version    Version tag for images (required)
#   -r, --registry   Container registry URL
#   -s, --scan       Enable security scanning (default: true)
#   -p, --push       Push images to registry (default: false)
#   -f, --force      Force rebuild without cache
#   -h, --help       Show this help message

set -euo pipefail

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_LOG="${PROJECT_ROOT}/logs/deployment/build.log"
VERSION=""
REGISTRY="localhost:5000"
ENABLE_SCAN=true
PUSH_IMAGES=false
FORCE_REBUILD=false
BUILD_PARALLEL=true
MAX_PARALLEL_BUILDS=4

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*" | tee -a "${BUILD_LOG}"
}

error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" | tee -a "${BUILD_LOG}" >&2
}

warning() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] $*" | tee -a "${BUILD_LOG}"
}

# Service definitions with build contexts
declare -A SERVICES=(
    ["kgot-controller"]="alita-kgot-enhanced/kgot_core/controller"
    ["graph-store"]="alita-kgot-enhanced/kgot_core/graph_store"
    ["manager-agent"]="alita-kgot-enhanced/alita_core/manager_agent"
    ["web-agent"]="alita-kgot-enhanced/alita_core/web_agent"
    ["mcp-creation"]="alita-kgot-enhanced/alita_core/mcp_creation"
    ["monitoring"]="alita-kgot-enhanced/monitoring"
    ["validation"]="alita-kgot-enhanced/validation"
    ["multimodal"]="alita-kgot-enhanced/multimodal"
    ["optimization"]="alita-kgot-enhanced/optimization"
)

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -s|--scan)
                ENABLE_SCAN="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_IMAGES=true
                shift
                ;;
            -f|--force)
                FORCE_REBUILD=true
                shift
                ;;
            --no-parallel)
                BUILD_PARALLEL=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$VERSION" ]]; then
        error "Version is required. Use -v or --version to specify."
        exit 1
    fi
}

show_help() {
    cat << EOF
KGoT Production Deployment - Build Script

Usage: $0 [OPTIONS]

Options:
  -v, --version VERSION    Version tag for images (required)
  -r, --registry URL       Container registry URL (default: localhost:5000)
  -s, --scan BOOL         Enable security scanning (default: true)
  -p, --push              Push images to registry after build
  -f, --force             Force rebuild without using cache
  --no-parallel           Disable parallel building
  -h, --help              Show this help message

Examples:
  $0 -v 1.2.3 -p                    # Build version 1.2.3 and push to registry
  $0 -v dev -s false                # Build dev version without security scan
  $0 -v latest -r my-registry.com   # Build with custom registry

EOF
}

# Setup build environment
setup_build_env() {
    log "Setting up build environment"
    
    # Create log directory
    mkdir -p "$(dirname "${BUILD_LOG}")"
    
    # Verify Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or accessible"
        exit 1
    fi
    
    # Check if security scanning tools are available
    if [[ "$ENABLE_SCAN" == "true" ]]; then
        if ! command -v trivy >/dev/null 2>&1; then
            warning "Trivy not found, installing..."
            install_trivy
        fi
    fi
    
    # Login to registry if credentials are available
    if [[ -n "${DOCKER_REGISTRY_USER:-}" && -n "${DOCKER_REGISTRY_PASS:-}" ]]; then
        log "Logging into registry: $REGISTRY"
        echo "${DOCKER_REGISTRY_PASS}" | docker login "$REGISTRY" -u "${DOCKER_REGISTRY_USER}" --password-stdin
    fi
}

# Install Trivy security scanner
install_trivy() {
    log "Installing Trivy security scanner"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install trivy
        else
            error "Please install Trivy manually: https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
            exit 1
        fi
    else
        error "Unsupported OS for automatic Trivy installation"
        exit 1
    fi
}

# Build single Docker image
build_image() {
    local service="$1"
    local context_path="$2"
    local image_tag="${REGISTRY}/${service}:${VERSION}"
    
    log "Building $service ($image_tag)"
    
    # Check if Dockerfile exists
    local dockerfile_path="${PROJECT_ROOT}/${context_path}/Dockerfile"
    if [[ ! -f "$dockerfile_path" ]]; then
        error "Dockerfile not found: $dockerfile_path"
        return 1
    fi
    
    # Build arguments
    local build_args=(
        "build"
        "-t" "$image_tag"
        "-f" "$dockerfile_path"
    )
    
    # Add build context
    build_args+=("${PROJECT_ROOT}/${context_path}")
    
    # Add cache options
    if [[ "$FORCE_REBUILD" == "true" ]]; then
        build_args+=("--no-cache")
    fi
    
    # Add build metadata
    build_args+=(
        "--label" "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        "--label" "org.opencontainers.image.version=$VERSION"
        "--label" "org.opencontainers.image.source=https://github.com/kgot/alita-enhanced"
        "--label" "org.opencontainers.image.title=$service"
    )
    
    # Execute build
    if docker "${build_args[@]}" 2>&1 | tee -a "${BUILD_LOG}"; then
        log "Successfully built $service"
        return 0
    else
        error "Failed to build $service"
        return 1
    fi
}

# Run security scan on image
scan_image() {
    local service="$1"
    local image_tag="${REGISTRY}/${service}:${VERSION}"
    
    log "Running security scan for $service"
    
    local scan_output="${PROJECT_ROOT}/logs/deployment/scan_${service}_${VERSION}.json"
    
    # Run Trivy scan
    if trivy image \
        --format json \
        --output "$scan_output" \
        --severity HIGH,CRITICAL \
        --exit-code 1 \
        "$image_tag" 2>&1 | tee -a "${BUILD_LOG}"; then
        log "Security scan passed for $service"
        return 0
    else
        error "Security vulnerabilities found in $service"
        error "Scan report: $scan_output"
        
        # Show summary of critical vulnerabilities
        if command -v jq >/dev/null 2>&1 && [[ -f "$scan_output" ]]; then
            local critical_count
            critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$scan_output" 2>/dev/null || echo "0")
            local high_count
            high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$scan_output" 2>/dev/null || echo "0")
            
            error "Found $critical_count CRITICAL and $high_count HIGH severity vulnerabilities"
        fi
        
        return 1
    fi
}

# Push image to registry
push_image() {
    local service="$1"
    local image_tag="${REGISTRY}/${service}:${VERSION}"
    
    log "Pushing $service to registry"
    
    if docker push "$image_tag" 2>&1 | tee -a "${BUILD_LOG}"; then
        log "Successfully pushed $service"
        return 0
    else
        error "Failed to push $service"
        return 1
    fi
}

# Build single service (used for parallel execution)
build_service() {
    local service="$1"
    local context_path="$2"
    local success=true
    
    # Build image
    if ! build_image "$service" "$context_path"; then
        success=false
    fi
    
    # Security scan
    if [[ "$success" == "true" && "$ENABLE_SCAN" == "true" ]]; then
        if ! scan_image "$service"; then
            success=false
        fi
    fi
    
    # Push to registry
    if [[ "$success" == "true" && "$PUSH_IMAGES" == "true" ]]; then
        if ! push_image "$service"; then
            success=false
        fi
    fi
    
    if [[ "$success" == "true" ]]; then
        log "Successfully processed $service"
        echo "$service" >> "${PROJECT_ROOT}/logs/deployment/build_success.list"
    else
        error "Failed to process $service"
        echo "$service" >> "${PROJECT_ROOT}/logs/deployment/build_failed.list"
    fi
    
    return $([[ "$success" == "true" ]] && echo 0 || echo 1)
}

# Build all services
build_all_services() {
    log "Starting build process for ${#SERVICES[@]} services"
    
    # Clean up previous build status files
    rm -f "${PROJECT_ROOT}/logs/deployment/build_success.list"
    rm -f "${PROJECT_ROOT}/logs/deployment/build_failed.list"
    
    local pids=()
    local running_jobs=0
    
    for service in "${!SERVICES[@]}"; do
        local context_path="${SERVICES[$service]}"
        
        if [[ "$BUILD_PARALLEL" == "true" ]]; then
            # Wait if we've reached max parallel jobs
            while [[ $running_jobs -ge $MAX_PARALLEL_BUILDS ]]; do
                wait -n  # Wait for any job to complete
                ((running_jobs--))
            done
            
            # Start build in background
            build_service "$service" "$context_path" &
            pids+=($!)
            ((running_jobs++))
        else
            # Sequential build
            build_service "$service" "$context_path"
        fi
    done
    
    # Wait for all background jobs to complete
    if [[ "$BUILD_PARALLEL" == "true" ]]; then
        log "Waiting for all builds to complete..."
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    fi
}

# Generate build report
generate_build_report() {
    local success_count=0
    local failed_count=0
    
    if [[ -f "${PROJECT_ROOT}/logs/deployment/build_success.list" ]]; then
        success_count=$(wc -l < "${PROJECT_ROOT}/logs/deployment/build_success.list")
    fi
    
    if [[ -f "${PROJECT_ROOT}/logs/deployment/build_failed.list" ]]; then
        failed_count=$(wc -l < "${PROJECT_ROOT}/logs/deployment/build_failed.list")
    fi
    
    local total_count=$((success_count + failed_count))
    
    log "Build Summary:"
    log "  Total services: $total_count"
    log "  Successful: $success_count"
    log "  Failed: $failed_count"
    
    if [[ $failed_count -gt 0 ]]; then
        error "Some builds failed. Check build log: $BUILD_LOG"
        if [[ -f "${PROJECT_ROOT}/logs/deployment/build_failed.list" ]]; then
            error "Failed services:"
            while IFS= read -r service; do
                error "  - $service"
            done < "${PROJECT_ROOT}/logs/deployment/build_failed.list"
        fi
        return 1
    else
        log "All builds completed successfully!"
        return 0
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files"
    
    # Remove temporary build files
    find "${PROJECT_ROOT}/logs/deployment" -name "*.tmp" -delete 2>/dev/null || true
    
    # Prune unused Docker resources
    docker system prune -f >/dev/null 2>&1 || true
}

# Main execution
main() {
    local start_time
    start_time=$(date +%s)
    
    log "Starting KGoT build process"
    log "Version: $VERSION"
    log "Registry: $REGISTRY"
    log "Security scan: $ENABLE_SCAN"
    log "Push images: $PUSH_IMAGES"
    log "Force rebuild: $FORCE_REBUILD"
    log "Parallel build: $BUILD_PARALLEL"
    
    # Setup environment
    setup_build_env
    
    # Build all services
    build_all_services
    
    # Generate report
    if ! generate_build_report; then
        cleanup
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Build process completed in ${duration} seconds"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    parse_args "$@"
    main
fi 