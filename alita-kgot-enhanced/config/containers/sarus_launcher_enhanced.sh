#!/bin/bash

# Enhanced Sarus Launcher for KGoT-Alita System
# Supports complete service stack deployment in HPC environments
# Implements service dependencies, health monitoring, and resource management

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../.."
LOG_DIR="${CONFIG_DIR}/../logs/containerization"
SARUS_LOG_DIR="${LOG_DIR}/sarus"

# Service configuration
declare -A SERVICES=(
    ["neo4j"]="neo4j:5.15-community"
    ["rdf4j"]="eclipse/rdf4j:latest"
    ["redis"]="redis:7-alpine"
    ["python-executor"]="python:3.12-slim"
    ["alita-manager"]="alita-manager:latest"
    ["alita-web"]="alita-web-agent:latest"
    ["alita-mcp"]="alita-mcp-creation:latest"
    ["kgot-controller"]="kgot-controller:latest"
    ["kgot-graph-store"]="kgot-graph-store:latest"
    ["kgot-tools"]="kgot-integrated-tools:latest"
    ["multimodal-processor"]="multimodal-processor:latest"
    ["validation-service"]="validation-service:latest"
    ["optimization-service"]="optimization-service:latest"
)

# Service dependencies (services that need to start first)
declare -A SERVICE_DEPS=(
    ["alita-manager"]="neo4j redis"
    ["alita-web"]=""
    ["alita-mcp"]=""
    ["kgot-controller"]="neo4j redis"
    ["kgot-graph-store"]="neo4j rdf4j"
    ["kgot-tools"]="python-executor"
    ["multimodal-processor"]=""
    ["validation-service"]=""
    ["optimization-service"]="redis"
)

# Port mappings
declare -A SERVICE_PORTS=(
    ["neo4j"]="7474:7474 7687:7687"
    ["rdf4j"]="8080:8080"
    ["redis"]="6379:6379"
    ["python-executor"]="5000:5000"
    ["alita-manager"]="3000:3000"
    ["alita-web"]="3001:3001"
    ["alita-mcp"]="3002:3002"
    ["kgot-controller"]="3003:3003"
    ["kgot-graph-store"]="3004:3004"
    ["kgot-tools"]="3005:3005"
    ["multimodal-processor"]="3006:3006"
    ["validation-service"]="3007:3007"
    ["optimization-service"]="3008:3008"
)

# Resource limits (memory in MB, CPU cores)
declare -A SERVICE_RESOURCES=(
    ["neo4j"]="2048:2"
    ["rdf4j"]="1024:1"
    ["redis"]="512:1"
    ["python-executor"]="1024:1"
    ["alita-manager"]="1024:1"
    ["alita-web"]="2048:2"
    ["alita-mcp"]="1024:1"
    ["kgot-controller"]="2048:2"
    ["kgot-graph-store"]="1024:1"
    ["kgot-tools"]="2048:2"
    ["multimodal-processor"]="4096:4"
    ["validation-service"]="1024:1"
    ["optimization-service"]="1024:1"
)

# Global variables
RUNNING_SERVICES=()
COMMAND=""
SELECTED_SERVICES=""
HEALTH_CHECK_ENABLED=true
RESOURCE_MONITORING=true

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/sarus_launcher.log"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/sarus_launcher.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/sarus_launcher.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "${LOG_DIR}/sarus_launcher.log"
}

# Initialize logging directories
init_logging() {
    mkdir -p "${LOG_DIR}" "${SARUS_LOG_DIR}"
    log_info "Enhanced Sarus launcher initialized"
    log_info "Log directory: ${LOG_DIR}"
    log_info "Sarus log directory: ${SARUS_LOG_DIR}"
}

# Check if Sarus is available
check_sarus() {
    if ! command -v sarus &> /dev/null; then
        log_error "Sarus is not installed or not in PATH"
        exit 1
    fi
    
    local sarus_version
    sarus_version=$(sarus --version 2>&1 | head -n1)
    log_info "Sarus version: ${sarus_version}"
}

# Load environment variables
load_environment() {
    local env_file="${CONFIG_DIR}/../.env"
    if [[ -f "${env_file}" ]]; then
        log_info "Loading environment from ${env_file}"
        # shellcheck source=/dev/null
        source "${env_file}"
    else
        log_warning "Environment file not found: ${env_file}"
        log_info "Using default values"
    fi
    
    # Set defaults if not provided
    export NEO4J_PASSWORD="${NEO4J_PASSWORD:-defaultpassword}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-defaultpassword}"
    export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
    export LOG_LEVEL="${LOG_LEVEL:-info}"
}

# Build Sarus command for a service
build_sarus_command() {
    local service_name="$1"
    local image="${SERVICES[$service_name]}"
    local cmd=("sarus" "run")
    
    # Add container name
    cmd+=("--name" "${service_name}")
    
    # Add port mappings
    if [[ -n "${SERVICE_PORTS[$service_name]:-}" ]]; then
        IFS=' ' read -ra ports <<< "${SERVICE_PORTS[$service_name]}"
        for port in "${ports[@]}"; do
            cmd+=("--publish" "${port}")
        done
    fi
    
    # Add volume mounts
    local service_logs="${LOG_DIR}/${service_name}"
    mkdir -p "${service_logs}"
    cmd+=("--mount" "type=bind,source=${service_logs},destination=/app/logs")
    
    # Add configuration mount
    cmd+=("--mount" "type=bind,source=${CONFIG_DIR},destination=/app/config")
    
    # Add environment-specific mounts and settings
    case "${service_name}" in
        "neo4j")
            cmd+=("--env" "NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}")
            cmd+=("--env" "NEO4J_PLUGINS=[\"apoc\"]")
            cmd+=("--env" "NEO4J_dbms_security_procedures_unrestricted=apoc.*")
            cmd+=("--env" "NEO4J_apoc_export_file_enabled=true")
            cmd+=("--env" "NEO4J_apoc_import_file_enabled=true")
            ;;
        "redis")
            cmd+=("--env" "REDIS_PASSWORD=${REDIS_PASSWORD}")
            ;;
        "python-executor")
            cmd+=("--env" "FLASK_APP=python_executor.py")
            cmd+=("--env" "PYTHONPATH=/app")
            ;;
        "alita-"* | "kgot-"* | "*-service")
            cmd+=("--env" "NODE_ENV=production")
            cmd+=("--env" "LOG_LEVEL=${LOG_LEVEL}")
            cmd+=("--env" "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}")
            
            # Add database connections
            cmd+=("--env" "NEO4J_URI=bolt://neo4j:7687")
            cmd+=("--env" "NEO4J_USER=neo4j")
            cmd+=("--env" "NEO4J_PASSWORD=${NEO4J_PASSWORD}")
            cmd+=("--env" "REDIS_URL=redis://redis:6379")
            ;;
    esac
    
    # Add resource limits if supported by Sarus configuration
    if [[ -n "${SERVICE_RESOURCES[$service_name]:-}" ]]; then
        IFS=':' read -ra resources <<< "${SERVICE_RESOURCES[$service_name]}"
        local memory_mb="${resources[0]}"
        local cpu_cores="${resources[1]}"
        
        # Note: Sarus resource limits depend on the underlying container runtime
        # These may need to be adjusted based on your Sarus configuration
        cmd+=("--env" "MEMORY_LIMIT=${memory_mb}m")
        cmd+=("--env" "CPU_LIMIT=${cpu_cores}")
    fi
    
    # Add network configuration
    cmd+=("--net" "host")  # Use host networking for simplicity in HPC
    
    # Add the image
    cmd+=("${image}")
    
    # Add service-specific commands
    case "${service_name}" in
        "python-executor")
            cmd+=("bash" "-c" "pip3 install --no-cache-dir -r requirements.txt && waitress-serve --port=5000 python_executor:app")
            ;;
        "redis")
            cmd+=("redis-server" "--requirepass" "${REDIS_PASSWORD}")
            ;;
    esac
    
    echo "${cmd[@]}"
}

# Start a service
start_service() {
    local service_name="$1"
    
    if [[ -z "${SERVICES[$service_name]:-}" ]]; then
        log_error "Unknown service: ${service_name}"
        return 1
    fi
    
    # Check if service is already running
    if is_service_running "${service_name}"; then
        log_warning "Service ${service_name} is already running"
        return 0
    fi
    
    log_info "Starting service: ${service_name}"
    
    # Build and execute Sarus command
    local sarus_cmd
    sarus_cmd=$(build_sarus_command "${service_name}")
    
    log_info "Executing: ${sarus_cmd}"
    
    # Start service in background
    eval "${sarus_cmd}" > "${SARUS_LOG_DIR}/${service_name}.log" 2>&1 &
    local pid=$!
    
    # Store PID for tracking
    echo "${pid}" > "${SARUS_LOG_DIR}/${service_name}.pid"
    
    # Wait a moment and check if process started successfully
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
        log_success "Service ${service_name} started successfully (PID: ${pid})"
        RUNNING_SERVICES+=("${service_name}")
        return 0
    else
        log_error "Failed to start service ${service_name}"
        rm -f "${SARUS_LOG_DIR}/${service_name}.pid"
        return 1
    fi
}

# Stop a service
stop_service() {
    local service_name="$1"
    local pid_file="${SARUS_LOG_DIR}/${service_name}.pid"
    
    if [[ ! -f "${pid_file}" ]]; then
        log_warning "Service ${service_name} PID file not found"
        return 0
    fi
    
    local pid
    pid=$(cat "${pid_file}")
    
    if ! kill -0 "${pid}" 2>/dev/null; then
        log_warning "Service ${service_name} process not running"
        rm -f "${pid_file}"
        return 0
    fi
    
    log_info "Stopping service: ${service_name} (PID: ${pid})"
    
    # Send SIGTERM
    if kill -TERM "${pid}" 2>/dev/null; then
        # Wait for graceful shutdown
        local timeout=30
        while kill -0 "${pid}" 2>/dev/null && ((timeout > 0)); do
            sleep 1
            ((timeout--))
        done
        
        # Force kill if still running
        if kill -0 "${pid}" 2>/dev/null; then
            log_warning "Force killing service ${service_name}"
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        
        log_success "Service ${service_name} stopped"
    else
        log_error "Failed to stop service ${service_name}"
        return 1
    fi
    
    rm -f "${pid_file}"
    
    # Remove from running services array
    RUNNING_SERVICES=($(printf '%s\n' "${RUNNING_SERVICES[@]}" | grep -v "^${service_name}$" || true))
}

# Check if service is running
is_service_running() {
    local service_name="$1"
    local pid_file="${SARUS_LOG_DIR}/${service_name}.pid"
    
    if [[ ! -f "${pid_file}" ]]; then
        return 1
    fi
    
    local pid
    pid=$(cat "${pid_file}")
    kill -0 "${pid}" 2>/dev/null
}

# Wait for service dependencies
wait_for_dependencies() {
    local service_name="$1"
    local deps="${SERVICE_DEPS[$service_name]:-}"
    
    if [[ -z "${deps}" ]]; then
        return 0
    fi
    
    log_info "Waiting for dependencies of ${service_name}: ${deps}"
    
    local max_wait=300  # 5 minutes
    local elapsed=0
    
    while ((elapsed < max_wait)); do
        local all_ready=true
        
        for dep in ${deps}; do
            if ! is_service_running "${dep}"; then
                all_ready=false
                break
            fi
        done
        
        if "${all_ready}"; then
            log_success "All dependencies ready for ${service_name}"
            return 0
        fi
        
        sleep 5
        ((elapsed += 5))
    done
    
    log_error "Timeout waiting for dependencies of ${service_name}"
    return 1
}

# Health check for services
health_check() {
    local service_name="$1"
    
    if ! is_service_running "${service_name}"; then
        return 1
    fi
    
    # Service-specific health checks
    case "${service_name}" in
        "neo4j")
            # Check if Neo4j is responsive
            timeout 10 bash -c "echo 'RETURN 1;' | cypher-shell -u neo4j -p ${NEO4J_PASSWORD} --address bolt://localhost:7687" &>/dev/null
            ;;
        "redis")
            # Check if Redis is responsive
            timeout 10 redis-cli -a "${REDIS_PASSWORD}" ping &>/dev/null
            ;;
        "python-executor")
            # Check HTTP endpoint
            timeout 10 curl -sf "http://localhost:5000/health" &>/dev/null
            ;;
        "alita-"* | "kgot-"* | "*-service")
            # Check HTTP health endpoint
            local port
            IFS=' ' read -ra ports <<< "${SERVICE_PORTS[$service_name]}"
            port=$(echo "${ports[0]}" | cut -d':' -f2)
            timeout 10 curl -sf "http://localhost:${port}/health" &>/dev/null
            ;;
        *)
            # Default: just check if process is running
            return 0
            ;;
    esac
}

# Monitor service health
monitor_services() {
    log_info "Starting service health monitoring"
    
    while true; do
        for service in "${RUNNING_SERVICES[@]}"; do
            if ! health_check "${service}"; then
                log_error "Health check failed for service: ${service}"
                # Optionally restart service
                # restart_service "${service}"
            else
                log_info "Health check passed for service: ${service}"
            fi
        done
        
        sleep 30  # Check every 30 seconds
    done
}

# Deploy service stack
deploy_stack() {
    log_info "Deploying KGoT-Alita service stack with Sarus"
    
    # Determine services to deploy
    local services_to_deploy
    if [[ -n "${SELECTED_SERVICES}" ]]; then
        IFS=',' read -ra services_to_deploy <<< "${SELECTED_SERVICES}"
    else
        services_to_deploy=("neo4j" "redis" "rdf4j" "python-executor" "alita-manager" "kgot-controller" "kgot-graph-store" "kgot-tools")
    fi
    
    # Sort services by dependencies
    local sorted_services=()
    local remaining_services=("${services_to_deploy[@]}")
    
    while ((${#remaining_services[@]} > 0)); do
        local ready_services=()
        
        for service in "${remaining_services[@]}"; do
            local deps="${SERVICE_DEPS[$service]:-}"
            local deps_met=true
            
            for dep in ${deps}; do
                if [[ ! " ${sorted_services[*]} " =~ " ${dep} " ]]; then
                    deps_met=false
                    break
                fi
            done
            
            if "${deps_met}"; then
                ready_services+=("${service}")
            fi
        done
        
        if ((${#ready_services[@]} == 0)); then
            log_warning "Circular dependencies detected, starting remaining services: ${remaining_services[*]}"
            ready_services=("${remaining_services[@]}")
        fi
        
        for service in "${ready_services[@]}"; do
            if wait_for_dependencies "${service}"; then
                if start_service "${service}"; then
                    sorted_services+=("${service}")
                else
                    log_error "Failed to start ${service}, aborting deployment"
                    return 1
                fi
            else
                log_error "Dependencies not met for ${service}, aborting deployment"
                return 1
            fi
            
            # Remove from remaining services
            remaining_services=($(printf '%s\n' "${remaining_services[@]}" | grep -v "^${service}$"))
        done
    done
    
    log_success "Service stack deployment completed"
    
    # Start health monitoring if enabled
    if "${HEALTH_CHECK_ENABLED}"; then
        monitor_services &
        echo $! > "${SARUS_LOG_DIR}/monitor.pid"
    fi
}

# Stop service stack
stop_stack() {
    log_info "Stopping KGoT-Alita service stack"
    
    # Stop monitoring
    local monitor_pid_file="${SARUS_LOG_DIR}/monitor.pid"
    if [[ -f "${monitor_pid_file}" ]]; then
        local monitor_pid
        monitor_pid=$(cat "${monitor_pid_file}")
        if kill -0 "${monitor_pid}" 2>/dev/null; then
            kill "${monitor_pid}" 2>/dev/null || true
        fi
        rm -f "${monitor_pid_file}"
    fi
    
    # Stop services in reverse dependency order
    local services_to_stop=($(printf '%s\n' "${RUNNING_SERVICES[@]}" | tac))
    
    for service in "${services_to_stop[@]}"; do
        stop_service "${service}"
    done
    
    log_success "Service stack stopped"
}

# Show service status
show_status() {
    log_info "KGoT-Alita Service Status"
    echo
    printf "%-20s %-10s %-15s %-10s\n" "SERVICE" "STATUS" "IMAGE" "PID"
    echo "----------------------------------------------------------------"
    
    for service in "${!SERVICES[@]}"; do
        local status="STOPPED"
        local pid="N/A"
        
        if is_service_running "${service}"; then
            status="RUNNING"
            local pid_file="${SARUS_LOG_DIR}/${service}.pid"
            if [[ -f "${pid_file}" ]]; then
                pid=$(cat "${pid_file}")
            fi
            
            # Check health
            if health_check "${service}"; then
                status="HEALTHY"
            else
                status="UNHEALTHY"
            fi
        fi
        
        printf "%-20s %-10s %-15s %-10s\n" "${service}" "${status}" "${SERVICES[$service]}" "${pid}"
    done
    echo
}

# Show help
show_help() {
    cat << EOF
Enhanced Sarus Launcher for KGoT-Alita System

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Deploy the complete service stack
    stop            Stop the complete service stack
    status          Show service status
    start <service> Start a specific service
    stop <service>  Stop a specific service
    logs <service>  Show logs for a service
    help            Show this help message

Options:
    --services <list>     Comma-separated list of services to deploy
    --no-health-check    Disable health monitoring
    --no-monitoring      Disable resource monitoring

Examples:
    $0 deploy                                    # Deploy all services
    $0 deploy --services neo4j,redis            # Deploy specific services
    $0 start kgot-controller                     # Start KGoT controller
    $0 status                                    # Show service status
    $0 logs neo4j                               # Show Neo4j logs

Available Services:
$(printf "    %s\n" "${!SERVICES[@]}" | sort)
EOF
}

# Show service logs
show_logs() {
    local service_name="$1"
    local log_file="${SARUS_LOG_DIR}/${service_name}.log"
    
    if [[ ! -f "${log_file}" ]]; then
        log_error "Log file not found for service: ${service_name}"
        return 1
    fi
    
    log_info "Showing logs for service: ${service_name}"
    echo "Log file: ${log_file}"
    echo "----------------------------------------"
    tail -f "${log_file}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy|stop|status|help)
                COMMAND="$1"
                shift
                ;;
            start|logs)
                COMMAND="$1"
                if [[ $# -lt 2 ]]; then
                    log_error "Service name required for $1 command"
                    exit 1
                fi
                SELECTED_SERVICES="$2"
                shift 2
                ;;
            --services)
                SELECTED_SERVICES="$2"
                shift 2
                ;;
            --no-health-check)
                HEALTH_CHECK_ENABLED=false
                shift
                ;;
            --no-monitoring)
                RESOURCE_MONITORING=false
                shift
                ;;
            -h|--help)
                COMMAND="help"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ -z "${COMMAND}" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
}

# Signal handlers for cleanup
cleanup() {
    log_info "Received signal, cleaning up..."
    stop_stack
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main function
main() {
    init_logging
    check_sarus
    load_environment
    
    # Parse command line arguments
    parse_args "$@"
    
    # Execute command
    case "${COMMAND}" in
        deploy)
            deploy_stack
            ;;
        stop)
            stop_stack
            ;;
        start)
            start_service "${SELECTED_SERVICES}"
            ;;
        stop)
            stop_service "${SELECTED_SERVICES}"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${SELECTED_SERVICES}"
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 