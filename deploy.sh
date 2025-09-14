#!/bin/bash

# deploy.sh
set -e

echo "🚀 Deploying Comedy Discovery App..."

# Configuration
ENVIRONMENT=${1:-production}
NAMESPACE="comedy-discovery"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if [ "$ENVIRONMENT" != "development" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed"
            exit 1
        fi
        
        # Check if kubectl can connect to cluster
        if ! kubectl cluster-info &> /dev/null; then
            log_error "Cannot connect to Kubernetes cluster"
            exit 1
        fi
    fi
    
    # Check if .env files exist
    if [ ! -f "backend/.env" ]; then
        if [ -f "backend/.env.example" ]; then
            log_warn "backend/.env not found. Copying from .env.example - please configure it."
            cp backend/.env.example backend/.env
        else
            log_error "backend/.env not found and no .env.example available"
            exit 1
        fi
    fi
    
    if [ ! -f "frontend/.env.local" ]; then
        if [ -f "frontend/.env.local.example" ]; then
            log_warn "frontend/.env.local not found. Copying from .env.local.example - please configure it."
            cp frontend/.env.local.example frontend/.env.local
        else
            log_error "frontend/.env.local not found and no .env.local.example available"
            exit 1
        fi
    fi
    
    log_info "Prerequisites check passed ✅"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build backend
    log_info "Building backend image..."
    if ! docker build -t comedy-app/backend:latest ./backend; then
        log_error "Failed to build backend image"
        exit 1
    fi
    
    # Build frontend
    log_info "Building frontend image..."
    if ! docker build -t comedy-app/frontend:latest ./frontend; then
        log_error "Failed to build frontend image"
        exit 1
    fi
    
    log_info "Docker images built successfully ✅"
}

# Start development environment
start_development() {
    log_info "Starting development environment..."
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose is not installed"
        exit 1
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Start services
    log_info "Starting services with docker-compose..."
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 10
    
    # Health check
    log_info "Performing health check..."
    
    # Check backend health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_info "Backend is healthy ✅"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Backend health check failed"
            docker-compose logs backend
            exit 1
        fi
        sleep 2
    done
    
    # Check frontend
    for i in {1..30}; do
        if curl -f http://localhost:3000 &> /dev/null; then
            log_info "Frontend is healthy ✅"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Frontend health check failed"
            docker-compose logs frontend
            exit 1
        fi
        sleep 2
    done
    
    log_info "🎉 Development environment is ready!"
    log_info "Access the application at:"
    log_info "  Frontend: http://localhost:3000"
    log_info "  Backend API: http://localhost:8000"
    log_info "  API Documentation: http://localhost:8000/docs"
    log_info "  Prometheus (if enabled): http://localhost:9090"
    log_info "  Grafana (if enabled): http://localhost:3001 (admin/admin123)"
}

# Deploy to Kubernetes
deploy_to_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    log_info "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Check if k8s directory exists
    if [ ! -d "k8s" ]; then
        log_error "k8s directory not found. Please create Kubernetes manifests."
        exit 1
    fi
    
    # Create secrets from environment files
    log_info "Creating secrets..."
    
    # Create secret from backend .env file
    kubectl create secret generic app-secrets \
        --from-env-file=backend/.env \
        --namespace=$NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    if ! kubectl apply -f k8s/ --namespace=$NAMESPACE; then
        log_error "Failed to apply Kubernetes manifests"
        exit 1
    fi
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    
    deployments=("redis" "comedy-backend" "comedy-frontend")
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for deployment: $deployment"
        if ! kubectl wait --for=condition=available --timeout=300s deployment/$deployment --namespace=$NAMESPACE; then
            log_error "Deployment $deployment failed to become ready"
            kubectl get pods --namespace=$NAMESPACE
            kubectl describe deployment $deployment --namespace=$NAMESPACE
            exit 1
        fi
    done
    
    log_info "All deployments are ready ✅"
}

# Run database migrations/setup
setup_database() {
    log_info "Setting up database..."
    
    # Check if database setup script exists
    if [ -f "scripts/setup-database.sql" ]; then
        log_info "Database setup script found"
        # You would run this against your Supabase instance
        log_warn "Please run the database setup script manually in your Supabase dashboard"
    else
        log_info "No database setup script found"
    fi
    
    log_info "Database setup instructions:"
    log_info "1. Log into your Supabase dashboard"
    log_info "2. Go to SQL Editor"
    log_info "3. Run the database schema from the setup guide"
    log_info "4. Insert seed data for comedians and media items"
    
    log_warn "Database setup requires manual steps - see README for details"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    if [ "$ENVIRONMENT" = "development" ]; then
        # Check local services
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_info "Backend health check passed ✅"
        else
            log_error "Backend health check failed ❌"
        fi
        
        if curl -f http://localhost:3000 &> /dev/null; then
            log_info "Frontend health check passed ✅"
        else
            log_error "Frontend health check failed ❌"
        fi
    else
        # Check Kubernetes services
        log_info "Checking pod status..."
        kubectl get pods --namespace=$NAMESPACE
        
        # Get service information
        log_info "Service information:"
        kubectl get services --namespace=$NAMESPACE
        
        # Try to get external IP
        EXTERNAL_IP=$(kubectl get service comedy-frontend-service --namespace=$NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        
        if [ "$EXTERNAL_IP" = "pending" ] || [ -z "$EXTERNAL_IP" ]; then
            log_warn "External IP not yet available. Check with:"
            log_warn "kubectl get services --namespace=$NAMESPACE"
        else
            log_info "Application should be available at: http://$EXTERNAL_IP"
        fi
    fi
}

# Cleanup function
cleanup() {
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "To stop the development environment, run:"
        log_info "docker-compose down"
    fi
}

# Show logs
show_logs() {
    if [ "$ENVIRONMENT" = "development" ]; then
        log_info "Showing recent logs..."
        docker-compose logs --tail=50
    else
        log_info "To view logs in Kubernetes, run:"
        log_info "kubectl logs -f deployment/comedy-backend --namespace=$NAMESPACE"
        log_info "kubectl logs -f deployment/comedy-frontend --namespace=$NAMESPACE"
    fi
}

# Main deployment flow
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT"
    
    # Set up signal handling for cleanup
    trap cleanup EXIT
    
    check_prerequisites
    build_images
    
    case $ENVIRONMENT in
        "development"|"dev")
            start_development
            ;;
        "production"|"prod"|"staging")
            deploy_to_k8s
            setup_database
            health_check
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Usage: ./deploy.sh [development|production|staging]"
            exit 1
            ;;
    esac
    
    log_info "🎉 Deployment completed for $ENVIRONMENT!"
    
    # Show relevant access information
    case $ENVIRONMENT in
        "development"|"dev")
            log_info ""
            log_info "=== ACCESS INFORMATION ==="
            log_info "Frontend: http://localhost:3000"
            log_info "Backend API: http://localhost:8000"
            log_info "API Documentation: http://localhost:8000/docs"
            log_info "Backend Health: http://localhost:8000/health"
            if docker-compose ps | grep -q prometheus; then
                log_info "Prometheus: http://localhost:9090"
            fi
            if docker-compose ps | grep -q grafana; then
                log_info "Grafana: http://localhost:3001 (admin/admin123)"
            fi
            log_info ""
            log_info "To view logs: docker-compose logs -f"
            log_info "To stop: docker-compose down"
            ;;
        *)
            log_info ""
            log_info "=== KUBERNETES DEPLOYMENT ==="
            log_info "Check status: kubectl get all --namespace=$NAMESPACE"
            log_info "View logs: kubectl logs -f deployment/comedy-backend --namespace=$NAMESPACE"
            log_info "Port forward (for testing): kubectl port-forward service/comedy-frontend-service 3000:3000 --namespace=$NAMESPACE"
            ;;
    esac
}

# Handle command line arguments
case "${1:-}" in
    "-h"|"--help"|"help")
        echo "Comedy Discovery App Deployment Script"
        echo ""
        echo "Examples:"
        echo "  $0 development     # Start local development"
        echo "  $0 production      # Deploy to production"
        echo "  $0 dev --logs      # Start dev and show logs"
        exit 0
        ;;
    "--logs"|"-l")
        SHOW_LOGS=true
        shift
        ;;
esac

# Run main function
if [[ "${SHOW_LOGS:-}" == "true" ]]; then
    main "$@"
    show_logs
else
    main "$@"
fi
        echo "Usage: $0 [ENVIRONMENT] [OPTIONS]"
        echo ""
        echo "Environments:"
        echo "  development, dev    Start local development environment"
        echo "  production, prod    Deploy to production Kubernetes"
        echo "  staging            Deploy to staging Kubernetes"
        echo ""
        echo "Options:"
        echo "  -h, --help         Show this help message"
        echo "  -l, --logs         Show logs after deployment"
        echo ""