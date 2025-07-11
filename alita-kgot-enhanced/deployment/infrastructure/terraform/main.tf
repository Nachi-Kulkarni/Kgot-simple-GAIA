# KGoT Production Infrastructure - Terraform Configuration
# =====================================================
#
# This Terraform configuration provisions the complete infrastructure
# for the KGoT-Alita enhanced system across multiple environments.
#
# Features:
# - Multi-environment support (staging, production)
# - Kubernetes cluster provisioning
# - Container registry setup
# - Monitoring and logging infrastructure
# - Security and networking configuration

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
  
  # Backend configuration for state management
  backend "s3" {
    # Configure via environment variables or terraform init -backend-config
    # bucket = "kgot-terraform-state"
    # key    = "infrastructure/terraform.tfstate"
    # region = "us-west-2"
  }
}

# Variables
variable "environment" {
  description = "Environment name (staging/production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be staging or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "kgot-alita"
}

variable "domain" {
  description = "Base domain for the environment"
  type        = string
}

variable "enable_monitoring" {
  description = "Enable monitoring infrastructure"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "node_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 3
}

variable "node_instance_type" {
  description = "Instance type for worker nodes"
  type        = string
  default     = "t3.medium"
}

variable "container_registry" {
  description = "Container registry configuration"
  type = object({
    type     = string  # "ecr", "gcr", "dockerhub", "local"
    url      = string
    username = optional(string)
    password = optional(string)
  })
  default = {
    type = "local"
    url  = "localhost:5000"
  }
}

# Application secrets variables
variable "openrouter_api_key" {
  description = "OpenRouter API key"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Google API key"
  type        = string
  sensitive   = true
}

variable "neo4j_password" {
  description = "Neo4j database password"
  type        = string
  sensitive   = true
}

variable "redis_password" {
  description = "Redis password"
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "JWT secret for authentication"
  type        = string
  sensitive   = true
}

variable "session_secret" {
  description = "Session secret for web sessions"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub token for repository access"
  type        = string
  sensitive   = true
  default     = ""
}

variable "SERPAPI_API_KEY" {
  description = "SerpAPI key for search functionality"
  type        = string
  sensitive   = true
  default     = ""
}

variable "aws_access_key_id" {
  description = "AWS access key ID"
  type        = string
  sensitive   = true
  default     = ""
}

variable "aws_secret_access_key" {
  description = "AWS secret access key"
  type        = string
  sensitive   = true
  default     = ""
}

# Local values for common tags and naming
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    CreatedAt   = timestamp()
  }
  
  name_prefix = "${var.project_name}-${var.environment}"
  
  # Environment-specific configurations
  env_config = {
    staging = {
      replicas = {
        controller    = 2
        graph_store   = 2
        manager_agent = 1
        web_agent     = 2
        monitoring    = 1
      }
      resources = {
        requests = {
          cpu    = "100m"
          memory = "128Mi"
        }
        limits = {
          cpu    = "500m"
          memory = "512Mi"
        }
      }
    }
    production = {
      replicas = {
        controller    = 3
        graph_store   = 3
        manager_agent = 2
        web_agent     = 4
        monitoring    = 2
      }
      resources = {
        requests = {
          cpu    = "200m"
          memory = "256Mi"
        }
        limits = {
          cpu    = "1000m"
          memory = "1Gi"
        }
      }
    }
  }
}

# Kubernetes Cluster Configuration
resource "kubernetes_namespace" "kgot" {
  metadata {
    name = "${local.name_prefix}"
    
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"      = var.project_name
      "app.kubernetes.io/component" = "namespace"
    })
  }
}

# Blue-Green Deployment Namespaces
resource "kubernetes_namespace" "kgot_blue" {
  metadata {
    name = "${local.name_prefix}-blue"
    
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"      = var.project_name
      "app.kubernetes.io/component" = "blue-environment"
    })
  }
}

resource "kubernetes_namespace" "kgot_green" {
  metadata {
    name = "${local.name_prefix}-green"
    
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"      = var.project_name
      "app.kubernetes.io/component" = "green-environment"
    })
  }
}

# ConfigMap for environment configuration
resource "kubernetes_config_map" "environment_config" {
  metadata {
    name      = "environment-config"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  data = {
    "config.yaml" = yamlencode({
      environment = var.environment
      domain      = var.domain
      services    = local.env_config[var.environment]
      monitoring = {
        enabled = var.enable_monitoring
        prometheus = {
          scrape_interval = "15s"
          retention       = "30d"
        }
      }
      logging = {
        enabled = var.enable_logging
        level   = var.environment == "production" ? "INFO" : "DEBUG"
      }
    })
  }
}

# Secret for container registry credentials
resource "kubernetes_secret" "registry_credentials" {
  count = var.container_registry.username != null ? 1 : 0
  
  metadata {
    name      = "registry-credentials"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  type = "kubernetes.io/dockerconfigjson"
  
  data = {
    ".dockerconfigjson" = jsonencode({
      auths = {
        (var.container_registry.url) = {
          username = var.container_registry.username
          password = var.container_registry.password
          auth     = base64encode("${var.container_registry.username}:${var.container_registry.password}")
        }
      }
    })
  }
}

# Application secrets for KGoT services
resource "kubernetes_secret" "application_secrets" {
  metadata {
    name      = "application-secrets"
    namespace = kubernetes_namespace.kgot.metadata[0].name
    
    labels = merge(local.common_tags, {
      "app.kubernetes.io/name"      = var.project_name
      "app.kubernetes.io/component" = "secrets"
    })
  }
  
  type = "Opaque"
  
  data = {
    # Core API Keys
    "OPENROUTER_API_KEY"  = var.openrouter_api_key

    "GOOGLE_API_KEY"      = var.google_api_key
    
    # Database credentials
    "NEO4J_PASSWORD"      = var.neo4j_password
    "REDIS_PASSWORD"      = var.redis_password
    
    # Security tokens
    "JWT_SECRET"          = var.jwt_secret
    "SESSION_SECRET"      = var.session_secret
    
    # External service keys
    "GITHUB_TOKEN"        = var.github_token
    "SERPAPI_API_KEY"         = var.SERPAPI_API_KEY
    
    # AWS credentials
    "AWS_ACCESS_KEY_ID"     = var.aws_access_key_id
    "AWS_SECRET_ACCESS_KEY" = var.aws_secret_access_key
  }
}

# Network Policies for Security
resource "kubernetes_network_policy" "deny_all" {
  metadata {
    name      = "deny-all"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  spec {
    pod_selector {}
    policy_types = ["Ingress", "Egress"]
  }
}

resource "kubernetes_network_policy" "allow_kgot_internal" {
  metadata {
    name      = "allow-kgot-internal"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  spec {
    pod_selector {
      match_labels = {
        "app.kubernetes.io/part-of" = var.project_name
      }
    }
    
    policy_types = ["Ingress", "Egress"]
    
    ingress {
      from {
        pod_selector {
          match_labels = {
            "app.kubernetes.io/part-of" = var.project_name
          }
        }
      }
    }
    
    egress {
      to {
        pod_selector {
          match_labels = {
            "app.kubernetes.io/part-of" = var.project_name
          }
        }
      }
    }
    
    # Allow egress to DNS and external services
    egress {
      ports {
        port     = "53"
        protocol = "TCP"
      }
      ports {
        port     = "53"
        protocol = "UDP"
      }
    }
    
    egress {
      ports {
        port     = "80"
        protocol = "TCP"
      }
      ports {
        port     = "443"
        protocol = "TCP"
      }
    }
  }
}

# RBAC Configuration
resource "kubernetes_service_account" "kgot_deployer" {
  metadata {
    name      = "kgot-deployer"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
}

resource "kubernetes_cluster_role" "kgot_deployer" {
  metadata {
    name = "${local.name_prefix}-deployer"
  }
  
  rule {
    api_groups = [""]
    resources  = ["pods", "services", "configmaps", "secrets"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }
  
  rule {
    api_groups = ["apps"]
    resources  = ["deployments", "replicasets"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }
  
  rule {
    api_groups = ["networking.k8s.io"]
    resources  = ["ingresses", "networkpolicies"]
    verbs      = ["get", "list", "watch", "create", "update", "patch", "delete"]
  }
}

resource "kubernetes_cluster_role_binding" "kgot_deployer" {
  metadata {
    name = "${local.name_prefix}-deployer"
  }
  
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.kgot_deployer.metadata[0].name
  }
  
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.kgot_deployer.metadata[0].name
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
}

# Persistent Volumes for Data Storage
resource "kubernetes_persistent_volume_claim" "graph_store_data" {
  metadata {
    name      = "graph-store-data"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  spec {
    access_modes = ["ReadWriteOnce"]
    
    resources {
      requests = {
        storage = var.environment == "production" ? "100Gi" : "20Gi"
      }
    }
    
    storage_class_name = "fast"
  }
}

resource "kubernetes_persistent_volume_claim" "monitoring_data" {
  count = var.enable_monitoring ? 1 : 0
  
  metadata {
    name      = "monitoring-data"
    namespace = kubernetes_namespace.kgot.metadata[0].name
  }
  
  spec {
    access_modes = ["ReadWriteOnce"]
    
    resources {
      requests = {
        storage = var.environment == "production" ? "50Gi" : "10Gi"
      }
    }
    
    storage_class_name = "standard"
  }
}

# Monitoring Infrastructure (Prometheus & Grafana)
resource "helm_release" "prometheus" {
  count = var.enable_monitoring ? 1 : 0
  
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "51.2.0"
  namespace  = kubernetes_namespace.kgot.metadata[0].name
  
  values = [
    yamlencode({
      prometheus = {
        prometheusSpec = {
          retention    = "30d"
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = var.environment == "production" ? "50Gi" : "10Gi"
                  }
                }
              }
            }
          }
        }
      }
      
      grafana = {
        adminPassword = "admin"  # Change this in production!
        ingress = {
          enabled = true
          hosts   = ["grafana.${var.domain}"]
        }
      }
      
      alertmanager = {
        enabled = true
      }
    })
  ]
  
  depends_on = [kubernetes_namespace.kgot]
}

# Logging Infrastructure (ELK Stack)
resource "helm_release" "elasticsearch" {
  count = var.enable_logging ? 1 : 0
  
  name       = "elasticsearch"
  repository = "https://helm.elastic.co"
  chart      = "elasticsearch"
  version    = "8.5.1"
  namespace  = kubernetes_namespace.kgot.metadata[0].name
  
  values = [
    yamlencode({
      replicas = var.environment == "production" ? 3 : 1
      
      resources = {
        requests = {
          cpu    = "100m"
          memory = "1Gi"
        }
        limits = {
          cpu    = "1000m"
          memory = "2Gi"
        }
      }
      
      volumeClaimTemplate = {
        accessModes = ["ReadWriteOnce"]
        resources = {
          requests = {
            storage = var.environment == "production" ? "30Gi" : "10Gi"
          }
        }
      }
    })
  ]
  
  depends_on = [kubernetes_namespace.kgot]
}

resource "helm_release" "kibana" {
  count = var.enable_logging ? 1 : 0
  
  name       = "kibana"
  repository = "https://helm.elastic.co"
  chart      = "kibana"
  version    = "8.5.1"
  namespace  = kubernetes_namespace.kgot.metadata[0].name
  
  values = [
    yamlencode({
      ingress = {
        enabled = true
        hosts = [
          {
            host = "kibana.${var.domain}"
            paths = [
              {
                path = "/"
              }
            ]
          }
        ]
      }
    })
  ]
  
  depends_on = [helm_release.elasticsearch]
}

# Ingress Controller
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.7.1"
  namespace  = kubernetes_namespace.kgot.metadata[0].name
  
  values = [
    yamlencode({
      controller = {
        service = {
          type = "LoadBalancer"
        }
        
        config = {
          enable-real-ip = "true"
          use-forwarded-headers = "true"
        }
        
        metrics = {
          enabled = var.enable_monitoring
        }
      }
    })
  ]
  
  depends_on = [kubernetes_namespace.kgot]
}

# Outputs
output "namespace" {
  description = "Kubernetes namespace for KGoT deployment"
  value       = kubernetes_namespace.kgot.metadata[0].name
}

output "blue_namespace" {
  description = "Blue environment namespace"
  value       = kubernetes_namespace.kgot_blue.metadata[0].name
}

output "green_namespace" {
  description = "Green environment namespace"
  value       = kubernetes_namespace.kgot_green.metadata[0].name
}

output "service_account" {
  description = "Service account for deployments"
  value       = kubernetes_service_account.kgot_deployer.metadata[0].name
}

output "domain" {
  description = "Domain for the environment"
  value       = var.domain
}

output "monitoring_enabled" {
  description = "Whether monitoring is enabled"
  value       = var.enable_monitoring
}

output "logging_enabled" {
  description = "Whether logging is enabled"
  value       = var.enable_logging
}