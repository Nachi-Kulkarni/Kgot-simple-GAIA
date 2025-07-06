#!/usr/bin/env python3
"""
Advanced API Gateway for Alita-KGoT Enhanced System

This module implements a production-grade API Gateway that serves as the centralized
entry point for all external requests to the unified Alita-KGoT system.

Features:
- Intelligent rate limiting with multi-tier strategies
- Comprehensive API analytics and monitoring
- Secure third-party integration management
- Dynamic routing with circuit breaker patterns
- Real-time metrics and health monitoring
- JWT and API key authentication
- Request/response transformation
- Cost-based throttling

Source Paper Principles:
- KGoT's emphasis on control and monitoring (Section 3.2, 3.6)
- RAG-MCP's focus on efficient, streamlined interaction (Section 1.2)

Author: Alita-KGoT Enhanced System
Date: 2024
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import hashlib
import hmac
import base64
from urllib.parse import urlparse

# FastAPI and middleware imports
from fastapi import FastAPI, Request, Response, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import RequestResponseEndpoint
import uvicorn

# Redis and caching
import redis.asyncio as redis
from redis.exceptions import RedisError

# JWT and security
import jwt
from passlib.context import CryptContext

# HTTP client for proxying
import httpx

# Logging and monitoring
import logging
from core.enhanced_logging_system import StructuredLogger, LogLevel, LogCategory
from monitoring.production_monitoring import ProductionMonitoringSystem
from core.shared_state_utilities import EnhancedSharedStateManager, StateScope

# Configuration
import os
from pathlib import Path


class RateLimitTier(Enum):
    """Rate limiting tiers for different user types"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    SYSTEM = "system"


class AuthMethod(Enum):
    """Supported authentication methods"""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    HMAC = "hmac"
    NONE = "none"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration for different tiers"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    cost_limit_per_hour: float
    concurrent_requests: int


@dataclass
class APIKeyInfo:
    """API key information and metadata"""
    key_id: str
    user_id: str
    tier: RateLimitTier
    permissions: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    metadata: Dict[str, Any]


@dataclass
class RequestMetrics:
    """Request metrics for analytics"""
    request_id: str
    timestamp: datetime
    method: str
    path: str
    user_id: Optional[str]
    api_key_id: Optional[str]
    response_time_ms: float
    status_code: int
    request_size: int
    response_size: int
    cost: float
    error_type: Optional[str]
    user_agent: Optional[str]
    ip_address: str
    tier: Optional[RateLimitTier]


class GatewayConfig:
    """Configuration management for the API Gateway"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.jwt_secret = os.getenv("JWT_SECRET", "your-secret-key")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        
        # Rate limiting configurations
        self.rate_limits = {
            RateLimitTier.FREE: RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                burst_limit=20,
                cost_limit_per_hour=1.0,
                concurrent_requests=2
            ),
            RateLimitTier.BASIC: RateLimitConfig(
                requests_per_minute=50,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=100,
                cost_limit_per_hour=10.0,
                concurrent_requests=5
            ),
            RateLimitTier.PREMIUM: RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_limit=500,
                cost_limit_per_hour=100.0,
                concurrent_requests=20
            ),
            RateLimitTier.ENTERPRISE: RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=25000,
                requests_per_day=250000,
                burst_limit=2000,
                cost_limit_per_hour=1000.0,
                concurrent_requests=100
            ),
            RateLimitTier.SYSTEM: RateLimitConfig(
                requests_per_minute=10000,
                requests_per_hour=100000,
                requests_per_day=1000000,
                burst_limit=20000,
                cost_limit_per_hour=10000.0,
                concurrent_requests=1000
            )
        }
        
        # Service endpoints
        self.service_endpoints = {
            "/api/alita": "http://localhost:8000",
            "/api/kgot": "http://localhost:16000",
            "/api/mcp-brainstorming": "http://localhost:8001",
            "/api/web-agent": "http://localhost:8000",
            "/api/unified-system": "http://localhost:9000",
            "/api/monitoring": "http://localhost:9001"
        }
        
        # Circuit breaker settings
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_timeout = 30
        self.circuit_breaker_recovery_timeout = 60
        
        # Analytics settings
        self.analytics_batch_size = 100
        self.analytics_flush_interval = 30
        
        # Security settings
        self.allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        self.api_key_header = "X-API-Key"
        self.hmac_header = "X-HMAC-Signature"
        

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with multi-tier support"""
    
    def __init__(self, app, redis_client: redis.Redis, config: GatewayConfig, logger: StructuredLogger):
        super().__init__(app)
        self.redis = redis_client
        self.config = config
        self.logger = logger
        self.active_requests = defaultdict(int)
        
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through rate limiting"""
        start_time = time.time()
        
        # Extract user identification
        user_id = getattr(request.state, 'user_id', None)
        api_key_id = getattr(request.state, 'api_key_id', None)
        tier = getattr(request.state, 'tier', RateLimitTier.FREE)
        ip_address = request.client.host
        
        # Create rate limit key
        rate_limit_key = self._create_rate_limit_key(user_id, api_key_id, ip_address)
        
        try:
            # Check rate limits
            rate_limit_result = await self._check_rate_limits(rate_limit_key, tier, request)
            
            if not rate_limit_result["allowed"]:
                self.logger.log_security_event(
                    "RATE_LIMIT_EXCEEDED",
                    f"Rate limit exceeded for {rate_limit_key}",
                    {
                        "user_id": user_id,
                        "api_key_id": api_key_id,
                        "ip_address": ip_address,
                        "tier": tier.value,
                        "limit_type": rate_limit_result["limit_type"]
                    }
                )
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "limit_type": rate_limit_result["limit_type"],
                        "retry_after": rate_limit_result["retry_after"]
                    },
                    headers={"Retry-After": str(rate_limit_result["retry_after"])}
                )
            
            # Check concurrent requests
            if self.active_requests[rate_limit_key] >= self.config.rate_limits[tier].concurrent_requests:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many concurrent requests"},
                    headers={"Retry-After": "1"}
                )
            
            # Increment active requests
            self.active_requests[rate_limit_key] += 1
            
            # Process request
            response = await call_next(request)
            
            # Update rate limit counters
            await self._update_rate_limit_counters(rate_limit_key, tier)
            
            # Add rate limit headers
            remaining = await self._get_remaining_requests(rate_limit_key, tier)
            response.headers["X-RateLimit-Limit"] = str(self.config.rate_limits[tier].requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
            
            return response
            
        except Exception as e:
            self.logger.log_error("RATE_LIMITING_ERROR", e, {"rate_limit_key": rate_limit_key})
            # Allow request to proceed on rate limiting errors
            return await call_next(request)
            
        finally:
            # Decrement active requests
            if rate_limit_key in self.active_requests:
                self.active_requests[rate_limit_key] = max(0, self.active_requests[rate_limit_key] - 1)
    
    def _create_rate_limit_key(self, user_id: Optional[str], api_key_id: Optional[str], ip_address: str) -> str:
        """Create a unique rate limit key"""
        if user_id:
            return f"user:{user_id}"
        elif api_key_id:
            return f"api_key:{api_key_id}"
        else:
            return f"ip:{ip_address}"
    
    async def _check_rate_limits(self, key: str, tier: RateLimitTier, request: Request) -> Dict[str, Any]:
        """Check all rate limit types"""
        config = self.config.rate_limits[tier]
        current_time = int(time.time())
        
        # Check minute limit
        minute_key = f"rate_limit:{key}:minute:{current_time // 60}"
        minute_count = await self.redis.get(minute_key) or 0
        minute_count = int(minute_count)
        
        if minute_count >= config.requests_per_minute:
            return {
                "allowed": False,
                "limit_type": "per_minute",
                "retry_after": 60 - (current_time % 60)
            }
        
        # Check hour limit
        hour_key = f"rate_limit:{key}:hour:{current_time // 3600}"
        hour_count = await self.redis.get(hour_key) or 0
        hour_count = int(hour_count)
        
        if hour_count >= config.requests_per_hour:
            return {
                "allowed": False,
                "limit_type": "per_hour",
                "retry_after": 3600 - (current_time % 3600)
            }
        
        # Check day limit
        day_key = f"rate_limit:{key}:day:{current_time // 86400}"
        day_count = await self.redis.get(day_key) or 0
        day_count = int(day_count)
        
        if day_count >= config.requests_per_day:
            return {
                "allowed": False,
                "limit_type": "per_day",
                "retry_after": 86400 - (current_time % 86400)
            }
        
        # Check cost limit (if applicable)
        cost_key = f"cost_limit:{key}:hour:{current_time // 3600}"
        hour_cost = await self.redis.get(cost_key) or 0
        hour_cost = float(hour_cost)
        
        estimated_cost = self._estimate_request_cost(request)
        if hour_cost + estimated_cost > config.cost_limit_per_hour:
            return {
                "allowed": False,
                "limit_type": "cost_limit",
                "retry_after": 3600 - (current_time % 3600)
            }
        
        return {"allowed": True}
    
    async def _update_rate_limit_counters(self, key: str, tier: RateLimitTier):
        """Update rate limit counters in Redis"""
        current_time = int(time.time())
        
        # Update counters with expiration
        minute_key = f"rate_limit:{key}:minute:{current_time // 60}"
        hour_key = f"rate_limit:{key}:hour:{current_time // 3600}"
        day_key = f"rate_limit:{key}:day:{current_time // 86400}"
        
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 120)  # 2 minutes
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7200)  # 2 hours
        pipe.incr(day_key)
        pipe.expire(day_key, 172800)  # 2 days
        await pipe.execute()
    
    async def _get_remaining_requests(self, key: str, tier: RateLimitTier) -> int:
        """Get remaining requests for the current minute"""
        current_time = int(time.time())
        minute_key = f"rate_limit:{key}:minute:{current_time // 60}"
        minute_count = await self.redis.get(minute_key) or 0
        minute_count = int(minute_count)
        
        return max(0, self.config.rate_limits[tier].requests_per_minute - minute_count)
    
    def _estimate_request_cost(self, request: Request) -> float:
        """Estimate the cost of a request based on endpoint and payload"""
        # Basic cost estimation - can be enhanced with ML models
        base_cost = 0.01
        
        # Higher cost for complex endpoints
        if "/api/kgot" in str(request.url):
            base_cost *= 5
        elif "/api/mcp-brainstorming" in str(request.url):
            base_cost *= 3
        
        # Factor in payload size
        content_length = request.headers.get("content-length", "0")
        payload_factor = min(10, int(content_length) / 1000)  # Cap at 10x
        
        return base_cost * (1 + payload_factor)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware supporting multiple auth methods"""
    
    def __init__(self, app, redis_client: redis.Redis, config: GatewayConfig, logger: StructuredLogger):
        super().__init__(app)
        self.redis = redis_client
        self.config = config
        self.logger = logger
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/", "/health", "/metrics", "/docs", "/openapi.json",
            "/api/auth/login", "/api/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through authentication"""
        
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        try:
            # Determine authentication method
            auth_method = self._determine_auth_method(request)
            
            if auth_method == AuthMethod.NONE:
                # No authentication provided - use anonymous access
                request.state.user_id = None
                request.state.api_key_id = None
                request.state.tier = RateLimitTier.FREE
                request.state.permissions = ["read"]
                
            elif auth_method == AuthMethod.API_KEY:
                auth_result = await self._authenticate_api_key(request)
                if not auth_result["success"]:
                    return self._create_auth_error_response(auth_result["error"])
                
                request.state.user_id = auth_result["user_id"]
                request.state.api_key_id = auth_result["api_key_id"]
                request.state.tier = auth_result["tier"]
                request.state.permissions = auth_result["permissions"]
                
            elif auth_method == AuthMethod.JWT:
                auth_result = await self._authenticate_jwt(request)
                if not auth_result["success"]:
                    return self._create_auth_error_response(auth_result["error"])
                
                request.state.user_id = auth_result["user_id"]
                request.state.api_key_id = None
                request.state.tier = auth_result["tier"]
                request.state.permissions = auth_result["permissions"]
                
            elif auth_method == AuthMethod.HMAC:
                auth_result = await self._authenticate_hmac(request)
                if not auth_result["success"]:
                    return self._create_auth_error_response(auth_result["error"])
                
                request.state.user_id = auth_result["user_id"]
                request.state.api_key_id = auth_result["api_key_id"]
                request.state.tier = auth_result["tier"]
                request.state.permissions = auth_result["permissions"]
            
            # Log successful authentication
            self.logger.log_security_event(
                "AUTHENTICATION_SUCCESS",
                f"User authenticated via {auth_method.value}",
                {
                    "user_id": request.state.user_id,
                    "api_key_id": request.state.api_key_id,
                    "auth_method": auth_method.value,
                    "tier": request.state.tier.value if hasattr(request.state, 'tier') else None
                }
            )
            
            return await call_next(request)
            
        except Exception as e:
            self.logger.log_error("AUTHENTICATION_ERROR", e)
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication system error"}
            )
    
    def _determine_auth_method(self, request: Request) -> AuthMethod:
        """Determine which authentication method to use"""
        if self.config.api_key_header in request.headers:
            return AuthMethod.API_KEY
        elif "authorization" in request.headers:
            auth_header = request.headers["authorization"]
            if auth_header.startswith("Bearer "):
                return AuthMethod.JWT
        elif self.config.hmac_header in request.headers:
            return AuthMethod.HMAC
        
        return AuthMethod.NONE
    
    async def _authenticate_api_key(self, request: Request) -> Dict[str, Any]:
        """Authenticate using API key"""
        api_key = request.headers.get(self.config.api_key_header)
        if not api_key:
            return {"success": False, "error": "API key required"}
        
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Look up API key info in Redis
        key_data = await self.redis.get(f"api_key:{key_hash}")
        if not key_data:
            return {"success": False, "error": "Invalid API key"}
        
        try:
            key_info = json.loads(key_data)
            
            # Check if key is active
            if not key_info.get("is_active", False):
                return {"success": False, "error": "API key is inactive"}
            
            # Check expiration
            if key_info.get("expires_at"):
                expires_at = datetime.fromisoformat(key_info["expires_at"])
                if datetime.now() > expires_at:
                    return {"success": False, "error": "API key has expired"}
            
            return {
                "success": True,
                "user_id": key_info["user_id"],
                "api_key_id": key_info["key_id"],
                "tier": RateLimitTier(key_info["tier"]),
                "permissions": key_info["permissions"]
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return {"success": False, "error": "Invalid API key data"}
    
    async def _authenticate_jwt(self, request: Request) -> Dict[str, Any]:
        """Authenticate using JWT token"""
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return {"success": False, "error": "Invalid authorization header"}
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            # Check token expiration
            if "exp" in payload and payload["exp"] < time.time():
                return {"success": False, "error": "Token has expired"}
            
            return {
                "success": True,
                "user_id": payload.get("user_id"),
                "tier": RateLimitTier(payload.get("tier", "free")),
                "permissions": payload.get("permissions", ["read"])
            }
            
        except jwt.InvalidTokenError as e:
            return {"success": False, "error": f"Invalid token: {str(e)}"}
    
    async def _authenticate_hmac(self, request: Request) -> Dict[str, Any]:
        """Authenticate using HMAC signature"""
        signature = request.headers.get(self.config.hmac_header)
        if not signature:
            return {"success": False, "error": "HMAC signature required"}
        
        # Get API key for HMAC verification
        api_key = request.headers.get(self.config.api_key_header)
        if not api_key:
            return {"success": False, "error": "API key required for HMAC"}
        
        # Get request body for signature verification
        body = await request.body()
        
        # Look up API key secret
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = await self.redis.get(f"api_key:{key_hash}")
        if not key_data:
            return {"success": False, "error": "Invalid API key"}
        
        try:
            key_info = json.loads(key_data)
            secret = key_info.get("secret")
            if not secret:
                return {"success": False, "error": "HMAC not supported for this key"}
            
            # Verify HMAC signature
            expected_signature = hmac.new(
                secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return {"success": False, "error": "Invalid HMAC signature"}
            
            return {
                "success": True,
                "user_id": key_info["user_id"],
                "api_key_id": key_info["key_id"],
                "tier": RateLimitTier(key_info["tier"]),
                "permissions": key_info["permissions"]
            }
            
        except (json.JSONDecodeError, KeyError) as e:
            return {"success": False, "error": "Invalid API key data"}
    
    def _create_auth_error_response(self, error_message: str) -> JSONResponse:
        """Create standardized authentication error response"""
        return JSONResponse(
            status_code=401,
            content={"error": "Authentication failed", "message": error_message}
        )


class AnalyticsMiddleware(BaseHTTPMiddleware):
    """Analytics middleware for comprehensive request/response monitoring"""
    
    def __init__(self, app, redis_client: redis.Redis, config: GatewayConfig, 
                 logger: StructuredLogger, monitoring_system: ProductionMonitoringSystem):
        super().__init__(app)
        self.redis = redis_client
        self.config = config
        self.logger = logger
        self.monitoring_system = monitoring_system
        self.metrics_buffer = deque(maxlen=self.config.analytics_batch_size)
        self.last_flush = time.time()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through analytics collection"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Capture request details
        request_size = int(request.headers.get("content-length", "0"))
        
        try:
            response = await call_next(request)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Capture response details
            response_size = 0
            if hasattr(response, 'body'):
                response_size = len(response.body) if response.body else 0
            
            # Create metrics record
            metrics = RequestMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                method=request.method,
                path=request.url.path,
                user_id=getattr(request.state, 'user_id', None),
                api_key_id=getattr(request.state, 'api_key_id', None),
                response_time_ms=response_time_ms,
                status_code=response.status_code,
                request_size=request_size,
                response_size=response_size,
                cost=self._calculate_request_cost(request, response_time_ms),
                error_type=None if response.status_code < 400 else "http_error",
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host,
                tier=getattr(request.state, 'tier', None)
            )
            
            # Buffer metrics for batch processing
            await self._buffer_metrics(metrics)
            
            # Add analytics headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            # Handle errors in request processing
            response_time_ms = (time.time() - start_time) * 1000
            
            error_metrics = RequestMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                method=request.method,
                path=request.url.path,
                user_id=getattr(request.state, 'user_id', None),
                api_key_id=getattr(request.state, 'api_key_id', None),
                response_time_ms=response_time_ms,
                status_code=500,
                request_size=request_size,
                response_size=0,
                cost=0.0,
                error_type=type(e).__name__,
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host,
                tier=getattr(request.state, 'tier', None)
            )
            
            await self._buffer_metrics(error_metrics)
            
            # Re-raise the exception
            raise e
    
    async def _buffer_metrics(self, metrics: RequestMetrics):
        """Buffer metrics for batch processing"""
        self.metrics_buffer.append(asdict(metrics))
        
        # Flush if buffer is full or time threshold reached
        current_time = time.time()
        if (len(self.metrics_buffer) >= self.config.analytics_batch_size or 
            current_time - self.last_flush >= self.config.analytics_flush_interval):
            await self._flush_metrics()
            self.last_flush = current_time
    
    async def _flush_metrics(self):
        """Flush metrics buffer to storage"""
        if not self.metrics_buffer:
            return
        
        try:
            # Convert buffer to list
            metrics_batch = list(self.metrics_buffer)
            self.metrics_buffer.clear()
            
            # Store in Redis for real-time analytics
            pipe = self.redis.pipeline()
            for metrics in metrics_batch:
                # Store individual metric
                pipe.lpush("gateway:metrics", json.dumps(metrics, default=str))
                
                # Update aggregated counters
                timestamp = int(time.time())
                hour_key = f"gateway:stats:hour:{timestamp // 3600}"
                day_key = f"gateway:stats:day:{timestamp // 86400}"
                
                pipe.hincrby(hour_key, "total_requests", 1)
                pipe.hincrby(hour_key, f"status_{metrics['status_code']}", 1)
                pipe.hincrbyfloat(hour_key, "total_response_time", metrics['response_time_ms'])
                pipe.hincrbyfloat(hour_key, "total_cost", metrics['cost'])
                pipe.expire(hour_key, 86400 * 7)  # Keep for 7 days
                
                pipe.hincrby(day_key, "total_requests", 1)
                pipe.hincrby(day_key, f"status_{metrics['status_code']}", 1)
                pipe.hincrbyfloat(day_key, "total_response_time", metrics['response_time_ms'])
                pipe.hincrbyfloat(day_key, "total_cost", metrics['cost'])
                pipe.expire(day_key, 86400 * 30)  # Keep for 30 days
            
            await pipe.execute()
            
            # Send to production monitoring system
            if self.monitoring_system:
                await self._send_to_monitoring(metrics_batch)
            
        except Exception as e:
            self.logger.log_error("ANALYTICS_FLUSH_ERROR", e)
    
    async def _send_to_monitoring(self, metrics_batch: List[Dict[str, Any]]):
        """Send metrics to production monitoring system"""
        try:
            # Aggregate metrics for monitoring
            total_requests = len(metrics_batch)
            avg_response_time = sum(m['response_time_ms'] for m in metrics_batch) / total_requests
            error_count = sum(1 for m in metrics_batch if m['status_code'] >= 400)
            total_cost = sum(m['cost'] for m in metrics_batch)
            
            # Send aggregated metrics
            await self.monitoring_system.log_api_metrics({
                "timestamp": datetime.now(),
                "total_requests": total_requests,
                "avg_response_time_ms": avg_response_time,
                "error_rate": error_count / total_requests,
                "total_cost": total_cost,
                "requests_by_tier": self._aggregate_by_tier(metrics_batch)
            })
            
        except Exception as e:
            self.logger.log_error("MONITORING_SEND_ERROR", e)
    
    def _aggregate_by_tier(self, metrics_batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate request counts by tier"""
        tier_counts = defaultdict(int)
        for metrics in metrics_batch:
            tier = metrics.get('tier')
            if tier:
                tier_counts[tier] += 1
        return dict(tier_counts)
    
    def _calculate_request_cost(self, request: Request, response_time_ms: float) -> float:
        """Calculate the cost of processing a request"""
        # Base cost calculation
        base_cost = 0.001  # $0.001 per request
        
        # Time-based cost (longer requests cost more)
        time_cost = response_time_ms * 0.00001  # $0.00001 per ms
        
        # Endpoint-based cost multiplier
        path = request.url.path
        if "/api/kgot" in path:
            base_cost *= 10  # KGoT operations are more expensive
        elif "/api/mcp-brainstorming" in path:
            base_cost *= 5
        elif "/api/unified-system" in path:
            base_cost *= 3
        
        return base_cost + time_cost


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker middleware for service resilience"""
    
    def __init__(self, app, redis_client: redis.Redis, config: GatewayConfig, logger: StructuredLogger):
        super().__init__(app)
        self.redis = redis_client
        self.config = config
        self.logger = logger
        self.circuit_states = {}  # In-memory circuit state cache
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request through circuit breaker"""
        service = self._get_service_from_path(request.url.path)
        
        if service:
            circuit_state = await self._get_circuit_state(service)
            
            if circuit_state == "open":
                self.logger.log_operation(
                    "CIRCUIT_BREAKER_OPEN",
                    f"Circuit breaker open for service: {service}"
                )
                return JSONResponse(
                    status_code=503,
                    content={"error": "Service temporarily unavailable", "service": service}
                )
            
            elif circuit_state == "half_open":
                # Allow limited requests in half-open state
                if not await self._allow_half_open_request(service):
                    return JSONResponse(
                        status_code=503,
                        content={"error": "Service in recovery mode", "service": service}
                    )
        
        try:
            response = await call_next(request)
            
            # Record success for circuit breaker
            if service and response.status_code < 500:
                await self._record_success(service)
            elif service and response.status_code >= 500:
                await self._record_failure(service)
            
            return response
            
        except Exception as e:
            # Record failure for circuit breaker
            if service:
                await self._record_failure(service)
            raise e
    
    def _get_service_from_path(self, path: str) -> Optional[str]:
        """Extract service name from request path"""
        for endpoint_prefix in self.config.service_endpoints.keys():
            if path.startswith(endpoint_prefix):
                return endpoint_prefix.split("/")[-1]  # Extract service name
        return None
    
    async def _get_circuit_state(self, service: str) -> str:
        """Get current circuit breaker state for service"""
        try:
            state_data = await self.redis.get(f"circuit_breaker:{service}")
            if state_data:
                state_info = json.loads(state_data)
                
                # Check if circuit should transition states
                current_time = time.time()
                
                if state_info["state"] == "open":
                    if current_time - state_info["opened_at"] > self.config.circuit_breaker_recovery_timeout:
                        # Transition to half-open
                        await self._set_circuit_state(service, "half_open")
                        return "half_open"
                
                return state_info["state"]
            
            return "closed"  # Default state
            
        except Exception as e:
            self.logger.log_error("CIRCUIT_BREAKER_STATE_ERROR", e)
            return "closed"  # Fail safe to closed state
    
    async def _set_circuit_state(self, service: str, state: str):
        """Set circuit breaker state for service"""
        try:
            state_info = {
                "state": state,
                "opened_at": time.time() if state == "open" else None,
                "failure_count": 0 if state == "closed" else None
            }
            
            await self.redis.set(
                f"circuit_breaker:{service}",
                json.dumps(state_info),
                ex=3600  # Expire after 1 hour
            )
            
            self.circuit_states[service] = state
            
        except Exception as e:
            self.logger.log_error("CIRCUIT_BREAKER_SET_STATE_ERROR", e)
    
    async def _record_success(self, service: str):
        """Record successful request for circuit breaker"""
        try:
            # Reset failure count on success
            await self.redis.delete(f"circuit_breaker:{service}:failures")
            
            # If in half-open state, transition to closed
            current_state = await self._get_circuit_state(service)
            if current_state == "half_open":
                await self._set_circuit_state(service, "closed")
                self.logger.log_operation(
                    "CIRCUIT_BREAKER_CLOSED",
                    f"Circuit breaker closed for service: {service}"
                )
            
        except Exception as e:
            self.logger.log_error("CIRCUIT_BREAKER_SUCCESS_ERROR", e)
    
    async def _record_failure(self, service: str):
        """Record failed request for circuit breaker"""
        try:
            failure_key = f"circuit_breaker:{service}:failures"
            failure_count = await self.redis.incr(failure_key)
            await self.redis.expire(failure_key, self.config.circuit_breaker_timeout)
            
            if failure_count >= self.config.circuit_breaker_failure_threshold:
                await self._set_circuit_state(service, "open")
                self.logger.log_operation(
                    "CIRCUIT_BREAKER_OPENED",
                    f"Circuit breaker opened for service: {service} (failures: {failure_count})"
                )
            
        except Exception as e:
            self.logger.log_error("CIRCUIT_BREAKER_FAILURE_ERROR", e)
    
    async def _allow_half_open_request(self, service: str) -> bool:
        """Check if request is allowed in half-open state"""
        try:
            # Allow one request per second in half-open state
            half_open_key = f"circuit_breaker:{service}:half_open_requests"
            current_requests = await self.redis.get(half_open_key) or 0
            current_requests = int(current_requests)
            
            if current_requests >= 1:
                return False
            
            await self.redis.incr(half_open_key)
            await self.redis.expire(half_open_key, 1)  # Reset every second
            
            return True
            
        except Exception as e:
            self.logger.log_error("CIRCUIT_BREAKER_HALF_OPEN_ERROR", e)
            return False


class AdvancedAPIGateway:
    """Advanced API Gateway for the Alita-KGoT Enhanced System"""
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.app = FastAPI(
            title="Alita-KGoT Advanced API Gateway",
            description="Production-grade API Gateway with intelligent rate limiting, analytics, and security",
            version="1.0.0"
        )
        
        # Initialize components
        self.redis_client = None
        self.logger = None
        self.monitoring_system = None
        self.state_manager = None
        self.http_client = None
        
        # Middleware instances
        self.rate_limiting_middleware = None
        self.auth_middleware = None
        self.analytics_middleware = None
        self.circuit_breaker_middleware = None
        
        # Service health status
        self.service_health = {}
        
    async def initialize(self):
        """Initialize the API Gateway with all dependencies"""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Initialize logger
            self.logger = StructuredLogger(
                name="advanced_api_gateway",
                level=LogLevel.INFO,
                redis_client=self.redis_client
            )
            
            # Initialize monitoring system
            self.monitoring_system = ProductionMonitoringSystem(
                redis_client=self.redis_client,
                logger=self.logger
            )
            
            # Initialize state manager
            self.state_manager = EnhancedSharedStateManager(
                redis_url=self.config.redis_url,
                enable_streaming=True,
                enable_analytics=True
            )
            await self.state_manager.connect()
            
            # Initialize HTTP client for proxying
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
            )
            
            # Setup middleware
            await self._setup_middleware()
            
            # Setup routes
            self._setup_routes()
            
            # Start background tasks
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self.logger.log_operation(
                "GATEWAY_INITIALIZED",
                "Advanced API Gateway initialized successfully"
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error("GATEWAY_INITIALIZATION_ERROR", e)
            raise e
    
    async def _setup_middleware(self):
        """Setup all middleware components"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Initialize custom middleware
        self.circuit_breaker_middleware = CircuitBreakerMiddleware(
            self.app, self.redis_client, self.config, self.logger
        )
        
        self.analytics_middleware = AnalyticsMiddleware(
            self.app, self.redis_client, self.config, self.logger, self.monitoring_system
        )
        
        self.rate_limiting_middleware = RateLimitingMiddleware(
            self.app, self.redis_client, self.config, self.logger
        )
        
        self.auth_middleware = AuthenticationMiddleware(
            self.app, self.redis_client, self.config, self.logger
        )
        
        # Add middleware in order (last added is executed first)
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.circuit_breaker_middleware.dispatch)
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.analytics_middleware.dispatch)
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.rate_limiting_middleware.dispatch)
        self.app.add_middleware(BaseHTTPMiddleware, dispatch=self.auth_middleware.dispatch)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            """Gateway root endpoint"""
            return {
                "service": "Alita-KGoT Advanced API Gateway",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.now().isoformat(),
                "services": list(self.config.service_endpoints.keys())
            }
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check"""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "redis": await self._check_redis_health(),
                    "services": self.service_health,
                    "middleware": {
                        "authentication": True,
                        "rate_limiting": True,
                        "analytics": True,
                        "circuit_breaker": True
                    }
                }
            }
            
            # Determine overall health
            if not health_status["components"]["redis"]:
                health_status["status"] = "degraded"
            
            unhealthy_services = [k for k, v in self.service_health.items() if not v]
            if unhealthy_services:
                health_status["status"] = "degraded"
                health_status["unhealthy_services"] = unhealthy_services
            
            status_code = 200 if health_status["status"] == "healthy" else 503
            return JSONResponse(content=health_status, status_code=status_code)
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get gateway metrics"""
            try:
                # Get real-time metrics from Redis
                current_hour = int(time.time()) // 3600
                hour_key = f"gateway:stats:hour:{current_hour}"
                
                hour_stats = await self.redis_client.hgetall(hour_key)
                
                # Calculate derived metrics
                total_requests = int(hour_stats.get("total_requests", 0))
                total_response_time = float(hour_stats.get("total_response_time", 0))
                avg_response_time = total_response_time / total_requests if total_requests > 0 else 0
                
                error_count = sum(
                    int(hour_stats.get(f"status_{code}", 0))
                    for code in range(400, 600)
                )
                error_rate = error_count / total_requests if total_requests > 0 else 0
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "period": "current_hour",
                    "metrics": {
                        "total_requests": total_requests,
                        "avg_response_time_ms": round(avg_response_time, 2),
                        "error_rate": round(error_rate, 4),
                        "total_cost": float(hour_stats.get("total_cost", 0)),
                        "status_codes": {
                            code: int(hour_stats.get(f"status_{code}", 0))
                            for code in [200, 201, 400, 401, 403, 404, 429, 500, 502, 503]
                            if hour_stats.get(f"status_{code}", 0) != "0"
                        }
                    }
                }
                
            except Exception as e:
                self.logger.log_error("METRICS_ERROR", e)
                return JSONResponse(
                    status_code=500,
                    content={"error": "Failed to retrieve metrics"}
                )
        
        # Authentication endpoints
        @self.app.post("/api/auth/api-key")
        async def create_api_key(request: dict):
            """Create a new API key"""
            try:
                user_id = request.get("user_id")
                tier = request.get("tier", "free")
                permissions = request.get("permissions", ["read"])
                expires_in_days = request.get("expires_in_days")
                
                if not user_id:
                    raise HTTPException(status_code=400, detail="user_id is required")
                
                # Generate API key
                api_key = self._generate_api_key()
                key_id = str(uuid.uuid4())
                
                # Calculate expiration
                expires_at = None
                if expires_in_days:
                    expires_at = datetime.now() + timedelta(days=expires_in_days)
                
                # Store API key info
                key_info = {
                    "key_id": key_id,
                    "user_id": user_id,
                    "tier": tier,
                    "permissions": permissions,
                    "created_at": datetime.now().isoformat(),
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "is_active": True,
                    "metadata": request.get("metadata", {})
                }
                
                # Hash API key for storage
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                await self.redis_client.set(
                    f"api_key:{key_hash}",
                    json.dumps(key_info),
                    ex=86400 * 365 if not expires_at else None  # 1 year default
                )
                
                # Store key ID mapping
                await self.redis_client.set(
                    f"api_key_id:{key_id}",
                    key_hash,
                    ex=86400 * 365 if not expires_at else None
                )
                
                self.logger.log_security_event(
                    "API_KEY_CREATED",
                    f"API key created for user {user_id}",
                    {"user_id": user_id, "key_id": key_id, "tier": tier}
                )
                
                return {
                    "api_key": api_key,
                    "key_id": key_id,
                    "tier": tier,
                    "permissions": permissions,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
                
            except Exception as e:
                self.logger.log_error("API_KEY_CREATION_ERROR", e)
                raise HTTPException(status_code=500, detail="Failed to create API key")
        
        # Proxy routes for all services
        @self.app.api_route("/api/{service_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(request: Request, service_path: str):
            """Proxy requests to backend services"""
            return await self._proxy_to_service(request, service_path)
    
    async def _proxy_to_service(self, request: Request, service_path: str) -> Response:
        """Proxy request to appropriate backend service"""
        try:
            # Determine target service
            target_url = self._get_target_url(service_path)
            if not target_url:
                raise HTTPException(status_code=404, detail="Service not found")
            
            # Prepare request
            method = request.method
            headers = dict(request.headers)
            
            # Remove hop-by-hop headers
            hop_by_hop_headers = {
                "connection", "keep-alive", "proxy-authenticate",
                "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
            }
            headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}
            
            # Add gateway headers
            headers["X-Gateway-Request-ID"] = getattr(request.state, 'request_id', str(uuid.uuid4()))
            headers["X-Gateway-User-ID"] = getattr(request.state, 'user_id', "anonymous")
            headers["X-Gateway-Tier"] = getattr(request.state, 'tier', RateLimitTier.FREE).value
            
            # Get request body
            body = await request.body()
            
            # Make proxied request
            response = await self.http_client.request(
                method=method,
                url=target_url,
                headers=headers,
                content=body,
                params=dict(request.query_params)
            )
            
            # Prepare response headers
            response_headers = dict(response.headers)
            response_headers = {k: v for k, v in response_headers.items() if k.lower() not in hop_by_hop_headers}
            
            # Add gateway response headers
            response_headers["X-Gateway-Service"] = self._get_service_name(service_path)
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
                media_type=response.headers.get("content-type")
            )
            
        except httpx.RequestError as e:
            self.logger.log_error("PROXY_REQUEST_ERROR", e, {"service_path": service_path})
            raise HTTPException(status_code=502, detail="Bad Gateway")
        
        except httpx.TimeoutException as e:
            self.logger.log_error("PROXY_TIMEOUT_ERROR", e, {"service_path": service_path})
            raise HTTPException(status_code=504, detail="Gateway Timeout")
        
        except Exception as e:
            self.logger.log_error("PROXY_GENERAL_ERROR", e, {"service_path": service_path})
            raise HTTPException(status_code=500, detail="Internal Server Error")
    
    def _get_target_url(self, service_path: str) -> Optional[str]:
        """Get target URL for service path"""
        for endpoint_prefix, base_url in self.config.service_endpoints.items():
            if service_path.startswith(endpoint_prefix.lstrip("/")):
                # Replace the prefix with the base URL
                remaining_path = service_path[len(endpoint_prefix.lstrip("/")):]
                return f"{base_url.rstrip('/')}/{remaining_path.lstrip('/')}"
        
        return None
    
    def _get_service_name(self, service_path: str) -> str:
        """Get service name from path"""
        for endpoint_prefix in self.config.service_endpoints.keys():
            if service_path.startswith(endpoint_prefix.lstrip("/")):
                return endpoint_prefix.split("/")[-1]
        return "unknown"
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        import secrets
        return f"ak_{secrets.token_urlsafe(32)}"
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity"""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def _health_check_loop(self):
        """Background task to check service health"""
        while True:
            try:
                for endpoint_prefix, base_url in self.config.service_endpoints.items():
                    service_name = endpoint_prefix.split("/")[-1]
                    
                    try:
                        # Simple health check
                        response = await self.http_client.get(
                            f"{base_url}/health",
                            timeout=5
                        )
                        self.service_health[service_name] = response.status_code == 200
                        
                    except Exception:
                        self.service_health[service_name] = False
                
                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error("HEALTH_CHECK_LOOP_ERROR", e)
                await asyncio.sleep(30)
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while True:
            try:
                # Cleanup old metrics (keep last 1000 entries)
                await self.redis_client.ltrim("gateway:metrics", 0, 999)
                
                # Cleanup expired rate limit keys
                current_time = int(time.time())
                
                # Remove old minute keys (older than 2 minutes)
                old_minute = (current_time // 60) - 2
                pattern = f"rate_limit:*:minute:{old_minute}"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                if self.logger:
                    self.logger.log_error("CLEANUP_LOOP_ERROR", e)
                await asyncio.sleep(300)
    
    async def shutdown(self):
        """Graceful shutdown of the gateway"""
        try:
            if self.logger:
                self.logger.log_operation("GATEWAY_SHUTDOWN", "Shutting down Advanced API Gateway")
            
            # Flush remaining analytics
            if self.analytics_middleware:
                await self.analytics_middleware._flush_metrics()
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Close state manager
            if self.state_manager:
                await self.state_manager.disconnect()
                
        except Exception as e:
            if self.logger:
                self.logger.log_error("GATEWAY_SHUTDOWN_ERROR", e)


class GatewayManager:
    """Manager class for running the Advanced API Gateway"""
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.gateway = None
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the API Gateway server"""
        try:
            # Initialize gateway
            self.gateway = AdvancedAPIGateway(self.config)
            await self.gateway.initialize()
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.gateway.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True,
                loop="asyncio"
            )
            
            server = uvicorn.Server(config)
            
            # Setup signal handlers for graceful shutdown
            import signal
            
            def signal_handler(signum, frame):
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start server
            await server.serve()
            
        except Exception as e:
            if self.gateway and self.gateway.logger:
                self.gateway.logger.log_error("GATEWAY_START_ERROR", e)
            raise e
    
    async def shutdown(self):
        """Shutdown the gateway"""
        if self.gateway:
            await self.gateway.shutdown()


# CLI and main execution
if __name__ == "__main__":
    import argparse
    import sys
    
    def main():
        """Main entry point for the Advanced API Gateway"""
        parser = argparse.ArgumentParser(description="Alita-KGoT Advanced API Gateway")
        parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
        parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
        parser.add_argument("--redis-url", help="Redis URL")
        parser.add_argument("--jwt-secret", help="JWT secret key")
        parser.add_argument("--config-file", help="Configuration file path")
        
        args = parser.parse_args()
        
        # Create configuration
        config = GatewayConfig()
        
        if args.redis_url:
            config.redis_url = args.redis_url
        if args.jwt_secret:
            config.jwt_secret = args.jwt_secret
        
        # Load configuration from file if provided
        if args.config_file:
            try:
                import yaml
                with open(args.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Update config with file values
                for key, value in file_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        
            except Exception as e:
                print(f"Error loading config file: {e}")
                sys.exit(1)
        
        # Create and start gateway manager
        manager = GatewayManager(config)
        
        try:
            asyncio.run(manager.start(args.host, args.port))
        except KeyboardInterrupt:
            print("\nShutting down gateway...")
        except Exception as e:
            print(f"Gateway error: {e}")
            sys.exit(1)
    
    main()