# Task 15 Deployment Guide
## KGoT Surfer Agent + Alita Web Agent Integration

This guide provides step-by-step instructions for deploying the complete **Task 15: Integrate KGoT Surfer Agent with Alita Web Agent** implementation.

## 🚀 **Quick Start**

### **1. Prerequisites**

**System Requirements:**
- Python 3.9+ with pip
- Node.js 18+ with npm
- Docker & Docker Compose (optional)
- Git

**API Keys Required:**
```bash
# Required for OpenRouter integration (user preference)
OPENROUTER_API_KEY="your_openrouter_api_key"

# Required for Google Search
GOOGLE_API_KEY="your_google_api_key"
GOOGLE_SEARCH_ENGINE_ID="your_search_engine_id"

# Required for GitHub search
GITHUB_TOKEN="your_github_token"

# Optional for advanced features
NEO4J_PASSWORD="your_neo4j_password"
REDIS_PASSWORD="your_redis_password"
```

### **2. Environment Setup**

**Clone and Setup:**
```bash
# Navigate to the project
cd alita-kgot-enhanced

# Copy environment template
cp env.template .env

# Edit with your API keys
nano .env
```

**Python Dependencies:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install additional integration dependencies
pip install langchain langchain-openai
pip install transformers torch
pip install requests aiohttp websockets
pip install playwright
pip install networkx neo4j rdflib
```

**Node.js Dependencies:**
```bash
# Install Alita Web Agent dependencies
cd alita_core/web_agent
npm install
cd ../..

# Install Manager Agent dependencies
cd alita_core/manager_agent
npm install
cd ../..
```

### **3. Service Initialization**

**Initialize Graph Store:**
```bash
# For NetworkX (default, no setup required)
echo "NetworkX backend ready"

# For Neo4j (if using)
docker run -d \
  --name neo4j-kgot \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \
  neo4j:latest

# For RDF4j (if using)
docker run -d \
  --name rdf4j-kgot \
  -p 8080:8080 \
  eclipse/rdf4j-workbench:latest
```

**Initialize Browser for KGoT Tools:**
```bash
# Install Playwright browsers
playwright install chromium firefox webkit
```

**Create Log Directories:**
```bash
mkdir -p logs/{alita,kgot,system,web_agent,integration}
chmod 755 logs/*
```

## 🔧 **Configuration**

### **Model Configuration (Per User Rules)**

Edit `config/models/model_config.json`:
```json
{
  "model_providers": {
    "openrouter": {
      "base_url": "https://openrouter.ai/api/v1",
      "models": {
        "orchestration": "x-ai/grok-4",
        "webagent": "anthropic/claude-4-sonnet", 
        "vision": "openai/o3"
      }
    }
  }
}
```

### **Integration Configuration**

Create `config/integration/task15_config.json`:
```json
{
  "kgot_surfer_config": {
    "model_name": "anthropic/claude-4-sonnet",
    "temperature": 0.1,
    "max_iterations": 12,
    "enable_browser": true
  },
  "alita_web_agent_config": {
    "endpoint": "http://localhost:3001",
    "timeout": 30,
    "enable_playwright": true,
    "enable_graph_integration": true
  },
  "graph_store_config": {
    "backend": "networkx",
    "enable_persistence": true,
    "max_nodes": 10000
  },
  "mcp_validation_config": {
    "enable_validation": true,
    "confidence_threshold": 0.7,
    "validation_timeout": 10
  }
}
```

## 🏃‍♂️ **Running the Integration**

### **Method 1: Manual Service Start**

**Terminal 1 - Alita Web Agent:**
```bash
cd alita_core/web_agent
npm start
# Runs on http://localhost:3001
```

**Terminal 2 - Alita Manager Agent:**
```bash
cd alita_core/manager_agent
npm start
# Runs on http://localhost:3000
```

**Terminal 3 - Integration Service:**
```bash
# Run the integration
python web_integration/kgot_surfer_alita_web.py

# Or run with test
python test_integration_task15.py
```

### **Method 2: Docker Compose (Recommended)**

```bash
# Start all services
cd config/containers
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f alita-web
docker-compose logs -f alita-manager
```

### **Method 3: Development Mode**

```bash
# Start in development mode with auto-reload
cd alita_core/web_agent
npm run dev &

cd ../manager_agent
npm run dev &

# Run integration with debug logging
export LOG_LEVEL=debug
python web_integration/kgot_surfer_alita_web.py
```

## 🧪 **Testing & Validation**

### **Run Integration Tests**

```bash
# Run comprehensive test suite
python test_integration_task15.py

# Expected output:
# 🎉 Task 15 Integration: ALL TESTS PASSED! ✅
```

### **Manual Testing Scenarios**

**Test 1: Basic Search Integration**
```python
import asyncio
from web_integration.kgot_surfer_alita_web import create_kgot_surfer_alita_integration

async def test_basic_search():
    integration = await create_kgot_surfer_alita_integration()
    
    result = await integration.execute_integrated_search(
        query="Latest developments in AI agents 2025",
        context={"search_type": "comprehensive"}
    )
    
    print(f"Search completed: {result['status']}")
    print(f"Results found: {len(result.get('results', []))}")

asyncio.run(test_basic_search())
```

**Test 2: Granular Navigation**
```python
async def test_navigation():
    integration = await create_kgot_surfer_alita_integration()
    
    # Test page navigation tools
    navigation_result = await integration.kgot_surfer_agent._run(
        "Visit https://example.com and use page down to navigate through content"
    )
    
    print(f"Navigation result: {navigation_result}")

asyncio.run(test_navigation())
```

**Test 3: GitHub Integration**
```python
async def test_github_search():
    integration = await create_kgot_surfer_alita_integration()
    
    github_result = await integration.execute_integrated_search(
        query="langchain agents python repositories",
        context={"search_type": "github", "focus": "repositories"}
    )
    
    print(f"GitHub search: {github_result['status']}")

asyncio.run(test_github_search())
```

## 📊 **Monitoring & Logging**

### **Log Analysis**

```bash
# Real-time log monitoring
tail -f logs/integration/combined.log

# Search for specific operations
grep "INTEGRATION_SUCCESS" logs/integration/combined.log

# Monitor error rates
grep "ERROR" logs/*/error.log | wc -l
```

### **Health Checks**

```bash
# Check service health
curl http://localhost:3001/health  # Alita Web Agent
curl http://localhost:3000/health  # Alita Manager Agent

# Check integration status
curl -X POST http://localhost:3001/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "system status"}'
```

### **Performance Monitoring**

Configure Grafana dashboard (optional):
```bash
# Start monitoring stack
cd config/monitoring
docker-compose up -d prometheus grafana

# Access Grafana: http://localhost:3000
# Default: admin/admin
```

## 🔧 **Troubleshooting**

### **Common Issues**

**1. Import Errors**
```bash
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/knowledge-graph-of-thoughts"
```

**2. API Key Issues**
```bash
# Verify OpenRouter API key
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

**3. Browser Issues**
```bash
# Reinstall Playwright browsers
playwright install --force chromium
```

**4. Graph Store Connection**
```bash
# Test NetworkX backend
python -c "import networkx as nx; print('NetworkX OK')"

# Test Neo4j connection
python -c "from neo4j import GraphDatabase; print('Neo4j OK')"
```

### **Debug Mode**

Enable comprehensive debugging:
```bash
export DEBUG=alita:*,kgot:*
export LOG_LEVEL=debug
export VERBOSE_LOGGING=true

python web_integration/kgot_surfer_alita_web.py
```

## 📈 **Production Deployment**

### **Environment Variables for Production**

```bash
# Production configuration
export NODE_ENV=production
export LOG_LEVEL=info
export ENABLE_RATE_LIMITING=true
export ENABLE_COMPRESSION=true
export MAX_CONCURRENT_REQUESTS=100

# Security settings
export JWT_SECRET="your_production_jwt_secret"
export SESSION_SECRET="your_production_session_secret"
export ALLOWED_ORIGINS="https://yourdomain.com"

# Performance settings
export ENABLE_CACHING=true
export CACHE_TTL=3600
export MAX_MEMORY_USAGE=2048
```

### **Production Docker Compose**

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  alita-web-prod:
    image: alita-web-agent:latest
    environment:
      - NODE_ENV=production
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M
    restart: unless-stopped
```

### **Scaling Configuration**

```bash
# Scale services
docker-compose up -d --scale alita-web=3 --scale alita-manager=2

# Load balancer configuration (nginx example)
upstream alita_web {
    server localhost:3001;
    server localhost:3002;
    server localhost:3003;
}
```

## 🔐 **Security Considerations**

### **API Security**

```javascript
// Rate limiting configuration
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use('/api/', limiter);
```

### **Input Validation**

```python
# Input sanitization for web integration
import bleach
from urllib.parse import urlparse

def validate_search_query(query: str) -> str:
    """Sanitize search query input"""
    # Remove potentially dangerous content
    clean_query = bleach.clean(query, strip=True)
    
    # Limit length
    if len(clean_query) > 500:
        raise ValueError("Query too long")
    
    return clean_query

def validate_url(url: str) -> bool:
    """Validate URL for web navigation"""
    parsed = urlparse(url)
    return parsed.scheme in ['http', 'https'] and parsed.netloc
```

## 📚 **API Usage Examples**

### **Integration API**

```python
# Complete integration usage example
from web_integration.kgot_surfer_alita_web import (
    create_kgot_surfer_alita_integration,
    WebIntegrationConfig
)

# Custom configuration
config = WebIntegrationConfig(
    alita_web_agent_endpoint="http://localhost:3001",
    graph_store_backend="neo4j",
    enable_mcp_validation=True,
    mcp_confidence_threshold=0.8
)

# Initialize integration
integration = await create_kgot_surfer_alita_integration(config)

# Execute complex search with context
result = await integration.execute_integrated_search(
    query="Find recent research papers on multi-agent systems",
    context={
        "search_type": "academic",
        "time_range": "2024-2025",
        "depth": "comprehensive",
        "include_github": True,
        "navigation_mode": "intelligent"
    }
)

# Access results
print(f"Status: {result['status']}")
print(f"Results: {len(result['results'])}")
print(f"Graph updates: {result['graph_updates']}")
print(f"MCP validation: {result['mcp_validation']}")
```

### **REST API Endpoints**

```bash
# Search endpoint
curl -X POST http://localhost:3001/search/enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning frameworks 2025",
    "context": {"type": "research", "depth": "detailed"}
  }'

# Navigation endpoint  
curl -X POST http://localhost:3001/navigation/enhance \
  -H "Content-Type: application/json" \
  -d '{
    "targetUrl": "https://arxiv.org/abs/2401.12345",
    "navigationIntent": "extract research details",
    "currentContext": {"domain": "AI research"}
  }'

# Agent query endpoint
curl -X POST http://localhost:3001/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Search for and analyze the latest AI safety papers",
    "context": {"task": "research_analysis"}
  }'
```

## 🎯 **Next Steps**

### **Phase 2 Preparation**

The Task 15 integration sets the foundation for subsequent phases:

1. **Enhanced Multimodal Processing** - Vision integration ready
2. **Advanced Knowledge Graph Operations** - Graph store established
3. **Intelligent Task Orchestration** - Agent framework prepared
4. **Production Scaling** - Infrastructure components deployed

### **Optimization Opportunities**

1. **Caching Layer**: Implement Redis caching for frequent searches
2. **Load Balancing**: Scale web agents horizontally
3. **Model Optimization**: Fine-tune model selection per task type
4. **Graph Optimization**: Implement graph pruning and clustering

---

## ✅ **Deployment Checklist**

- [ ] Environment variables configured
- [ ] API keys validated
- [ ] Dependencies installed
- [ ] Services started successfully
- [ ] Integration tests passed
- [ ] Health checks responding
- [ ] Logging operational
- [ ] Monitoring configured (optional)
- [ ] Security measures implemented
- [ ] Documentation reviewed

**🎉 Your Task 15 KGoT Surfer + Alita Web Agent integration is ready for production!** 