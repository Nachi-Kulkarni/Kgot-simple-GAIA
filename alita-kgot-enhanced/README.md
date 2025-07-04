# Alita-KGoT Enhanced System

An advanced AI assistant system that combines the **Alita** architecture with **Knowledge Graph of Thoughts (KGoT)** for superior reasoning, multimodal processing, and dynamic tool creation.

## 🌟 Key Features

### Core Architectures
- **Alita Architecture**: Manager Agent, Web Agent, and MCP Creation for comprehensive AI assistance
- **KGoT Architecture**: Knowledge graph-based reasoning with advanced thought processes
- **Integrated Orchestration**: Seamless coordination between all components

### Advanced Capabilities
- **Multimodal Processing**: Vision, audio, and document processing
- **Dynamic Tool Creation**: MCP-based tool generation
- **Knowledge Graph Reasoning**: Complex problem-solving with graph structures
- **Cost Optimization**: Intelligent model selection and usage tracking
- **Validation & QA**: Automated quality assurance and testing
- **Performance Optimization**: Resource management and efficiency tuning

## 🏗️ Architecture Overview

```
alita-kgot-enhanced/
├── alita_core/                    # Alita Architecture Components
│   ├── manager_agent/             # Central orchestrator (LangChain-based)
│   ├── web_agent/                 # Web interactions and scraping
│   └── mcp_creation/              # Dynamic MCP tool generation
├── kgot_core/                     # KGoT Architecture Components  
│   ├── controller/                # KGoT reasoning controller
│   ├── graph_store/               # Knowledge graph persistence
│   └── integrated_tools/          # Tool execution and management
├── multimodal/                    # Multimodal processing
│   ├── vision/                    # Image and video processing
│   ├── audio/                     # Audio processing and transcription
│   └── text_processing/           # Advanced text processing
├── validation/                    # Quality assurance
│   ├── testing/                   # Automated testing frameworks
│   ├── benchmarks/                # Performance benchmarking
│   └── quality_assurance/         # QA processes and validation
├── optimization/                  # Performance optimization
│   ├── cost_optimization/         # Cost management and tracking
│   ├── performance_tuning/        # Performance optimization
│   └── resource_management/       # Resource allocation and management
└── config/                        # Configuration management
    ├── models/                    # Model configurations
    ├── containers/                # Docker/container configurations
    └── logging/                   # Logging infrastructure
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Docker and Docker Compose
- OpenRouter API key
- 8GB+ RAM recommended

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd alita-kgot-enhanced
   cp env.template .env
   ```

2. **Configure Environment**
   Edit `.env` file with your API keys and configuration:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   NEO4J_PASSWORD=your_password_here
   REDIS_PASSWORD=your_password_here
   ```

3. **Install Dependencies**
   ```bash
   npm install
   ```

4. **Start with Docker (Recommended)**
   ```bash
   npm run docker:up
   ```

5. **Or Start Individual Services**
   ```bash
   # Start manager agent
   npm run start:manager
   
   # Start other services in separate terminals
   npm run start:web
   npm run start:mcp
   npm run start:kgot
   ```

### Verification

Check system status:
```bash
curl http://localhost:3000/health
curl http://localhost:3000/status
```

Access monitoring:
- Grafana: http://localhost:3000 (admin/your_grafana_password)
- Prometheus: http://localhost:9090
- Neo4j Browser: http://localhost:7474

## 📝 Usage Examples

### Basic Chat Interface

```javascript
const response = await fetch('http://localhost:3000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Help me research and summarize the latest developments in AI",
    sessionId: "user-session-123"
  })
});

const result = await response.json();
console.log(result.response);
```

### Web Agent Tasks

```javascript
const webTask = await fetch('http://localhost:3000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Search for recent papers on knowledge graphs and create a summary",
    sessionId: "research-session"
  })
});
```

### Knowledge Graph Operations

```javascript
const kgotTask = await fetch('http://localhost:3000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Create a knowledge graph of the relationships between different machine learning concepts",
    sessionId: "kg-session"
  })
});
```

## 🔧 Configuration

### Model Configuration

Edit `config/models/model_config.json` to customize:
- Model provider settings (OpenRouter)
- Model selection for different components
- Cost optimization parameters
- Fallback strategies

### Logging Configuration

Comprehensive Winston-based logging in `config/logging/winston_config.js`:
- Component-specific loggers
- Multiple log levels and transports
- Performance and cost tracking
- Error monitoring

### Container Configuration

Docker services configured in `config/containers/docker-compose.yml`:
- All system components
- Database services (Neo4j, RDF4J, Redis)
- Monitoring stack (Grafana, Prometheus)
- Network and volume management

## 📊 Monitoring & Observability

### Metrics and Monitoring
- **Grafana Dashboards**: System performance, costs, and usage metrics
- **Prometheus**: Metrics collection and alerting
- **Winston Logging**: Comprehensive logging across all components
- **Health Checks**: Automated service health monitoring

### Cost Tracking
- Real-time model usage tracking
- Daily and per-request cost limits
- Cost optimization recommendations
- Usage analytics and reporting

## 🧪 Testing & Validation

### Automated Testing
```bash
npm test                # Run all tests
npm run test:watch      # Watch mode for development
npm run test:coverage   # Generate coverage reports
```

### Quality Assurance
- Automated output validation
- Benchmark testing against known datasets
- Performance regression testing
- Security vulnerability scanning

## 🔍 Development

### Development Mode
```bash
npm run dev             # Start in development mode with hot reload
```

### Code Quality
```bash
npm run lint            # Check code quality
npm run lint:fix        # Auto-fix linting issues
npm run format          # Format code with Prettier
```

### Adding New Components

1. Create component directory in appropriate section
2. Implement following the established patterns:
   - Use Winston logging with component-specific logger
   - Follow JSDoc commenting standards
   - Implement health checks and status endpoints
   - Use LangChain for agent-based components
3. Add to Docker Compose configuration
4. Update monitoring and validation

## 🛠️ Troubleshooting

### Common Issues

**Service Not Starting**
- Check environment variables in `.env`
- Verify API keys are valid
- Check Docker container logs: `npm run docker:logs`

**Model API Errors**
- Verify OpenRouter API key and credits
- Check model availability and pricing
- Review cost limits in configuration

**Database Connection Issues**
- Ensure Neo4j and Redis are running
- Check connection strings and passwords
- Verify network connectivity between containers

### Logs and Debugging

View logs by component:
```bash
tail -f logs/alita/combined.log          # Manager agent logs
tail -f logs/kgot/combined.log           # KGoT controller logs
tail -f logs/system/combined.log         # System logs
```

Enable debug mode:
```bash
export DEBUG=alita:*
export LOG_LEVEL=debug
npm run dev
```

## 📚 Documentation

- [API Documentation](docs/api/README.md)
- [Architecture Deep Dive](docs/architecture/README.md)
- [Deployment Guide](docs/deployment/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

### Task-Specific Documentation

- [Task 4: KGoT Integrated Tools](docs/TASK_4_KGOT_INTEGRATED_TOOLS.md) - Complete implementation guide and architecture
- [KGoT Integrated Tools API Reference](docs/api/KGOT_INTEGRATED_TOOLS_API.md) - Comprehensive API documentation with examples
- [Task 5: KGoT Knowledge Extraction Methods](docs/TASK_5_KGOT_KNOWLEDGE_EXTRACTION.md) - Complete implementation guide for knowledge extraction
- [Task 5 Completion Summary](docs/TASK_5_COMPLETION_SUMMARY.md) - Implementation summary and validation results
- [Task 5 Knowledge Extraction API Reference](docs/api/TASK_5_KNOWLEDGE_EXTRACTION_API.md) - Comprehensive API documentation with examples

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Follow coding standards and add tests
4. Commit changes: `git commit -m 'Add amazing feature'`
5. Push to branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Alita Research Team** for the foundational assistant architecture
- **KGoT Research Team** for knowledge graph reasoning innovations
- **LangChain** for the agent framework
- **OpenRouter** for model access and routing
- **Neo4j** for graph database capabilities

## 📞 Support

- [GitHub Issues](https://github.com/your-org/alita-kgot-enhanced/issues)
- [Documentation](https://docs.your-domain.com)
- [Community Discord](https://discord.gg/your-server)

---

**Built with ❤️ for the future of AI assistance** 