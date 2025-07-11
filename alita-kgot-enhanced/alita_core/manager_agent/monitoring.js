/**
 * Prometheus Monitoring Module
 * Provides metrics collection for tool usage and system performance
 */

const promClient = require('prom-client');
const { loggers } = require('../../config/logging/winston_config');
const logger = loggers.managerAgent;

// Create a Registry which registers the metrics
const register = new promClient.Registry();

// Add a default label which is added to all metrics
register.setDefaultLabels({
  app: 'alita-kgot-enhanced',
  component: 'manager-agent'
});

// Enable the collection of default metrics
promClient.collectDefaultMetrics({ register });

// Custom metrics for tool usage
const toolUsageCounter = new promClient.Counter({
  name: 'alita_tool_usage_total',
  help: 'Total number of tool invocations',
  labelNames: ['tool_name', 'status'],
  registers: [register]
});

const toolExecutionDuration = new promClient.Histogram({
  name: 'alita_tool_execution_duration_seconds',
  help: 'Duration of tool execution in seconds',
  labelNames: ['tool_name'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30],
  registers: [register]
});

const agentRequestsCounter = new promClient.Counter({
  name: 'alita_agent_requests_total',
  help: 'Total number of agent requests',
  labelNames: ['endpoint', 'method', 'status_code'],
  registers: [register]
});

const agentResponseTime = new promClient.Histogram({
  name: 'alita_agent_response_time_seconds',
  help: 'Agent response time in seconds',
  labelNames: ['endpoint', 'method'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60],
  registers: [register]
});

const systemErrorsCounter = new promClient.Counter({
  name: 'alita_system_errors_total',
  help: 'Total number of system errors',
  labelNames: ['error_type', 'component'],
  registers: [register]
});

const activeConnectionsGauge = new promClient.Gauge({
  name: 'alita_active_connections',
  help: 'Number of active connections',
  registers: [register]
});

class MonitoringService {
  constructor() {
    this.register = register;
    this.metrics = {
      toolUsageCounter,
      toolExecutionDuration,
      agentRequestsCounter,
      agentResponseTime,
      systemErrorsCounter,
      activeConnectionsGauge
    };
    
    logger.logOperation('info', 'MONITORING_INITIALIZED', 'Prometheus monitoring service initialized');
  }

  /**
   * Record tool usage metrics
   */
  recordToolUsage(toolName, status = 'success', duration = null) {
    try {
      this.metrics.toolUsageCounter.inc({ tool_name: toolName, status });
      
      if (duration !== null) {
        this.metrics.toolExecutionDuration.observe({ tool_name: toolName }, duration);
      }
      
      logger.logOperation('debug', 'TOOL_METRIC_RECORDED', 'Tool usage metric recorded', {
        toolName,
        status,
        duration
      });
    } catch (error) {
      logger.logError('MONITORING_TOOL_METRIC_ERROR', error, { toolName, status, duration });
    }
  }

  /**
   * Record agent request metrics
   */
  recordAgentRequest(endpoint, method, statusCode, responseTime = null) {
    try {
      this.metrics.agentRequestsCounter.inc({ 
        endpoint, 
        method, 
        status_code: statusCode.toString() 
      });
      
      if (responseTime !== null) {
        this.metrics.agentResponseTime.observe({ endpoint, method }, responseTime);
      }
      
      logger.logOperation('debug', 'AGENT_REQUEST_METRIC_RECORDED', 'Agent request metric recorded', {
        endpoint,
        method,
        statusCode,
        responseTime
      });
    } catch (error) {
      logger.logError('MONITORING_REQUEST_METRIC_ERROR', error, { endpoint, method, statusCode, responseTime });
    }
  }

  /**
   * Record system error metrics
   */
  recordSystemError(errorType, component) {
    try {
      this.metrics.systemErrorsCounter.inc({ error_type: errorType, component });
      
      logger.logOperation('debug', 'SYSTEM_ERROR_METRIC_RECORDED', 'System error metric recorded', {
        errorType,
        component
      });
    } catch (error) {
      logger.logError('MONITORING_ERROR_METRIC_ERROR', error, { errorType, component });
    }
  }

  /**
   * Update active connections gauge
   */
  updateActiveConnections(count) {
    try {
      this.metrics.activeConnectionsGauge.set(count);
      
      logger.logOperation('debug', 'ACTIVE_CONNECTIONS_UPDATED', 'Active connections metric updated', {
        count
      });
    } catch (error) {
      logger.logError('MONITORING_CONNECTIONS_METRIC_ERROR', error, { count });
    }
  }

  /**
   * Get metrics for Prometheus scraping
   */
  async getMetrics() {
    try {
      return await this.register.metrics();
    } catch (error) {
      logger.logError('MONITORING_GET_METRICS_ERROR', error);
      throw error;
    }
  }

  /**
   * Create Express middleware for automatic request monitoring
   */
  createExpressMiddleware() {
    return (req, res, next) => {
      const startTime = Date.now();
      
      // Override res.end to capture response metrics
      const originalEnd = res.end;
      res.end = (...args) => {
        const responseTime = (Date.now() - startTime) / 1000;
        this.recordAgentRequest(
          req.route?.path || req.path || 'unknown',
          req.method,
          res.statusCode,
          responseTime
        );
        originalEnd.apply(res, args);
      };
      
      next();
    };
  }

  /**
   * Create tool execution wrapper for automatic monitoring
   */
  wrapToolExecution(toolName, toolFunction) {
    return async (...args) => {
      const startTime = Date.now();
      let status = 'success';
      
      try {
        const result = await toolFunction(...args);
        return result;
      } catch (error) {
        status = 'error';
        this.recordSystemError('tool_execution_error', 'manager_agent');
        throw error;
      } finally {
        const duration = (Date.now() - startTime) / 1000;
        this.recordToolUsage(toolName, status, duration);
      }
    };
  }
}

// Export singleton instance
const monitoringService = new MonitoringService();

module.exports = {
  MonitoringService,
  monitoring: monitoringService,
  register
};