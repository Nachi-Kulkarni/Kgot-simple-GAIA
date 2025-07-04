/**
 * Winston Logging Configuration for Alita-KGoT Enhanced System
 * Provides comprehensive logging infrastructure with different levels and transports
 * @module WinstonConfig
 */

const winston = require('winston');
const path = require('path');
const { format } = winston;

/**
 * Custom log format with timestamp, level, and structured metadata
 */
const customFormat = format.combine(
  format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss.SSS' }),
  format.errors({ stack: true }),
  format.json(),
  format.printf(({ timestamp, level, message, component, operation, metadata, stack }) => {
    const logObject = {
      timestamp,
      level: level.toUpperCase(),
      component: component || 'SYSTEM',
      operation: operation || 'GENERAL',
      message,
      ...(metadata && { metadata }),
      ...(stack && { stack })
    };
    return JSON.stringify(logObject);
  })
);

/**
 * Console format for development debugging
 */
const consoleFormat = format.combine(
  format.colorize(),
  format.timestamp({ format: 'HH:mm:ss' }),
  format.printf(({ timestamp, level, message, component, operation }) => {
    const comp = component ? `[${component}]` : '';
    const op = operation ? `[${operation}]` : '';
    return `${timestamp} ${level} ${comp}${op}: ${message}`;
  })
);

/**
 * Log levels configuration
 */
const logLevels = {
  error: 0,    // System errors, failures
  warn: 1,     // Warning conditions
  info: 2,     // General information about system operation
  http: 3,     // HTTP requests/responses
  verbose: 4,  // Verbose information
  debug: 5,    // Debug information
  silly: 6     // Everything
};

/**
 * Create logger instance with component-specific configuration
 * @param {string} component - Component name (alita, kgot, system, etc.)
 * @param {object} options - Additional configuration options
 * @returns {winston.Logger} Configured Winston logger instance
 */
function createLogger(component = 'SYSTEM', options = {}) {
  const {
    level = process.env.LOG_LEVEL || 'info',
    enableConsole = process.env.NODE_ENV !== 'production',
    enableFile = true,
    logDir = path.join(__dirname, '../../logs')
  } = options;

  const transports = [];

  // Console transport for development
  if (enableConsole) {
    transports.push(
      new winston.transports.Console({
        level: 'debug',
        format: consoleFormat,
        handleExceptions: true,
        handleRejections: true
      })
    );
  }

  // File transports for different log levels
  if (enableFile) {
    // Combined logs
    transports.push(
      new winston.transports.File({
        filename: path.join(logDir, `${component.toLowerCase()}/combined.log`),
        level: level,
        format: customFormat,
        maxsize: 10485760, // 10MB
        maxFiles: 5,
        tailable: true
      })
    );

    // Error logs
    transports.push(
      new winston.transports.File({
        filename: path.join(logDir, `${component.toLowerCase()}/error.log`),
        level: 'error',
        format: customFormat,
        maxsize: 10485760, // 10MB
        maxFiles: 5,
        tailable: true
      })
    );

    // HTTP/API logs for web components
    if (['alita', 'web_agent', 'mcp_creation'].includes(component.toLowerCase())) {
      transports.push(
        new winston.transports.File({
          filename: path.join(logDir, `${component.toLowerCase()}/http.log`),
          level: 'http',
          format: customFormat,
          maxsize: 10485760, // 10MB
          maxFiles: 3,
          tailable: true
        })
      );
    }
  }

  const logger = winston.createLogger({
    levels: logLevels,
    level: level,
    format: customFormat,
    transports: transports,
    exitOnError: false,
    defaultMeta: { component }
  });

  /**
   * Enhanced logging methods with operation context
   */
  logger.logOperation = function(level, operation, message, metadata = {}) {
    this.log(level, message, { operation, metadata });
  };

  logger.logError = function(operation, error, metadata = {}) {
    this.error(error.message || error, { 
      operation, 
      metadata: { ...metadata, errorType: error.constructor.name },
      stack: error.stack 
    });
  };

  logger.logApiCall = function(method, url, statusCode, duration, metadata = {}) {
    this.http(`${method} ${url} - ${statusCode} (${duration}ms)`, {
      operation: 'API_CALL',
      metadata: { method, url, statusCode, duration, ...metadata }
    });
  };

  logger.logModelUsage = function(modelId, tokens, cost, operation, metadata = {}) {
    this.info(`Model usage: ${modelId} - ${tokens} tokens ($${cost.toFixed(4)})`, {
      operation: 'MODEL_USAGE',
      metadata: { modelId, tokens, cost, operation, ...metadata }
    });
  };

  logger.logGraphOperation = function(operation, graphType, nodeCount, edgeCount, duration, metadata = {}) {
    this.info(`Graph operation: ${operation} on ${graphType} - ${nodeCount} nodes, ${edgeCount} edges (${duration}ms)`, {
      operation: 'GRAPH_OPERATION',
      metadata: { operation, graphType, nodeCount, edgeCount, duration, ...metadata }
    });
  };

  return logger;
}

/**
 * Pre-configured loggers for different components
 */
const loggers = {
  // Alita components
  alita: createLogger('ALITA'),
  managerAgent: createLogger('MANAGER_AGENT'),
  webAgent: createLogger('WEB_AGENT'),
  mcpCreation: createLogger('MCP_CREATION'),

  // KGoT components
  kgot: createLogger('KGOT'),
  kgotController: createLogger('KGOT_CONTROLLER'),
  controller: createLogger('CONTROLLER'),
  graphStore: createLogger('GRAPH_STORE'),
  integratedTools: createLogger('INTEGRATED_TOOLS'),

  // Extension components
  multimodal: createLogger('MULTIMODAL'),
  validation: createLogger('VALIDATION'),
  optimization: createLogger('OPTIMIZATION'),

  // System components
  system: createLogger('SYSTEM'),
  errors: createLogger('ERRORS')
};

/**
 * Middleware for Express.js applications to log HTTP requests
 */
function httpLoggingMiddleware(logger) {
  return (req, res, next) => {
    const startTime = Date.now();
    
    res.on('finish', () => {
      const duration = Date.now() - startTime;
      logger.logApiCall(
        req.method,
        req.originalUrl,
        res.statusCode,
        duration,
        {
          userAgent: req.get('user-agent'),
          ip: req.ip,
          contentLength: res.get('content-length')
        }
      );
    });

    next();
  };
}

module.exports = {
  createLogger,
  loggers,
  httpLoggingMiddleware,
  logLevels
}; 