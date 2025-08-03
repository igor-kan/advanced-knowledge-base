/**
 * Advanced Knowledge Base API Server
 * 
 * Unified API layer supporting:
 * - REST API endpoints
 * - GraphQL API
 * - WebSocket real-time API
 * - Multiple knowledge base implementations
 * - High-performance query processing
 * - Sophisticated relationship modeling
 */

const express = require('express');
const { ApolloServer } = require('apollo-server-express');
const { createServer } = require('http');
const { WebSocketServer } = require('ws');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const { v4: uuidv4 } = require('uuid');

// Knowledge Base Implementations
const Neo4jKnowledgeBase = require('../../implementations/neo4j-graph/src/KnowledgeBase');
const RedisGraphKB = require('../../implementations/redis-graph/src/RedisGraphKB');
const HypergraphKB = require('../../implementations/hypergraph/src/HypergraphKB');
const CustomKnowledgeEngine = require('../../implementations/custom-engine/src/CustomKnowledgeEngine');

// API Components
const RestRoutes = require('./rest-routes');
const GraphQLSchema = require('./graphql-schema');
const WebSocketHandler = require('./websocket-handler');
const AuthMiddleware = require('./middleware/auth');
const ValidationMiddleware = require('./middleware/validation');
const CachingMiddleware = require('./middleware/caching');

class KnowledgeBaseAPIServer {
  constructor(config = {}) {
    this.config = {
      port: config.port || 8080,
      graphqlPort: config.graphqlPort || 8081,
      websocketPort: config.websocketPort || 8082,
      host: config.host || '0.0.0.0',
      enableCors: config.enableCors !== false,
      enableCompression: config.enableCompression !== false,
      enableRateLimit: config.enableRateLimit !== false,
      enableAuth: config.enableAuth || false,
      enableMetrics: config.enableMetrics !== false,
      implementations: config.implementations || ['neo4j', 'redis', 'hypergraph', 'custom'],
      defaultImplementation: config.defaultImplementation || 'neo4j',
      ...config
    };
    
    // Express app
    this.app = express();
    this.httpServer = null;
    
    // Knowledge base implementations
    this.knowledgeBases = new Map();
    this.currentKB = null;
    
    // WebSocket server
    this.wsServer = null;
    this.wsConnections = new Map();
    
    // GraphQL server
    this.apolloServer = null;
    
    // Middleware instances
    this.authMiddleware = new AuthMiddleware(this.config);
    this.validationMiddleware = new ValidationMiddleware();
    this.cachingMiddleware = new CachingMiddleware(this.config);
    
    // Performance monitoring
    this.metrics = {
      requests: 0,
      errors: 0,
      responseTime: [],
      activeConnections: 0,
      queriesPerSecond: 0,
      memoryUsage: 0
    };
    
    this.isRunning = false;
  }

  /**
   * Initialize and start the API server
   */
  async start() {
    try {
      console.log('🚀 Starting Advanced Knowledge Base API Server...');
      
      // Initialize knowledge base implementations
      await this.initializeKnowledgeBases();
      
      // Setup middleware
      this.setupMiddleware();
      
      // Setup REST routes
      this.setupRestRoutes();
      
      // Setup GraphQL server
      await this.setupGraphQLServer();
      
      // Setup WebSocket server
      this.setupWebSocketServer();
      
      // Setup monitoring
      this.setupMonitoring();
      
      // Start HTTP server
      await this.startHttpServer();
      
      this.isRunning = true;
      console.log(`✅ API Server running on http://${this.config.host}:${this.config.port}`);
      console.log(`📊 GraphQL Server running on http://${this.config.host}:${this.config.graphqlPort}/graphql`);
      console.log(`🔌 WebSocket Server running on ws://${this.config.host}:${this.config.websocketPort}`);
      
    } catch (error) {
      console.error('❌ Failed to start API server:', error);
      throw error;
    }
  }

  /**
   * Initialize all knowledge base implementations
   */
  async initializeKnowledgeBases() {
    console.log('📚 Initializing knowledge base implementations...');
    
    const implementations = {
      neo4j: async () => {
        const kb = new Neo4jKnowledgeBase({
          uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
          username: process.env.NEO4J_USER || 'neo4j',
          password: process.env.NEO4J_PASSWORD || 'password'
        });
        await kb.connect();
        return kb;
      },
      
      redis: async () => {
        const kb = new RedisGraphKB({
          host: process.env.REDIS_HOST || 'localhost',
          port: process.env.REDIS_PORT || 6379,
          password: process.env.REDIS_PASSWORD
        });
        await kb.connect();
        return kb;
      },
      
      hypergraph: async () => {
        const kb = new HypergraphKB({
          maxHyperedgeSize: 1000,
          enableCompression: true
        });
        return kb;
      },
      
      custom: async () => {
        const kb = new CustomKnowledgeEngine({
          storageDir: process.env.STORAGE_DIR || './data/custom',
          maxMemoryMB: 4096,
          enableParallelProcessing: true
        });
        await kb.initialize();
        return kb;
      }
    };
    
    // Initialize selected implementations
    for (const impl of this.config.implementations) {
      if (implementations[impl]) {
        try {
          console.log(`  Initializing ${impl}...`);
          const kb = await implementations[impl]();
          this.knowledgeBases.set(impl, kb);
          console.log(`  ✅ ${impl} initialized`);
        } catch (error) {
          console.warn(`  ⚠️  Failed to initialize ${impl}:`, error.message);
        }
      }
    }
    
    // Set default implementation
    this.currentKB = this.knowledgeBases.get(this.config.defaultImplementation);
    if (!this.currentKB && this.knowledgeBases.size > 0) {
      this.currentKB = this.knowledgeBases.values().next().value;
    }
    
    if (!this.currentKB) {
      throw new Error('No knowledge base implementations available');
    }
    
    console.log(`📚 Using ${this.config.defaultImplementation} as default implementation`);
  }

  /**
   * Setup Express middleware
   */
  setupMiddleware() {
    // Security
    this.app.use(helmet({
      contentSecurityPolicy: false // Allow GraphQL Playground
    }));
    
    // CORS
    if (this.config.enableCors) {
      this.app.use(cors({
        origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type', 'Authorization', 'X-KB-Implementation']
      }));
    }
    
    // Compression
    if (this.config.enableCompression) {
      this.app.use(compression());
    }
    
    // Logging
    this.app.use(morgan('combined'));
    
    // Rate limiting
    if (this.config.enableRateLimit) {
      const limiter = rateLimit({
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 1000, // limit each IP to 1000 requests per windowMs
        message: 'Too many requests from this IP'
      });
      this.app.use(limiter);
    }
    
    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    
    // Custom middleware
    this.app.use(this.requestTracker.bind(this));
    this.app.use(this.implementationSelector.bind(this));
    
    if (this.config.enableAuth) {
      this.app.use('/api', this.authMiddleware.authenticate());
    }
  }

  /**
   * Setup REST API routes
   */
  setupRestRoutes() {
    console.log('🔗 Setting up REST API routes...');
    
    const restRoutes = new RestRoutes(this);
    
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        implementations: Array.from(this.knowledgeBases.keys()),
        currentImplementation: this.config.defaultImplementation,
        metrics: this.getMetrics()
      });
    });
    
    // API Info
    this.app.get('/api', (req, res) => {
      res.json({
        name: 'Advanced Knowledge Base API',
        version: '1.0.0',
        description: 'High-performance knowledge base with sophisticated relationship modeling',
        endpoints: {
          rest: `http://${this.config.host}:${this.config.port}/api`,
          graphql: `http://${this.config.host}:${this.config.graphqlPort}/graphql`,
          websocket: `ws://${this.config.host}:${this.config.websocketPort}`
        },
        implementations: Array.from(this.knowledgeBases.keys()),
        documentation: '/api/docs'
      });
    });
    
    // Node operations
    this.app.use('/api/nodes', restRoutes.nodeRoutes());
    
    // Edge operations  
    this.app.use('/api/edges', restRoutes.edgeRoutes());
    
    // Pattern matching
    this.app.use('/api/patterns', restRoutes.patternRoutes());
    
    // Graph traversal
    this.app.use('/api/traverse', restRoutes.traversalRoutes());
    
    // Hypergraph operations
    this.app.use('/api/hypergraph', restRoutes.hypergraphRoutes());
    
    // Bulk operations
    this.app.use('/api/bulk', restRoutes.bulkRoutes());
    
    // Analytics
    this.app.use('/api/analytics', restRoutes.analyticsRoutes());
    
    // Admin operations
    this.app.use('/api/admin', restRoutes.adminRoutes());
    
    // Implementation switching
    this.app.post('/api/implementation', async (req, res) => {
      try {
        const { implementation } = req.body;
        
        if (!this.knowledgeBases.has(implementation)) {
          return res.status(400).json({
            error: 'Invalid implementation',
            available: Array.from(this.knowledgeBases.keys())
          });
        }
        
        this.currentKB = this.knowledgeBases.get(implementation);
        this.config.defaultImplementation = implementation;
        
        res.json({
          success: true,
          currentImplementation: implementation,
          message: `Switched to ${implementation} implementation`
        });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
    
    // Error handling
    this.app.use((error, req, res, next) => {
      console.error('API Error:', error);
      this.metrics.errors++;
      
      res.status(error.status || 500).json({
        error: error.message,
        timestamp: new Date().toISOString(),
        requestId: req.requestId
      });
    });
    
    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        error: 'Endpoint not found',
        path: req.originalUrl,
        method: req.method,
        availableEndpoints: [
          'GET /health',
          'GET /api',
          'POST /api/nodes',
          'GET /api/nodes/:id',
          'POST /api/edges',
          'POST /api/patterns/search',
          'POST /api/traverse',
          'POST /api/bulk/nodes',
          'GET /api/analytics/stats'
        ]
      });
    });
  }

  /**
   * Setup GraphQL server
   */
  async setupGraphQLServer() {
    console.log('🎯 Setting up GraphQL server...');
    
    const schema = new GraphQLSchema(this);
    
    this.apolloServer = new ApolloServer({
      typeDefs: schema.typeDefs,
      resolvers: schema.resolvers,
      context: ({ req }) => ({
        knowledgeBase: this.currentKB,
        knowledgeBases: this.knowledgeBases,
        requestId: req.headers['x-request-id'] || uuidv4(),
        implementation: req.headers['x-kb-implementation'] || this.config.defaultImplementation
      }),
      plugins: [
        {
          requestDidStart() {
            return {
              willSendResponse(requestContext) {
                // Add custom headers
                requestContext.response.http.setHeader('X-KB-Implementation', 
                  requestContext.context.implementation);
              }
            };
          }
        }
      ],
      introspection: true,
      playground: process.env.NODE_ENV !== 'production'
    });
    
    await this.apolloServer.start();
    this.apolloServer.applyMiddleware({ 
      app: this.app, 
      path: '/graphql',
      cors: this.config.enableCors
    });
  }

  /**
   * Setup WebSocket server
   */
  setupWebSocketServer() {
    console.log('🔌 Setting up WebSocket server...');
    
    this.httpServer = createServer(this.app);
    
    this.wsServer = new WebSocketServer({
      server: this.httpServer,
      port: this.config.websocketPort
    });
    
    const wsHandler = new WebSocketHandler(this);
    
    this.wsServer.on('connection', (ws, request) => {
      const connectionId = uuidv4();
      
      console.log(`🔌 New WebSocket connection: ${connectionId}`);
      this.metrics.activeConnections++;
      
      this.wsConnections.set(connectionId, {
        socket: ws,
        connectedAt: Date.now(),
        requestCount: 0
      });
      
      ws.on('message', async (message) => {
        try {
          const data = JSON.parse(message.toString());
          const response = await wsHandler.handleMessage(data, connectionId);
          ws.send(JSON.stringify(response));
          
          this.wsConnections.get(connectionId).requestCount++;
        } catch (error) {
          ws.send(JSON.stringify({
            error: error.message,
            timestamp: new Date().toISOString()
          }));
        }
      });
      
      ws.on('close', () => {
        console.log(`🔌 WebSocket connection closed: ${connectionId}`);
        this.wsConnections.delete(connectionId);
        this.metrics.activeConnections--;
      });
      
      // Send welcome message
      ws.send(JSON.stringify({
        type: 'connection',
        connectionId,
        message: 'Connected to Advanced Knowledge Base API',
        availableCommands: [
          'createNode',
          'createEdge', 
          'findPattern',
          'traverse',
          'subscribe',
          'getMetrics'
        ]
      }));
    });
  }

  /**
   * Setup monitoring and metrics
   */
  setupMonitoring() {
    if (!this.config.enableMetrics) return;
    
    console.log('📊 Setting up monitoring...');
    
    // Collect metrics every 10 seconds
    setInterval(() => {
      this.collectMetrics();
    }, 10000);
    
    // Memory usage monitoring
    setInterval(() => {
      const usage = process.memoryUsage();
      this.metrics.memoryUsage = usage.heapUsed;
      
      // Warn if memory usage is high
      if (usage.heapUsed > 1024 * 1024 * 1024) { // 1GB
        console.warn('⚠️  High memory usage detected:', 
          Math.round(usage.heapUsed / 1024 / 1024), 'MB');
      }
    }, 30000);
    
    // Metrics endpoint
    this.app.get('/metrics', (req, res) => {
      res.json(this.getMetrics());
    });
  }

  /**
   * Start HTTP server
   */
  async startHttpServer() {
    return new Promise((resolve, reject) => {
      this.httpServer = this.httpServer || createServer(this.app);
      
      this.httpServer.listen(this.config.port, this.config.host, (error) => {
        if (error) {
          reject(error);
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Middleware: Track requests
   */
  requestTracker(req, res, next) {
    req.requestId = uuidv4();
    req.startTime = Date.now();
    
    this.metrics.requests++;
    
    res.on('finish', () => {
      const duration = Date.now() - req.startTime;
      this.metrics.responseTime.push(duration);
      
      // Keep only last 1000 response times
      if (this.metrics.responseTime.length > 1000) {
        this.metrics.responseTime.shift();
      }
    });
    
    next();
  }

  /**
   * Middleware: Select knowledge base implementation
   */
  implementationSelector(req, res, next) {
    const requestedImpl = req.headers['x-kb-implementation'];
    
    if (requestedImpl && this.knowledgeBases.has(requestedImpl)) {
      req.knowledgeBase = this.knowledgeBases.get(requestedImpl);
      req.implementation = requestedImpl;
    } else {
      req.knowledgeBase = this.currentKB;
      req.implementation = this.config.defaultImplementation;
    }
    
    next();
  }

  /**
   * Collect performance metrics
   */
  collectMetrics() {
    // Calculate queries per second
    const now = Date.now();
    if (!this.lastMetricsTime) {
      this.lastMetricsTime = now;
      this.lastRequestCount = this.metrics.requests;
      return;
    }
    
    const timeDiff = (now - this.lastMetricsTime) / 1000;
    const requestDiff = this.metrics.requests - this.lastRequestCount;
    
    this.metrics.queriesPerSecond = requestDiff / timeDiff;
    
    this.lastMetricsTime = now;
    this.lastRequestCount = this.metrics.requests;
  }

  /**
   * Get comprehensive metrics
   */
  getMetrics() {
    const avgResponseTime = this.metrics.responseTime.length > 0 ?
      this.metrics.responseTime.reduce((a, b) => a + b, 0) / this.metrics.responseTime.length : 0;
    
    return {
      server: {
        uptime: process.uptime(),
        requests: this.metrics.requests,
        errors: this.metrics.errors,
        activeConnections: this.metrics.activeConnections,
        queriesPerSecond: this.metrics.queriesPerSecond,
        averageResponseTime: Math.round(avgResponseTime),
        memoryUsage: this.metrics.memoryUsage
      },
      implementations: Object.fromEntries(
        Array.from(this.knowledgeBases.entries()).map(([name, kb]) => [
          name,
          kb.getStatistics ? kb.getStatistics() : { status: 'active' }
        ])
      ),
      system: {
        nodeVersion: process.version,
        platform: process.platform,
        arch: process.arch,
        cpuUsage: process.cpuUsage(),
        memoryUsage: process.memoryUsage()
      }
    };
  }

  /**
   * Broadcast message to all WebSocket connections
   */
  broadcast(message) {
    const data = JSON.stringify({
      type: 'broadcast',
      timestamp: new Date().toISOString(),
      ...message
    });
    
    for (const connection of this.wsConnections.values()) {
      if (connection.socket.readyState === 1) { // OPEN
        connection.socket.send(data);
      }
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    console.log('🛑 Shutting down API server...');
    
    this.isRunning = false;
    
    // Close WebSocket connections
    for (const connection of this.wsConnections.values()) {
      connection.socket.close();
    }
    
    // Close knowledge base connections
    for (const [name, kb] of this.knowledgeBases) {
      try {
        if (kb.close) {
          await kb.close();
        }
        console.log(`✅ Closed ${name} implementation`);
      } catch (error) {
        console.warn(`⚠️  Error closing ${name}:`, error.message);
      }
    }
    
    // Close HTTP server
    if (this.httpServer) {
      await new Promise((resolve) => {
        this.httpServer.close(resolve);
      });
    }
    
    console.log('✅ API server shutdown complete');
  }

  /**
   * Get current knowledge base instance
   */
  getKnowledgeBase(implementation) {
    if (implementation && this.knowledgeBases.has(implementation)) {
      return this.knowledgeBases.get(implementation);
    }
    return this.currentKB;
  }
}

// Export the server class
module.exports = KnowledgeBaseAPIServer;

// Start server if run directly
if (require.main === module) {
  const server = new KnowledgeBaseAPIServer({
    port: process.env.PORT || 8080,
    graphqlPort: process.env.GRAPHQL_PORT || 8081,
    websocketPort: process.env.WEBSOCKET_PORT || 8082,
    enableAuth: process.env.ENABLE_AUTH === 'true',
    enableMetrics: process.env.ENABLE_METRICS !== 'false',
    implementations: process.env.KB_IMPLEMENTATIONS?.split(',') || 
      ['neo4j', 'redis', 'hypergraph', 'custom'],
    defaultImplementation: process.env.DEFAULT_IMPLEMENTATION || 'neo4j'
  });

  // Graceful shutdown
  process.on('SIGTERM', async () => {
    await server.shutdown();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    await server.shutdown();
    process.exit(0);
  });

  // Start the server
  server.start().catch((error) => {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  });
}