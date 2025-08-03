/**
 * Redis Graph Knowledge Base Implementation
 * 
 * High-speed in-memory graph database implementation with:
 * - Ultra-fast atomic operations
 * - Real-time relationship updates
 * - Memory-optimized data structures
 * - Sub-millisecond query response times
 * - Redis ecosystem integration
 */

const redis = require('redis');
const { Graph } = require('redisgraph');
const { v4: uuidv4 } = require('uuid');
const EventEmitter = require('events');

class RedisGraphKnowledgeBase extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      host: config.host || 'localhost',
      port: config.port || 6379,
      password: config.password || null,
      database: config.database || 0,
      graphName: config.graphName || 'knowledge',
      maxRetries: config.maxRetries || 3,
      retryDelayOnFailover: config.retryDelayOnFailover || 100,
      enableAutoPipelining: config.enableAutoPipelining || true,
      maxRetriesPerRequest: config.maxRetriesPerRequest || 3,
      ...config
    };
    
    this.client = null;
    this.graph = null;
    this.isConnected = false;
    
    // High-performance caching layers
    this.nodeIndex = new Map(); // Fast node lookup
    this.edgeIndex = new Map(); // Fast edge lookup
    this.typeIndex = new Map(); // Nodes grouped by type
    this.relationshipIndex = new Map(); // Edges grouped by type
    
    // Performance counters
    this.metrics = {
      operations: 0,
      cacheHits: 0,
      cacheMisses: 0,
      queryTime: 0,
      memoryUsage: 0
    };
    
    // Batch operation buffer
    this.batchBuffer = [];
    this.batchSize = config.batchSize || 1000;
    this.batchTimeout = config.batchTimeout || 100;
    this.batchTimer = null;
  }

  /**
   * Connect to Redis and initialize graph
   */
  async connect() {
    try {
      this.client = redis.createClient({
        socket: {
          host: this.config.host,
          port: this.config.port
        },
        password: this.config.password,
        database: this.config.database,
        enableAutoPipelining: this.config.enableAutoPipelining,
        maxRetriesPerRequest: this.config.maxRetriesPerRequest,
        retryDelayOnFailover: this.config.retryDelayOnFailover
      });

      this.client.on('error', (error) => {
        this.emit('error', error);
      });

      this.client.on('connect', () => {
        this.emit('connected');
      });

      await this.client.connect();
      
      // Initialize RedisGraph
      this.graph = new Graph(this.config.graphName, this.client);
      this.isConnected = true;

      // Create initial indexes for performance
      await this.createIndexes();
      
      // Start batch processing
      this.startBatchProcessor();

      return true;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to connect to Redis: ${error.message}`);
    }
  }

  /**
   * Create performance indexes
   */
  async createIndexes() {
    const indexes = [
      // Node indexes
      'CREATE INDEX ON :AtomicInfo(id)',
      'CREATE INDEX ON :AtomicInfo(type)',
      'CREATE INDEX ON :AtomicInfo(domain)',
      'CREATE INDEX ON :AtomicInfo(created)',
      
      // Group indexes
      'CREATE INDEX ON :Group(id)',
      'CREATE INDEX ON :Group(hierarchy)',
      
      // Composite indexes
      'CREATE INDEX ON :AtomicInfo(type, domain)'
    ];

    for (const indexQuery of indexes) {
      try {
        await this.graph.query(indexQuery);
      } catch (error) {
        // Index might already exist
        console.warn(`Index creation warning: ${error.message}`);
      }
    }
  }

  /**
   * Create atomic information node with ultra-fast insertion
   */
  async createNode(nodeData) {
    if (!this.isConnected) {
      throw new Error('Not connected to Redis');
    }

    const nodeId = nodeData.id || uuidv4();
    const timestamp = Date.now();
    
    const node = {
      id: nodeId,
      type: nodeData.type || 'generic',
      data: nodeData.data || {},
      metadata: {
        created: timestamp,
        updated: timestamp,
        version: 1,
        confidence: nodeData.metadata?.confidence || 1.0,
        ...nodeData.metadata
      },
      properties: nodeData.properties || {},
      ...nodeData
    };

    // Build labels dynamically
    const labels = ['AtomicInfo', node.type];
    if (nodeData.labels) {
      labels.push(...nodeData.labels);
    }

    // Optimize for Redis Graph syntax
    const labelString = labels.join(':');
    const propertiesString = this.buildPropertiesString(node);

    const query = `
      CREATE (n:${labelString} {${propertiesString}})
      RETURN n
    `;

    const startTime = process.hrtime.bigint();
    
    try {
      const result = await this.graph.query(query, {
        id: node.id,
        type: node.type,
        data: JSON.stringify(node.data),
        metadata: JSON.stringify(node.metadata),
        properties: JSON.stringify(node.properties),
        created: node.metadata.created,
        updated: node.metadata.updated,
        version: node.metadata.version,
        confidence: node.metadata.confidence
      });

      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000; // Convert to ms
      this.metrics.operations++;

      // Update local indexes for ultra-fast lookups
      this.nodeIndex.set(nodeId, node);
      
      if (!this.typeIndex.has(node.type)) {
        this.typeIndex.set(node.type, new Set());
      }
      this.typeIndex.get(node.type).add(nodeId);

      this.emit('nodeCreated', node);
      return node;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to create node: ${error.message}`);
    }
  }

  /**
   * Create high-performance relationship
   */
  async createEdge(edgeData) {
    if (!this.isConnected) {
      throw new Error('Not connected to Redis');
    }

    const edgeId = edgeData.id || uuidv4();
    const timestamp = Date.now();

    const edge = {
      id: edgeId,
      type: edgeData.type || 'RELATES_TO',
      from: edgeData.from,
      to: edgeData.to,
      properties: edgeData.properties || {},
      metadata: {
        created: timestamp,
        updated: timestamp,
        strength: edgeData.properties?.strength || 1.0,
        confidence: edgeData.metadata?.confidence || 1.0,
        ...edgeData.metadata
      },
      weight: edgeData.weight || 1.0,
      bidirectional: edgeData.bidirectional || false
    };

    const propertiesString = this.buildPropertiesString(edge);
    
    let query = `
      MATCH (from:AtomicInfo {id: $fromId})
      MATCH (to:AtomicInfo {id: $toId})
      CREATE (from)-[r:${edge.type} {${propertiesString}}]->(to)
    `;

    // Add reverse relationship if bidirectional
    if (edge.bidirectional) {
      query += `
        CREATE (to)-[r2:${edge.type} {${propertiesString.replace('id: $id', 'id: $reverseId')}}]->(from)
        RETURN r, r2
      `;
    } else {
      query += ' RETURN r';
    }

    const startTime = process.hrtime.bigint();

    try {
      const result = await this.graph.query(query, {
        fromId: edge.from,
        toId: edge.to,
        id: edge.id,
        reverseId: edge.id + '_reverse',
        type: edge.type,
        properties: JSON.stringify(edge.properties),
        metadata: JSON.stringify(edge.metadata),
        strength: edge.metadata.strength,
        weight: edge.weight,
        created: edge.metadata.created,
        updated: edge.metadata.updated,
        confidence: edge.metadata.confidence
      });

      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000;
      this.metrics.operations++;

      // Update relationship index
      this.edgeIndex.set(edgeId, edge);
      
      if (!this.relationshipIndex.has(edge.type)) {
        this.relationshipIndex.set(edge.type, new Set());
      }
      this.relationshipIndex.get(edge.type).add(edgeId);

      this.emit('edgeCreated', edge);
      return edge;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to create edge: ${error.message}`);
    }
  }

  /**
   * Batch create nodes for maximum performance
   */
  async batchCreateNodes(nodes) {
    if (!Array.isArray(nodes) || nodes.length === 0) {
      return [];
    }

    const queries = [];
    const parameters = {};
    
    nodes.forEach((nodeData, index) => {
      const nodeId = nodeData.id || uuidv4();
      const timestamp = Date.now();
      
      const node = {
        id: nodeId,
        type: nodeData.type || 'generic',
        data: nodeData.data || {},
        metadata: {
          created: timestamp,
          updated: timestamp,
          version: 1,
          confidence: nodeData.metadata?.confidence || 1.0,
          ...nodeData.metadata
        },
        properties: nodeData.properties || {},
        ...nodeData
      };

      const labels = ['AtomicInfo', node.type];
      const labelString = labels.join(':');
      const propertiesString = this.buildPropertiesString(node, index);

      queries.push(`CREATE (n${index}:${labelString} {${propertiesString}})`);
      
      // Add parameters for this node
      Object.entries(node).forEach(([key, value]) => {
        const paramKey = `${key}_${index}`;
        parameters[paramKey] = typeof value === 'object' ? JSON.stringify(value) : value;
      });
    });

    const batchQuery = queries.join(' ') + ` RETURN ${nodes.map((_, i) => `n${i}`).join(', ')}`;

    const startTime = process.hrtime.bigint();

    try {
      const result = await this.graph.query(batchQuery, parameters);
      
      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000;
      this.metrics.operations += nodes.length;

      // Update indexes
      nodes.forEach((nodeData, index) => {
        const nodeId = nodeData.id || parameters[`id_${index}`];
        this.nodeIndex.set(nodeId, nodeData);
        
        if (!this.typeIndex.has(nodeData.type)) {
          this.typeIndex.set(nodeData.type, new Set());
        }
        this.typeIndex.get(nodeData.type).add(nodeId);
      });

      this.emit('batchNodesCreated', nodes);
      return nodes;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Batch node creation failed: ${error.message}`);
    }
  }

  /**
   * Ultra-fast node retrieval with caching
   */
  async getNode(nodeId, includeRelationships = false) {
    // Check cache first for instant response
    if (this.nodeIndex.has(nodeId)) {
      this.metrics.cacheHits++;
      const cachedNode = this.nodeIndex.get(nodeId);
      
      if (!includeRelationships) {
        return cachedNode;
      }
    }

    this.metrics.cacheMisses++;
    
    let query = `MATCH (n:AtomicInfo {id: $nodeId})`;
    
    if (includeRelationships) {
      query += `
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN n, collect({rel: r, node: connected}) as relationships
      `;
    } else {
      query += ' RETURN n';
    }

    const startTime = process.hrtime.bigint();

    try {
      const result = await this.graph.query(query, { nodeId });
      
      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000;
      this.metrics.operations++;

      if (result.data.length === 0) {
        return null;
      }

      const nodeData = result.data[0][0];
      
      if (includeRelationships && result.data[0][1]) {
        nodeData.relationships = result.data[0][1];
      }

      // Update cache
      this.nodeIndex.set(nodeId, nodeData);

      return nodeData;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to get node: ${error.message}`);
    }
  }

  /**
   * High-speed pattern matching
   */
  async findPattern(pattern, options = {}) {
    const limit = options.limit || 100;
    const offset = options.offset || 0;

    let query = 'MATCH ';
    const parameters = {};

    // Build node patterns
    const nodePatterns = [];
    Object.entries(pattern.nodes || {}).forEach(([alias, nodeSpec]) => {
      let nodePattern = `(${alias}:AtomicInfo`;
      
      if (nodeSpec.properties) {
        const conditions = [];
        Object.entries(nodeSpec.properties).forEach(([key, value]) => {
          const paramName = `${alias}_${key}`;
          parameters[paramName] = value;
          conditions.push(`${key}: $${paramName}`);
        });
        
        if (conditions.length > 0) {
          nodePattern += ` {${conditions.join(', ')}}`;
        }
      }
      
      nodePattern += ')';
      nodePatterns.push(nodePattern);
    });

    query += nodePatterns.join(', ');

    // Build relationship patterns
    if (pattern.edges && pattern.edges.length > 0) {
      const edgePatterns = [];
      pattern.edges.forEach((edge, index) => {
        const edgeAlias = `e${index}`;
        let edgePattern = `(${edge.from})-[${edgeAlias}:${edge.type || 'RELATES_TO'}`;
        
        if (edge.properties) {
          const conditions = [];
          Object.entries(edge.properties).forEach(([key, value]) => {
            const paramName = `${edgeAlias}_${key}`;
            parameters[paramName] = value;
            conditions.push(`${key}: $${paramName}`);
          });
          
          if (conditions.length > 0) {
            edgePattern += ` {${conditions.join(', ')}}`;
          }
        }
        
        edgePattern += `]->(${edge.to})`;
        edgePatterns.push(edgePattern);
      });
      
      if (edgePatterns.length > 0) {
        query += ', ' + edgePatterns.join(', ');
      }
    }

    // Add constraints
    const whereConditions = [];
    if (pattern.constraints) {
      if (pattern.constraints.temporal?.after) {
        whereConditions.push('n.created > $afterTime');
        parameters.afterTime = new Date(pattern.constraints.temporal.after).getTime();
      }
      
      if (pattern.constraints.confidence?.min) {
        whereConditions.push('n.confidence >= $minConfidence');
        parameters.minConfidence = pattern.constraints.confidence.min;
      }
    }

    if (whereConditions.length > 0) {
      query += ` WHERE ${whereConditions.join(' AND ')}`;
    }

    // Return clause
    const returnItems = Object.keys(pattern.nodes || {});
    if (pattern.edges) {
      returnItems.push(...pattern.edges.map((_, index) => `e${index}`));
    }
    
    query += ` RETURN ${returnItems.join(', ')}`;
    query += ` SKIP ${offset} LIMIT ${limit}`;

    const startTime = process.hrtime.bigint();

    try {
      const result = await this.graph.query(query, parameters);
      
      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000;
      this.metrics.operations++;

      return result.data.map(row => {
        const match = {};
        returnItems.forEach((item, index) => {
          match[item] = row[index];
        });
        return match;
      });
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Pattern matching failed: ${error.message}`);
    }
  }

  /**
   * Real-time graph traversal
   */
  async traverse(startNodeId, options = {}) {
    const maxDepth = options.maxDepth || 3;
    const direction = options.direction || 'BOTH'; // IN, OUT, BOTH
    const relationshipTypes = options.relationshipTypes || [];
    
    let relationshipFilter = '';
    if (relationshipTypes.length > 0) {
      relationshipFilter = ':' + relationshipTypes.join('|');
    }

    const query = `
      MATCH path = (start:AtomicInfo {id: $startNodeId})-[${relationshipFilter}*1..${maxDepth}]-(end)
      RETURN path, length(path) as depth
      ORDER BY depth ASC
      LIMIT ${options.limit || 1000}
    `;

    const startTime = process.hrtime.bigint();

    try {
      const result = await this.graph.query(query, { startNodeId });
      
      const endTime = process.hrtime.bigint();
      this.metrics.queryTime += Number(endTime - startTime) / 1000000;
      this.metrics.operations++;

      return result.data.map(row => ({
        path: row[0],
        depth: row[1]
      }));
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Graph traversal failed: ${error.message}`);
    }
  }

  /**
   * Get real-time performance metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      averageQueryTime: this.metrics.operations > 0 ? this.metrics.queryTime / this.metrics.operations : 0,
      cacheHitRate: this.metrics.cacheHits / (this.metrics.cacheHits + this.metrics.cacheMisses) || 0,
      indexSizes: {
        nodes: this.nodeIndex.size,
        edges: this.edgeIndex.size,
        types: this.typeIndex.size,
        relationships: this.relationshipIndex.size
      }
    };
  }

  /**
   * Start batch processor for high-throughput operations
   */
  startBatchProcessor() {
    this.batchTimer = setInterval(() => {
      if (this.batchBuffer.length > 0) {
        this.processBatch();
      }
    }, this.batchTimeout);
  }

  /**
   * Process batched operations
   */
  async processBatch() {
    if (this.batchBuffer.length === 0) return;

    const batch = this.batchBuffer.splice(0, this.batchSize);
    
    try {
      // Group operations by type
      const nodeOps = batch.filter(op => op.type === 'createNode');
      const edgeOps = batch.filter(op => op.type === 'createEdge');

      // Execute batched operations
      if (nodeOps.length > 0) {
        await this.batchCreateNodes(nodeOps.map(op => op.data));
      }

      if (edgeOps.length > 0) {
        // Process edges in batch
        await this.batchCreateEdges(edgeOps.map(op => op.data));
      }

      this.emit('batchProcessed', { nodes: nodeOps.length, edges: edgeOps.length });
    } catch (error) {
      this.emit('error', error);
    }
  }

  /**
   * Add operation to batch queue
   */
  addToBatch(operation) {
    this.batchBuffer.push(operation);
    
    if (this.batchBuffer.length >= this.batchSize) {
      this.processBatch();
    }
  }

  /**
   * Build properties string for queries
   */
  buildPropertiesString(obj, index = '') {
    const properties = [];
    const suffix = index !== '' ? `_${index}` : '';
    
    Object.entries(obj).forEach(([key, value]) => {
      if (key !== 'labels' && value !== undefined) {
        properties.push(`${key}: $${key}${suffix}`);
      }
    });
    
    return properties.join(', ');
  }

  /**
   * Batch create edges
   */
  async batchCreateEdges(edges) {
    // Implementation for batch edge creation
    const queries = [];
    const parameters = {};
    
    edges.forEach((edgeData, index) => {
      const edgeId = edgeData.id || uuidv4();
      const query = `
        MATCH (from${index}:AtomicInfo {id: $fromId_${index}})
        MATCH (to${index}:AtomicInfo {id: $toId_${index}})
        CREATE (from${index})-[r${index}:${edgeData.type || 'RELATES_TO'} {id: $id_${index}}]->(to${index})
      `;
      
      queries.push(query);
      parameters[`fromId_${index}`] = edgeData.from;
      parameters[`toId_${index}`] = edgeData.to;
      parameters[`id_${index}`] = edgeId;
    });

    if (queries.length > 0) {
      const batchQuery = queries.join(' ');
      await this.graph.query(batchQuery, parameters);
    }
  }

  /**
   * Close connection and cleanup
   */
  async close() {
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
    }
    
    if (this.client) {
      await this.client.quit();
    }
    
    this.isConnected = false;
    this.emit('disconnected');
  }

  /**
   * Clear all caches
   */
  clearCache() {
    this.nodeIndex.clear();
    this.edgeIndex.clear();
    this.typeIndex.clear();
    this.relationshipIndex.clear();
  }
}

module.exports = RedisGraphKnowledgeBase;