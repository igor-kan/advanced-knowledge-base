/**
 * Neo4j Knowledge Base Implementation
 * 
 * High-performance graph database implementation supporting:
 * - Atomic information units as nodes
 * - Sophisticated relationship modeling
 * - Relationship properties and metadata
 * - Hierarchical group structures
 * - Billions-scale performance optimizations
 */

const neo4j = require('neo4j-driver');
const { v4: uuidv4, v1: uuidv1 } = require('uuid');
const EventEmitter = require('events');

class Neo4jKnowledgeBase extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      uri: config.uri || 'bolt://localhost:7687',
      username: config.username || 'neo4j',
      password: config.password || 'password',
      database: config.database || 'knowledge',
      maxConnectionPoolSize: config.maxConnectionPoolSize || 100,
      connectionTimeout: config.connectionTimeout || 30000,
      maxTransactionRetryTime: config.maxTransactionRetryTime || 30000,
      ...config
    };
    
    this.driver = null;
    this.session = null;
    this.isConnected = false;
    
    // Performance optimization caches
    this.nodeCache = new Map();
    this.edgeCache = new Map();
    this.groupCache = new Map();
    this.queryCache = new Map();
    
    // Statistics tracking
    this.stats = {
      nodesCreated: 0,
      edgesCreated: 0,
      groupsCreated: 0,
      queriesExecuted: 0,
      cacheHits: 0,
      cacheMisses: 0
    };
  }

  /**
   * Initialize connection to Neo4j database
   */
  async connect() {
    try {
      this.driver = neo4j.driver(
        this.config.uri,
        neo4j.auth.basic(this.config.username, this.config.password),
        {
          maxConnectionPoolSize: this.config.maxConnectionPoolSize,
          connectionTimeout: this.config.connectionTimeout,
          maxTransactionRetryTime: this.config.maxTransactionRetryTime,
          disableLosslessIntegers: true
        }
      );

      // Verify connectivity
      await this.driver.verifyConnectivity();
      
      // Create database if it doesn't exist
      const session = this.driver.session({ database: 'system' });
      try {
        await session.run(`CREATE DATABASE ${this.config.database} IF NOT EXISTS`);
      } catch (error) {
        // Database might already exist or user lacks permissions
        console.warn('Could not create database:', error.message);
      } finally {
        await session.close();
      }

      this.session = this.driver.session({ database: this.config.database });
      this.isConnected = true;

      // Create indexes for performance
      await this.createIndexes();
      
      this.emit('connected');
      return true;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to connect to Neo4j: ${error.message}`);
    }
  }

  /**
   * Create performance indexes
   */
  async createIndexes() {
    const indexes = [
      // Node indexes
      'CREATE INDEX node_id_index IF NOT EXISTS FOR (n:AtomicInfo) ON (n.id)',
      'CREATE INDEX node_type_index IF NOT EXISTS FOR (n:AtomicInfo) ON (n.type)',
      'CREATE INDEX node_domain_index IF NOT EXISTS FOR (n:AtomicInfo) ON (n.domain)',
      'CREATE INDEX node_created_index IF NOT EXISTS FOR (n:AtomicInfo) ON (n.created)',
      
      // Edge indexes
      'CREATE INDEX edge_id_index IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.id)',
      'CREATE INDEX edge_type_index IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.type)',
      'CREATE INDEX edge_strength_index IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.strength)',
      
      // Group indexes
      'CREATE INDEX group_id_index IF NOT EXISTS FOR (g:Group) ON (g.id)',
      'CREATE INDEX group_hierarchy_index IF NOT EXISTS FOR (g:Group) ON (g.hierarchy)',
      
      // Composite indexes for complex queries
      'CREATE INDEX node_type_domain_index IF NOT EXISTS FOR (n:AtomicInfo) ON (n.type, n.domain)',
      'CREATE INDEX edge_type_strength_index IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.type, r.strength)'
    ];

    for (const indexQuery of indexes) {
      try {
        await this.session.run(indexQuery);
      } catch (error) {
        console.warn(`Index creation warning: ${error.message}`);
      }
    }
  }

  /**
   * Create an atomic information unit (node)
   */
  async createNode(nodeData) {
    if (!this.isConnected) {
      throw new Error('Not connected to database');
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
        source: nodeData.metadata?.source || 'user',
        ...nodeData.metadata
      },
      properties: nodeData.properties || {},
      labels: nodeData.labels || [],
      ...nodeData
    };

    // Build dynamic labels
    const labels = ['AtomicInfo', node.type];
    if (node.labels && Array.isArray(node.labels)) {
      labels.push(...node.labels);
    }
    const labelString = labels.map(label => `:${label}`).join('');

    const query = `
      CREATE (n${labelString} {
        id: $id,
        type: $type,
        data: $data,
        metadata: $metadata,
        properties: $properties,
        created: $created,
        updated: $updated,
        version: $version
      })
      RETURN n
    `;

    try {
      const result = await this.session.run(query, {
        id: node.id,
        type: node.type,
        data: JSON.stringify(node.data),
        metadata: JSON.stringify(node.metadata),
        properties: JSON.stringify(node.properties),
        created: node.metadata.created,
        updated: node.metadata.updated,
        version: node.metadata.version
      });

      const createdNode = result.records[0].get('n').properties;
      
      // Cache the node
      this.nodeCache.set(nodeId, createdNode);
      this.stats.nodesCreated++;
      
      this.emit('nodeCreated', createdNode);
      return createdNode;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to create node: ${error.message}`);
    }
  }

  /**
   * Create a sophisticated relationship (edge)
   */
  async createEdge(edgeData) {
    if (!this.isConnected) {
      throw new Error('Not connected to database');
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
        version: 1,
        strength: edgeData.properties?.strength || 1.0,
        confidence: edgeData.metadata?.confidence || 1.0,
        source: edgeData.metadata?.source || 'user',
        temporal: edgeData.metadata?.temporal || null,
        context: edgeData.metadata?.context || null,
        ...edgeData.metadata
      },
      bidirectional: edgeData.bidirectional || false,
      weight: edgeData.weight || 1.0,
      ...edgeData
    };

    const query = `
      MATCH (from:AtomicInfo {id: $fromId})
      MATCH (to:AtomicInfo {id: $toId})
      CREATE (from)-[r:${edge.type} {
        id: $id,
        type: $type,
        properties: $properties,
        metadata: $metadata,
        strength: $strength,
        weight: $weight,
        bidirectional: $bidirectional,
        created: $created,
        updated: $updated,
        version: $version
      }]->(to)
      ${edge.bidirectional ? `CREATE (to)-[r2:${edge.type} {
        id: $id + '_reverse',
        type: $type,
        properties: $properties,
        metadata: $metadata,
        strength: $strength,
        weight: $weight,
        bidirectional: $bidirectional,
        created: $created,
        updated: $updated,
        version: $version,
        reverse_of: $id
      }]->(from)` : ''}
      RETURN r${edge.bidirectional ? ', r2' : ''}
    `;

    try {
      const result = await this.session.run(query, {
        fromId: edge.from,
        toId: edge.to,
        id: edge.id,
        type: edge.type,
        properties: JSON.stringify(edge.properties),
        metadata: JSON.stringify(edge.metadata),
        strength: edge.metadata.strength,
        weight: edge.weight,
        bidirectional: edge.bidirectional,
        created: edge.metadata.created,
        updated: edge.metadata.updated,
        version: edge.metadata.version
      });

      const createdEdge = result.records[0].get('r').properties;
      
      // Cache the edge
      this.edgeCache.set(edgeId, createdEdge);
      this.stats.edgesCreated++;
      
      this.emit('edgeCreated', createdEdge);
      return createdEdge;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to create edge: ${error.message}`);
    }
  }

  /**
   * Create a hierarchical group
   */
  async createGroup(groupData) {
    if (!this.isConnected) {
      throw new Error('Not connected to database');
    }

    const groupId = groupData.id || uuidv4();
    const timestamp = Date.now();

    const group = {
      id: groupId,
      name: groupData.name || `Group_${groupId}`,
      type: groupData.type || 'collection',
      hierarchy: groupData.hierarchy || 'root',
      members: groupData.members || [],
      properties: groupData.properties || {},
      metadata: {
        created: timestamp,
        updated: timestamp,
        version: 1,
        memberCount: groupData.members?.length || 0,
        ...groupData.metadata
      },
      ...groupData
    };

    // Create the group node
    const createGroupQuery = `
      CREATE (g:Group {
        id: $id,
        name: $name,
        type: $type,
        hierarchy: $hierarchy,
        properties: $properties,
        metadata: $metadata,
        created: $created,
        updated: $updated,
        version: $version,
        memberCount: $memberCount
      })
      RETURN g
    `;

    try {
      const groupResult = await this.session.run(createGroupQuery, {
        id: group.id,
        name: group.name,
        type: group.type,
        hierarchy: group.hierarchy,
        properties: JSON.stringify(group.properties),
        metadata: JSON.stringify(group.metadata),
        created: group.metadata.created,
        updated: group.metadata.updated,
        version: group.metadata.version,
        memberCount: group.metadata.memberCount
      });

      // Add members to the group
      if (group.members && group.members.length > 0) {
        const addMembersQuery = `
          MATCH (g:Group {id: $groupId})
          UNWIND $members AS memberId
          MATCH (n:AtomicInfo {id: memberId})
          CREATE (g)-[:CONTAINS {
            id: $groupId + '_contains_' + memberId,
            created: $created
          }]->(n)
        `;

        await this.session.run(addMembersQuery, {
          groupId: group.id,
          members: group.members,
          created: timestamp
        });
      }

      const createdGroup = groupResult.records[0].get('g').properties;
      
      // Cache the group
      this.groupCache.set(groupId, createdGroup);
      this.stats.groupsCreated++;
      
      this.emit('groupCreated', createdGroup);
      return createdGroup;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to create group: ${error.message}`);
    }
  }

  /**
   * Create meta-relationships (relationships between relationships)
   */
  async createMetaRelationship(metaData) {
    const metaId = metaData.id || uuidv4();
    const timestamp = Date.now();

    // Create a special node to represent the meta-relationship
    const metaNode = await this.createNode({
      id: metaId,
      type: 'MetaRelationship',
      data: {
        sourceEdgeId: metaData.sourceEdge,
        targetEdgeId: metaData.targetEdge,
        relationshipType: metaData.type || 'RELATES_TO',
        ...metaData.data
      },
      metadata: {
        created: timestamp,
        isMetaRelationship: true,
        ...metaData.metadata
      }
    });

    // Connect the meta-relationship to the source edges
    const connectQuery = `
      MATCH ()-[r1]->() WHERE r1.id = $sourceEdgeId
      MATCH ()-[r2]->() WHERE r2.id = $targetEdgeId
      MATCH (meta:MetaRelationship {id: $metaId})
      CREATE (meta)-[:DESCRIBES_EDGE {id: $metaId + '_describes_' + $sourceEdgeId}]->(r1)
      CREATE (meta)-[:DESCRIBES_EDGE {id: $metaId + '_describes_' + $targetEdgeId}]->(r2)
      RETURN meta
    `;

    await this.session.run(connectQuery, {
      sourceEdgeId: metaData.sourceEdge,
      targetEdgeId: metaData.targetEdge,
      metaId: metaId
    });

    return metaNode;
  }

  /**
   * Find patterns in the knowledge graph
   */
  async findPattern(pattern) {
    const cacheKey = JSON.stringify(pattern);
    if (this.queryCache.has(cacheKey)) {
      this.stats.cacheHits++;
      return this.queryCache.get(cacheKey);
    }

    let query = 'MATCH ';
    const parameters = {};
    const nodePairs = [];

    // Build node patterns
    Object.entries(pattern.nodes || {}).forEach(([alias, nodeSpec]) => {
      const labels = nodeSpec.labels ? nodeSpec.labels.map(l => `:${l}`).join('') : ':AtomicInfo';
      query += `(${alias}${labels}`;
      
      if (nodeSpec.properties) {
        const conditions = Object.entries(nodeSpec.properties).map(([key, value]) => {
          const paramName = `${alias}_${key}`;
          parameters[paramName] = value;
          return `${key}: $${paramName}`;
        }).join(', ');
        query += ` {${conditions}}`;
      }
      query += ')';
    });

    // Build edge patterns
    if (pattern.edges && pattern.edges.length > 0) {
      pattern.edges.forEach((edge, index) => {
        const edgeAlias = `e${index}`;
        const edgeType = edge.type ? `:${edge.type}` : '';
        
        if (index === 0) {
          query += `, (${edge.from})-[${edgeAlias}${edgeType}`;
        } else {
          query += `(${edge.from})-[${edgeAlias}${edgeType}`;
        }

        if (edge.properties) {
          const conditions = Object.entries(edge.properties).map(([key, value]) => {
            const paramName = `${edgeAlias}_${key}`;
            parameters[paramName] = value;
            return `${key}: $${paramName}`;
          }).join(', ');
          query += ` {${conditions}}`;
        }

        query += `]->(${edge.to})`;
        if (index < pattern.edges.length - 1) {
          query += ', ';
        }
      });
    }

    // Add WHERE clauses for constraints
    if (pattern.constraints) {
      const whereConditions = [];
      
      if (pattern.constraints.temporal) {
        if (pattern.constraints.temporal.after) {
          whereConditions.push('n.created > $afterTime');
          parameters.afterTime = new Date(pattern.constraints.temporal.after).getTime();
        }
        if (pattern.constraints.temporal.before) {
          whereConditions.push('n.created < $beforeTime');
          parameters.beforeTime = new Date(pattern.constraints.temporal.before).getTime();
        }
      }

      if (pattern.constraints.confidence) {
        whereConditions.push('n.metadata.confidence >= $minConfidence');
        parameters.minConfidence = pattern.constraints.confidence.min;
      }

      if (whereConditions.length > 0) {
        query += ' WHERE ' + whereConditions.join(' AND ');
      }
    }

    // Return clause
    const returnItems = Object.keys(pattern.nodes || {});
    if (pattern.edges) {
      returnItems.push(...pattern.edges.map((_, index) => `e${index}`));
    }
    query += ` RETURN ${returnItems.join(', ')}`;

    // Add limit if specified
    if (pattern.limit) {
      query += ` LIMIT ${pattern.limit}`;
    }

    try {
      const result = await this.session.run(query, parameters);
      const matches = result.records.map(record => {
        const match = {};
        returnItems.forEach(item => {
          const value = record.get(item);
          match[item] = value.properties || value;
        });
        return match;
      });

      // Cache the result
      this.queryCache.set(cacheKey, matches);
      this.stats.queriesExecuted++;
      this.stats.cacheMisses++;

      return matches;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Pattern matching failed: ${error.message}`);
    }
  }

  /**
   * Find paths between nodes
   */
  async findPaths(fromNodeId, toNodeId, options = {}) {
    const maxDepth = options.maxDepth || 5;
    const algorithm = options.algorithm || 'shortestPath';
    
    let query;
    
    switch (algorithm) {
      case 'shortestPath':
        query = `
          MATCH (start:AtomicInfo {id: $fromId})
          MATCH (end:AtomicInfo {id: $toId})
          MATCH path = shortestPath((start)-[*1..${maxDepth}]-(end))
          RETURN path, length(path) as pathLength
          ORDER BY pathLength ASC
          LIMIT ${options.limit || 10}
        `;
        break;
        
      case 'allShortestPaths':
        query = `
          MATCH (start:AtomicInfo {id: $fromId})
          MATCH (end:AtomicInfo {id: $toId})
          MATCH path = allShortestPaths((start)-[*1..${maxDepth}]-(end))
          RETURN path, length(path) as pathLength
          ORDER BY pathLength ASC
          LIMIT ${options.limit || 100}
        `;
        break;

      default:
        throw new Error(`Unknown path finding algorithm: ${algorithm}`);
    }

    try {
      const result = await this.session.run(query, {
        fromId: fromNodeId,
        toId: toNodeId
      });

      const paths = result.records.map(record => ({
        path: record.get('path'),
        length: record.get('pathLength').toNumber()
      }));

      this.stats.queriesExecuted++;
      return paths;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Path finding failed: ${error.message}`);
    }
  }

  /**
   * Get node with relationships
   */
  async getNode(nodeId, includeRelationships = true) {
    // Check cache first
    if (this.nodeCache.has(nodeId)) {
      this.stats.cacheHits++;
      const cached = this.nodeCache.get(nodeId);
      if (!includeRelationships) {
        return cached;
      }
    }

    let query = `MATCH (n:AtomicInfo {id: $nodeId})`;
    
    if (includeRelationships) {
      query += `
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN n, collect({
          relationship: r,
          connectedNode: connected
        }) as relationships
      `;
    } else {
      query += ` RETURN n`;
    }

    try {
      const result = await this.session.run(query, { nodeId });
      
      if (result.records.length === 0) {
        return null;
      }

      const node = result.records[0].get('n').properties;
      
      if (includeRelationships) {
        node.relationships = result.records[0].get('relationships').map(rel => ({
          relationship: rel.relationship?.properties,
          connectedNode: rel.connectedNode?.properties
        })).filter(rel => rel.relationship);
      }

      // Update cache
      this.nodeCache.set(nodeId, node);
      this.stats.queriesExecuted++;
      
      return node;
    } catch (error) {
      this.emit('error', error);
      throw new Error(`Failed to get node: ${error.message}`);
    }
  }

  /**
   * Get statistics about the knowledge base
   */
  async getStatistics() {
    const query = `
      CALL apoc.meta.stats() YIELD labels, relTypesCount, nodeCount, relCount
      RETURN labels, relTypesCount, nodeCount, relCount
    `;

    try {
      const result = await this.session.run(query);
      const dbStats = result.records[0];
      
      return {
        database: {
          nodeCount: dbStats.get('nodeCount').toNumber(),
          relationshipCount: dbStats.get('relCount').toNumber(),
          labels: dbStats.get('labels'),
          relationshipTypes: dbStats.get('relTypesCount')
        },
        runtime: this.stats,
        cache: {
          nodeCache: this.nodeCache.size,
          edgeCache: this.edgeCache.size,
          groupCache: this.groupCache.size,
          queryCache: this.queryCache.size
        }
      };
    } catch (error) {
      // Fallback if APOC is not available
      const basicQuery = `
        MATCH (n) RETURN count(n) as nodeCount
        UNION ALL
        MATCH ()-[r]->() RETURN count(r) as relCount
      `;
      
      const result = await this.session.run(basicQuery);
      return {
        database: {
          nodeCount: result.records[0].get('nodeCount').toNumber(),
          relationshipCount: result.records[1].get('relCount').toNumber()
        },
        runtime: this.stats,
        cache: {
          nodeCache: this.nodeCache.size,
          edgeCache: this.edgeCache.size,
          groupCache: this.groupCache.size,
          queryCache: this.queryCache.size
        }
      };
    }
  }

  /**
   * Close connection
   */
  async close() {
    if (this.session) {
      await this.session.close();
    }
    if (this.driver) {
      await this.driver.close();
    }
    this.isConnected = false;
    this.emit('disconnected');
  }

  /**
   * Clear all caches
   */
  clearCache() {
    this.nodeCache.clear();
    this.edgeCache.clear();
    this.groupCache.clear();
    this.queryCache.clear();
  }
}

module.exports = Neo4jKnowledgeBase;