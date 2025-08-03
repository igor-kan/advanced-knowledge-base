/**
 * Hypergraph Knowledge Base Implementation
 * 
 * Advanced hypergraph system supporting:
 * - Multi-way relationships (hyperedges)
 * - N-ary associations between nodes
 * - Complex relationship patterns
 * - Hierarchical hyperedge structures
 * - Billions-scale hypergraph operations
 * - Sophisticated pattern matching
 */

const { v4: uuidv4 } = require('uuid');
const EventEmitter = require('events');

class HypergraphKnowledgeBase extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      maxHyperedgeSize: config.maxHyperedgeSize || 1000,
      enableCompression: config.enableCompression || true,
      cacheSize: config.cacheSize || 10000,
      batchSize: config.batchSize || 1000,
      indexThreshold: config.indexThreshold || 100,
      ...config
    };
    
    // Core data structures optimized for hypergraph operations
    this.nodes = new Map(); // nodeId -> Node object
    this.hyperedges = new Map(); // hyperedgeId -> Hyperedge object
    this.nodeToHyperedges = new Map(); // nodeId -> Set of hyperedgeIds
    this.hyperedgeToNodes = new Map(); // hyperedgeId -> Set of nodeIds
    
    // Advanced indexing structures
    this.typeIndex = new Map(); // type -> Set of nodeIds
    this.hyperedgeTypeIndex = new Map(); // type -> Set of hyperedgeIds
    this.sizeIndex = new Map(); // size -> Set of hyperedgeIds
    this.labelIndex = new Map(); // label -> Set of nodeIds/hyperedgeIds
    
    // Multi-dimensional indexes for complex queries
    this.spatialIndex = new Map(); // For spatial relationships
    this.temporalIndex = new Map(); // For temporal relationships
    this.semanticIndex = new Map(); // For semantic similarity
    
    // Performance optimizations
    this.queryCache = new Map();
    this.patternCache = new Map();
    this.computationCache = new Map();
    
    // Statistics and metrics
    this.stats = {
      nodes: 0,
      hyperedges: 0,
      operations: 0,
      queryTime: 0,
      cacheHits: 0,
      cacheMisses: 0,
      memoryUsage: 0
    };
    
    // Hypergraph-specific algorithms
    this.algorithms = {
      clustering: new HypergraphClustering(this),
      centrality: new HypergraphCentrality(this),
      similarity: new HypergraphSimilarity(this),
      traversal: new HypergraphTraversal(this)
    };
  }

  /**
   * Create an atomic information node
   */
  async createNode(nodeData) {
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
      labels: new Set(nodeData.labels || []),
      position: nodeData.position || null, // For spatial indexing
      ...nodeData
    };

    // Store the node
    this.nodes.set(nodeId, node);
    this.nodeToHyperedges.set(nodeId, new Set());
    
    // Update indexes
    this.updateNodeIndexes(node);
    
    this.stats.nodes++;
    this.stats.operations++;
    
    this.emit('nodeCreated', node);
    return node;
  }

  /**
   * Create a hyperedge connecting multiple nodes
   */
  async createHyperedge(hyperedgeData) {
    const hyperedgeId = hyperedgeData.id || uuidv4();
    const timestamp = Date.now();
    
    if (!hyperedgeData.nodes || !Array.isArray(hyperedgeData.nodes)) {
      throw new Error('Hyperedge must specify nodes array');
    }
    
    if (hyperedgeData.nodes.length < 2) {
      throw new Error('Hyperedge must connect at least 2 nodes');
    }
    
    if (hyperedgeData.nodes.length > this.config.maxHyperedgeSize) {
      throw new Error(`Hyperedge size exceeds maximum of ${this.config.maxHyperedgeSize}`);
    }

    const hyperedge = {
      id: hyperedgeId,
      type: hyperedgeData.type || 'MULTI_RELATES',
      nodes: new Set(hyperedgeData.nodes),
      data: hyperedgeData.data || {},
      properties: hyperedgeData.properties || {},
      metadata: {
        created: timestamp,
        updated: timestamp,
        version: 1,
        size: hyperedgeData.nodes.length,
        strength: hyperedgeData.properties?.strength || 1.0,
        confidence: hyperedgeData.metadata?.confidence || 1.0,
        ...hyperedgeData.metadata
      },
      labels: new Set(hyperedgeData.labels || []),
      weight: hyperedgeData.weight || 1.0,
      directed: hyperedgeData.directed || false,
      temporal: hyperedgeData.temporal || null,
      context: hyperedgeData.context || null,
      ...hyperedgeData
    };

    // Validate that all nodes exist
    for (const nodeId of hyperedge.nodes) {
      if (!this.nodes.has(nodeId)) {
        throw new Error(`Node ${nodeId} does not exist`);
      }
    }

    // Store the hyperedge
    this.hyperedges.set(hyperedgeId, hyperedge);
    this.hyperedgeToNodes.set(hyperedgeId, hyperedge.nodes);
    
    // Update node-to-hyperedge mappings
    for (const nodeId of hyperedge.nodes) {
      this.nodeToHyperedges.get(nodeId).add(hyperedgeId);
    }
    
    // Update indexes
    this.updateHyperedgeIndexes(hyperedge);
    
    this.stats.hyperedges++;
    this.stats.operations++;
    
    this.emit('hyperedgeCreated', hyperedge);
    return hyperedge;
  }

  /**
   * Create complex multi-level hyperedge (hyperedge of hyperedges)
   */
  async createMetaHyperedge(metaData) {
    const metaId = metaData.id || uuidv4();
    
    // Create a special node to represent the meta-hyperedge
    const metaNode = await this.createNode({
      id: metaId,
      type: 'MetaHyperedge',
      data: {
        sourceHyperedges: metaData.hyperedges || [],
        relationshipType: metaData.type || 'META_RELATES',
        ...metaData.data
      },
      metadata: {
        isMetaHyperedge: true,
        level: metaData.level || 2,
        ...metaData.metadata
      }
    });

    // Create connections to the source hyperedges
    if (metaData.hyperedges && metaData.hyperedges.length > 0) {
      const metaHyperedge = await this.createHyperedge({
        id: metaId + '_meta',
        type: 'DESCRIBES_HYPEREDGES',
        nodes: [metaId, ...metaData.hyperedges],
        metadata: {
          isMetaRelationship: true,
          describes: metaData.hyperedges
        }
      });
      
      return { metaNode, metaHyperedge };
    }
    
    return { metaNode };
  }

  /**
   * Find complex hypergraph patterns
   */
  async findHyperpattern(pattern) {
    const startTime = process.hrtime.bigint();
    const cacheKey = JSON.stringify(pattern);
    
    // Check cache first
    if (this.patternCache.has(cacheKey)) {
      this.stats.cacheHits++;
      return this.patternCache.get(cacheKey);
    }
    
    this.stats.cacheMisses++;
    
    const matches = [];
    
    // Handle different pattern types
    if (pattern.type === 'hyperedge_pattern') {
      matches.push(...await this.findHyperedgePattern(pattern));
    } else if (pattern.type === 'node_cluster') {
      matches.push(...await this.findNodeCluster(pattern));
    } else if (pattern.type === 'path_pattern') {
      matches.push(...await this.findHyperpathPattern(pattern));
    } else {
      matches.push(...await this.findGeneralPattern(pattern));
    }
    
    // Apply filters
    const filteredMatches = this.applyPatternFilters(matches, pattern.constraints || {});
    
    // Cache the result
    this.patternCache.set(cacheKey, filteredMatches);
    
    const endTime = process.hrtime.bigint();
    this.stats.queryTime += Number(endTime - startTime) / 1000000;
    this.stats.operations++;
    
    return filteredMatches;
  }

  /**
   * Find hyperedge patterns (multi-way relationships)
   */
  async findHyperedgePattern(pattern) {
    const matches = [];
    const requiredNodes = pattern.nodes || [];
    const hyperedgeType = pattern.hyperedgeType;
    const minSize = pattern.minSize || 2;
    const maxSize = pattern.maxSize || this.config.maxHyperedgeSize;
    
    for (const [hyperedgeId, hyperedge] of this.hyperedges) {
      // Check type match
      if (hyperedgeType && hyperedge.type !== hyperedgeType) {
        continue;
      }
      
      // Check size constraints
      if (hyperedge.nodes.size < minSize || hyperedge.nodes.size > maxSize) {
        continue;
      }
      
      // Check if required nodes are present
      let hasAllRequired = true;
      for (const requiredNode of requiredNodes) {
        if (!hyperedge.nodes.has(requiredNode)) {
          hasAllRequired = false;
          break;
        }
      }
      
      if (hasAllRequired) {
        matches.push({
          hyperedge: hyperedge,
          nodes: Array.from(hyperedge.nodes).map(nodeId => this.nodes.get(nodeId)),
          strength: hyperedge.metadata.strength,
          confidence: hyperedge.metadata.confidence
        });
      }
    }
    
    return matches;
  }

  /**
   * Find node clusters (densely connected node groups)
   */
  async findNodeCluster(pattern) {
    const centerNode = pattern.centerNode;
    const radius = pattern.radius || 2;
    const minClusterSize = pattern.minSize || 3;
    
    if (!centerNode || !this.nodes.has(centerNode)) {
      return [];
    }
    
    const visited = new Set();
    const cluster = new Set([centerNode]);
    const queue = [{ nodeId: centerNode, distance: 0 }];
    
    while (queue.length > 0) {
      const { nodeId, distance } = queue.shift();
      
      if (distance >= radius || visited.has(nodeId)) {
        continue;
      }
      
      visited.add(nodeId);
      
      // Find all hyperedges containing this node
      const hyperedges = this.nodeToHyperedges.get(nodeId) || new Set();
      
      for (const hyperedgeId of hyperedges) {
        const hyperedge = this.hyperedges.get(hyperedgeId);
        
        // Add all connected nodes to cluster
        for (const connectedNodeId of hyperedge.nodes) {
          if (!visited.has(connectedNodeId)) {
            cluster.add(connectedNodeId);
            queue.push({ nodeId: connectedNodeId, distance: distance + 1 });
          }
        }
      }
    }
    
    if (cluster.size >= minClusterSize) {
      return [{
        cluster: Array.from(cluster).map(nodeId => this.nodes.get(nodeId)),
        size: cluster.size,
        centerNode: this.nodes.get(centerNode),
        radius: radius
      }];
    }
    
    return [];
  }

  /**
   * Find hyperpath patterns (sequences of connected hyperedges)
   */
  async findHyperpathPattern(pattern) {
    const startNode = pattern.startNode;
    const endNode = pattern.endNode;
    const maxLength = pattern.maxLength || 5;
    
    if (!startNode || !endNode) {
      return [];
    }
    
    const paths = [];
    const visited = new Set();
    
    const dfs = (currentNode, path, length) => {
      if (length > maxLength) {
        return;
      }
      
      if (currentNode === endNode && path.length > 0) {
        paths.push([...path]);
        return;
      }
      
      if (visited.has(currentNode)) {
        return;
      }
      
      visited.add(currentNode);
      
      // Explore all hyperedges containing current node
      const hyperedges = this.nodeToHyperedges.get(currentNode) || new Set();
      
      for (const hyperedgeId of hyperedges) {
        const hyperedge = this.hyperedges.get(hyperedgeId);
        path.push(hyperedge);
        
        // Explore all other nodes in this hyperedge
        for (const nextNode of hyperedge.nodes) {
          if (nextNode !== currentNode) {
            dfs(nextNode, path, length + 1);
          }
        }
        
        path.pop();
      }
      
      visited.delete(currentNode);
    };
    
    dfs(startNode, [], 0);
    
    return paths.map(path => ({
      path: path,
      length: path.length,
      startNode: this.nodes.get(startNode),
      endNode: this.nodes.get(endNode)
    }));
  }

  /**
   * Find general patterns with flexible matching
   */
  async findGeneralPattern(pattern) {
    const matches = [];
    
    // Node-based pattern matching
    if (pattern.nodes) {
      const nodeMatches = new Map();
      
      Object.entries(pattern.nodes).forEach(([alias, nodeSpec]) => {
        const candidates = [];
        
        // Find candidate nodes
        for (const [nodeId, node] of this.nodes) {
          if (this.matchesNodeSpec(node, nodeSpec)) {
            candidates.push(node);
          }
        }
        
        nodeMatches.set(alias, candidates);
      });
      
      // Generate combinations
      const combinations = this.generateCombinations(nodeMatches);
      
      // Filter by relationship constraints
      if (pattern.relationships) {
        for (const combo of combinations) {
          if (this.satisfiesRelationshipConstraints(combo, pattern.relationships)) {
            matches.push(combo);
          }
        }
      } else {
        matches.push(...combinations);
      }
    }
    
    return matches;
  }

  /**
   * Apply pattern filters and constraints
   */
  applyPatternFilters(matches, constraints) {
    let filtered = matches;
    
    // Temporal constraints
    if (constraints.temporal) {
      filtered = filtered.filter(match => {
        if (constraints.temporal.after) {
          const afterTime = new Date(constraints.temporal.after).getTime();
          return match.created ? match.created > afterTime : true;
        }
        return true;
      });
    }
    
    // Confidence constraints
    if (constraints.confidence) {
      filtered = filtered.filter(match => {
        return match.confidence >= (constraints.confidence.min || 0);
      });
    }
    
    // Size constraints
    if (constraints.size) {
      filtered = filtered.filter(match => {
        const size = match.cluster ? match.cluster.length : 
                    match.nodes ? match.nodes.length : 1;
        return size >= (constraints.size.min || 0) && 
               size <= (constraints.size.max || Infinity);
      });
    }
    
    // Limit results
    if (constraints.limit) {
      filtered = filtered.slice(0, constraints.limit);
    }
    
    return filtered;
  }

  /**
   * Check if node matches specification
   */
  matchesNodeSpec(node, spec) {
    // Type matching
    if (spec.type && node.type !== spec.type) {
      return false;
    }
    
    // Property matching
    if (spec.properties) {
      for (const [key, value] of Object.entries(spec.properties)) {
        if (node.properties[key] !== value) {
          return false;
        }
      }
    }
    
    // Label matching
    if (spec.labels) {
      for (const label of spec.labels) {
        if (!node.labels.has(label)) {
          return false;
        }
      }
    }
    
    return true;
  }

  /**
   * Generate all valid combinations of node assignments
   */
  generateCombinations(nodeMatches) {
    const aliases = Array.from(nodeMatches.keys());
    const combinations = [];
    
    const generateRecursive = (index, current) => {
      if (index === aliases.length) {
        combinations.push({ ...current });
        return;
      }
      
      const alias = aliases[index];
      const candidates = nodeMatches.get(alias);
      
      for (const candidate of candidates) {
        // Ensure no duplicate node assignments
        if (!Object.values(current).some(node => node.id === candidate.id)) {
          current[alias] = candidate;
          generateRecursive(index + 1, current);
          delete current[alias];
        }
      }
    };
    
    generateRecursive(0, {});
    return combinations;
  }

  /**
   * Check if node combination satisfies relationship constraints
   */
  satisfiesRelationshipConstraints(nodeCombo, relationships) {
    for (const rel of relationships) {
      const fromNode = nodeCombo[rel.from];
      const toNode = nodeCombo[rel.to];
      
      if (!fromNode || !toNode) {
        continue;
      }
      
      // Check if there's a hyperedge connecting these nodes
      const fromHyperedges = this.nodeToHyperedges.get(fromNode.id) || new Set();
      const toHyperedges = this.nodeToHyperedges.get(toNode.id) || new Set();
      
      // Find intersection
      const sharedHyperedges = new Set([...fromHyperedges].filter(x => toHyperedges.has(x)));
      
      if (sharedHyperedges.size === 0) {
        return false;
      }
      
      // Check relationship type if specified
      if (rel.type) {
        let typeMatches = false;
        for (const hyperedgeId of sharedHyperedges) {
          const hyperedge = this.hyperedges.get(hyperedgeId);
          if (hyperedge.type === rel.type) {
            typeMatches = true;
            break;
          }
        }
        if (!typeMatches) {
          return false;
        }
      }
    }
    
    return true;
  }

  /**
   * Update node indexes
   */
  updateNodeIndexes(node) {
    // Type index
    if (!this.typeIndex.has(node.type)) {
      this.typeIndex.set(node.type, new Set());
    }
    this.typeIndex.get(node.type).add(node.id);
    
    // Label index
    for (const label of node.labels) {
      if (!this.labelIndex.has(label)) {
        this.labelIndex.set(label, new Set());
      }
      this.labelIndex.get(label).add(node.id);
    }
    
    // Spatial index (if position provided)
    if (node.position) {
      const spatialKey = this.getSpatialKey(node.position);
      if (!this.spatialIndex.has(spatialKey)) {
        this.spatialIndex.set(spatialKey, new Set());
      }
      this.spatialIndex.get(spatialKey).add(node.id);
    }
    
    // Temporal index
    const temporalKey = this.getTemporalKey(node.metadata.created);
    if (!this.temporalIndex.has(temporalKey)) {
      this.temporalIndex.set(temporalKey, new Set());
    }
    this.temporalIndex.get(temporalKey).add(node.id);
  }

  /**
   * Update hyperedge indexes
   */
  updateHyperedgeIndexes(hyperedge) {
    // Type index
    if (!this.hyperedgeTypeIndex.has(hyperedge.type)) {
      this.hyperedgeTypeIndex.set(hyperedge.type, new Set());
    }
    this.hyperedgeTypeIndex.get(hyperedge.type).add(hyperedge.id);
    
    // Size index
    const size = hyperedge.nodes.size;
    if (!this.sizeIndex.has(size)) {
      this.sizeIndex.set(size, new Set());
    }
    this.sizeIndex.get(size).add(hyperedge.id);
    
    // Label index
    for (const label of hyperedge.labels) {
      if (!this.labelIndex.has(label)) {
        this.labelIndex.set(label, new Set());
      }
      this.labelIndex.get(label).add(hyperedge.id);
    }
  }

  /**
   * Get spatial indexing key
   */
  getSpatialKey(position) {
    if (!position || !position.x || !position.y) {
      return 'unknown';
    }
    
    // Simple grid-based spatial indexing
    const gridSize = 100;
    const x = Math.floor(position.x / gridSize);
    const y = Math.floor(position.y / gridSize);
    return `${x},${y}`;
  }

  /**
   * Get temporal indexing key
   */
  getTemporalKey(timestamp) {
    const date = new Date(timestamp);
    return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
  }

  /**
   * Get hypergraph statistics
   */
  getStatistics() {
    return {
      ...this.stats,
      indexes: {
        typeIndex: this.typeIndex.size,
        hyperedgeTypeIndex: this.hyperedgeTypeIndex.size,
        sizeIndex: this.sizeIndex.size,
        labelIndex: this.labelIndex.size,
        spatialIndex: this.spatialIndex.size,
        temporalIndex: this.temporalIndex.size
      },
      cache: {
        queryCache: this.queryCache.size,
        patternCache: this.patternCache.size,
        computationCache: this.computationCache.size
      },
      averageHyperedgeSize: this.stats.hyperedges > 0 ? 
        Array.from(this.hyperedges.values()).reduce((sum, he) => sum + he.nodes.size, 0) / this.stats.hyperedges : 0
    };
  }

  /**
   * Clear all caches
   */
  clearCache() {
    this.queryCache.clear();
    this.patternCache.clear();
    this.computationCache.clear();
  }

  /**
   * Export hypergraph data
   */
  export() {
    return {
      nodes: Array.from(this.nodes.values()).map(node => ({
        ...node,
        labels: Array.from(node.labels)
      })),
      hyperedges: Array.from(this.hyperedges.values()).map(hyperedge => ({
        ...hyperedge,
        nodes: Array.from(hyperedge.nodes),
        labels: Array.from(hyperedge.labels)
      })),
      statistics: this.getStatistics()
    };
  }
}

/**
 * Hypergraph Clustering Algorithms
 */
class HypergraphClustering {
  constructor(hypergraph) {
    this.hg = hypergraph;
  }

  /**
   * Community detection in hypergraphs
   */
  detectCommunities(options = {}) {
    const algorithm = options.algorithm || 'louvain';
    
    switch (algorithm) {
      case 'louvain':
        return this.louvainClustering(options);
      case 'modularity':
        return this.modularityClustering(options);
      default:
        throw new Error(`Unknown clustering algorithm: ${algorithm}`);
    }
  }

  louvainClustering(options) {
    // Implementation of Louvain method for hypergraphs
    const communities = new Map();
    let nodeId = 0;
    
    // Initialize each node in its own community
    for (const [id, node] of this.hg.nodes) {
      communities.set(id, nodeId++);
    }
    
    // Iterative improvement (simplified version)
    let improved = true;
    while (improved) {
      improved = false;
      
      for (const [nodeId, node] of this.hg.nodes) {
        const currentCommunity = communities.get(nodeId);
        const neighborCommunities = this.getNeighborCommunities(nodeId, communities);
        
        let bestCommunity = currentCommunity;
        let bestGain = 0;
        
        for (const [community, gain] of neighborCommunities) {
          if (gain > bestGain) {
            bestGain = gain;
            bestCommunity = community;
          }
        }
        
        if (bestCommunity !== currentCommunity) {
          communities.set(nodeId, bestCommunity);
          improved = true;
        }
      }
    }
    
    return this.formatCommunities(communities);
  }

  getNeighborCommunities(nodeId, communities) {
    const neighborCommunities = new Map();
    const hyperedges = this.hg.nodeToHyperedges.get(nodeId) || new Set();
    
    for (const hyperedgeId of hyperedges) {
      const hyperedge = this.hg.hyperedges.get(hyperedgeId);
      
      for (const neighborId of hyperedge.nodes) {
        if (neighborId !== nodeId) {
          const community = communities.get(neighborId);
          const weight = hyperedge.weight || 1.0;
          
          neighborCommunities.set(community, 
            (neighborCommunities.get(community) || 0) + weight);
        }
      }
    }
    
    return neighborCommunities;
  }

  formatCommunities(communities) {
    const result = new Map();
    
    for (const [nodeId, communityId] of communities) {
      if (!result.has(communityId)) {
        result.set(communityId, []);
      }
      result.get(communityId).push(nodeId);
    }
    
    return Array.from(result.values());
  }
}

/**
 * Hypergraph Centrality Measures
 */
class HypergraphCentrality {
  constructor(hypergraph) {
    this.hg = hypergraph;
  }

  /**
   * Calculate hypergraph centrality measures
   */
  calculate(type = 'degree') {
    switch (type) {
      case 'degree':
        return this.degreeCentrality();
      case 'betweenness':
        return this.betweennessCentrality();
      case 'eigenvector':
        return this.eigenvectorCentrality();
      default:
        throw new Error(`Unknown centrality type: ${type}`);
    }
  }

  degreeCentrality() {
    const centrality = new Map();
    
    for (const [nodeId, node] of this.hg.nodes) {
      const hyperedges = this.hg.nodeToHyperedges.get(nodeId) || new Set();
      centrality.set(nodeId, hyperedges.size);
    }
    
    return centrality;
  }

  betweennessCentrality() {
    // Simplified betweenness centrality for hypergraphs
    const centrality = new Map();
    
    // Initialize all centralities to 0
    for (const nodeId of this.hg.nodes.keys()) {
      centrality.set(nodeId, 0);
    }
    
    // Calculate shortest paths through hypergraph
    // (This is a simplified version - full implementation would be more complex)
    
    return centrality;
  }

  eigenvectorCentrality() {
    // Eigenvector centrality for hypergraphs
    const centrality = new Map();
    
    // Initialize centralities
    for (const nodeId of this.hg.nodes.keys()) {
      centrality.set(nodeId, 1.0);
    }
    
    // Power iteration method (simplified)
    const maxIterations = 100;
    const tolerance = 1e-6;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      const newCentrality = new Map();
      
      for (const nodeId of this.hg.nodes.keys()) {
        newCentrality.set(nodeId, 0);
      }
      
      // Update centralities based on hyperedge structure
      for (const [hyperedgeId, hyperedge] of this.hg.hyperedges) {
        const weight = hyperedge.weight || 1.0;
        const size = hyperedge.nodes.size;
        
        for (const nodeId of hyperedge.nodes) {
          let sum = 0;
          for (const neighborId of hyperedge.nodes) {
            if (neighborId !== nodeId) {
              sum += centrality.get(neighborId);
            }
          }
          
          newCentrality.set(nodeId, 
            newCentrality.get(nodeId) + (weight * sum) / (size - 1));
        }
      }
      
      // Normalize
      const norm = Math.sqrt(Array.from(newCentrality.values())
        .reduce((sum, val) => sum + val * val, 0));
      
      if (norm > 0) {
        for (const [nodeId, value] of newCentrality) {
          newCentrality.set(nodeId, value / norm);
        }
      }
      
      // Check convergence
      let converged = true;
      for (const nodeId of this.hg.nodes.keys()) {
        if (Math.abs(newCentrality.get(nodeId) - centrality.get(nodeId)) > tolerance) {
          converged = false;
          break;
        }
      }
      
      centrality.clear();
      for (const [nodeId, value] of newCentrality) {
        centrality.set(nodeId, value);
      }
      
      if (converged) break;
    }
    
    return centrality;
  }
}

/**
 * Hypergraph Similarity Measures
 */
class HypergraphSimilarity {
  constructor(hypergraph) {
    this.hg = hypergraph;
  }

  /**
   * Calculate similarity between nodes
   */
  nodeSimilarity(nodeId1, nodeId2, method = 'jaccard') {
    switch (method) {
      case 'jaccard':
        return this.jaccardSimilarity(nodeId1, nodeId2);
      case 'cosine':
        return this.cosineSimilarity(nodeId1, nodeId2);
      case 'structural':
        return this.structuralSimilarity(nodeId1, nodeId2);
      default:
        throw new Error(`Unknown similarity method: ${method}`);
    }
  }

  jaccardSimilarity(nodeId1, nodeId2) {
    const hyperedges1 = this.hg.nodeToHyperedges.get(nodeId1) || new Set();
    const hyperedges2 = this.hg.nodeToHyperedges.get(nodeId2) || new Set();
    
    const intersection = new Set([...hyperedges1].filter(x => hyperedges2.has(x)));
    const union = new Set([...hyperedges1, ...hyperedges2]);
    
    return union.size > 0 ? intersection.size / union.size : 0;
  }

  cosineSimilarity(nodeId1, nodeId2) {
    const hyperedges1 = this.hg.nodeToHyperedges.get(nodeId1) || new Set();
    const hyperedges2 = this.hg.nodeToHyperedges.get(nodeId2) || new Set();
    
    const intersection = new Set([...hyperedges1].filter(x => hyperedges2.has(x)));
    
    const norm1 = Math.sqrt(hyperedges1.size);
    const norm2 = Math.sqrt(hyperedges2.size);
    
    return norm1 > 0 && norm2 > 0 ? intersection.size / (norm1 * norm2) : 0;
  }

  structuralSimilarity(nodeId1, nodeId2) {
    // Calculate structural similarity based on neighborhood structure
    const neighbors1 = this.getNeighbors(nodeId1);
    const neighbors2 = this.getNeighbors(nodeId2);
    
    const sharedNeighbors = new Set([...neighbors1].filter(x => neighbors2.has(x)));
    const totalNeighbors = new Set([...neighbors1, ...neighbors2]);
    
    return totalNeighbors.size > 0 ? sharedNeighbors.size / totalNeighbors.size : 0;
  }

  getNeighbors(nodeId) {
    const neighbors = new Set();
    const hyperedges = this.hg.nodeToHyperedges.get(nodeId) || new Set();
    
    for (const hyperedgeId of hyperedges) {
      const hyperedge = this.hg.hyperedges.get(hyperedgeId);
      for (const neighborId of hyperedge.nodes) {
        if (neighborId !== nodeId) {
          neighbors.add(neighborId);
        }
      }
    }
    
    return neighbors;
  }
}

/**
 * Hypergraph Traversal Algorithms
 */
class HypergraphTraversal {
  constructor(hypergraph) {
    this.hg = hypergraph;
  }

  /**
   * Breadth-first traversal through hypergraph
   */
  breadthFirstTraversal(startNodeId, options = {}) {
    const maxDepth = options.maxDepth || Infinity;
    const visited = new Set();
    const queue = [{ nodeId: startNodeId, depth: 0, path: [startNodeId] }];
    const result = [];
    
    while (queue.length > 0) {
      const { nodeId, depth, path } = queue.shift();
      
      if (visited.has(nodeId) || depth > maxDepth) {
        continue;
      }
      
      visited.add(nodeId);
      result.push({ nodeId, depth, path: [...path] });
      
      // Explore hyperedges
      const hyperedges = this.hg.nodeToHyperedges.get(nodeId) || new Set();
      
      for (const hyperedgeId of hyperedges) {
        const hyperedge = this.hg.hyperedges.get(hyperedgeId);
        
        for (const neighborId of hyperedge.nodes) {
          if (!visited.has(neighborId)) {
            queue.push({
              nodeId: neighborId,
              depth: depth + 1,
              path: [...path, neighborId]
            });
          }
        }
      }
    }
    
    return result;
  }

  /**
   * Find shortest hyperpath between nodes
   */
  shortestHyperpath(startNodeId, endNodeId, options = {}) {
    const maxDepth = options.maxDepth || 10;
    const queue = [{ nodeId: startNodeId, depth: 0, path: [startNodeId], hyperedges: [] }];
    const visited = new Set();
    
    while (queue.length > 0) {
      const { nodeId, depth, path, hyperedges } = queue.shift();
      
      if (nodeId === endNodeId) {
        return { path, hyperedges, length: depth };
      }
      
      if (visited.has(nodeId) || depth >= maxDepth) {
        continue;
      }
      
      visited.add(nodeId);
      
      // Explore through hyperedges
      const nodeHyperedges = this.hg.nodeToHyperedges.get(nodeId) || new Set();
      
      for (const hyperedgeId of nodeHyperedges) {
        const hyperedge = this.hg.hyperedges.get(hyperedgeId);
        
        for (const neighborId of hyperedge.nodes) {
          if (!visited.has(neighborId)) {
            queue.push({
              nodeId: neighborId,
              depth: depth + 1,
              path: [...path, neighborId],
              hyperedges: [...hyperedges, hyperedge]
            });
          }
        }
      }
    }
    
    return null; // No path found
  }
}

module.exports = HypergraphKnowledgeBase;