/**
 * GraphQL Schema for Advanced Knowledge Base
 * 
 * Comprehensive GraphQL API supporting:
 * - Node and edge operations
 * - Complex pattern matching
 * - Hypergraph operations
 * - Real-time subscriptions
 * - Performance monitoring
 */

const { gql } = require('apollo-server-express');
const { PubSub } = require('graphql-subscriptions');

const pubsub = new PubSub();

class GraphQLSchema {
  constructor(apiServer) {
    this.apiServer = apiServer;
    
    this.typeDefs = gql`
      scalar JSON
      scalar Date
      scalar BigInt

      # Core Types
      type Node {
        id: ID!
        type: String!
        data: JSON
        properties: JSON
        metadata: NodeMetadata!
        relationships: [Edge!]
        groups: [Group!]
        createdAt: Date!
        updatedAt: Date!
      }

      type Edge {
        id: ID!
        type: String!
        from: Node!
        to: Node!
        properties: JSON
        metadata: EdgeMetadata!
        weight: Float
        bidirectional: Boolean
        createdAt: Date!
        updatedAt: Date!
      }

      type Hyperedge {
        id: ID!
        type: String!
        nodes: [Node!]!
        properties: JSON
        metadata: HyperedgeMetadata!
        size: Int!
        weight: Float
        directed: Boolean
        createdAt: Date!
        updatedAt: Date!
      }

      type Group {
        id: ID!
        name: String!
        type: String!
        hierarchy: String
        members: [Node!]!
        properties: JSON
        metadata: GroupMetadata!
        memberCount: Int!
        createdAt: Date!
        updatedAt: Date!
      }

      # Metadata Types
      type NodeMetadata {
        created: BigInt!
        updated: BigInt!
        version: Int!
        confidence: Float!
        source: String
        tags: [String!]
      }

      type EdgeMetadata {
        created: BigInt!
        updated: BigInt!
        version: Int!
        strength: Float!
        confidence: Float!
        source: String
        temporal: TemporalInfo
        context: String
      }

      type HyperedgeMetadata {
        created: BigInt!
        updated: BigInt!
        version: Int!
        confidence: Float!
        source: String
        temporal: TemporalInfo
      }

      type GroupMetadata {
        created: BigInt!
        updated: BigInt!
        version: Int!
        memberCount: Int!
        description: String
      }

      type TemporalInfo {
        start: Date
        end: Date
        duration: Int
        recurring: Boolean
      }

      # Query Types
      type Query {
        # Node queries
        node(id: ID!): Node
        nodes(
          filter: NodeFilter
          sort: SortInput
          pagination: PaginationInput
        ): NodeConnection!
        
        # Edge queries
        edge(id: ID!): Edge
        edges(
          filter: EdgeFilter
          sort: SortInput
          pagination: PaginationInput
        ): EdgeConnection!
        
        # Hypergraph queries
        hyperedge(id: ID!): Hyperedge
        hyperedges(
          filter: HyperedgeFilter
          sort: SortInput
          pagination: PaginationInput
        ): HyperedgeConnection!
        
        # Pattern matching
        findPattern(pattern: PatternInput!): [PatternMatch!]!
        findHyperpattern(pattern: HyperpatternInput!): [HyperpatternMatch!]!
        
        # Graph traversal
        traverse(
          startNodeId: ID!
          options: TraversalOptions
        ): TraversalResult!
        
        # Path finding
        findPaths(
          fromNodeId: ID!
          toNodeId: ID!
          options: PathOptions
        ): [Path!]!
        
        # Analytics
        analytics: AnalyticsResult!
        nodeStatistics(nodeId: ID!): NodeStatistics!
        graphStatistics: GraphStatistics!
        
        # Search
        search(
          query: String!
          options: SearchOptions
        ): SearchResult!
        
        # Implementation info
        implementations: [ImplementationInfo!]!
        currentImplementation: ImplementationInfo!
      }

      # Mutation Types
      type Mutation {
        # Node operations
        createNode(input: CreateNodeInput!): Node!
        updateNode(id: ID!, input: UpdateNodeInput!): Node!
        deleteNode(id: ID!): Boolean!
        
        # Edge operations
        createEdge(input: CreateEdgeInput!): Edge!
        updateEdge(id: ID!, input: UpdateEdgeInput!): Edge!
        deleteEdge(id: ID!): Boolean!
        
        # Hyperedge operations
        createHyperedge(input: CreateHyperedgeInput!): Hyperedge!
        updateHyperedge(id: ID!, input: UpdateHyperedgeInput!): Hyperedge!
        deleteHyperedge(id: ID!): Boolean!
        
        # Group operations
        createGroup(input: CreateGroupInput!): Group!
        updateGroup(id: ID!, input: UpdateGroupInput!): Group!
        deleteGroup(id: ID!): Boolean!
        addToGroup(groupId: ID!, nodeIds: [ID!]!): Group!
        removeFromGroup(groupId: ID!, nodeIds: [ID!]!): Group!
        
        # Batch operations
        batchCreateNodes(inputs: [CreateNodeInput!]!): [Node!]!
        batchCreateEdges(inputs: [CreateEdgeInput!]!): [Edge!]!
        
        # Meta operations
        createMetaRelationship(input: MetaRelationshipInput!): Edge!
        createMetaHyperedge(input: MetaHyperedgeInput!): Hyperedge!
        
        # Implementation switching
        switchImplementation(implementation: String!): Boolean!
        
        # Cache management
        clearCache: Boolean!
        
        # Data management
        exportData(format: ExportFormat!): ExportResult!
        importData(input: ImportInput!): ImportResult!
      }

      # Subscription Types
      type Subscription {
        # Real-time updates
        nodeCreated: Node!
        nodeUpdated: Node!
        nodeDeleted: ID!
        
        edgeCreated: Edge!
        edgeUpdated: Edge!
        edgeDeleted: ID!
        
        hyperedgeCreated: Hyperedge!
        hyperedgeUpdated: Hyperedge!
        hyperedgeDeleted: ID!
        
        # Pattern monitoring
        patternMatched(pattern: PatternInput!): PatternMatch!
        
        # Analytics updates
        metricsUpdated: AnalyticsResult!
        
        # System events
        systemEvent: SystemEvent!
      }

      # Input Types
      input CreateNodeInput {
        id: ID
        type: String!
        data: JSON
        properties: JSON
        metadata: NodeMetadataInput
        labels: [String!]
        position: PositionInput
      }

      input UpdateNodeInput {
        type: String
        data: JSON
        properties: JSON
        metadata: NodeMetadataInput
        labels: [String!]
        position: PositionInput
      }

      input CreateEdgeInput {
        id: ID
        type: String!
        from: ID!
        to: ID!
        properties: JSON
        metadata: EdgeMetadataInput
        weight: Float
        bidirectional: Boolean
      }

      input UpdateEdgeInput {
        type: String
        properties: JSON
        metadata: EdgeMetadataInput
        weight: Float
        bidirectional: Boolean
      }

      input CreateHyperedgeInput {
        id: ID
        type: String!
        nodes: [ID!]!
        properties: JSON
        metadata: HyperedgeMetadataInput
        weight: Float
        directed: Boolean
        temporal: TemporalInfoInput
        context: String
      }

      input UpdateHyperedgeInput {
        type: String
        nodes: [ID!]
        properties: JSON
        metadata: HyperedgeMetadataInput
        weight: Float
        directed: Boolean
        temporal: TemporalInfoInput
        context: String
      }

      input CreateGroupInput {
        id: ID
        name: String!
        type: String!
        hierarchy: String
        members: [ID!]
        properties: JSON
        metadata: GroupMetadataInput
      }

      input UpdateGroupInput {
        name: String
        type: String
        hierarchy: String
        properties: JSON
        metadata: GroupMetadataInput
      }

      input NodeMetadataInput {
        confidence: Float
        source: String
        tags: [String!]
      }

      input EdgeMetadataInput {
        strength: Float
        confidence: Float
        source: String
        temporal: TemporalInfoInput
        context: String
      }

      input HyperedgeMetadataInput {
        confidence: Float
        source: String
        temporal: TemporalInfoInput
      }

      input GroupMetadataInput {
        description: String
      }

      input TemporalInfoInput {
        start: Date
        end: Date
        duration: Int
        recurring: Boolean
      }

      input PositionInput {
        x: Float!
        y: Float!
        z: Float
      }

      # Filter Types
      input NodeFilter {
        type: String
        types: [String!]
        labels: [String!]
        properties: JSON
        metadata: NodeMetadataFilter
        createdAfter: Date
        createdBefore: Date
        updatedAfter: Date
        updatedBefore: Date
      }

      input EdgeFilter {
        type: String
        types: [String!]
        from: ID
        to: ID
        properties: JSON
        metadata: EdgeMetadataFilter
        weightMin: Float
        weightMax: Float
        bidirectional: Boolean
        createdAfter: Date
        createdBefore: Date
      }

      input HyperedgeFilter {
        type: String
        types: [String!]
        minSize: Int
        maxSize: Int
        properties: JSON
        directed: Boolean
        createdAfter: Date
        createdBefore: Date
      }

      input NodeMetadataFilter {
        confidenceMin: Float
        confidenceMax: Float
        source: String
        sources: [String!]
        tags: [String!]
      }

      input EdgeMetadataFilter {
        strengthMin: Float
        strengthMax: Float
        confidenceMin: Float
        confidenceMax: Float
        source: String
        sources: [String!]
        context: String
      }

      # Pattern Types
      input PatternInput {
        nodes: JSON
        edges: [EdgePatternInput!]
        constraints: PatternConstraints
        limit: Int
      }

      input EdgePatternInput {
        from: String!
        to: String!
        type: String
        properties: JSON
      }

      input PatternConstraints {
        temporal: TemporalConstraints
        confidence: ConfidenceConstraints
        size: SizeConstraints
      }

      input TemporalConstraints {
        after: Date
        before: Date
      }

      input ConfidenceConstraints {
        min: Float
        max: Float
      }

      input SizeConstraints {
        min: Int
        max: Int
      }

      input HyperpatternInput {
        type: String
        nodes: [ID!]
        hyperedgeType: String
        minSize: Int
        maxSize: Int
        constraints: PatternConstraints
      }

      # Traversal Types
      input TraversalOptions {
        algorithm: TraversalAlgorithm
        maxDepth: Int
        direction: TraversalDirection
        relationshipTypes: [String!]
        weightProperty: String
        limit: Int
      }

      enum TraversalAlgorithm {
        BREADTH_FIRST
        DEPTH_FIRST
        DIJKSTRA
        A_STAR
        PARALLEL
      }

      enum TraversalDirection {
        OUTGOING
        INCOMING
        BOTH
      }

      input PathOptions {
        algorithm: PathAlgorithm
        maxDepth: Int
        weightProperty: String
        limit: Int
      }

      enum PathAlgorithm {
        SHORTEST_PATH
        ALL_SHORTEST_PATHS
        K_SHORTEST_PATHS
        DIJKSTRA
      }

      input SearchOptions {
        type: SearchType
        limit: Int
        offset: Int
        fuzzy: Boolean
        exactMatch: Boolean
      }

      enum SearchType {
        FULL_TEXT
        SEMANTIC
        STRUCTURAL
        HYBRID
      }

      # Pagination Types
      input PaginationInput {
        limit: Int
        offset: Int
        cursor: String
      }

      input SortInput {
        field: String!
        direction: SortDirection
      }

      enum SortDirection {
        ASC
        DESC
      }

      # Connection Types
      type NodeConnection {
        nodes: [Node!]!
        totalCount: Int!
        pageInfo: PageInfo!
      }

      type EdgeConnection {
        edges: [Edge!]!
        totalCount: Int!
        pageInfo: PageInfo!
      }

      type HyperedgeConnection {
        hyperedges: [Hyperedge!]!
        totalCount: Int!
        pageInfo: PageInfo!
      }

      type PageInfo {
        hasNextPage: Boolean!
        hasPreviousPage: Boolean!
        startCursor: String
        endCursor: String
      }

      # Result Types
      type PatternMatch {
        nodes: [Node!]!
        edges: [Edge!]!
        score: Float!
        confidence: Float!
      }

      type HyperpatternMatch {
        hyperedge: Hyperedge!
        nodes: [Node!]!
        score: Float!
        confidence: Float!
      }

      type TraversalResult {
        nodes: [Node!]!
        edges: [Edge!]!
        paths: [Path!]!
        depth: Int!
        algorithm: TraversalAlgorithm!
      }

      type Path {
        nodes: [Node!]!
        edges: [Edge!]!
        length: Int!
        weight: Float
        cost: Float
      }

      type SearchResult {
        nodes: [Node!]!
        edges: [Edge!]!
        hyperedges: [Hyperedge!]!
        totalCount: Int!
        searchTime: Float!
        suggestions: [String!]!
      }

      # Analytics Types
      type AnalyticsResult {
        nodeCount: Int!
        edgeCount: Int!
        hyperedgeCount: Int!
        groupCount: Int!
        density: Float!
        clustering: Float!
        diameter: Int
        averagePathLength: Float
        centralityMetrics: CentralityMetrics!
        performanceMetrics: PerformanceMetrics!
      }

      type NodeStatistics {
        degree: Int!
        inDegree: Int!
        outDegree: Int!
        clustering: Float!
        centrality: NodeCentrality!
        neighborhoods: [Node!]!
      }

      type GraphStatistics {
        basic: BasicStatistics!
        structural: StructuralStatistics!
        temporal: TemporalStatistics!
        performance: PerformanceStatistics!
      }

      type BasicStatistics {
        nodeCount: Int!
        edgeCount: Int!
        hyperedgeCount: Int!
        averageDegree: Float!
        density: Float!
      }

      type StructuralStatistics {
        diameter: Int
        radius: Int
        clustering: Float!
        transitivity: Float!
        assortativity: Float
      }

      type TemporalStatistics {
        timespan: TimeSpan!
        activityPeaks: [ActivityPeak!]!
        growthRate: Float!
      }

      type PerformanceStatistics {
        queryTime: PerformanceMetric!
        throughput: PerformanceMetric!
        memoryUsage: MemoryUsage!
        cacheHitRate: Float!
      }

      type CentralityMetrics {
        degree: [NodeCentrality!]!
        betweenness: [NodeCentrality!]!
        closeness: [NodeCentrality!]!
        eigenvector: [NodeCentrality!]!
        pagerank: [NodeCentrality!]!
      }

      type NodeCentrality {
        nodeId: ID!
        value: Float!
        rank: Int!
      }

      type PerformanceMetrics {
        queryTime: Float!
        throughput: Float!
        memoryUsage: Float!
        cacheHitRate: Float!
        errorRate: Float!
      }

      type PerformanceMetric {
        average: Float!
        min: Float!
        max: Float!
        p95: Float!
        p99: Float!
      }

      type MemoryUsage {
        heap: Float!
        external: Float!
        buffers: Float!
        total: Float!
      }

      type TimeSpan {
        start: Date!
        end: Date!
        duration: Int!
      }

      type ActivityPeak {
        timestamp: Date!
        value: Float!
        type: String!
      }

      # System Types
      type ImplementationInfo {
        name: String!
        version: String!
        status: ImplementationStatus!
        capabilities: [String!]!
        performance: PerformanceMetrics!
        configuration: JSON
      }

      enum ImplementationStatus {
        ACTIVE
        INACTIVE
        ERROR
        INITIALIZING
      }

      type SystemEvent {
        type: SystemEventType!
        timestamp: Date!
        data: JSON
        severity: EventSeverity!
      }

      enum SystemEventType {
        NODE_CREATED
        NODE_UPDATED
        NODE_DELETED
        EDGE_CREATED
        EDGE_UPDATED
        EDGE_DELETED
        HYPEREDGE_CREATED
        HYPEREDGE_UPDATED
        HYPEREDGE_DELETED
        PATTERN_MATCHED
        SYSTEM_ERROR
        PERFORMANCE_ALERT
      }

      enum EventSeverity {
        INFO
        WARNING
        ERROR
        CRITICAL
      }

      # Data Management Types
      input MetaRelationshipInput {
        sourceEdge: ID!
        targetEdge: ID!
        type: String!
        data: JSON
        metadata: JSON
      }

      input MetaHyperedgeInput {
        hyperedges: [ID!]!
        type: String!
        level: Int
        data: JSON
        metadata: JSON
      }

      enum ExportFormat {
        JSON
        CSV
        GRAPHML
        GEXF
        RDF
        CYPHER
      }

      input ImportInput {
        format: ExportFormat!
        data: String!
        options: JSON
      }

      type ExportResult {
        format: ExportFormat!
        data: String!
        size: Int!
        timestamp: Date!
      }

      type ImportResult {
        success: Boolean!
        nodesImported: Int!
        edgesImported: Int!
        hyperedgesImported: Int!
        errors: [String!]!
        timestamp: Date!
      }
    `;

    this.resolvers = {
      Query: {
        // Node queries
        node: async (parent, { id }, { knowledgeBase }) => {
          return await knowledgeBase.getNode(id, { includeRelationships: true });
        },

        nodes: async (parent, { filter, sort, pagination }, { knowledgeBase }) => {
          const result = await knowledgeBase.findNodes(filter, { sort, pagination });
          return {
            nodes: result.nodes,
            totalCount: result.totalCount,
            pageInfo: result.pageInfo
          };
        },

        // Edge queries
        edge: async (parent, { id }, { knowledgeBase }) => {
          return await knowledgeBase.getEdge(id);
        },

        edges: async (parent, { filter, sort, pagination }, { knowledgeBase }) => {
          const result = await knowledgeBase.findEdges(filter, { sort, pagination });
          return {
            edges: result.edges,
            totalCount: result.totalCount,
            pageInfo: result.pageInfo
          };
        },

        // Pattern matching
        findPattern: async (parent, { pattern }, { knowledgeBase }) => {
          return await knowledgeBase.findPattern(pattern);
        },

        findHyperpattern: async (parent, { pattern }, { knowledgeBase }) => {
          if (knowledgeBase.findHyperpattern) {
            return await knowledgeBase.findHyperpattern(pattern);
          }
          throw new Error('Hyperpattern search not supported by current implementation');
        },

        // Graph traversal
        traverse: async (parent, { startNodeId, options }, { knowledgeBase }) => {
          return await knowledgeBase.traverse(startNodeId, options);
        },

        // Path finding
        findPaths: async (parent, { fromNodeId, toNodeId, options }, { knowledgeBase }) => {
          return await knowledgeBase.findPaths(fromNodeId, toNodeId, options);
        },

        // Analytics
        analytics: async (parent, args, { knowledgeBase }) => {
          return await knowledgeBase.getAnalytics();
        },

        nodeStatistics: async (parent, { nodeId }, { knowledgeBase }) => {
          return await knowledgeBase.getNodeStatistics(nodeId);
        },

        graphStatistics: async (parent, args, { knowledgeBase }) => {
          return await knowledgeBase.getGraphStatistics();
        },

        // Search
        search: async (parent, { query, options }, { knowledgeBase }) => {
          return await knowledgeBase.search(query, options);
        },

        // Implementation info
        implementations: async (parent, args, { knowledgeBases }) => {
          return Array.from(knowledgeBases.entries()).map(([name, kb]) => ({
            name,
            version: '1.0.0',
            status: 'ACTIVE',
            capabilities: kb.getCapabilities ? kb.getCapabilities() : [],
            performance: kb.getPerformanceMetrics ? kb.getPerformanceMetrics() : {},
            configuration: kb.getConfiguration ? kb.getConfiguration() : {}
          }));
        },

        currentImplementation: async (parent, args, { implementation, knowledgeBase }) => {
          return {
            name: implementation,
            version: '1.0.0',
            status: 'ACTIVE',
            capabilities: knowledgeBase.getCapabilities ? knowledgeBase.getCapabilities() : [],
            performance: knowledgeBase.getPerformanceMetrics ? knowledgeBase.getPerformanceMetrics() : {},
            configuration: knowledgeBase.getConfiguration ? knowledgeBase.getConfiguration() : {}
          };
        }
      },

      Mutation: {
        // Node operations
        createNode: async (parent, { input }, { knowledgeBase }) => {
          const node = await knowledgeBase.createNode(input);
          
          // Publish subscription
          pubsub.publish('NODE_CREATED', { nodeCreated: node });
          
          return node;
        },

        updateNode: async (parent, { id, input }, { knowledgeBase }) => {
          const node = await knowledgeBase.updateNode(id, input);
          
          // Publish subscription
          pubsub.publish('NODE_UPDATED', { nodeUpdated: node });
          
          return node;
        },

        deleteNode: async (parent, { id }, { knowledgeBase }) => {
          const success = await knowledgeBase.deleteNode(id);
          
          if (success) {
            // Publish subscription
            pubsub.publish('NODE_DELETED', { nodeDeleted: id });
          }
          
          return success;
        },

        // Edge operations
        createEdge: async (parent, { input }, { knowledgeBase }) => {
          const edge = await knowledgeBase.createEdge(input);
          
          // Publish subscription
          pubsub.publish('EDGE_CREATED', { edgeCreated: edge });
          
          return edge;
        },

        updateEdge: async (parent, { id, input }, { knowledgeBase }) => {
          const edge = await knowledgeBase.updateEdge(id, input);
          
          // Publish subscription
          pubsub.publish('EDGE_UPDATED', { edgeUpdated: edge });
          
          return edge;
        },

        deleteEdge: async (parent, { id }, { knowledgeBase }) => {
          const success = await knowledgeBase.deleteEdge(id);
          
          if (success) {
            // Publish subscription
            pubsub.publish('EDGE_DELETED', { edgeDeleted: id });
          }
          
          return success;
        },

        // Hyperedge operations
        createHyperedge: async (parent, { input }, { knowledgeBase }) => {
          if (!knowledgeBase.createHyperedge) {
            throw new Error('Hyperedge operations not supported by current implementation');
          }
          
          const hyperedge = await knowledgeBase.createHyperedge(input);
          
          // Publish subscription
          pubsub.publish('HYPEREDGE_CREATED', { hyperedgeCreated: hyperedge });
          
          return hyperedge;
        },

        // Batch operations
        batchCreateNodes: async (parent, { inputs }, { knowledgeBase }) => {
          const nodes = await knowledgeBase.batchCreateNodes(inputs);
          
          // Publish batch creation event
          pubsub.publish('SYSTEM_EVENT', {
            systemEvent: {
              type: 'BATCH_NODES_CREATED',
              timestamp: new Date(),
              data: { count: nodes.length },
              severity: 'INFO'
            }
          });
          
          return nodes;
        },

        batchCreateEdges: async (parent, { inputs }, { knowledgeBase }) => {
          const edges = await knowledgeBase.batchCreateEdges(inputs);
          
          // Publish batch creation event
          pubsub.publish('SYSTEM_EVENT', {
            systemEvent: {
              type: 'BATCH_EDGES_CREATED',
              timestamp: new Date(),
              data: { count: edges.length },
              severity: 'INFO'
            }
          });
          
          return edges;
        },

        // Meta operations
        createMetaRelationship: async (parent, { input }, { knowledgeBase }) => {
          if (!knowledgeBase.createMetaRelationship) {
            throw new Error('Meta relationships not supported by current implementation');
          }
          
          return await knowledgeBase.createMetaRelationship(input);
        },

        // Implementation switching
        switchImplementation: async (parent, { implementation }, context) => {
          if (!context.knowledgeBases.has(implementation)) {
            throw new Error(`Implementation '${implementation}' not available`);
          }
          
          // This would need to be handled at the server level
          return true;
        },

        // Cache management
        clearCache: async (parent, args, { knowledgeBase }) => {
          if (knowledgeBase.clearCache) {
            knowledgeBase.clearCache();
          }
          return true;
        },

        // Data export
        exportData: async (parent, { format }, { knowledgeBase }) => {
          if (!knowledgeBase.export) {
            throw new Error('Data export not supported by current implementation');
          }
          
          const data = await knowledgeBase.export(format);
          return {
            format,
            data: JSON.stringify(data),
            size: JSON.stringify(data).length,
            timestamp: new Date()
          };
        }
      },

      Subscription: {
        nodeCreated: {
          subscribe: () => pubsub.asyncIterator(['NODE_CREATED'])
        },
        
        nodeUpdated: {
          subscribe: () => pubsub.asyncIterator(['NODE_UPDATED'])
        },
        
        nodeDeleted: {
          subscribe: () => pubsub.asyncIterator(['NODE_DELETED'])
        },
        
        edgeCreated: {
          subscribe: () => pubsub.asyncIterator(['EDGE_CREATED'])
        },
        
        edgeUpdated: {
          subscribe: () => pubsub.asyncIterator(['EDGE_UPDATED'])
        },
        
        edgeDeleted: {
          subscribe: () => pubsub.asyncIterator(['EDGE_DELETED'])
        },
        
        hyperedgeCreated: {
          subscribe: () => pubsub.asyncIterator(['HYPEREDGE_CREATED'])
        },
        
        systemEvent: {
          subscribe: () => pubsub.asyncIterator(['SYSTEM_EVENT'])
        },
        
        metricsUpdated: {
          subscribe: () => pubsub.asyncIterator(['METRICS_UPDATED'])
        }
      },

      // Scalar resolvers
      JSON: require('graphql-type-json'),
      Date: {
        serialize: (date) => date.toISOString(),
        parseValue: (value) => new Date(value),
        parseLiteral: (ast) => new Date(ast.value)
      },
      BigInt: {
        serialize: (value) => value.toString(),
        parseValue: (value) => BigInt(value),
        parseLiteral: (ast) => BigInt(ast.value)
      }
    };
  }
}

module.exports = GraphQLSchema;