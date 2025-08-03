//! Core types and traits for hybrid ultra-fast knowledge graph
//!
//! This module defines the fundamental types, traits, and constants used
//! throughout the hybrid implementation. It combines the best of Rust's
//! type safety with C++'s performance optimizations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

// Re-export commonly used types
pub use crate::error::{HybridError, HybridResult};

/// Unique identifier for graph nodes (64-bit for maximum range)
pub type NodeId = u64;

/// Unique identifier for graph edges (64-bit for maximum range)
pub type EdgeId = u64;

/// Edge weight type optimized for SIMD operations
pub type Weight = f32;

/// Hash type for fast lookups and comparisons
pub type Hash = u64;

/// Property value that can hold various data types efficiently
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Null/empty value
    Null,
    /// Boolean value
    Bool(bool),
    /// 32-bit signed integer
    Int32(i32),
    /// 64-bit signed integer  
    Int64(i64),
    /// 32-bit floating point (SIMD-friendly)
    Float32(f32),
    /// 64-bit floating point
    Float64(f64),
    /// UTF-8 string
    String(String),
    /// Binary data
    Bytes(Vec<u8>),
    /// Array of 32-bit integers (SIMD-optimized)
    Int32Array(SmallVec<[i32; 8]>),
    /// Array of 64-bit integers
    Int64Array(SmallVec<[i64; 4]>),
    /// Array of 32-bit floats (SIMD-optimized)
    Float32Array(SmallVec<[f32; 8]>),
    /// Array of 64-bit floats
    Float64Array(SmallVec<[f64; 4]>),
    /// Array of strings
    StringArray(SmallVec<[String; 4]>),
}

impl PropertyValue {
    /// Get the memory size of this property value
    pub fn memory_size(&self) -> usize {
        match self {
            PropertyValue::Null => 0,
            PropertyValue::Bool(_) => 1,
            PropertyValue::Int32(_) => 4,
            PropertyValue::Int64(_) => 8,
            PropertyValue::Float32(_) => 4,
            PropertyValue::Float64(_) => 8,
            PropertyValue::String(s) => s.len(),
            PropertyValue::Bytes(b) => b.len(),
            PropertyValue::Int32Array(arr) => arr.len() * 4,
            PropertyValue::Int64Array(arr) => arr.len() * 8,
            PropertyValue::Float32Array(arr) => arr.len() * 4,
            PropertyValue::Float64Array(arr) => arr.len() * 8,
            PropertyValue::StringArray(arr) => arr.iter().map(|s| s.len()).sum(),
        }
    }
    
    /// Check if this is a numeric type suitable for SIMD operations
    pub fn is_simd_compatible(&self) -> bool {
        matches!(
            self,
            PropertyValue::Float32(_) 
                | PropertyValue::Float32Array(_)
                | PropertyValue::Int32(_)
                | PropertyValue::Int32Array(_)
        )
    }
}

/// Property map for flexible node/edge attributes
pub type PropertyMap = HashMap<String, PropertyValue>;

/// Node data with properties and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    /// Human-readable label for the node
    pub label: String,
    /// Key-value properties
    pub properties: PropertyMap,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
}

impl NodeData {
    /// Create new node data with label and properties
    pub fn new(label: String, properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        Self {
            label,
            properties,
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Get the total memory size of this node
    pub fn memory_size(&self) -> usize {
        self.label.len() 
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 32 // Timestamps and overhead
    }
    
    /// Update properties and modification time
    pub fn update_properties(&mut self, new_properties: PropertyMap) {
        self.properties = new_properties;
        self.updated_at = SystemTime::now();
    }
}

/// Edge data with properties and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    /// Key-value properties
    pub properties: PropertyMap,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
}

impl EdgeData {
    /// Create new edge data with properties
    pub fn new(properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        Self {
            properties,
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Get the total memory size of this edge
    pub fn memory_size(&self) -> usize {
        self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 24 // Timestamps and overhead
    }
}

/// Hyperedge data for N-ary relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperedgeData {
    /// Nodes connected by this hyperedge
    pub nodes: SmallVec<[NodeId; 8]>,
    /// Key-value properties
    pub properties: PropertyMap,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
}

impl HyperedgeData {
    /// Create new hyperedge data
    pub fn new(nodes: SmallVec<[NodeId; 8]>, properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        Self {
            nodes,
            properties,
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Get the total memory size of this hyperedge
    pub fn memory_size(&self) -> usize {
        self.nodes.len() * 8 
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 32
    }
}

/// Direction for edge traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeDirection {
    /// Follow outgoing edges
    Outgoing,
    /// Follow incoming edges  
    Incoming,
    /// Follow edges in both directions
    Both,
}

/// Path representation for shortest path algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Sequence of nodes in the path
    pub nodes: Vec<NodeId>,
    /// Sequence of edges in the path
    pub edges: Vec<EdgeId>,
    /// Edge weights along the path
    pub weights: Vec<Weight>,
    /// Total path weight
    pub total_weight: Weight,
    /// Path length (number of edges)
    pub length: usize,
    /// Time taken to compute this path
    pub computation_time: Duration,
    /// Confidence score for approximate algorithms
    pub confidence: f32,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            weights: Vec::new(),
            total_weight: 0.0,
            length: 0,
            computation_time: Duration::ZERO,
            confidence: 1.0,
        }
    }
    
    /// Check if this path is valid (nodes and edges match)
    pub fn is_valid(&self) -> bool {
        self.nodes.len() > 0 
            && (self.nodes.len() == self.edges.len() + 1)
            && (self.weights.len() == self.edges.len())
            && (self.length == self.edges.len())
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a graph traversal operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    /// Nodes visited during traversal
    pub nodes: Vec<NodeId>,
    /// Edges traversed
    pub edges: Vec<EdgeId>,
    /// Depth of each visited node
    pub depths: Vec<u32>,
    /// Distances for weighted traversals
    pub distances: Vec<Weight>,
    /// Total number of nodes visited
    pub nodes_visited: usize,
    /// Total number of edges traversed
    pub edges_traversed: usize,
    /// Time taken for traversal
    pub duration: Duration,
    /// Memory used during traversal
    pub memory_used: usize,
    /// Confidence score
    pub confidence: f32,
    /// Performance metrics
    pub simd_operations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl TraversalResult {
    /// Create a new empty traversal result
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            depths: Vec::new(),
            distances: Vec::new(),
            nodes_visited: 0,
            edges_traversed: 0,
            duration: Duration::ZERO,
            memory_used: 0,
            confidence: 1.0,
            simd_operations: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl Default for TraversalResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern node for graph pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    /// Unique identifier within the pattern
    pub id: String,
    /// Optional type filter
    pub type_filter: Option<String>,
    /// Property constraints
    pub property_filters: PropertyMap,
    /// Optional degree constraints
    pub min_degree: Option<usize>,
    pub max_degree: Option<usize>,
}

impl PatternNode {
    /// Create a new pattern node with ID
    pub fn new(id: String) -> Self {
        Self {
            id,
            type_filter: None,
            property_filters: PropertyMap::new(),
            min_degree: None,
            max_degree: None,
        }
    }
    
    /// Add type filter
    pub fn with_type(mut self, type_name: String) -> Self {
        self.type_filter = Some(type_name);
        self
    }
    
    /// Add property filter
    pub fn with_property(mut self, key: String, value: PropertyValue) -> Self {
        self.property_filters.insert(key, value);
        self
    }
    
    /// Add degree constraints
    pub fn with_degree_range(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        self.min_degree = min;
        self.max_degree = max;
        self
    }
}

/// Pattern edge for graph pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    /// Source node ID within pattern
    pub from: String,
    /// Target node ID within pattern
    pub to: String,
    /// Optional edge type filter
    pub type_filter: Option<String>,
    /// Edge direction constraint
    pub direction: EdgeDirection,
    /// Optional weight range constraint
    pub weight_range: Option<(Weight, Weight)>,
    /// Property constraints
    pub property_filters: PropertyMap,
}

impl PatternEdge {
    /// Create a new pattern edge
    pub fn new(from: String, to: String) -> Self {
        Self {
            from,
            to,
            type_filter: None,
            direction: EdgeDirection::Outgoing,
            weight_range: None,
            property_filters: PropertyMap::new(),
        }
    }
    
    /// Set edge direction
    pub fn with_direction(mut self, direction: EdgeDirection) -> Self {
        self.direction = direction;
        self
    }
    
    /// Add type filter
    pub fn with_type(mut self, type_name: String) -> Self {
        self.type_filter = Some(type_name);
        self
    }
    
    /// Add weight range constraint
    pub fn with_weight_range(mut self, min: Weight, max: Weight) -> Self {
        self.weight_range = Some((min, max));
        self
    }
}

/// Complete pattern for subgraph matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern nodes
    pub nodes: Vec<PatternNode>,
    /// Pattern edges
    pub edges: Vec<PatternEdge>,
    /// Query constraints
    pub constraints: PatternConstraints,
}

impl Pattern {
    /// Create a new empty pattern
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            constraints: PatternConstraints::default(),
        }
    }
    
    /// Add a node to the pattern
    pub fn add_node(mut self, node: PatternNode) -> Self {
        self.nodes.push(node);
        self
    }
    
    /// Add an edge to the pattern
    pub fn add_edge(mut self, edge: PatternEdge) -> Self {
        self.edges.push(edge);
        self
    }
    
    /// Set pattern constraints
    pub fn with_constraints(mut self, constraints: PatternConstraints) -> Self {
        self.constraints = constraints;
        self
    }
    
    /// Validate pattern structure
    pub fn is_valid(&self) -> bool {
        // Check that all edge endpoints reference existing nodes
        let node_ids: std::collections::HashSet<_> = self.nodes.iter().map(|n| &n.id).collect();
        
        self.edges.iter().all(|e| {
            node_ids.contains(&e.from) && node_ids.contains(&e.to)
        })
    }
}

impl Default for Pattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Constraints for pattern matching queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConstraints {
    /// Maximum number of results to return
    pub max_results: Option<usize>,
    /// Query timeout
    pub timeout: Option<Duration>,
    /// Minimum confidence score for approximate matching
    pub min_confidence: Option<f32>,
    /// Temporal constraints
    pub temporal_start: Option<SystemTime>,
    pub temporal_end: Option<SystemTime>,
    /// Enable approximate matching
    pub allow_approximate: bool,
    /// Maximum memory usage for query
    pub memory_limit: Option<usize>,
}

impl Default for PatternConstraints {
    fn default() -> Self {
        Self {
            max_results: Some(1000),
            timeout: Some(Duration::from_secs(30)),
            min_confidence: Some(0.8),
            temporal_start: None,
            temporal_end: None,
            allow_approximate: false,
            memory_limit: None,
        }
    }
}

/// Pattern match result with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatch {
    /// Node bindings from pattern to actual graph
    pub node_bindings: HashMap<String, NodeId>,
    /// Edge bindings from pattern to actual graph
    pub edge_bindings: HashMap<String, EdgeId>,
    /// Confidence score [0.0, 1.0]
    pub score: f32,
    /// Time taken to find this match
    pub computation_time: Duration,
    /// Additional metadata
    pub metadata: PropertyMap,
}

impl PatternMatch {
    /// Create a new pattern match
    pub fn new() -> Self {
        Self {
            node_bindings: HashMap::new(),
            edge_bindings: HashMap::new(),
            score: 1.0,
            computation_time: Duration::ZERO,
            metadata: PropertyMap::new(),
        }
    }
    
    /// Add node binding
    pub fn bind_node(mut self, pattern_id: String, node_id: NodeId) -> Self {
        self.node_bindings.insert(pattern_id, node_id);
        self
    }
    
    /// Add edge binding
    pub fn bind_edge(mut self, pattern_id: String, edge_id: EdgeId) -> Self {
        self.edge_bindings.insert(pattern_id, edge_id);
        self
    }
    
    /// Set confidence score
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score.clamp(0.0, 1.0);
        self
    }
}

impl Default for PatternMatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Algorithm types for centrality computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CentralityAlgorithm {
    /// Degree centrality (fastest)
    Degree,
    /// Betweenness centrality
    Betweenness,
    /// Closeness centrality
    Closeness,
    /// Eigenvector centrality
    Eigenvector,
    /// PageRank centrality
    PageRank,
    /// Katz centrality
    Katz,
}

/// Algorithm types for graph traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraversalAlgorithm {
    /// Breadth-first search
    BreadthFirst,
    /// Depth-first search
    DepthFirst,
    /// Dijkstra's shortest path
    Dijkstra,
    /// A* pathfinding
    AStar,
    /// Bidirectional search
    Bidirectional,
}

/// Community detection algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    /// Louvain modularity optimization
    Louvain,
    /// Label propagation
    LabelPropagation,
    /// Fast greedy modularity
    FastGreedy,
    /// Walktrap algorithm
    WalkTrap,
}

/// Graph statistics with atomic counters for thread safety
#[derive(Debug)]
pub struct GraphStatistics {
    /// Total number of nodes
    pub node_count: AtomicU64,
    /// Total number of edges
    pub edge_count: AtomicU64,
    /// Total number of hyperedges
    pub hyperedge_count: AtomicU64,
    /// Total memory usage in bytes
    pub memory_usage: AtomicUsize,
    /// Number of operations performed
    pub operations_performed: AtomicU64,
    /// Number of queries executed
    pub queries_executed: AtomicU64,
    /// Average query time in nanoseconds
    pub average_query_time_ns: AtomicU64,
    /// Cache hit ratio (0.0 to 1.0, stored as u32)
    pub cache_hit_ratio_x1000: AtomicU64,
    /// SIMD operations performed
    pub simd_operations: AtomicU64,
    /// Start time for uptime calculation
    pub start_time: SystemTime,
}

impl GraphStatistics {
    /// Create new statistics with all counters at zero
    pub fn new() -> Self {
        Self {
            node_count: AtomicU64::new(0),
            edge_count: AtomicU64::new(0),
            hyperedge_count: AtomicU64::new(0),
            memory_usage: AtomicUsize::new(0),
            operations_performed: AtomicU64::new(0),
            queries_executed: AtomicU64::new(0),
            average_query_time_ns: AtomicU64::new(0),
            cache_hit_ratio_x1000: AtomicU64::new(0),
            simd_operations: AtomicU64::new(0),
            start_time: SystemTime::now(),
        }
    }
    
    /// Get cache hit ratio as a float [0.0, 1.0]
    pub fn cache_hit_ratio(&self) -> f32 {
        self.cache_hit_ratio_x1000.load(Ordering::Relaxed) as f32 / 1000.0
    }
    
    /// Set cache hit ratio from a float [0.0, 1.0]
    pub fn set_cache_hit_ratio(&self, ratio: f32) {
        let ratio_x1000 = (ratio.clamp(0.0, 1.0) * 1000.0) as u64;
        self.cache_hit_ratio_x1000.store(ratio_x1000, Ordering::Relaxed);
    }
    
    /// Get uptime since creation
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed().unwrap_or(Duration::ZERO)
    }
    
    /// Reset all counters
    pub fn reset(&self) {
        self.node_count.store(0, Ordering::Relaxed);
        self.edge_count.store(0, Ordering::Relaxed);
        self.hyperedge_count.store(0, Ordering::Relaxed);
        self.memory_usage.store(0, Ordering::Relaxed);
        self.operations_performed.store(0, Ordering::Relaxed);
        self.queries_executed.store(0, Ordering::Relaxed);
        self.average_query_time_ns.store(0, Ordering::Relaxed);
        self.cache_hit_ratio_x1000.store(0, Ordering::Relaxed);
        self.simd_operations.store(0, Ordering::Relaxed);
    }
}

impl Default for GraphStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for components that can be optimized
pub trait Optimizable {
    /// Optimize component for current workload
    type OptimizationResult;
    
    /// Perform optimization
    fn optimize(&mut self) -> HybridResult<Self::OptimizationResult>;
    
    /// Check if optimization is needed
    fn needs_optimization(&self) -> bool;
}

/// Trait for components that support SIMD operations
pub trait SimdCapable {
    /// Check if SIMD is supported for this component
    fn supports_simd(&self) -> bool;
    
    /// Get optimal SIMD width for operations
    fn optimal_simd_width(&self) -> usize;
    
    /// Enable or disable SIMD operations
    fn set_simd_enabled(&mut self, enabled: bool);
}

/// Trait for components with cache-aware operations
pub trait CacheAware {
    /// Prefetch data into cache
    fn prefetch(&self, hint: CachePrefetchHint);
    
    /// Get cache hit ratio
    fn cache_hit_ratio(&self) -> f32;
    
    /// Clear internal caches
    fn clear_caches(&mut self);
}

/// Cache prefetch hints for optimal memory access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePrefetchHint {
    /// Prefetch for immediate use (L1 cache)
    Temporal,
    /// Prefetch for future use (L2/L3 cache)
    NonTemporal,
    /// Prefetch for write operations
    Write,
    /// Prefetch for read operations
    Read,
}

/// Constants for SIMD-optimized operations
pub mod simd_constants {
    /// AVX-512 vector width in f32 elements
    pub const AVX512_F32_WIDTH: usize = 16;
    /// AVX2 vector width in f32 elements
    pub const AVX2_F32_WIDTH: usize = 8;
    /// SSE vector width in f32 elements
    pub const SSE_F32_WIDTH: usize = 4;
    
    /// AVX-512 vector width in i32 elements
    pub const AVX512_I32_WIDTH: usize = 16;
    /// AVX2 vector width in i32 elements
    pub const AVX2_I32_WIDTH: usize = 8;
    /// SSE vector width in i32 elements
    pub const SSE_I32_WIDTH: usize = 4;
    
    /// Alignment requirement for AVX-512
    pub const AVX512_ALIGNMENT: usize = 64;
    /// Alignment requirement for AVX2
    pub const AVX2_ALIGNMENT: usize = 32;
    /// Alignment requirement for SSE
    pub const SSE_ALIGNMENT: usize = 16;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_property_value_memory_size() {
        assert_eq!(PropertyValue::Null.memory_size(), 0);
        assert_eq!(PropertyValue::Bool(true).memory_size(), 1);
        assert_eq!(PropertyValue::Int32(42).memory_size(), 4);
        assert_eq!(PropertyValue::String("hello".to_string()).memory_size(), 5);
    }
    
    #[test]
    fn test_property_value_simd_compatibility() {
        assert!(PropertyValue::Float32(1.0).is_simd_compatible());
        assert!(PropertyValue::Int32(42).is_simd_compatible());
        assert!(!PropertyValue::String("test".to_string()).is_simd_compatible());
    }
    
    #[test]
    fn test_node_data_creation() {
        let mut props = PropertyMap::new();
        props.insert("age".to_string(), PropertyValue::Int32(30));
        props.insert("name".to_string(), PropertyValue::String("Alice".to_string()));
        
        let node = NodeData::new("Person".to_string(), props);
        assert_eq!(node.label, "Person");
        assert_eq!(node.properties.len(), 2);
        assert!(node.memory_size() > 0);
    }
    
    #[test]
    fn test_pattern_validation() {
        let mut pattern = Pattern::new();
        pattern = pattern.add_node(PatternNode::new("a".to_string()));
        pattern = pattern.add_node(PatternNode::new("b".to_string()));
        pattern = pattern.add_edge(PatternEdge::new("a".to_string(), "b".to_string()));
        
        assert!(pattern.is_valid());
        
        // Add edge with invalid endpoint
        pattern = pattern.add_edge(PatternEdge::new("a".to_string(), "c".to_string()));
        assert!(!pattern.is_valid());
    }
    
    #[test]
    fn test_path_validation() {
        let mut path = Path::new();
        assert!(!path.is_valid()); // Empty path is invalid
        
        path.nodes = vec![1, 2, 3];
        path.edges = vec![10, 20];
        path.weights = vec![1.0, 2.0];
        path.length = 2;
        
        assert!(path.is_valid());
    }
    
    #[test]
    fn test_graph_statistics() {
        let stats = GraphStatistics::new();
        
        stats.node_count.store(100, Ordering::Relaxed);
        stats.edge_count.store(200, Ordering::Relaxed);
        
        assert_eq!(stats.node_count.load(Ordering::Relaxed), 100);
        assert_eq!(stats.edge_count.load(Ordering::Relaxed), 200);
        
        stats.set_cache_hit_ratio(0.85);
        assert!((stats.cache_hit_ratio() - 0.85).abs() < 0.001);
        
        stats.reset();
        assert_eq!(stats.node_count.load(Ordering::Relaxed), 0);
    }
}