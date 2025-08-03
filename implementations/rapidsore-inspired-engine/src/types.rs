//! Core data types optimized for 2025 research insights
//!
//! This module implements data structures based on the latest research from:
//! - IndraDB: Lock-free node/edge representations
//! - Kuzu: Columnar storage-friendly layouts
//! - RapidStore: Decoupled read/write optimized structures
//! - 2025 arXiv benchmarks: Memory-aligned and cache-friendly designs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use ahash::AHashMap;

/// High-performance node identifier optimized for billion-scale graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub u128);

impl NodeId {
    /// Create a new node ID from u64 (common case optimization)
    #[inline]
    pub fn from_u64(id: u64) -> Self {
        Self(id as u128)
    }
    
    /// Create a new node ID from u128 (full range)
    #[inline]
    pub const fn from_u128(id: u128) -> Self {
        Self(id)
    }
    
    /// Convert to u64 (lossy, but common case)
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }
    
    /// Convert to u128 (lossless)
    #[inline]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
    
    /// Generate a random node ID (for testing/sharding)
    #[inline]
    pub fn random() -> Self {
        Self(fastrand::u128(..))
    }
    
    /// Check if this is a special/reserved ID
    #[inline]
    pub const fn is_reserved(self) -> bool {
        self.0 < 1000 // Reserve first 1000 IDs for system use
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node:{}", self.0)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        Self::from_u64(id)
    }
}

impl From<u128> for NodeId {
    fn from(id: u128) -> Self {
        Self::from_u128(id)
    }
}

/// High-performance edge identifier optimized for massive edge counts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct EdgeId(pub u128);

impl EdgeId {
    /// Create a new edge ID
    #[inline]
    pub const fn new(id: u128) -> Self {
        Self(id)
    }
    
    /// Convert to u128
    #[inline]
    pub const fn as_u128(self) -> u128 {
        self.0
    }
    
    /// Generate a random edge ID
    #[inline]
    pub fn random() -> Self {
        Self(fastrand::u128(..))
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "edge:{}", self.0)
    }
}

/// Node data structure optimized for columnar storage (Kuzu-inspired)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier (128-bit for massive scale)
    pub id: NodeId,
    /// Node type for schema optimization
    pub node_type: String,
    /// Flexible data storage
    pub data: NodeData,
    /// Metadata for advanced features
    pub metadata: NodeMetadata,
}

impl Node {
    /// Create a new node with minimal data
    pub fn new(id: NodeId, node_type: impl Into<String>) -> Self {
        Self {
            id,
            node_type: node_type.into(),
            data: NodeData::Empty,
            metadata: NodeMetadata::default(),
        }
    }
    
    /// Create a node with text data (common case)
    pub fn with_text(id: NodeId, node_type: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id,
            node_type: node_type.into(),
            data: NodeData::Text(text.into()),
            metadata: NodeMetadata::default(),
        }
    }
    
    /// Create a node with properties
    pub fn with_properties(
        id: NodeId,
        node_type: impl Into<String>,
        properties: HashMap<String, PropertyValue>,
    ) -> Self {
        Self {
            id,
            node_type: node_type.into(),
            data: NodeData::Properties(properties),
            metadata: NodeMetadata::default(),
        }
    }
    
    /// Get memory footprint for optimization
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<NodeId>()
            + self.node_type.len()
            + self.data.memory_size()
            + self.metadata.memory_size()
    }
}

/// Flexible node data storage optimized for different use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeData {
    /// No data (minimal memory footprint)
    Empty,
    /// Text content (UTF-8 optimized)
    Text(String),
    /// Structured properties (key-value pairs)
    Properties(HashMap<String, PropertyValue>),
    /// Binary blob (for embeddings, images, etc.)
    Binary(Vec<u8>),
    /// Numeric vector (for ML/AI applications)
    Vector(Vec<f64>),
    /// JSON-like structured data
    Structured(serde_json::Value),
}

impl NodeData {
    /// Calculate memory footprint
    pub fn memory_size(&self) -> usize {
        match self {
            NodeData::Empty => 0,
            NodeData::Text(s) => s.len(),
            NodeData::Properties(map) => {
                map.iter()
                    .map(|(k, v)| k.len() + v.memory_size())
                    .sum()
            }
            NodeData::Binary(data) => data.len(),
            NodeData::Vector(vec) => vec.len() * 8, // f64 = 8 bytes
            NodeData::Structured(value) => {
                // Rough estimate for JSON value
                serde_json::to_string(value).map_or(0, |s| s.len())
            }
        }
    }
    
    /// Check if data is empty
    pub fn is_empty(&self) -> bool {
        matches!(self, NodeData::Empty)
    }
}

impl Default for NodeData {
    fn default() -> Self {
        NodeData::Empty
    }
}

/// Edge data structure optimized for high-throughput operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique edge identifier
    pub id: EdgeId,
    /// Source node
    pub from: NodeId,
    /// Target node
    pub to: NodeId,
    /// Edge type for schema optimization
    pub edge_type: String,
    /// Optional weight for algorithms
    pub weight: Option<f64>,
    /// Edge-specific data
    pub data: EdgeData,
    /// Metadata for advanced features
    pub metadata: EdgeMetadata,
}

impl Edge {
    /// Create a simple edge between two nodes
    pub fn new(
        id: EdgeId,
        from: NodeId,
        to: NodeId,
        edge_type: impl Into<String>,
    ) -> Self {
        Self {
            id,
            from,
            to,
            edge_type: edge_type.into(),
            weight: None,
            data: EdgeData::Empty,
            metadata: EdgeMetadata::default(),
        }
    }
    
    /// Create a weighted edge
    pub fn weighted(
        id: EdgeId,
        from: NodeId,
        to: NodeId,
        edge_type: impl Into<String>,
        weight: f64,
    ) -> Self {
        Self {
            id,
            from,
            to,
            edge_type: edge_type.into(),
            weight: Some(weight),
            data: EdgeData::Empty,
            metadata: EdgeMetadata::default(),
        }
    }
    
    /// Get the effective weight (1.0 if not specified)
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight.unwrap_or(1.0)
    }
    
    /// Check if this is a self-loop
    #[inline]
    pub fn is_self_loop(&self) -> bool {
        self.from == self.to
    }
    
    /// Get memory footprint
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<EdgeId>()
            + std::mem::size_of::<NodeId>() * 2
            + self.edge_type.len()
            + std::mem::size_of::<Option<f64>>()
            + self.data.memory_size()
            + self.metadata.memory_size()
    }
}

/// Edge data storage optimized for different relationship types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeData {
    /// No additional data
    Empty,
    /// Simple text label or description
    Text(String),
    /// Structured properties
    Properties(HashMap<String, PropertyValue>),
    /// Temporal information (for time-series graphs)
    Temporal {
        timestamp: SystemTime,
        duration: Option<std::time::Duration>,
    },
    /// Probabilistic edge with confidence
    Probabilistic {
        confidence: f64,
        source: String,
    },
    /// Relationship strength/intensity
    Weighted {
        strength: f64,
        normalized: bool,
    },
}

impl EdgeData {
    /// Calculate memory footprint
    pub fn memory_size(&self) -> usize {
        match self {
            EdgeData::Empty => 0,
            EdgeData::Text(s) => s.len(),
            EdgeData::Properties(map) => {
                map.iter()
                    .map(|(k, v)| k.len() + v.memory_size())
                    .sum()
            }
            EdgeData::Temporal { .. } => std::mem::size_of::<SystemTime>() + std::mem::size_of::<Option<std::time::Duration>>(),
            EdgeData::Probabilistic { source, .. } => std::mem::size_of::<f64>() + source.len(),
            EdgeData::Weighted { .. } => std::mem::size_of::<f64>() + std::mem::size_of::<bool>(),
        }
    }
}

impl Default for EdgeData {
    fn default() -> Self {
        EdgeData::Empty
    }
}

/// Flexible property value type for structured data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer value (64-bit signed)
    Int(i64),
    /// Floating point value (64-bit)
    Float(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<PropertyValue>),
    /// Nested object
    Object(HashMap<String, PropertyValue>),
    /// Binary data
    Binary(Vec<u8>),
    /// Timestamp
    Timestamp(SystemTime),
}

impl PropertyValue {
    /// Calculate memory footprint
    pub fn memory_size(&self) -> usize {
        match self {
            PropertyValue::Null => 0,
            PropertyValue::Bool(_) => 1,
            PropertyValue::Int(_) => 8,
            PropertyValue::Float(_) => 8,
            PropertyValue::String(s) => s.len(),
            PropertyValue::Array(arr) => arr.iter().map(|v| v.memory_size()).sum(),
            PropertyValue::Object(map) => {
                map.iter()
                    .map(|(k, v)| k.len() + v.memory_size())
                    .sum()
            }
            PropertyValue::Binary(data) => data.len(),
            PropertyValue::Timestamp(_) => std::mem::size_of::<SystemTime>(),
        }
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => s.clone(),
            PropertyValue::Array(_) => "[array]".to_string(),
            PropertyValue::Object(_) => "{object}".to_string(),
            PropertyValue::Binary(data) => format!("[binary:{}]", data.len()),
            PropertyValue::Timestamp(ts) => format!("{:?}", ts),
        }
    }
}

/// Node metadata for advanced features and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Version for optimistic concurrency control
    pub version: u64,
    /// Tags for indexing and categorization
    pub tags: Vec<String>,
    /// Custom properties
    pub properties: AHashMap<String, PropertyValue>,
    /// Access statistics for optimization
    pub access_count: AtomicU64,
    /// Schema version for evolution
    pub schema_version: u32,
}

impl NodeMetadata {
    /// Create new metadata with current timestamp
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            tags: Vec::new(),
            properties: AHashMap::new(),
            access_count: AtomicU64::new(0),
            schema_version: 1,
        }
    }
    
    /// Record access for optimization
    #[inline]
    pub fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get access count
    #[inline]
    pub fn get_access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
    
    /// Update modification timestamp and version
    pub fn touch(&mut self) {
        self.updated_at = SystemTime::now();
        self.version += 1;
    }
    
    /// Calculate memory footprint
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<SystemTime>() * 2
            + std::mem::size_of::<u64>()
            + self.tags.iter().map(|t| t.len()).sum::<usize>()
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + std::mem::size_of::<AtomicU64>()
            + std::mem::size_of::<u32>()
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge metadata for relationship-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp  
    pub updated_at: SystemTime,
    /// Version for optimistic concurrency control
    pub version: u64,
    /// Custom properties
    pub properties: AHashMap<String, PropertyValue>,
    /// Relationship strength/confidence
    pub strength: f64,
    /// Bidirectional flag
    pub bidirectional: bool,
    /// Schema version
    pub schema_version: u32,
}

impl EdgeMetadata {
    /// Create new metadata with current timestamp
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            properties: AHashMap::new(),
            strength: 1.0,
            bidirectional: false,
            schema_version: 1,
        }
    }
    
    /// Update modification timestamp and version
    pub fn touch(&mut self) {
        self.updated_at = SystemTime::now();
        self.version += 1;
    }
    
    /// Calculate memory footprint
    pub fn memory_size(&self) -> usize {
        std::mem::size_of::<SystemTime>() * 2
            + std::mem::size_of::<u64>()
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + std::mem::size_of::<f64>()
            + std::mem::size_of::<bool>()
            + std::mem::size_of::<u32>()
    }
}

impl Default for EdgeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Path representation for shortest path queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Sequence of nodes in the path
    pub nodes: Vec<NodeId>,
    /// Sequence of edges connecting the nodes
    pub edges: Vec<EdgeId>,
    /// Total path weight/cost
    pub weight: f64,
    /// Path metadata
    pub metadata: PathMetadata,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            weight: 0.0,
            metadata: PathMetadata::default(),
        }
    }
    
    /// Create a path from a sequence of nodes and edges
    pub fn from_sequence(nodes: Vec<NodeId>, edges: Vec<EdgeId>, weight: f64) -> Self {
        Self {
            nodes,
            edges,
            weight,
            metadata: PathMetadata::default(),
        }
    }
    
    /// Add a hop to the path
    pub fn add_hop(&mut self, node: NodeId, edge: Option<EdgeId>, edge_weight: f64) {
        self.nodes.push(node);
        if let Some(edge_id) = edge {
            self.edges.push(edge_id);
            self.weight += edge_weight;
        }
    }
    
    /// Get path length (number of edges)
    #[inline]
    pub fn length(&self) -> usize {
        self.edges.len()
    }
    
    /// Check if path is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    
    /// Get the first node (source)
    pub fn source(&self) -> Option<NodeId> {
        self.nodes.first().copied()
    }
    
    /// Get the last node (target)
    pub fn target(&self) -> Option<NodeId> {
        self.nodes.last().copied()
    }
    
    /// Reverse the path
    pub fn reverse(&mut self) {
        self.nodes.reverse();
        self.edges.reverse();
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// Path metadata for algorithm-specific information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathMetadata {
    /// Algorithm used to find this path
    pub algorithm: String,
    /// Computation time in microseconds
    pub computation_time_us: u64,
    /// Number of nodes explored during search
    pub nodes_explored: usize,
    /// Whether this is guaranteed to be optimal
    pub is_optimal: bool,
}

/// Graph statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: u64,
    /// Total number of edges
    pub edge_count: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Average node degree
    pub avg_degree: f64,
    /// Maximum node degree
    pub max_degree: u64,
    /// Graph density (0.0 to 1.0)
    pub density: f64,
    /// Number of connected components
    pub connected_components: u64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Last update timestamp
    pub last_updated: SystemTime,
}

impl GraphStats {
    /// Create new stats with current timestamp
    pub fn new() -> Self {
        Self {
            last_updated: SystemTime::now(),
            ..Default::default()
        }
    }
    
    /// Update statistics
    pub fn update(&mut self, nodes: u64, edges: u64, memory: u64) {
        self.node_count = nodes;
        self.edge_count = edges;
        self.memory_usage = memory;
        self.avg_degree = if nodes > 0 {
            (edges * 2) as f64 / nodes as f64
        } else {
            0.0
        };
        self.density = if nodes > 1 {
            edges as f64 / ((nodes * (nodes - 1)) as f64 / 2.0)
        } else {
            0.0
        };
        self.last_updated = SystemTime::now();
    }
}

/// Ultra-fast hash function optimized for node/edge IDs
#[inline]
pub fn fast_hash<T: AsRef<[u8]>>(data: T) -> u64 {
    // Use ahash for consistent high performance
    use std::hash::{Hash, Hasher};
    let mut hasher = ahash::AHasher::default();
    data.as_ref().hash(&mut hasher);
    hasher.finish()
}

/// Specialized hash function for NodeId (optimized hot path)
#[inline]
pub fn hash_node_id(id: NodeId) -> u64 {
    // Use simple bit manipulation for maximum speed on u128
    let high = (id.0 >> 64) as u64;
    let low = id.0 as u64;
    high.wrapping_mul(0x9e3779b97f4a7c15) ^ low.wrapping_mul(0x517cc1b727220a95)
}

/// Specialized hash function for EdgeId  
#[inline]
pub fn hash_edge_id(id: EdgeId) -> u64 {
    hash_node_id(NodeId(id.0))
}

/// Batch operation types for high-throughput scenarios
#[derive(Debug, Clone)]
pub enum BatchOperation {
    /// Insert multiple nodes
    InsertNodes(Vec<Node>),
    /// Insert multiple edges
    InsertEdges(Vec<Edge>),
    /// Update multiple nodes
    UpdateNodes(Vec<Node>),
    /// Update multiple edges
    UpdateEdges(Vec<Edge>),
    /// Delete multiple nodes (by ID)
    DeleteNodes(Vec<NodeId>),
    /// Delete multiple edges (by ID)
    DeleteEdges(Vec<EdgeId>),
    /// Mixed operations
    Mixed(Vec<SingleOperation>),
}

/// Single operation for mixed batches
#[derive(Debug, Clone)]
pub enum SingleOperation {
    /// Insert a node
    InsertNode(Node),
    /// Insert an edge
    InsertEdge(Edge),
    /// Update a node
    UpdateNode(Node),
    /// Update an edge
    UpdateEdge(Edge),
    /// Delete a node
    DeleteNode(NodeId),
    /// Delete an edge
    DeleteEdge(EdgeId),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_id_operations() {
        let id1 = NodeId::from_u64(12345);
        let id2 = NodeId::from_u128(12345);
        assert_eq!(id1, id2);
        
        let id3 = NodeId::random();
        assert_ne!(id1, id3);
        
        assert!(!id1.is_reserved());
        assert!(NodeId::from_u64(100).is_reserved());
    }
    
    #[test]
    fn test_node_creation() {
        let id = NodeId::from_u64(1);
        let node = Node::new(id, "Person");
        
        assert_eq!(node.id, id);
        assert_eq!(node.node_type, "Person");
        assert!(matches!(node.data, NodeData::Empty));
        
        let text_node = Node::with_text(id, "Document", "Hello world");
        assert!(matches!(text_node.data, NodeData::Text(_)));
    }
    
    #[test]
    fn test_edge_creation() {
        let id = EdgeId::new(1);
        let from = NodeId::from_u64(1);
        let to = NodeId::from_u64(2);
        
        let edge = Edge::new(id, from, to, "KNOWS");
        assert_eq!(edge.weight(), 1.0); // Default weight
        assert!(!edge.is_self_loop());
        
        let weighted_edge = Edge::weighted(id, from, to, "LIKES", 0.8);
        assert_eq!(weighted_edge.weight(), 0.8);
        
        let self_loop = Edge::new(id, from, from, "SELF");
        assert!(self_loop.is_self_loop());
    }
    
    #[test]
    fn test_path_operations() {
        let mut path = Path::new();
        assert!(path.is_empty());
        assert_eq!(path.length(), 0);
        
        let node1 = NodeId::from_u64(1);
        let node2 = NodeId::from_u64(2);
        let edge1 = EdgeId::new(1);
        
        path.add_hop(node1, None, 0.0);
        path.add_hop(node2, Some(edge1), 1.5);
        
        assert_eq!(path.length(), 1);
        assert_eq!(path.weight, 1.5);
        assert_eq!(path.source(), Some(node1));
        assert_eq!(path.target(), Some(node2));
    }
    
    #[test]
    fn test_memory_calculations() {
        let node = Node::with_text(NodeId::from_u64(1), "Test", "Sample text");
        let memory_size = node.memory_size();
        assert!(memory_size > 0);
        
        let edge = Edge::weighted(
            EdgeId::new(1),
            NodeId::from_u64(1),
            NodeId::from_u64(2),
            "CONNECTS",
            1.0,
        );
        let edge_memory = edge.memory_size();
        assert!(edge_memory > 0);
    }
    
    #[test]
    fn test_hash_functions() {
        let node_id = NodeId::from_u64(12345);
        let hash1 = hash_node_id(node_id);
        let hash2 = hash_node_id(node_id);
        assert_eq!(hash1, hash2); // Deterministic
        
        let edge_id = EdgeId::new(67890);
        let edge_hash = hash_edge_id(edge_id);
        assert_ne!(hash1, edge_hash); // Different values should hash differently
    }
    
    #[test]
    fn test_graph_stats() {
        let mut stats = GraphStats::new();
        stats.update(1000, 5000, 1024 * 1024);
        
        assert_eq!(stats.node_count, 1000);
        assert_eq!(stats.edge_count, 5000);
        assert_eq!(stats.avg_degree, 10.0); // (5000 * 2) / 1000
        assert!(stats.density > 0.0);
    }
    
    #[test]
    fn test_property_values() {
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), PropertyValue::String("Alice".to_string()));
        properties.insert("age".to_string(), PropertyValue::Int(30));
        properties.insert("active".to_string(), PropertyValue::Bool(true));
        
        let node = Node::with_properties(NodeId::from_u64(1), "Person", properties);
        
        match &node.data {
            NodeData::Properties(props) => {
                assert_eq!(props.len(), 3);
                assert!(props.contains_key("name"));
            }
            _ => panic!("Expected properties"),
        }
    }
}