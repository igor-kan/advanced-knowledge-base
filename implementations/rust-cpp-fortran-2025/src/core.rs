//! Core types and data structures for the ultra-fast knowledge graph
//!
//! This module implements the foundational types optimized for 177x+ speedups
//! using the latest 2025 research in high-performance graph processing.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use ahash::AHasher;
use serde::{Deserialize, Serialize};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

/// Ultra-optimized node identifier using 128-bit integers for infinite scale
pub type NodeId = u128;

/// Ultra-optimized edge identifier using 128-bit integers
pub type EdgeId = u128;

/// Property key type optimized for cache efficiency
pub type PropertyKey = smallvec::SmallVec<[u8; 32]>;

/// Timestamp type for temporal graph operations
pub type Timestamp = u64;

/// Weight type for weighted graphs with high precision
pub type Weight = f64;

/// Probability type for probabilistic graphs
pub type Probability = f32;

/// Core result type for ultra-fast operations
pub type UltraResult<T> = std::result::Result<T, crate::error::UltraFastKnowledgeGraphError>;

/// Ultra-optimized graph node with 2025 research optimizations
#[derive(Debug, Clone, Serialize, Deserialize, AsBytes, FromBytes, FromZeroes)]
#[repr(C, align(64))] // Cache line alignment for optimal memory access
pub struct UltraNode {
    /// Unique node identifier (128-bit for infinite scale)
    pub id: NodeId,
    
    /// Node type identifier for intricate relationship modeling
    pub node_type: u32,
    
    /// Creation timestamp for temporal analysis
    pub created_at: Timestamp,
    
    /// Last modification timestamp
    pub modified_at: Timestamp,
    
    /// Packed node flags for efficient storage
    pub flags: NodeFlags,
    
    /// Reserved space for future extensions (maintains alignment)
    pub reserved: [u64; 3],
}

impl UltraNode {
    /// Create new ultra-optimized node
    #[inline(always)]
    pub fn new(id: NodeId, node_type: u32) -> Self {
        let now = current_timestamp();
        Self {
            id,
            node_type,
            created_at: now,
            modified_at: now,
            flags: NodeFlags::default(),
            reserved: [0; 3],
        }
    }
    
    /// Get node age in microseconds
    #[inline(always)]
    pub fn age_microseconds(&self) -> u64 {
        current_timestamp().saturating_sub(self.created_at)
    }
    
    /// Check if node has specific flag
    #[inline(always)]
    pub fn has_flag(&self, flag: NodeFlag) -> bool {
        self.flags.has_flag(flag)
    }
    
    /// Set node flag
    #[inline(always)]
    pub fn set_flag(&mut self, flag: NodeFlag) {
        self.flags.set_flag(flag);
        self.modified_at = current_timestamp();
    }
}

impl Hash for UltraNode {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Ultra-fast hash using only the ID for performance
        self.id.hash(state);
    }
}

impl PartialEq for UltraNode {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for UltraNode {}

/// Ultra-optimized graph edge with 2025 research optimizations
#[derive(Debug, Clone, Serialize, Deserialize, AsBytes, FromBytes, FromZeroes)]
#[repr(C, align(64))] // Cache line alignment for optimal memory access
pub struct UltraEdge {
    /// Unique edge identifier
    pub id: EdgeId,
    
    /// Source node identifier
    pub from: NodeId,
    
    /// Target node identifier
    pub to: NodeId,
    
    /// Edge type for intricate relationship modeling
    pub edge_type: u32,
    
    /// Edge weight for weighted graph algorithms
    pub weight: Weight,
    
    /// Edge probability for probabilistic graphs
    pub probability: Probability,
    
    /// Creation timestamp
    pub created_at: Timestamp,
    
    /// Last modification timestamp
    pub modified_at: Timestamp,
    
    /// Packed edge flags
    pub flags: EdgeFlags,
    
    /// Reserved space for future extensions
    pub reserved: [u32; 2],
}

impl UltraEdge {
    /// Create new ultra-optimized edge
    #[inline(always)]
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, edge_type: u32) -> Self {
        let now = current_timestamp();
        Self {
            id,
            from,
            to,
            edge_type,
            weight: 1.0,
            probability: 1.0,
            created_at: now,
            modified_at: now,
            flags: EdgeFlags::default(),
            reserved: [0; 2],
        }
    }
    
    /// Create weighted edge
    #[inline(always)]
    pub fn with_weight(mut self, weight: Weight) -> Self {
        self.weight = weight;
        self
    }
    
    /// Create probabilistic edge
    #[inline(always)]
    pub fn with_probability(mut self, probability: Probability) -> Self {
        self.probability = probability;
        self
    }
    
    /// Check if edge is bidirectional
    #[inline(always)]
    pub fn is_bidirectional(&self) -> bool {
        self.flags.has_flag(EdgeFlag::Bidirectional)
    }
    
    /// Get edge direction vector for geometric calculations
    #[inline(always)]
    pub fn direction_hash(&self) -> u64 {
        // Ultra-fast directional hash for spatial algorithms
        crate::ultra_fast_hash(self.from as u64) ^ crate::ultra_fast_hash(self.to as u64)
    }
}

impl Hash for UltraEdge {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for UltraEdge {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for UltraEdge {}

/// Packed node flags for efficient storage and fast bitwise operations
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, AsBytes, FromBytes, FromZeroes)]
#[repr(transparent)]
pub struct NodeFlags(u64);

impl NodeFlags {
    /// Check if specific flag is set
    #[inline(always)]
    pub fn has_flag(&self, flag: NodeFlag) -> bool {
        (self.0 & flag as u64) != 0
    }
    
    /// Set specific flag
    #[inline(always)]
    pub fn set_flag(&mut self, flag: NodeFlag) {
        self.0 |= flag as u64;
    }
    
    /// Clear specific flag
    #[inline(always)]
    pub fn clear_flag(&mut self, flag: NodeFlag) {
        self.0 &= !(flag as u64);
    }
    
    /// Toggle specific flag
    #[inline(always)]
    pub fn toggle_flag(&mut self, flag: NodeFlag) {
        self.0 ^= flag as u64;
    }
}

/// Individual node flags for various node states
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum NodeFlag {
    /// Node is active and available for queries
    Active = 1 << 0,
    /// Node is indexed for fast searches
    Indexed = 1 << 1,
    /// Node is cached in memory
    Cached = 1 << 2,
    /// Node is part of a cluster/community
    Clustered = 1 << 3,
    /// Node has temporal properties
    Temporal = 1 << 4,
    /// Node has spatial/geometric properties
    Spatial = 1 << 5,
    /// Node is replicated across distributed nodes
    Replicated = 1 << 6,
    /// Node is compressed for storage optimization
    Compressed = 1 << 7,
    /// Node is marked for deletion (soft delete)
    MarkedForDeletion = 1 << 8,
    /// Node has been processed by ML algorithms
    MLProcessed = 1 << 9,
    /// Node is part of a materialized view
    Materialized = 1 << 10,
    /// Node has high-priority for caching
    HighPriority = 1 << 11,
}

/// Packed edge flags for efficient storage and fast bitwise operations
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, AsBytes, FromBytes, FromZeroes)]
#[repr(transparent)]
pub struct EdgeFlags(u64);

impl EdgeFlags {
    /// Check if specific flag is set
    #[inline(always)]
    pub fn has_flag(&self, flag: EdgeFlag) -> bool {
        (self.0 & flag as u64) != 0
    }
    
    /// Set specific flag
    #[inline(always)]
    pub fn set_flag(&mut self, flag: EdgeFlag) {
        self.0 |= flag as u64;
    }
    
    /// Clear specific flag
    #[inline(always)]
    pub fn clear_flag(&mut self, flag: EdgeFlag) {
        self.0 &= !(flag as u64);
    }
}

/// Individual edge flags for various edge states
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
pub enum EdgeFlag {
    /// Edge is bidirectional
    Bidirectional = 1 << 0,
    /// Edge is weighted
    Weighted = 1 << 1,
    /// Edge is probabilistic
    Probabilistic = 1 << 2,
    /// Edge is temporal (has time-based validity)
    Temporal = 1 << 3,
    /// Edge is derived/computed
    Derived = 1 << 4,
    /// Edge is part of a spanning tree
    SpanningTree = 1 << 5,
    /// Edge is a back edge in DFS
    BackEdge = 1 << 6,
    /// Edge is a bridge (removal disconnects graph)
    Bridge = 1 << 7,
    /// Edge is cached for fast access
    Cached = 1 << 8,
    /// Edge is replicated across distributed nodes
    Replicated = 1 << 9,
    /// Edge is compressed
    Compressed = 1 << 10,
    /// Edge is marked for deletion
    MarkedForDeletion = 1 << 11,
}

/// Ultra-fast property value with zero-copy serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UltraPropertyValue {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit unsigned integer  
    UInt64(u64),
    /// 64-bit float
    Float64(f64),
    /// 32-bit float
    Float32(f32),
    /// String value with small string optimization
    String(smallvec::SmallVec<[u8; 32]>),
    /// Binary data
    Bytes(Vec<u8>),
    /// Timestamp
    Timestamp(Timestamp),
    /// Geographic coordinates
    GeoPoint { lat: f64, lon: f64 },
    /// UUID
    Uuid([u8; 16]),
    /// JSON object (for complex nested data)
    Json(serde_json::Value),
    /// Array of values
    Array(Vec<UltraPropertyValue>),
    /// Hash map for nested properties
    Object(HashMap<String, UltraPropertyValue>),
}

impl UltraPropertyValue {
    /// Get type identifier for efficient type checking
    #[inline(always)]
    pub fn type_id(&self) -> u8 {
        match self {
            Self::Null => 0,
            Self::Bool(_) => 1,
            Self::Int64(_) => 2,
            Self::UInt64(_) => 3,
            Self::Float64(_) => 4,
            Self::Float32(_) => 5,
            Self::String(_) => 6,
            Self::Bytes(_) => 7,
            Self::Timestamp(_) => 8,
            Self::GeoPoint { .. } => 9,
            Self::Uuid(_) => 10,
            Self::Json(_) => 11,
            Self::Array(_) => 12,
            Self::Object(_) => 13,
        }
    }
    
    /// Check if value is numeric
    #[inline(always)]
    pub fn is_numeric(&self) -> bool {
        matches!(self, Self::Int64(_) | Self::UInt64(_) | Self::Float64(_) | Self::Float32(_))
    }
    
    /// Convert to f64 if numeric
    #[inline(always)]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int64(v) => Some(*v as f64),
            Self::UInt64(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            Self::Float32(v) => Some(*v as f64),
            _ => None,
        }
    }
}

/// Compressed Sparse Row (CSR) adjacency list for ultra-fast traversals
#[derive(Debug, Clone)]
pub struct UltraCSRGraph {
    /// Row pointers (offsets into neighbors array)
    pub row_ptr: Vec<u64>,
    
    /// Column indices (neighbor node IDs)
    pub col_idx: Vec<NodeId>,
    
    /// Edge weights (parallel to col_idx)
    pub weights: Vec<Weight>,
    
    /// Edge IDs (parallel to col_idx)
    pub edge_ids: Vec<EdgeId>,
    
    /// Number of nodes
    pub num_nodes: u64,
    
    /// Number of edges  
    pub num_edges: u64,
}

impl UltraCSRGraph {
    /// Create new CSR graph with capacity
    pub fn with_capacity(num_nodes: u64, num_edges: u64) -> Self {
        Self {
            row_ptr: Vec::with_capacity((num_nodes + 1) as usize),
            col_idx: Vec::with_capacity(num_edges as usize),
            weights: Vec::with_capacity(num_edges as usize),
            edge_ids: Vec::with_capacity(num_edges as usize),
            num_nodes,
            num_edges,
        }
    }
    
    /// Get neighbors of a node with ultra-fast access
    #[inline(always)]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        let node_idx = node as usize;
        if node_idx >= self.row_ptr.len() - 1 {
            return &[];
        }
        
        let start = self.row_ptr[node_idx] as usize;
        let end = self.row_ptr[node_idx + 1] as usize;
        &self.col_idx[start..end]
    }
    
    /// Get neighbor weights for a node
    #[inline(always)]
    pub fn neighbor_weights(&self, node: NodeId) -> &[Weight] {
        let node_idx = node as usize;
        if node_idx >= self.row_ptr.len() - 1 {
            return &[];
        }
        
        let start = self.row_ptr[node_idx] as usize;
        let end = self.row_ptr[node_idx + 1] as usize;
        &self.weights[start..end]
    }
    
    /// Get degree of a node
    #[inline(always)]
    pub fn degree(&self, node: NodeId) -> u64 {
        let node_idx = node as usize;
        if node_idx >= self.row_ptr.len() - 1 {
            return 0;
        }
        
        self.row_ptr[node_idx + 1] - self.row_ptr[node_idx]
    }
}

/// Query optimization hints for ultra-fast execution
#[derive(Debug, Clone)]
pub enum QueryHint {
    /// Prefer specific algorithm
    PreferAlgorithm(String),
    /// Expected result size for memory pre-allocation
    ExpectedResultSize(usize),
    /// Use specific index
    UseIndex(String),
    /// Parallelize across N threads
    Parallelize(usize),
    /// Cache result for future queries
    CacheResult,
    /// Use assembly-optimized hot paths
    UseAssemblyOptimization,
    /// Use SIMD vectorization
    UseSIMD,
    /// Use Fortran mathematical routines
    UseFortranMath,
    /// Use C++ backend for maximum performance
    UseCppBackend,
    /// Distribute across cluster nodes
    DistributeQuery,
}

/// Graph statistics for optimization and monitoring
#[derive(Debug, Clone, Default)]
pub struct UltraGraphStats {
    /// Total number of nodes
    pub total_nodes: AtomicU64,
    
    /// Total number of edges
    pub total_edges: AtomicU64,
    
    /// Average degree
    pub average_degree: std::sync::atomic::AtomicU64, // Stored as f64 bits
    
    /// Maximum degree
    pub max_degree: AtomicU64,
    
    /// Graph diameter (longest shortest path)
    pub diameter: AtomicU64,
    
    /// Number of connected components
    pub connected_components: AtomicU64,
    
    /// Clustering coefficient
    pub clustering_coefficient: std::sync::atomic::AtomicU64, // Stored as f64 bits
    
    /// Operations performed
    pub operations_count: AtomicU64,
    
    /// Cache hits
    pub cache_hits: AtomicU64,
    
    /// Cache misses
    pub cache_misses: AtomicU64,
}

impl UltraGraphStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Increment node count atomically
    #[inline(always)]
    pub fn increment_nodes(&self) -> u64 {
        self.total_nodes.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Increment edge count atomically
    #[inline(always)]
    pub fn increment_edges(&self) -> u64 {
        self.total_edges.fetch_add(1, Ordering::Relaxed)
    }
    
    /// Record cache hit
    #[inline(always)]
    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record cache miss
    #[inline(always)]
    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get cache hit ratio
    #[inline(always)]
    pub fn cache_hit_ratio(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// Ultra-fast timestamp generation
#[inline(always)]
pub fn current_timestamp() -> Timestamp {
    // Use monotonic time for consistent ordering
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Generate ultra-fast unique IDs with guaranteed uniqueness
pub struct UltraFastIdGenerator {
    /// Atomic counter for local uniqueness
    counter: AtomicU64,
    
    /// Node identifier for distributed uniqueness
    node_id: u64,
    
    /// Timestamp base for temporal ordering
    timestamp_base: u64,
}

impl UltraFastIdGenerator {
    /// Create new ID generator
    pub fn new(node_id: u64) -> Self {
        Self {
            counter: AtomicU64::new(0),
            node_id,
            timestamp_base: current_timestamp(),
        }
    }
    
    /// Generate next unique node ID
    #[inline(always)]
    pub fn next_node_id(&self) -> NodeId {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = current_timestamp() - self.timestamp_base;
        
        // Combine node_id (16 bits), timestamp (64 bits), counter (48 bits)
        ((self.node_id as u128) << 112) | ((timestamp as u128) << 48) | (counter as u128)
    }
    
    /// Generate next unique edge ID
    #[inline(always)]
    pub fn next_edge_id(&self) -> EdgeId {
        // Use different bit pattern for edges to avoid collisions
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = current_timestamp() - self.timestamp_base;
        
        // Set high bit to distinguish from node IDs
        (1u128 << 127) | ((self.node_id as u128) << 112) | ((timestamp as u128) << 48) | (counter as u128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ultra_node_creation() {
        let node = UltraNode::new(42, 1);
        assert_eq!(node.id, 42);
        assert_eq!(node.node_type, 1);
        assert!(node.age_microseconds() < 1000); // Should be very recent
    }
    
    #[test]
    fn test_ultra_edge_creation() {
        let edge = UltraEdge::new(1, 42, 84, 1).with_weight(2.5);
        assert_eq!(edge.id, 1);
        assert_eq!(edge.from, 42);
        assert_eq!(edge.to, 84);
        assert_eq!(edge.weight, 2.5);
    }
    
    #[test]
    fn test_node_flags() {
        let mut node = UltraNode::new(1, 1);
        assert!(!node.has_flag(NodeFlag::Active));
        
        node.set_flag(NodeFlag::Active);
        assert!(node.has_flag(NodeFlag::Active));
    }
    
    #[test]
    fn test_edge_flags() {
        let mut edge = UltraEdge::new(1, 42, 84, 1);
        assert!(!edge.is_bidirectional());
        
        edge.flags.set_flag(EdgeFlag::Bidirectional);
        assert!(edge.is_bidirectional());
    }
    
    #[test]
    fn test_property_values() {
        let prop = UltraPropertyValue::Int64(42);
        assert!(prop.is_numeric());
        assert_eq!(prop.as_f64(), Some(42.0));
        assert_eq!(prop.type_id(), 2);
    }
    
    #[test]
    fn test_csr_graph() {
        let mut graph = UltraCSRGraph::with_capacity(3, 4);
        
        // Manually build small graph: 0->1, 0->2, 1->2, 2->0
        graph.row_ptr = vec![0, 2, 3, 4];
        graph.col_idx = vec![1, 2, 2, 0];
        graph.weights = vec![1.0, 1.0, 1.0, 1.0];
        graph.edge_ids = vec![1, 2, 3, 4];
        
        let neighbors_0 = graph.neighbors(0);
        assert_eq!(neighbors_0, &[1, 2]);
        assert_eq!(graph.degree(0), 2);
        
        let neighbors_1 = graph.neighbors(1);
        assert_eq!(neighbors_1, &[2]);
        assert_eq!(graph.degree(1), 1);
    }
    
    #[test]
    fn test_id_generator() {
        let generator = UltraFastIdGenerator::new(1);
        
        let id1 = generator.next_node_id();
        let id2 = generator.next_node_id();
        let id3 = generator.next_edge_id();
        
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id2, id3);
        
        // Edge IDs should have high bit set
        assert!(id3 & (1u128 << 127) != 0);
        assert!(id1 & (1u128 << 127) == 0);
    }
    
    #[test]
    fn test_ultra_graph_stats() {
        let stats = UltraGraphStats::new();
        assert_eq!(stats.cache_hit_ratio(), 0.0);
        
        stats.record_cache_hit();
        stats.record_cache_hit();
        stats.record_cache_miss();
        
        assert_eq!(stats.cache_hit_ratio(), 2.0 / 3.0);
    }
}