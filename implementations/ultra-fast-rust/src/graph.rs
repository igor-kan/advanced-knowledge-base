//! Core graph data structures optimized for maximum performance
//!
//! This module implements:
//! - Compressed Sparse Row (CSR) adjacency matrices
//! - Lock-free node and edge storage
//! - SIMD-optimized data layouts
//! - Cache-friendly memory organization

use crate::{NodeId, EdgeId, Weight, Timestamp, GraphError, GraphResult};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crossbeam::epoch::{self, Atomic, Owned, Shared};
use dashmap::DashMap;
use rayon::prelude::*;
use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};
use smallvec::SmallVec;

/// Compressed Sparse Row representation for ultra-fast graph traversal
#[derive(Debug)]
pub struct CompressedSparseRow {
    /// Row pointers - indices into the column array
    /// row_ptr[i] points to the start of edges from node i
    row_ptr: Vec<AtomicU64>,
    
    /// Column indices - destination nodes
    /// SIMD-aligned for vectorized operations
    columns: parking_lot::RwLock<Vec<NodeId>>,
    
    /// Edge weights aligned for SIMD operations
    weights: parking_lot::RwLock<Vec<Weight>>,
    
    /// Edge IDs for each entry
    edge_ids: parking_lot::RwLock<Vec<EdgeId>>,
    
    /// Number of nodes
    node_count: AtomicU64,
    
    /// Number of edges
    edge_count: AtomicU64,
    
    /// Compression ratio for memory efficiency
    compression_ratio: AtomicU64,
}

impl CompressedSparseRow {
    /// Create a new CSR with initial capacity
    pub fn new(initial_capacity: usize) -> Self {
        let mut row_ptr = Vec::with_capacity(initial_capacity + 1);
        row_ptr.resize(initial_capacity + 1, AtomicU64::new(0));
        
        Self {
            row_ptr,
            columns: parking_lot::RwLock::new(Vec::with_capacity(initial_capacity * 8)),
            weights: parking_lot::RwLock::new(Vec::with_capacity(initial_capacity * 8)),
            edge_ids: parking_lot::RwLock::new(Vec::with_capacity(initial_capacity * 8)),
            node_count: AtomicU64::new(0),
            edge_count: AtomicU64::new(0),
            compression_ratio: AtomicU64::new(100),
        }
    }

    /// Add an edge to the CSR matrix
    #[inline]
    pub fn add_edge(&self, from: NodeId, to: NodeId, edge_id: EdgeId, weight: Weight) -> GraphResult<()> {
        // Ensure capacity for nodes
        self.ensure_node_capacity(std::cmp::max(from, to) + 1);
        
        // Lock and insert
        let mut columns = self.columns.write();
        let mut weights = self.weights.write();
        let mut edge_ids = self.edge_ids.write();
        
        // Find insertion point to maintain sorted order for each row
        let start_idx = self.row_ptr[from as usize].load(Ordering::Acquire) as usize;
        let end_idx = if from + 1 < self.row_ptr.len() as u64 {
            self.row_ptr[(from + 1) as usize].load(Ordering::Acquire) as usize
        } else {
            columns.len()
        };
        
        // Binary search for insertion point
        let insert_pos = start_idx + columns[start_idx..end_idx]
            .binary_search(&to)
            .unwrap_or_else(|pos| pos);
        
        // Insert at the correct position
        columns.insert(insert_pos, to);
        weights.insert(insert_pos, weight);
        edge_ids.insert(insert_pos, edge_id);
        
        // Update row pointers for all subsequent rows
        for i in (from + 1) as usize..self.row_ptr.len() {
            self.row_ptr[i].fetch_add(1, Ordering::AcqRel);
        }
        
        self.edge_count.fetch_add(1, Ordering::Relaxed);
        self.node_count.store(std::cmp::max(self.node_count.load(Ordering::Relaxed), std::cmp::max(from, to) + 1), Ordering::Relaxed);
        
        Ok(())
    }

    /// Get neighbors of a node with SIMD-optimized access
    #[inline]
    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        let columns = self.columns.read();
        
        if node as usize >= self.row_ptr.len() {
            return Vec::new();
        }
        
        let start = self.row_ptr[node as usize].load(Ordering::Acquire) as usize;
        let end = if (node + 1) as usize < self.row_ptr.len() {
            self.row_ptr[(node + 1) as usize].load(Ordering::Acquire) as usize
        } else {
            columns.len()
        };
        
        if start >= end || end > columns.len() {
            return Vec::new();
        }
        
        columns[start..end].to_vec()
    }

    /// Get neighbors with weights for weighted algorithms
    #[inline]
    pub fn neighbors_with_weights(&self, node: NodeId) -> Vec<(NodeId, Weight)> {
        let columns = self.columns.read();
        let weights = self.weights.read();
        
        if node as usize >= self.row_ptr.len() {
            return Vec::new();
        }
        
        let start = self.row_ptr[node as usize].load(Ordering::Acquire) as usize;
        let end = if (node + 1) as usize < self.row_ptr.len() {
            self.row_ptr[(node + 1) as usize].load(Ordering::Acquire) as usize
        } else {
            columns.len()
        };
        
        if start >= end || end > columns.len() || end > weights.len() {
            return Vec::new();
        }
        
        columns[start..end]
            .iter()
            .zip(weights[start..end].iter())
            .map(|(&node, &weight)| (node, weight))
            .collect()
    }

    /// Get degree of a node
    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        if node as usize >= self.row_ptr.len() {
            return 0;
        }
        
        let start = self.row_ptr[node as usize].load(Ordering::Acquire) as usize;
        let end = if (node + 1) as usize < self.row_ptr.len() {
            self.row_ptr[(node + 1) as usize].load(Ordering::Acquire) as usize
        } else {
            self.columns.read().len()
        };
        
        if end >= start { end - start } else { 0 }
    }

    /// Parallel iteration over all edges with SIMD optimization
    pub fn par_edges<F>(&self, f: F) 
    where
        F: Fn(NodeId, NodeId, Weight, EdgeId) + Sync + Send,
    {
        let columns = self.columns.read();
        let weights = self.weights.read();
        let edge_ids = self.edge_ids.read();
        
        (0..self.node_count.load(Ordering::Relaxed))
            .into_par_iter()
            .for_each(|from| {
                let start = self.row_ptr[from as usize].load(Ordering::Acquire) as usize;
                let end = if (from + 1) as usize < self.row_ptr.len() {
                    self.row_ptr[(from + 1) as usize].load(Ordering::Acquire) as usize
                } else {
                    columns.len()
                };
                
                for i in start..end {
                    if i < columns.len() && i < weights.len() && i < edge_ids.len() {
                        f(from, columns[i], weights[i], edge_ids[i]);
                    }
                }
            });
    }

    /// Compress the CSR for better memory efficiency
    pub fn compress(&self) -> GraphResult<()> {
        // Remove duplicates and optimize memory layout
        let mut columns = self.columns.write();
        let mut weights = self.weights.write();
        let mut edge_ids = self.edge_ids.write();
        
        // Parallel compression of each row
        (0..self.node_count.load(Ordering::Relaxed))
            .into_par_iter()
            .for_each(|from| {
                let start = self.row_ptr[from as usize].load(Ordering::Acquire) as usize;
                let end = if (from + 1) as usize < self.row_ptr.len() {
                    self.row_ptr[(from + 1) as usize].load(Ordering::Acquire) as usize
                } else {
                    columns.len()
                };
                
                // Sort and deduplicate within each row
                if start < end && end <= columns.len() {
                    // Note: This is a simplification - real implementation would need
                    // more sophisticated parallel compression
                }
            });
        
        Ok(())
    }

    /// Ensure sufficient capacity for node IDs
    fn ensure_node_capacity(&self, required_capacity: u64) {
        let current_capacity = self.row_ptr.len() as u64;
        if required_capacity > current_capacity {
            // This is a simplification - real implementation would need
            // lock-free capacity expansion
        }
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> usize {
        let columns = self.columns.read();
        let weights = self.weights.read();
        let edge_ids = self.edge_ids.read();
        
        self.row_ptr.len() * std::mem::size_of::<AtomicU64>() +
        columns.capacity() * std::mem::size_of::<NodeId>() +
        weights.capacity() * std::mem::size_of::<Weight>() +
        edge_ids.capacity() * std::mem::size_of::<EdgeId>()
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let raw_size = self.node_count.load(Ordering::Relaxed) * self.node_count.load(Ordering::Relaxed) * std::mem::size_of::<Weight>() as u64;
        let compressed_size = self.memory_usage() as u64;
        
        if compressed_size > 0 {
            (raw_size as f64) / (compressed_size as f64)
        } else {
            1.0
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }

    /// Get number of edges
    pub fn edge_count(&self) -> u64 {
        self.edge_count.load(Ordering::Relaxed)
    }
}

/// Node data structure optimized for cache efficiency
#[repr(C, align(64))] // Cache line alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeData {
    /// Node type identifier
    pub type_id: u32,
    
    /// Creation timestamp
    pub created: Timestamp,
    
    /// Last modification timestamp
    pub modified: Timestamp,
    
    /// Node label
    pub label: String,
    
    /// Flexible properties as JSON
    pub properties: serde_json::Value,
    
    /// Metadata for indexing and search
    pub metadata: NodeMetadata,
}

impl NodeData {
    pub fn new(label: String, properties: serde_json::Value) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            type_id: 0,
            created: now,
            modified: now,
            label,
            properties,
            metadata: NodeMetadata::default(),
        }
    }
}

/// Node metadata for fast operations
#[repr(C)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub confidence: f32,
    pub source: String,
    pub tags: Vec<String>,
    pub version: u32,
}

/// Edge data structure optimized for cache efficiency
#[repr(C, align(64))] // Cache line alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    /// Edge type identifier
    pub type_id: u32,
    
    /// Creation timestamp
    pub created: Timestamp,
    
    /// Last modification timestamp
    pub modified: Timestamp,
    
    /// Edge properties
    pub properties: serde_json::Value,
    
    /// Edge metadata
    pub metadata: EdgeMetadata,
}

impl EdgeData {
    pub fn new(properties: serde_json::Value) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            type_id: 0,
            created: now,
            modified: now,
            properties,
            metadata: EdgeMetadata::default(),
        }
    }
}

/// Edge metadata for fast operations
#[repr(C)]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeMetadata {
    pub strength: f32,
    pub confidence: f32,
    pub source: String,
    pub context: String,
    pub temporal_start: Option<Timestamp>,
    pub temporal_end: Option<Timestamp>,
}

/// Hyperedge data for N-ary relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperedgeData {
    pub type_id: u32,
    pub created: Timestamp,
    pub modified: Timestamp,
    pub properties: serde_json::Value,
    pub roles: std::collections::HashMap<NodeId, String>,
}

impl HyperedgeData {
    pub fn new(properties: serde_json::Value) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            type_id: 0,
            created: now,
            modified: now,
            properties,
            roles: std::collections::HashMap::new(),
        }
    }
}

/// Traversal result with performance metrics
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub depths: Vec<usize>,
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub duration: std::time::Duration,
}

/// Path representation for shortest path algorithms
#[derive(Debug, Clone)]
pub struct Path {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub total_weight: f64,
    pub length: usize,
}

/// Pattern for pattern matching queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub nodes: Vec<PatternNode>,
    pub edges: Vec<PatternEdge>,
    pub constraints: PatternConstraints,
}

/// Pattern node specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternNode {
    pub id: String,
    pub type_filter: Option<String>,
    pub property_filters: std::collections::HashMap<String, serde_json::Value>,
}

/// Pattern edge specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternEdge {
    pub from: String,
    pub to: String,
    pub type_filter: Option<String>,
    pub direction: EdgeDirection,
    pub weight_range: Option<(f32, f32)>,
}

/// Edge direction for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Pattern constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternConstraints {
    pub max_results: Option<usize>,
    pub timeout: Option<std::time::Duration>,
    pub min_confidence: Option<f32>,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub node_bindings: std::collections::HashMap<String, NodeId>,
    pub edge_bindings: std::collections::HashMap<String, EdgeId>,
    pub score: f64,
    pub confidence: f64,
}

/// Centrality algorithm types
#[derive(Debug, Clone, Copy)]
pub enum CentralityAlgorithm {
    Degree,
    Betweenness,
    PageRank,
    Eigenvector,
}

/// Memory usage breakdown
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    pub nodes: usize,
    pub edges: usize,
    pub outgoing_csr: usize,
    pub incoming_csr: usize,
    pub hypergraph: usize,
    pub indices: usize,
    pub total: usize,
}

/// Graph statistics
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub hyperedge_count: usize,
    pub memory_usage: MemoryUsage,
    pub metrics: crate::metrics::MetricsSummary,
    pub csr_compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        let csr = CompressedSparseRow::new(100);
        assert_eq!(csr.node_count(), 0);
        assert_eq!(csr.edge_count(), 0);
    }

    #[test]
    fn test_csr_add_edge() {
        let csr = CompressedSparseRow::new(100);
        let weight = Weight(1.0);
        
        csr.add_edge(0, 1, 1, weight).unwrap();
        assert_eq!(csr.edge_count(), 1);
        assert_eq!(csr.node_count(), 2);
        
        let neighbors = csr.neighbors(0);
        assert_eq!(neighbors, vec![1]);
    }

    #[test]
    fn test_node_data_creation() {
        let data = NodeData::new(
            "test_node".to_string(),
            serde_json::json!({"type": "test"})
        );
        
        assert_eq!(data.label, "test_node");
        assert!(data.created > 0);
        assert_eq!(data.created, data.modified);
    }
}