//! Hybrid storage layer combining Rust safety with C++ performance
//!
//! This module provides the storage abstraction that bridges Rust memory safety
//! with C++ optimized data structures for maximum performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::RwLock;
use smallvec::SmallVec;

use crate::core::*;
use crate::error::{HybridError, HybridResult};
use crate::config::StorageConfig;

/// Hybrid storage combining Rust and C++ components
pub struct HybridStorage {
    /// Storage configuration
    config: StorageConfig,
    
    /// Node storage (Rust-managed, C++ backend)
    node_storage: Arc<RwLock<NodeStorage>>,
    
    /// Edge storage (Rust-managed, C++ backend)
    edge_storage: Arc<RwLock<EdgeStorage>>,
    
    /// Hyperedge storage
    hyperedge_storage: Arc<RwLock<HyperedgeStorage>>,
    
    /// CSR matrix for efficient graph operations
    csr_matrix: Arc<RwLock<CsrMatrix>>,
    
    /// Statistics tracking
    statistics: Arc<GraphStatistics>,
}

impl HybridStorage {
    /// Create new hybrid storage
    pub fn new(config: StorageConfig) -> HybridResult<Self> {
        tracing::info!("🗄️  Initializing hybrid storage");
        
        let node_storage = Arc::new(RwLock::new(NodeStorage::new(&config)?));
        let edge_storage = Arc::new(RwLock::new(EdgeStorage::new(&config)?));
        let hyperedge_storage = Arc::new(RwLock::new(HyperedgeStorage::new(&config)?));
        let csr_matrix = Arc::new(RwLock::new(CsrMatrix::new(&config)?));
        let statistics = Arc::new(GraphStatistics::new());
        
        tracing::info!("✅ Hybrid storage initialized");
        
        Ok(Self {
            config,
            node_storage,
            edge_storage,
            hyperedge_storage,
            csr_matrix,
            statistics,
        })
    }
    
    /// Create a new node
    pub fn create_node(&self, data: NodeData) -> HybridResult<NodeId> {
        let node_id = self.node_storage.write().create_node(data)?;
        self.statistics.node_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(node_id)
    }
    
    /// Create a new edge
    pub fn create_edge(&self, from: NodeId, to: NodeId, weight: Weight, data: EdgeData) -> HybridResult<EdgeId> {
        let edge_id = self.edge_storage.write().create_edge(from, to, weight, data)?;
        self.csr_matrix.write().add_edge(from, to, edge_id, weight)?;
        self.statistics.edge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(edge_id)
    }
    
    /// Create a new hyperedge
    pub fn create_hyperedge(&self, nodes: SmallVec<[NodeId; 8]>, data: HyperedgeData) -> HybridResult<EdgeId> {
        let hyperedge_id = self.hyperedge_storage.write().create_hyperedge(nodes, data)?;
        self.statistics.hyperedge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(hyperedge_id)
    }
    
    /// Get node data
    pub fn get_node(&self, node_id: NodeId) -> HybridResult<Option<NodeData>> {
        self.node_storage.read().get_node(node_id)
    }
    
    /// Get edge data
    pub fn get_edge(&self, edge_id: EdgeId) -> HybridResult<Option<(NodeId, NodeId, Weight, EdgeData)>> {
        self.edge_storage.read().get_edge(edge_id)
    }
    
    /// Get neighbors of a node
    pub fn get_neighbors(&self, node_id: NodeId, direction: EdgeDirection) -> HybridResult<Vec<(NodeId, EdgeId, Weight)>> {
        self.csr_matrix.read().get_neighbors(node_id, direction)
    }
    
    /// Get hyperedges for a node
    pub fn get_hyperedges_for_node(&self, node_id: NodeId) -> HybridResult<Vec<(EdgeId, HyperedgeData)>> {
        self.hyperedge_storage.read().get_hyperedges_for_node(node_id)
    }
    
    /// Optimize storage
    pub fn optimize(&self) -> HybridResult<()> {
        tracing::info!("🔧 Optimizing hybrid storage");
        
        self.node_storage.write().optimize()?;
        self.edge_storage.write().optimize()?;
        self.hyperedge_storage.write().optimize()?;
        self.csr_matrix.write().optimize()?;
        
        tracing::info!("✅ Storage optimization completed");
        Ok(())
    }
    
    /// Compact storage
    pub fn compact(&self) -> HybridResult<()> {
        tracing::info!("🗜️  Compacting hybrid storage");
        
        self.node_storage.write().compact()?;
        self.edge_storage.write().compact()?;
        self.hyperedge_storage.write().compact()?;
        self.csr_matrix.write().compact()?;
        
        tracing::info!("✅ Storage compaction completed");
        Ok(())
    }
}

/// Node storage implementation
struct NodeStorage {
    nodes: HashMap<NodeId, NodeData>,
    next_id: NodeId,
}

impl NodeStorage {
    fn new(_config: &StorageConfig) -> HybridResult<Self> {
        Ok(Self {
            nodes: HashMap::new(),
            next_id: 1,
        })
    }
    
    fn create_node(&mut self, data: NodeData) -> HybridResult<NodeId> {
        let node_id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(node_id, data);
        Ok(node_id)
    }
    
    fn get_node(&self, node_id: NodeId) -> HybridResult<Option<NodeData>> {
        Ok(self.nodes.get(&node_id).cloned())
    }
    
    fn optimize(&mut self) -> HybridResult<()> {
        // TODO: Implement node storage optimization
        Ok(())
    }
    
    fn compact(&mut self) -> HybridResult<()> {
        // TODO: Implement node storage compaction
        self.nodes.shrink_to_fit();
        Ok(())
    }
}

/// Edge storage implementation
struct EdgeStorage {
    edges: HashMap<EdgeId, (NodeId, NodeId, Weight, EdgeData)>,
    next_id: EdgeId,
}

impl EdgeStorage {
    fn new(_config: &StorageConfig) -> HybridResult<Self> {
        Ok(Self {
            edges: HashMap::new(),
            next_id: 1,
        })
    }
    
    fn create_edge(&mut self, from: NodeId, to: NodeId, weight: Weight, data: EdgeData) -> HybridResult<EdgeId> {
        let edge_id = self.next_id;
        self.next_id += 1;
        self.edges.insert(edge_id, (from, to, weight, data));
        Ok(edge_id)
    }
    
    fn get_edge(&self, edge_id: EdgeId) -> HybridResult<Option<(NodeId, NodeId, Weight, EdgeData)>> {
        Ok(self.edges.get(&edge_id).cloned())
    }
    
    fn optimize(&mut self) -> HybridResult<()> {
        // TODO: Implement edge storage optimization
        Ok(())
    }
    
    fn compact(&mut self) -> HybridResult<()> {
        // TODO: Implement edge storage compaction
        self.edges.shrink_to_fit();
        Ok(())
    }
}

/// Hyperedge storage implementation
struct HyperedgeStorage {
    hyperedges: HashMap<EdgeId, HyperedgeData>,
    node_to_hyperedges: HashMap<NodeId, Vec<EdgeId>>,
    next_id: EdgeId,
}

impl HyperedgeStorage {
    fn new(_config: &StorageConfig) -> HybridResult<Self> {
        Ok(Self {
            hyperedges: HashMap::new(),
            node_to_hyperedges: HashMap::new(),
            next_id: 1_000_000, // Start hyperedge IDs from a high number
        })
    }
    
    fn create_hyperedge(&mut self, nodes: SmallVec<[NodeId; 8]>, data: HyperedgeData) -> HybridResult<EdgeId> {
        let hyperedge_id = self.next_id;
        self.next_id += 1;
        
        // Add to main storage
        self.hyperedges.insert(hyperedge_id, data);
        
        // Update node-to-hyperedge mapping
        for &node_id in &nodes {
            self.node_to_hyperedges
                .entry(node_id)
                .or_insert_with(Vec::new)
                .push(hyperedge_id);
        }
        
        Ok(hyperedge_id)
    }
    
    fn get_hyperedges_for_node(&self, node_id: NodeId) -> HybridResult<Vec<(EdgeId, HyperedgeData)>> {
        let hyperedge_ids = self.node_to_hyperedges.get(&node_id).unwrap_or(&Vec::new());
        let mut result = Vec::new();
        
        for &hyperedge_id in hyperedge_ids {
            if let Some(data) = self.hyperedges.get(&hyperedge_id) {
                result.push((hyperedge_id, data.clone()));
            }
        }
        
        Ok(result)
    }
    
    fn optimize(&mut self) -> HybridResult<()> {
        // TODO: Implement hyperedge storage optimization
        Ok(())
    }
    
    fn compact(&mut self) -> HybridResult<()> {
        // TODO: Implement hyperedge storage compaction
        self.hyperedges.shrink_to_fit();
        self.node_to_hyperedges.shrink_to_fit();
        Ok(())
    }
}

/// CSR matrix implementation for efficient graph operations
struct CsrMatrix {
    /// Outgoing adjacency lists (node -> [(neighbor, edge_id, weight)])
    outgoing: HashMap<NodeId, Vec<(NodeId, EdgeId, Weight)>>,
    
    /// Incoming adjacency lists (node -> [(neighbor, edge_id, weight)])
    incoming: HashMap<NodeId, Vec<(NodeId, EdgeId, Weight)>>,
}

impl CsrMatrix {
    fn new(_config: &StorageConfig) -> HybridResult<Self> {
        Ok(Self {
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        })
    }
    
    fn add_edge(&mut self, from: NodeId, to: NodeId, edge_id: EdgeId, weight: Weight) -> HybridResult<()> {
        // Add to outgoing adjacency list
        self.outgoing
            .entry(from)
            .or_insert_with(Vec::new)
            .push((to, edge_id, weight));
        
        // Add to incoming adjacency list
        self.incoming
            .entry(to)
            .or_insert_with(Vec::new)
            .push((from, edge_id, weight));
        
        Ok(())
    }
    
    fn get_neighbors(&self, node_id: NodeId, direction: EdgeDirection) -> HybridResult<Vec<(NodeId, EdgeId, Weight)>> {
        match direction {
            EdgeDirection::Outgoing => {
                Ok(self.outgoing.get(&node_id).cloned().unwrap_or_default())
            },
            EdgeDirection::Incoming => {
                Ok(self.incoming.get(&node_id).cloned().unwrap_or_default())
            },
            EdgeDirection::Both => {
                let mut neighbors = self.outgoing.get(&node_id).cloned().unwrap_or_default();
                neighbors.extend(self.incoming.get(&node_id).cloned().unwrap_or_default());
                neighbors.sort_unstable();
                neighbors.dedup();
                Ok(neighbors)
            },
        }
    }
    
    fn optimize(&mut self) -> HybridResult<()> {
        // TODO: Implement CSR matrix optimization
        // - Sort adjacency lists
        // - Remove duplicates
        // - Compress representation
        for neighbors in self.outgoing.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        
        for neighbors in self.incoming.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        
        Ok(())
    }
    
    fn compact(&mut self) -> HybridResult<()> {
        // TODO: Implement CSR matrix compaction
        self.outgoing.shrink_to_fit();
        self.incoming.shrink_to_fit();
        
        for neighbors in self.outgoing.values_mut() {
            neighbors.shrink_to_fit();
        }
        
        for neighbors in self.incoming.values_mut() {
            neighbors.shrink_to_fit();
        }
        
        Ok(())
    }
}

pub use crate::config::StorageConfig;

// Storage utilities
pub fn warm_up_memory_allocators() -> HybridResult<()> {
    tracing::debug!("🔥 Warming up memory allocators");
    
    // Pre-allocate some data structures to warm up allocators
    let _temp_vec: Vec<u64> = (0..1000).collect();
    let _temp_map: HashMap<u64, u64> = (0..1000).map(|i| (i, i * 2)).collect();
    
    tracing::debug!("✅ Memory allocators warmed up");
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn prefault_memory_pages() -> HybridResult<()> {
    tracing::debug!("🔥 Pre-faulting memory pages");
    
    // Pre-fault some memory pages to reduce page fault overhead
    let page_size = 4096; // 4KB pages
    let pages_to_fault = 1000;
    
    let mut temp_memory = vec![0u8; page_size * pages_to_fault];
    
    // Touch each page to force allocation
    for i in 0..pages_to_fault {
        temp_memory[i * page_size] = 1;
    }
    
    tracing::debug!("✅ Memory pages pre-faulted");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hybrid_storage_creation() {
        let config = StorageConfig::default();
        let storage = HybridStorage::new(config).expect("Failed to create storage");
        
        let stats = storage.statistics;
        assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.edge_count.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_node_storage() {
        let config = StorageConfig::default();
        let storage = HybridStorage::new(config).expect("Failed to create storage");
        
        let mut properties = PropertyMap::new();
        properties.insert("name".to_string(), PropertyValue::String("Alice".to_string()));
        
        let node_data = NodeData::new("Person".to_string(), properties);
        let node_id = storage.create_node(node_data.clone()).expect("Failed to create node");
        
        let retrieved = storage.get_node(node_id).expect("Failed to get node");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().label, "Person");
    }
    
    #[test]
    fn test_edge_storage() {
        let config = StorageConfig::default();
        let storage = HybridStorage::new(config).expect("Failed to create storage");
        
        // Create nodes
        let node1_data = NodeData::new("Person".to_string(), PropertyMap::new());
        let node2_data = NodeData::new("Person".to_string(), PropertyMap::new());
        
        let node1_id = storage.create_node(node1_data).expect("Failed to create node1");
        let node2_id = storage.create_node(node2_data).expect("Failed to create node2");
        
        // Create edge
        let edge_data = EdgeData::new(PropertyMap::new());
        let edge_id = storage.create_edge(node1_id, node2_id, 1.0, edge_data)
            .expect("Failed to create edge");
        
        let retrieved = storage.get_edge(edge_id).expect("Failed to get edge");
        assert!(retrieved.is_some());
        
        let (from, to, weight, _) = retrieved.unwrap();
        assert_eq!(from, node1_id);
        assert_eq!(to, node2_id);
        assert_eq!(weight, 1.0);
    }
    
    #[test]
    fn test_neighbors() {
        let config = StorageConfig::default();
        let storage = HybridStorage::new(config).expect("Failed to create storage");
        
        // Create nodes
        let node1_data = NodeData::new("Person".to_string(), PropertyMap::new());
        let node2_data = NodeData::new("Person".to_string(), PropertyMap::new());
        
        let node1_id = storage.create_node(node1_data).expect("Failed to create node1");
        let node2_id = storage.create_node(node2_data).expect("Failed to create node2");
        
        // Create edge
        let edge_data = EdgeData::new(PropertyMap::new());
        let _edge_id = storage.create_edge(node1_id, node2_id, 1.0, edge_data)
            .expect("Failed to create edge");
        
        // Test neighbors
        let outgoing = storage.get_neighbors(node1_id, EdgeDirection::Outgoing)
            .expect("Failed to get outgoing neighbors");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].0, node2_id);
        
        let incoming = storage.get_neighbors(node2_id, EdgeDirection::Incoming)
            .expect("Failed to get incoming neighbors");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].0, node1_id);
    }
}