//! Main hybrid knowledge graph implementation
//!
//! This module implements the core HybridKnowledgeGraph that orchestrates
//! Rust, C++, and Assembly components for maximum performance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::core::*;
use crate::error::{HybridError, HybridResult};
use crate::metrics::PerformanceMetrics;
use crate::storage::{HybridStorage, StorageConfig};
use crate::algorithms::{HybridAlgorithms, AlgorithmConfig};
use crate::query::{HybridQueryEngine, QueryConfig};
use crate::simd::SIMDOperations;

/// Main hybrid knowledge graph combining Rust safety with C++ performance
pub struct HybridKnowledgeGraph {
    /// Graph configuration
    config: HybridGraphConfig,
    
    /// Hybrid storage layer (Rust + C++ bridge)
    storage: Arc<HybridStorage>,
    
    /// Algorithm implementations (Rust + C++ + Assembly)
    algorithms: Arc<HybridAlgorithms>,
    
    /// Query engine with cost-based optimization
    query_engine: Arc<HybridQueryEngine>,
    
    /// SIMD operation dispatcher
    simd_ops: Arc<SIMDOperations>,
    
    /// Graph statistics with atomic counters
    statistics: Arc<GraphStatistics>,
    
    /// Read-write lock for structural changes
    structure_lock: RwLock<()>,
}

/// Configuration for hybrid knowledge graph
#[derive(Debug, Clone)]
pub struct HybridGraphConfig {
    /// Initial node capacity
    pub initial_node_capacity: usize,
    
    /// Initial edge capacity
    pub initial_edge_capacity: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Enable huge pages for memory allocation
    pub enable_huge_pages: bool,
    
    /// Thread pool size (0 = auto-detect)
    pub thread_pool_size: usize,
    
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Enable distributed mode
    pub enable_distributed: bool,
    
    /// Storage configuration
    pub storage_config: StorageConfig,
    
    /// Algorithm configuration
    pub algorithm_config: AlgorithmConfig,
    
    /// Query engine configuration
    pub query_config: QueryConfig,
}

impl Default for HybridGraphConfig {
    fn default() -> Self {
        Self {
            initial_node_capacity: 1_000_000,
            initial_edge_capacity: 10_000_000,
            enable_simd: true,
            enable_huge_pages: true,
            thread_pool_size: 0,
            memory_limit: None,
            enable_gpu: false,
            enable_distributed: false,
            storage_config: StorageConfig::default(),
            algorithm_config: AlgorithmConfig::default(),
            query_config: QueryConfig::default(),
        }
    }
}

impl HybridKnowledgeGraph {
    /// Create a new hybrid knowledge graph with configuration
    pub fn new(config: HybridGraphConfig) -> HybridResult<Self> {
        tracing::info!("🚀 Initializing Hybrid Knowledge Graph");
        
        // Initialize thread pool
        let thread_pool_size = if config.thread_pool_size == 0 {
            std::thread::available_parallelism()?.get()
        } else {
            config.thread_pool_size
        };
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_pool_size)
            .thread_name(|index| format!("hybrid-kg-{}", index))
            .build_global()?;
        
        tracing::info!("✅ Thread pool initialized with {} threads", thread_pool_size);
        
        // Initialize storage layer
        let storage = Arc::new(HybridStorage::new(config.storage_config.clone())?);
        tracing::info!("✅ Hybrid storage initialized");
        
        // Initialize algorithm implementations
        let algorithms = Arc::new(HybridAlgorithms::new(
            config.algorithm_config.clone(),
            Arc::clone(&storage)
        )?);
        tracing::info!("✅ Hybrid algorithms initialized");
        
        // Initialize query engine
        let query_engine = Arc::new(HybridQueryEngine::new(
            config.query_config.clone(),
            Arc::clone(&storage),
            Arc::clone(&algorithms)
        )?);
        tracing::info!("✅ Query engine initialized");
        
        // Initialize SIMD operations
        let simd_ops = Arc::new(SIMDOperations::new(config.enable_simd)?);
        tracing::info!("✅ SIMD operations initialized (width: {})", 
                      crate::cpu_features::SIMD_WIDTH);
        
        // Initialize statistics
        let statistics = Arc::new(GraphStatistics::new());
        
        Ok(Self {
            config,
            storage,
            algorithms,
            query_engine,
            simd_ops,
            statistics,
            structure_lock: RwLock::new(()),
        })
    }
    
    /// Create a node with data and return its ID
    pub fn create_node(&self, data: NodeData) -> HybridResult<NodeId> {
        let start = Instant::now();
        
        let node_id = self.storage.create_node(data)?;
        
        // Update statistics
        self.statistics.node_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Update average operation time
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        tracing::debug!("Created node {} in {}μs", node_id, duration_ns / 1000);
        
        Ok(node_id)
    }
    
    /// Create multiple nodes in batch for optimal performance
    pub fn batch_create_nodes(&self, nodes: Vec<NodeData>) -> HybridResult<Vec<NodeId>> {
        let start = Instant::now();
        let batch_size = nodes.len();
        
        tracing::info!("Creating batch of {} nodes", batch_size);
        
        // Use parallel processing for large batches
        let node_ids = if batch_size > 1000 {
            // Parallel batch creation using rayon
            nodes.into_par_iter()
                .map(|data| self.storage.create_node(data))
                .collect::<HybridResult<Vec<_>>>()?
        } else {
            // Sequential for small batches to avoid overhead
            let mut node_ids = Vec::with_capacity(batch_size);
            for data in nodes {
                node_ids.push(self.storage.create_node(data)?);
            }
            node_ids
        };
        
        // Update statistics
        self.statistics.node_count.fetch_add(batch_size as u64, std::sync::atomic::Ordering::Relaxed);
        self.statistics.operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration = start.elapsed();
        let nodes_per_sec = (batch_size as f64 / duration.as_secs_f64()) as u64;
        
        tracing::info!("✅ Created {} nodes in {}ms ({} nodes/sec)", 
                      batch_size, duration.as_millis(), nodes_per_sec);
        
        Ok(node_ids)
    }
    
    /// Create an edge between two nodes
    pub fn create_edge(&self, from: NodeId, to: NodeId, weight: Weight, data: EdgeData) -> HybridResult<EdgeId> {
        let start = Instant::now();
        
        let edge_id = self.storage.create_edge(from, to, weight, data)?;
        
        // Update statistics
        self.statistics.edge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        tracing::debug!("Created edge {} ({} -> {}) in {}μs", edge_id, from, to, duration_ns / 1000);
        
        Ok(edge_id)
    }
    
    /// Get node data by ID
    pub fn get_node(&self, node_id: NodeId) -> HybridResult<Option<NodeData>> {
        let start = Instant::now();
        
        let result = self.storage.get_node(node_id)?;
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        Ok(result)
    }
    
    /// Get edge data by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> HybridResult<Option<(NodeId, NodeId, Weight, EdgeData)>> {
        let start = Instant::now();
        
        let result = self.storage.get_edge(edge_id)?;
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        Ok(result)
    }
    
    /// Get neighbors of a node in a specific direction
    pub fn get_neighbors(&self, node_id: NodeId, direction: EdgeDirection) -> HybridResult<Vec<(NodeId, EdgeId, Weight)>> {
        let start = Instant::now();
        
        let neighbors = self.storage.get_neighbors(node_id, direction)?;
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        tracing::debug!("Retrieved {} neighbors for node {} in {}μs", 
                       neighbors.len(), node_id, duration_ns / 1000);
        
        Ok(neighbors)
    }
    
    /// Perform breadth-first search traversal
    pub fn traverse_bfs(&self, start_node: NodeId, max_depth: u32) -> HybridResult<TraversalResult> {
        let start = Instant::now();
        
        let result = self.algorithms.breadth_first_search(start_node, max_depth)?;
        
        // Update statistics
        self.statistics.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.simd_operations.fetch_add(
            result.simd_operations as u64, 
            std::sync::atomic::Ordering::Relaxed
        );
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        tracing::info!("BFS traversal: {} nodes, {} edges in {}ms", 
                      result.nodes_visited, result.edges_traversed, duration_ns / 1_000_000);
        
        Ok(result)
    }
    
    /// Find shortest path between two nodes using SIMD-optimized Dijkstra
    pub fn shortest_path(&self, from: NodeId, to: NodeId) -> HybridResult<Option<Path>> {
        let start = Instant::now();
        
        let result = self.algorithms.shortest_path(from, to)?;
        
        // Update statistics
        self.statistics.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        if let Some(ref path) = result {
            tracing::info!("Shortest path: {} hops, weight {:.3} in {}μs", 
                          path.length, path.total_weight, duration_ns / 1000);
        }
        
        Ok(result)
    }
    
    /// Compute centrality measures using SIMD optimization
    pub fn compute_centrality(&self, algorithm: CentralityAlgorithm) -> HybridResult<HashMap<NodeId, f64>> {
        let start = Instant::now();
        
        tracing::info!("Computing {:?} centrality", algorithm);
        
        let result = self.algorithms.compute_centrality(algorithm)?;
        
        // Update statistics
        self.statistics.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration = start.elapsed();
        tracing::info!("✅ Centrality computed for {} nodes in {}ms", 
                      result.len(), duration.as_millis());
        
        Ok(result)
    }
    
    /// Detect communities using advanced algorithms
    pub fn detect_communities(&self, algorithm: CommunityAlgorithm) -> HybridResult<Vec<Vec<NodeId>>> {
        let start = Instant::now();
        
        tracing::info!("Detecting communities using {:?}", algorithm);
        
        let result = self.algorithms.detect_communities(algorithm)?;
        
        let duration = start.elapsed();
        tracing::info!("✅ Detected {} communities in {}ms", 
                      result.len(), duration.as_millis());
        
        Ok(result)
    }
    
    /// Find pattern matches in the graph
    pub fn find_pattern(&self, pattern: Pattern) -> HybridResult<Vec<PatternMatch>> {
        let start = Instant::now();
        
        tracing::info!("Finding pattern with {} nodes, {} edges", 
                      pattern.nodes.len(), pattern.edges.len());
        
        let result = self.query_engine.find_pattern(pattern)?;
        
        // Update statistics
        self.statistics.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration = start.elapsed();
        tracing::info!("✅ Found {} pattern matches in {}ms", 
                      result.len(), duration.as_millis());
        
        Ok(result)
    }
    
    /// Create a hyperedge connecting multiple nodes
    pub fn create_hyperedge(&self, nodes: SmallVec<[NodeId; 8]>, data: HyperedgeData) -> HybridResult<EdgeId> {
        let start = Instant::now();
        
        let hyperedge_id = self.storage.create_hyperedge(nodes, data)?;
        
        // Update statistics
        self.statistics.hyperedge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_average_time(duration_ns);
        
        tracing::debug!("Created hyperedge {} in {}μs", hyperedge_id, duration_ns / 1000);
        
        Ok(hyperedge_id)
    }
    
    /// Get hyperedges connected to a specific node
    pub fn get_hyperedges_for_node(&self, node_id: NodeId) -> HybridResult<Vec<(EdgeId, HyperedgeData)>> {
        self.storage.get_hyperedges_for_node(node_id)
    }
    
    /// Get comprehensive graph statistics
    pub fn get_statistics(&self) -> Arc<GraphStatistics> {
        Arc::clone(&self.statistics)
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> HybridResult<PerformanceMetrics> {
        Ok(PerformanceMetrics {
            node_count: self.statistics.node_count.load(std::sync::atomic::Ordering::Relaxed),
            edge_count: self.statistics.edge_count.load(std::sync::atomic::Ordering::Relaxed),
            hyperedge_count: self.statistics.hyperedge_count.load(std::sync::atomic::Ordering::Relaxed),
            memory_usage: self.statistics.memory_usage.load(std::sync::atomic::Ordering::Relaxed),
            operations_performed: self.statistics.operations_performed.load(std::sync::atomic::Ordering::Relaxed),
            queries_executed: self.statistics.queries_executed.load(std::sync::atomic::Ordering::Relaxed),
            average_query_time: Duration::from_nanos(
                self.statistics.average_query_time_ns.load(std::sync::atomic::Ordering::Relaxed)
            ),
            cache_hit_ratio: self.statistics.cache_hit_ratio(),
            simd_operations: self.statistics.simd_operations.load(std::sync::atomic::Ordering::Relaxed),
            uptime: self.statistics.uptime(),
            simd_width: crate::cpu_features::SIMD_WIDTH,
            has_avx512: crate::cpu_features::HAS_AVX512,
            thread_count: rayon::current_num_threads(),
        })
    }
    
    /// Optimize the graph for current workload
    pub fn optimize(&mut self) -> HybridResult<()> {
        let start = Instant::now();
        
        tracing::info!("🔧 Optimizing graph for current workload");
        
        // Acquire write lock for structural changes
        let _lock = self.structure_lock.write();
        
        // Optimize storage layer
        self.storage.optimize()?;
        
        // Optimize algorithms
        Arc::get_mut(&mut self.algorithms)
            .ok_or_else(|| HybridError::ConcurrentAccess("Cannot optimize algorithms while in use".into()))?
            .optimize()?;
        
        // Optimize query engine
        Arc::get_mut(&mut self.query_engine)
            .ok_or_else(|| HybridError::ConcurrentAccess("Cannot optimize query engine while in use".into()))?
            .optimize()?;
        
        let duration = start.elapsed();
        tracing::info!("✅ Graph optimization completed in {}ms", duration.as_millis());
        
        Ok(())
    }
    
    /// Compact and defragment the graph storage
    pub fn compact(&mut self) -> HybridResult<()> {
        let start = Instant::now();
        
        tracing::info!("🗜️  Compacting graph storage");
        
        let _lock = self.structure_lock.write();
        
        let old_memory = self.statistics.memory_usage.load(std::sync::atomic::Ordering::Relaxed);
        
        self.storage.compact()?;
        
        let new_memory = self.statistics.memory_usage.load(std::sync::atomic::Ordering::Relaxed);
        let saved_memory = old_memory.saturating_sub(new_memory);
        let compression_ratio = if old_memory > 0 {
            new_memory as f64 / old_memory as f64
        } else {
            1.0
        };
        
        let duration = start.elapsed();
        tracing::info!("✅ Compaction completed in {}ms, saved {}MB ({:.1}% compression)", 
                      duration.as_millis(), 
                      saved_memory / 1024 / 1024,
                      (1.0 - compression_ratio) * 100.0);
        
        Ok(())
    }
    
    /// Reset performance statistics
    pub fn reset_statistics(&self) {
        self.statistics.reset();
        tracing::info!("📊 Performance statistics reset");
    }
    
    /// Update average query time using exponential moving average
    fn update_average_time(&self, duration_ns: u64) {
        let current_avg = self.statistics.average_query_time_ns.load(std::sync::atomic::Ordering::Relaxed);
        
        // Exponential moving average with alpha = 0.1
        let new_avg = if current_avg == 0 {
            duration_ns
        } else {
            ((current_avg as f64 * 0.9) + (duration_ns as f64 * 0.1)) as u64
        };
        
        self.statistics.average_query_time_ns.store(new_avg, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for HybridKnowledgeGraph {
    fn drop(&mut self) {
        tracing::info!("🔄 Shutting down Hybrid Knowledge Graph");
        
        let stats = self.get_performance_metrics().unwrap_or_default();
        
        tracing::info!("📊 Final Statistics:");
        tracing::info!("  Nodes: {}", stats.node_count);
        tracing::info!("  Edges: {}", stats.edge_count);
        tracing::info!("  Hyperedges: {}", stats.hyperedge_count);
        tracing::info!("  Memory: {}MB", stats.memory_usage / 1024 / 1024);
        tracing::info!("  Operations: {}", stats.operations_performed);
        tracing::info!("  Queries: {}", stats.queries_executed);
        tracing::info!("  Avg Query Time: {}μs", stats.average_query_time.as_micros());
        tracing::info!("  Cache Hit Ratio: {:.1}%", stats.cache_hit_ratio * 100.0);
        tracing::info!("  SIMD Operations: {}", stats.simd_operations);
        tracing::info!("  Uptime: {}s", stats.uptime.as_secs());
        
        tracing::info!("✅ Hybrid Knowledge Graph shutdown complete");
    }
}

/// Trait for components that can report their optimization status
pub trait OptimizationStatus {
    /// Check if the component needs optimization
    fn needs_optimization(&self) -> bool;
    
    /// Get optimization recommendations
    fn get_optimization_recommendations(&self) -> Vec<String>;
}

impl OptimizationStatus for HybridKnowledgeGraph {
    fn needs_optimization(&self) -> bool {
        // Check various conditions that indicate optimization might be beneficial
        let stats = self.statistics.as_ref();
        
        // High memory usage
        if let Some(limit) = self.config.memory_limit {
            let usage = stats.memory_usage.load(std::sync::atomic::Ordering::Relaxed);
            if usage as f64 / limit as f64 > 0.8 {
                return true;
            }
        }
        
        // Low cache hit ratio
        if stats.cache_hit_ratio() < 0.7 {
            return true;
        }
        
        // High average query time
        let avg_time_ms = stats.average_query_time_ns.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        if avg_time_ms > 100 {
            return true;
        }
        
        false
    }
    
    fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let stats = self.statistics.as_ref();
        
        // Memory optimization
        if let Some(limit) = self.config.memory_limit {
            let usage = stats.memory_usage.load(std::sync::atomic::Ordering::Relaxed);
            let usage_ratio = usage as f64 / limit as f64;
            
            if usage_ratio > 0.9 {
                recommendations.push("Critical: Memory usage > 90%, consider compaction or memory limit increase".into());
            } else if usage_ratio > 0.8 {
                recommendations.push("Warning: Memory usage > 80%, compaction recommended".into());
            }
        }
        
        // Cache optimization
        let hit_ratio = stats.cache_hit_ratio();
        if hit_ratio < 0.5 {
            recommendations.push("Critical: Cache hit ratio < 50%, consider memory reorganization".into());
        } else if hit_ratio < 0.7 {
            recommendations.push("Warning: Cache hit ratio < 70%, optimization recommended".into());
        }
        
        // Query performance
        let avg_time_ms = stats.average_query_time_ns.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000;
        if avg_time_ms > 1000 {
            recommendations.push("Critical: Average query time > 1s, index optimization needed".into());
        } else if avg_time_ms > 100 {
            recommendations.push("Warning: Average query time > 100ms, consider query optimization".into());
        }
        
        // SIMD utilization
        let total_ops = stats.operations_performed.load(std::sync::atomic::Ordering::Relaxed);
        let simd_ops = stats.simd_operations.load(std::sync::atomic::Ordering::Relaxed);
        
        if total_ops > 0 {
            let simd_ratio = simd_ops as f64 / total_ops as f64;
            if simd_ratio < 0.3 {
                recommendations.push("Info: Low SIMD utilization, consider enabling more vectorized operations".into());
            }
        }
        
        if recommendations.is_empty() {
            recommendations.push("All systems operating optimally".into());
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hybrid_graph_creation() {
        let config = HybridGraphConfig::default();
        let graph = HybridKnowledgeGraph::new(config).expect("Failed to create graph");
        
        let stats = graph.get_statistics();
        assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(stats.edge_count.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    
    #[tokio::test]
    async fn test_node_creation() {
        let config = HybridGraphConfig::default();
        let graph = HybridKnowledgeGraph::new(config).expect("Failed to create graph");
        
        let mut properties = PropertyMap::new();
        properties.insert("name".to_string(), PropertyValue::String("Alice".to_string()));
        properties.insert("age".to_string(), PropertyValue::Int32(30));
        
        let node_data = NodeData::new("Person".to_string(), properties);
        let node_id = graph.create_node(node_data).expect("Failed to create node");
        
        assert!(node_id > 0);
        
        let stats = graph.get_statistics();
        assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_batch_node_creation() {
        let config = HybridGraphConfig::default();
        let graph = HybridKnowledgeGraph::new(config).expect("Failed to create graph");
        
        let mut nodes = Vec::new();
        for i in 0..1000 {
            let mut properties = PropertyMap::new();
            properties.insert("id".to_string(), PropertyValue::Int32(i));
            properties.insert("name".to_string(), PropertyValue::String(format!("Node_{}", i)));
            
            nodes.push(NodeData::new("TestNode".to_string(), properties));
        }
        
        let node_ids = graph.batch_create_nodes(nodes).expect("Failed to create batch nodes");
        assert_eq!(node_ids.len(), 1000);
        
        let stats = graph.get_statistics();
        assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 1000);
    }
    
    #[tokio::test]
    async fn test_edge_creation() {
        let config = HybridGraphConfig::default();
        let graph = HybridKnowledgeGraph::new(config).expect("Failed to create graph");
        
        // Create two nodes
        let node1_data = NodeData::new("Person".to_string(), PropertyMap::new());
        let node2_data = NodeData::new("Person".to_string(), PropertyMap::new());
        
        let node1_id = graph.create_node(node1_data).expect("Failed to create node1");
        let node2_id = graph.create_node(node2_data).expect("Failed to create node2");
        
        // Create edge
        let mut edge_properties = PropertyMap::new();
        edge_properties.insert("type".to_string(), PropertyValue::String("KNOWS".to_string()));
        
        let edge_data = EdgeData::new(edge_properties);
        let edge_id = graph.create_edge(node1_id, node2_id, 1.0, edge_data)
            .expect("Failed to create edge");
        
        assert!(edge_id > 0);
        
        let stats = graph.get_statistics();
        assert_eq!(stats.edge_count.load(std::sync::atomic::Ordering::Relaxed), 1);
    }
    
    #[tokio::test]
    async fn test_optimization_status() {
        let config = HybridGraphConfig::default();
        let graph = HybridKnowledgeGraph::new(config).expect("Failed to create graph");
        
        // Initially should not need optimization
        assert!(!graph.needs_optimization());
        
        let recommendations = graph.get_optimization_recommendations();
        assert!(!recommendations.is_empty());
        assert_eq!(recommendations[0], "All systems operating optimally");
    }
}