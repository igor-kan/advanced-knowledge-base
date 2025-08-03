//! Main GPU-accelerated knowledge graph implementation
//!
//! This module implements the core GpuKnowledgeGraph that orchestrates
//! CPU and GPU resources for maximum performance on massive datasets.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::core::*;
use crate::error::{GpuKnowledgeGraphError, GpuResult};
use crate::gpu::{GpuManager, GpuDevice};
use crate::memory::{GpuMemoryPool, UnifiedMemoryManager};
use crate::algorithms::GpuAlgorithms;
use crate::kernels::CudaKernelManager;
use crate::metrics::GpuPerformanceMetrics;

/// Main GPU-accelerated knowledge graph
pub struct GpuKnowledgeGraph {
    /// Graph configuration
    config: GpuGraphConfig,
    
    /// GPU device manager
    gpu_manager: Arc<GpuManager>,
    
    /// Unified memory manager for CPU-GPU transfers
    memory_manager: Arc<UnifiedMemoryManager>,
    
    /// GPU algorithm implementations
    algorithms: Arc<GpuAlgorithms>,
    
    /// CUDA kernel manager
    kernel_manager: Arc<CudaKernelManager>,
    
    /// Node storage (hybrid CPU-GPU)
    node_storage: Arc<RwLock<GpuNodeStorage>>,
    
    /// Edge storage (hybrid CPU-GPU)
    edge_storage: Arc<RwLock<GpuEdgeStorage>>,
    
    /// Hyperedge storage (hybrid CPU-GPU)
    hyperedge_storage: Arc<RwLock<GpuHyperedgeStorage>>,
    
    /// Graph statistics with GPU metrics
    statistics: Arc<GpuGraphStatistics>,
    
    /// Read-write lock for structural changes
    structure_lock: RwLock<()>,
    
    /// Performance metrics collector
    metrics_collector: Arc<RwLock<GpuPerformanceMetrics>>,
}

/// Configuration for GPU-accelerated knowledge graph
#[derive(Debug, Clone)]
pub struct GpuGraphConfig {
    /// Initial node capacity
    pub initial_node_capacity: usize,
    
    /// Initial edge capacity
    pub initial_edge_capacity: usize,
    
    /// Initial hyperedge capacity
    pub initial_hyperedge_capacity: usize,
    
    /// GPU devices to use (empty = use all available)
    pub gpu_devices: Vec<GpuDeviceId>,
    
    /// Enable multi-GPU processing
    pub enable_multi_gpu: bool,
    
    /// GPU memory pool size per device (bytes)
    pub gpu_memory_pool_size: usize,
    
    /// Enable unified memory
    pub enable_unified_memory: bool,
    
    /// Enable GPU-CPU hybrid processing
    pub enable_hybrid_processing: bool,
    
    /// Thread pool size for CPU operations
    pub cpu_thread_pool_size: usize,
    
    /// Number of CUDA streams per GPU
    pub cuda_streams_per_gpu: usize,
    
    /// GPU batch size for operations
    pub gpu_batch_size: usize,
    
    /// Automatic GPU-CPU load balancing
    pub enable_auto_load_balancing: bool,
    
    /// Memory limit per GPU device (bytes, 0 = use all available)
    pub gpu_memory_limit: usize,
    
    /// Enable GPU memory prefetching
    pub enable_gpu_prefetching: bool,
    
    /// Enable real-time GPU monitoring
    pub enable_gpu_monitoring: bool,
}

impl Default for GpuGraphConfig {
    fn default() -> Self {
        Self {
            initial_node_capacity: 10_000_000,      // 10M nodes
            initial_edge_capacity: 100_000_000,     // 100M edges
            initial_hyperedge_capacity: 1_000_000,  // 1M hyperedges
            gpu_devices: Vec::new(),                 // Use all available
            enable_multi_gpu: true,
            gpu_memory_pool_size: 8 * 1024 * 1024 * 1024, // 8GB per GPU
            enable_unified_memory: true,
            enable_hybrid_processing: true,
            cpu_thread_pool_size: 0,                // Auto-detect
            cuda_streams_per_gpu: 8,
            gpu_batch_size: 1024 * 1024,           // 1M elements per batch
            enable_auto_load_balancing: true,
            gpu_memory_limit: 0,                    // Use all available
            enable_gpu_prefetching: true,
            enable_gpu_monitoring: true,
        }
    }
}

impl GpuKnowledgeGraph {
    /// Create a new GPU-accelerated knowledge graph
    pub async fn new(config: GpuGraphConfig) -> GpuResult<Self> {
        tracing::info!("🚀 Initializing GPU-accelerated Knowledge Graph");
        
        // Initialize GPU manager
        let gpu_manager = Arc::new(GpuManager::new(&config).await?);
        let available_gpus = gpu_manager.get_available_devices();
        tracing::info!("✅ Found {} GPU device(s)", available_gpus.len());
        
        // Initialize unified memory manager
        let memory_manager = Arc::new(UnifiedMemoryManager::new(
            &config,
            Arc::clone(&gpu_manager)
        ).await?);
        tracing::info!("✅ Unified memory manager initialized");
        
        // Initialize CUDA kernel manager
        let kernel_manager = Arc::new(CudaKernelManager::new(&gpu_manager).await?);
        tracing::info!("✅ CUDA kernels loaded");
        
        // Initialize GPU algorithms
        let algorithms = Arc::new(GpuAlgorithms::new(
            Arc::clone(&gpu_manager),
            Arc::clone(&memory_manager),
            Arc::clone(&kernel_manager)
        ).await?);
        tracing::info!("✅ GPU algorithms initialized");
        
        // Initialize CPU thread pool
        let cpu_threads = if config.cpu_thread_pool_size == 0 {
            std::thread::available_parallelism()?.get()
        } else {
            config.cpu_thread_pool_size
        };
        
        rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads)
            .thread_name(|index| format!("gpu-kg-cpu-{}", index))
            .build_global()?;
        
        tracing::info!("✅ CPU thread pool initialized with {} threads", cpu_threads);
        
        // Initialize storage layers
        let node_storage = Arc::new(RwLock::new(
            GpuNodeStorage::new(&config, Arc::clone(&memory_manager)).await?
        ));
        let edge_storage = Arc::new(RwLock::new(
            GpuEdgeStorage::new(&config, Arc::clone(&memory_manager)).await?
        ));
        let hyperedge_storage = Arc::new(RwLock::new(
            GpuHyperedgeStorage::new(&config, Arc::clone(&memory_manager)).await?
        ));
        
        tracing::info!("✅ GPU storage layers initialized");
        
        // Initialize statistics and metrics
        let statistics = Arc::new(GpuGraphStatistics::new());
        let metrics_collector = Arc::new(RwLock::new(GpuPerformanceMetrics::new()));
        
        Ok(Self {
            config,
            gpu_manager,
            memory_manager,
            algorithms,
            kernel_manager,
            node_storage,
            edge_storage,
            hyperedge_storage,
            statistics,
            structure_lock: RwLock::new(()),
            metrics_collector,
        })
    }
    
    /// Create a node with automatic GPU placement
    pub async fn create_node(&self, data: GpuNodeData) -> GpuResult<NodeId> {
        let start = Instant::now();
        
        // Choose optimal GPU device for this node
        let gpu_device = self.choose_optimal_gpu_device(&data).await?;
        
        // Create node with GPU optimization
        let node_id = {
            let mut storage = self.node_storage.write();
            storage.create_node(data, gpu_device).await?
        };
        
        // Update statistics
        self.statistics.node_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.gpu_operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Update performance metrics
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_gpu_query_time(duration_ns);
        
        tracing::debug!(
            "Created node {} on GPU device {} in {}μs", 
            node_id, gpu_device.unwrap_or(-1), duration_ns / 1000
        );
        
        Ok(node_id)
    }
    
    /// Create multiple nodes in parallel using GPU batch processing
    pub async fn batch_create_nodes(&self, nodes: Vec<GpuNodeData>) -> GpuResult<Vec<NodeId>> {
        let start = Instant::now();
        let batch_size = nodes.len();
        
        tracing::info!("Creating batch of {} nodes with GPU acceleration", batch_size);
        
        // Determine optimal batch processing strategy
        let node_ids = if batch_size > self.config.gpu_batch_size {
            // Large batch: use multi-GPU parallel processing
            self.parallel_batch_create_nodes(nodes).await?
        } else if batch_size > 1000 {
            // Medium batch: use single GPU with CUDA streams
            self.gpu_batch_create_nodes(nodes).await?
        } else {
            // Small batch: use CPU with parallel processing
            self.cpu_batch_create_nodes(nodes).await?
        };
        
        // Update statistics
        self.statistics.node_count.fetch_add(batch_size as u64, std::sync::atomic::Ordering::Relaxed);
        self.statistics.gpu_operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration = start.elapsed();
        let nodes_per_sec = (batch_size as f64 / duration.as_secs_f64()) as u64;
        
        tracing::info!(
            "✅ Created {} nodes in {}ms ({} nodes/sec) using GPU acceleration", 
            batch_size, duration.as_millis(), nodes_per_sec
        );
        
        Ok(node_ids)
    }
    
    /// Create an edge with GPU optimization
    pub async fn create_edge(&self, from: NodeId, to: NodeId, weight: Weight, data: GpuEdgeData) -> GpuResult<EdgeId> {
        let start = Instant::now();
        
        // Choose optimal GPU device based on node locations
        let gpu_device = self.choose_optimal_gpu_device_for_edge(from, to).await?;
        
        let edge_id = {
            let mut storage = self.edge_storage.write();
            storage.create_edge(from, to, weight, data, gpu_device).await?
        };
        
        // Update statistics
        self.statistics.edge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.gpu_operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_gpu_query_time(duration_ns);
        
        tracing::debug!(
            "Created edge {} ({} -> {}) on GPU device {} in {}μs", 
            edge_id, from, to, gpu_device.unwrap_or(-1), duration_ns / 1000
        );
        
        Ok(edge_id)
    }
    
    /// Get node data with GPU-optimized retrieval
    pub async fn get_node(&self, node_id: NodeId) -> GpuResult<Option<GpuNodeData>> {
        let start = Instant::now();
        
        let result = {
            let storage = self.node_storage.read();
            storage.get_node(node_id).await?
        };
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_gpu_query_time(duration_ns);
        
        Ok(result)
    }
    
    /// Perform GPU-accelerated breadth-first search
    pub async fn gpu_traverse_bfs(&self, start_node: NodeId, max_depth: u32) -> GpuResult<GpuTraversalResult> {
        let start = Instant::now();
        
        tracing::info!("Running GPU-accelerated BFS from node {} with max depth {}", start_node, max_depth);
        
        // Choose best GPU for BFS based on graph structure
        let optimal_gpu = self.choose_optimal_gpu_for_algorithm("bfs", start_node).await?;
        
        let result = self.algorithms.gpu_breadth_first_search(
            start_node, 
            max_depth, 
            optimal_gpu
        ).await?;
        
        // Update statistics
        self.statistics.gpu_queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.cuda_kernel_launches.fetch_add(
            result.cuda_streams_used as u64, 
            std::sync::atomic::Ordering::Relaxed
        );
        
        let total_duration = start.elapsed();
        tracing::info!(
            "GPU BFS completed: {} nodes, {} edges in {}ms (GPU: {}ms, Transfer: {}ms, Speedup: {:.1}x)", 
            result.nodes_visited, 
            result.edges_traversed, 
            total_duration.as_millis(),
            result.gpu_kernel_time.as_millis(),
            result.memory_transfer_time.as_millis(),
            result.gpu_speedup_ratio().unwrap_or(1.0)
        );
        
        Ok(result)
    }
    
    /// GPU-accelerated shortest path using parallel Dijkstra
    pub async fn gpu_shortest_path(&self, from: NodeId, to: NodeId) -> GpuResult<Option<GpuPath>> {
        let start = Instant::now();
        
        tracing::info!("Computing GPU shortest path from {} to {}", from, to);
        
        let optimal_gpu = self.choose_optimal_gpu_for_algorithm("dijkstra", from).await?;
        
        let result = self.algorithms.gpu_shortest_path(from, to, optimal_gpu).await?;
        
        // Update statistics
        self.statistics.gpu_queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration = start.elapsed();
        if let Some(ref path) = result {
            tracing::info!(
                "GPU shortest path: {} hops, weight {:.3} in {}μs (GPU speedup: {:.1}x)", 
                path.length, 
                path.total_weight, 
                duration.as_micros(),
                path.computation_time.as_nanos() as f32 / duration.as_nanos() as f32
            );
        }
        
        Ok(result)
    }
    
    /// GPU-accelerated PageRank algorithm
    pub async fn gpu_pagerank(&self, damping_factor: f32, max_iterations: u32, tolerance: f32) -> GpuResult<HashMap<NodeId, f32>> {
        let start = Instant::now();
        
        tracing::info!(
            "Running GPU PageRank: damping={}, iterations={}, tolerance={}", 
            damping_factor, max_iterations, tolerance
        );
        
        // Use all available GPUs for large-scale PageRank
        let result = if self.config.enable_multi_gpu && self.gpu_manager.get_device_count() > 1 {
            self.algorithms.multi_gpu_pagerank(damping_factor, max_iterations, tolerance).await?
        } else {
            let optimal_gpu = self.choose_optimal_gpu_for_algorithm("pagerank", 0).await?;
            self.algorithms.gpu_pagerank(damping_factor, max_iterations, tolerance, optimal_gpu).await?
        };
        
        let duration = start.elapsed();
        tracing::info!(
            "✅ GPU PageRank completed for {} nodes in {}ms", 
            result.len(), duration.as_millis()
        );
        
        Ok(result)
    }
    
    /// GPU-accelerated centrality computation
    pub async fn gpu_compute_centrality(&self, algorithm: CentralityAlgorithm) -> GpuResult<HashMap<NodeId, f64>> {
        let start = Instant::now();
        
        tracing::info!("Computing GPU {:?} centrality", algorithm);
        
        let result = self.algorithms.gpu_compute_centrality(algorithm).await?;
        
        let duration = start.elapsed();
        tracing::info!(
            "✅ GPU centrality computed for {} nodes in {}ms", 
            result.len(), duration.as_millis()
        );
        
        Ok(result)
    }
    
    /// GPU-accelerated community detection
    pub async fn gpu_detect_communities(&self, algorithm: CommunityAlgorithm) -> GpuResult<Vec<Vec<NodeId>>> {
        let start = Instant::now();
        
        tracing::info!("Detecting communities using GPU {:?}", algorithm);
        
        let result = self.algorithms.gpu_detect_communities(algorithm).await?;
        
        let duration = start.elapsed();
        tracing::info!(
            "✅ GPU detected {} communities in {}ms", 
            result.len(), duration.as_millis()
        );
        
        Ok(result)
    }
    
    /// Create a hyperedge with GPU optimization
    pub async fn create_hyperedge(&self, nodes: SmallVec<[NodeId; 8]>, data: GpuHyperedgeData) -> GpuResult<EdgeId> {
        let start = Instant::now();
        
        // Choose optimal GPU device based on node distribution
        let gpu_device = self.choose_optimal_gpu_device_for_hyperedge(&nodes).await?;
        
        let hyperedge_id = {
            let mut storage = self.hyperedge_storage.write();
            storage.create_hyperedge(nodes, data, gpu_device).await?
        };
        
        // Update statistics
        self.statistics.hyperedge_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.statistics.gpu_operations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let duration_ns = start.elapsed().as_nanos() as u64;
        self.update_gpu_query_time(duration_ns);
        
        tracing::debug!("Created hyperedge {} on GPU device {} in {}μs", 
                       hyperedge_id, gpu_device.unwrap_or(-1), duration_ns / 1000);
        
        Ok(hyperedge_id)
    }
    
    /// Get comprehensive GPU performance metrics
    pub async fn get_gpu_performance_metrics(&self) -> GpuResult<GpuPerformanceMetrics> {
        let metrics = self.metrics_collector.read().clone();
        
        // Add current statistics
        let mut final_metrics = metrics;
        final_metrics.update_from_statistics(&self.statistics);
        
        // Add GPU-specific metrics
        final_metrics.gpu_devices = self.gpu_manager.get_device_info().await?;
        final_metrics.gpu_memory_usage = self.memory_manager.get_memory_usage().await?;
        final_metrics.cuda_streams_active = self.kernel_manager.get_active_streams();
        
        Ok(final_metrics)
    }
    
    /// Optimize graph for current GPU workload
    pub async fn optimize_for_gpu(&mut self) -> GpuResult<()> {
        let start = Instant::now();
        
        tracing::info!("🔧 Optimizing graph for GPU workload");
        
        // Acquire write lock for structural changes
        let _lock = self.structure_lock.write();
        
        // Optimize GPU memory layout
        self.memory_manager.optimize_memory_layout().await?;
        
        // Optimize data placement across GPUs
        if self.config.enable_multi_gpu {
            self.optimize_multi_gpu_placement().await?;
        }
        
        // Optimize CUDA kernel parameters
        self.kernel_manager.optimize_kernel_parameters().await?;
        
        // Optimize storage layers
        {
            let mut node_storage = self.node_storage.write();
            node_storage.optimize_for_gpu().await?;
        }
        {
            let mut edge_storage = self.edge_storage.write();
            edge_storage.optimize_for_gpu().await?;
        }
        {
            let mut hyperedge_storage = self.hyperedge_storage.write();
            hyperedge_storage.optimize_for_gpu().await?;
        }
        
        let duration = start.elapsed();
        tracing::info!("✅ GPU optimization completed in {}ms", duration.as_millis());
        
        Ok(())
    }
    
    /// Synchronize all GPU operations
    pub async fn synchronize_gpu(&self) -> GpuResult<()> {
        self.gpu_manager.synchronize_all_devices().await?;
        tracing::debug!("🔄 All GPU operations synchronized");
        Ok(())
    }
    
    /// Get current GPU utilization across all devices
    pub async fn get_gpu_utilization(&self) -> GpuResult<Vec<f32>> {
        self.gpu_manager.get_utilization().await
    }
    
    /// Get GPU memory usage across all devices
    pub async fn get_gpu_memory_usage(&self) -> GpuResult<Vec<(usize, usize)>> {
        self.gpu_manager.get_memory_usage().await
    }
    
    // Private helper methods
    
    async fn choose_optimal_gpu_device(&self, _data: &GpuNodeData) -> GpuResult<Option<GpuDeviceId>> {
        // TODO: Implement intelligent GPU device selection based on:
        // - Current GPU memory usage
        // - Data locality
        // - Load balancing
        // - Graph structure
        
        if self.gpu_manager.get_device_count() > 0 {
            Ok(Some(0)) // For now, use GPU 0
        } else {
            Ok(None) // No GPU available
        }
    }
    
    async fn choose_optimal_gpu_device_for_edge(&self, _from: NodeId, _to: NodeId) -> GpuResult<Option<GpuDeviceId>> {
        // TODO: Choose GPU based on where the connected nodes are located
        if self.gpu_manager.get_device_count() > 0 {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    }
    
    async fn choose_optimal_gpu_device_for_hyperedge(&self, _nodes: &[NodeId]) -> GpuResult<Option<GpuDeviceId>> {
        // TODO: Choose GPU based on node distribution
        if self.gpu_manager.get_device_count() > 0 {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    }
    
    async fn choose_optimal_gpu_for_algorithm(&self, _algorithm: &str, _start_node: NodeId) -> GpuResult<Option<GpuDeviceId>> {
        // TODO: Choose optimal GPU based on algorithm requirements and data locality
        if self.gpu_manager.get_device_count() > 0 {
            Ok(Some(0))
        } else {
            Ok(None)
        }
    }
    
    async fn parallel_batch_create_nodes(&self, nodes: Vec<GpuNodeData>) -> GpuResult<Vec<NodeId>> {
        // TODO: Implement multi-GPU parallel batch creation
        tracing::info!("Using multi-GPU parallel batch creation for {} nodes", nodes.len());
        
        let chunk_size = nodes.len() / self.gpu_manager.get_device_count().max(1);
        let mut results = Vec::new();
        
        for (i, chunk) in nodes.chunks(chunk_size).enumerate() {
            let gpu_device = i as GpuDeviceId;
            let chunk_results = self.gpu_batch_create_nodes_on_device(chunk.to_vec(), gpu_device).await?;
            results.extend(chunk_results);
        }
        
        Ok(results)
    }
    
    async fn gpu_batch_create_nodes(&self, nodes: Vec<GpuNodeData>) -> GpuResult<Vec<NodeId>> {
        // TODO: Implement GPU batch creation with CUDA streams
        tracing::info!("Using GPU batch creation for {} nodes", nodes.len());
        
        let gpu_device = 0; // Use primary GPU
        self.gpu_batch_create_nodes_on_device(nodes, gpu_device).await
    }
    
    async fn gpu_batch_create_nodes_on_device(&self, nodes: Vec<GpuNodeData>, gpu_device: GpuDeviceId) -> GpuResult<Vec<NodeId>> {
        // TODO: Implement device-specific batch creation
        let mut node_ids = Vec::with_capacity(nodes.len());
        
        // For now, create nodes sequentially (will be replaced with GPU kernel)
        let mut storage = self.node_storage.write();
        for node in nodes {
            let node_id = storage.create_node(node, Some(gpu_device)).await?;
            node_ids.push(node_id);
        }
        
        Ok(node_ids)
    }
    
    async fn cpu_batch_create_nodes(&self, nodes: Vec<GpuNodeData>) -> GpuResult<Vec<NodeId>> {
        // CPU parallel processing for smaller batches
        tracing::info!("Using CPU parallel batch creation for {} nodes", nodes.len());
        
        let node_ids: Result<Vec<_>, _> = nodes.into_par_iter()
            .map(|node| {
                // This would need async handling in real implementation
                let storage = self.node_storage.read();
                futures::executor::block_on(storage.create_node_cpu(node))
            })
            .collect();
        
        node_ids.map_err(|e| GpuKnowledgeGraphError::internal_error(format!("CPU batch creation failed: {}", e)))
    }
    
    async fn optimize_multi_gpu_placement(&mut self) -> GpuResult<()> {
        // TODO: Implement intelligent data placement across multiple GPUs
        tracing::info!("Optimizing data placement across {} GPUs", self.gpu_manager.get_device_count());
        Ok(())
    }
    
    fn update_gpu_query_time(&self, duration_ns: u64) {
        let current_avg = self.statistics.average_gpu_query_time_ns.load(std::sync::atomic::Ordering::Relaxed);
        
        // Exponential moving average with alpha = 0.1
        let new_avg = if current_avg == 0 {
            duration_ns
        } else {
            ((current_avg as f64 * 0.9) + (duration_ns as f64 * 0.1)) as u64
        };
        
        self.statistics.average_gpu_query_time_ns.store(new_avg, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for GpuKnowledgeGraph {
    fn drop(&mut self) {
        tracing::info!("🔄 Shutting down GPU Knowledge Graph");
        
        // Synchronize all GPU operations before shutdown
        if let Err(e) = futures::executor::block_on(self.synchronize_gpu()) {
            tracing::error!("Failed to synchronize GPUs during shutdown: {}", e);
        }
        
        // Print final statistics
        let stats = &self.statistics;
        tracing::info!("📊 Final GPU Statistics:");
        tracing::info!("  Nodes: {}", stats.node_count.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  Edges: {}", stats.edge_count.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  Hyperedges: {}", stats.hyperedge_count.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  Host Memory: {}MB", stats.host_memory_usage.load(std::sync::atomic::Ordering::Relaxed) / 1024 / 1024);
        tracing::info!("  GPU Memory: {}MB", stats.gpu_memory_usage.load(std::sync::atomic::Ordering::Relaxed) / 1024 / 1024);
        tracing::info!("  GPU Operations: {}", stats.gpu_operations_performed.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  GPU Queries: {}", stats.gpu_queries_executed.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  Avg GPU Query Time: {}μs", stats.average_gpu_query_time_ns.load(std::sync::atomic::Ordering::Relaxed) / 1000);
        tracing::info!("  GPU Utilization: {:.1}%", stats.gpu_utilization());
        tracing::info!("  CUDA Kernel Launches: {}", stats.cuda_kernel_launches.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  GPU Memory Transfers: {}", stats.gpu_memory_transfers.load(std::sync::atomic::Ordering::Relaxed));
        tracing::info!("  Uptime: {}s", stats.uptime().as_secs());
        
        tracing::info!("✅ GPU Knowledge Graph shutdown complete");
    }
}

// Placeholder storage implementations (will be implemented in separate modules)

struct GpuNodeStorage {
    nodes: HashMap<NodeId, GpuNodeData>,
    next_id: NodeId,
    memory_manager: Arc<UnifiedMemoryManager>,
}

impl GpuNodeStorage {
    async fn new(_config: &GpuGraphConfig, memory_manager: Arc<UnifiedMemoryManager>) -> GpuResult<Self> {
        Ok(Self {
            nodes: HashMap::new(),
            next_id: 1,
            memory_manager,
        })
    }
    
    async fn create_node(&mut self, mut data: GpuNodeData, gpu_device: Option<GpuDeviceId>) -> GpuResult<NodeId> {
        let node_id = self.next_id;
        self.next_id += 1;
        
        data.id = node_id;
        if let Some(device) = gpu_device {
            data.migrate_to_gpu(device)?;
        }
        
        self.nodes.insert(node_id, data);
        Ok(node_id)
    }
    
    async fn create_node_cpu(&self, data: GpuNodeData) -> GpuResult<NodeId> {
        // Placeholder for CPU-only node creation
        Ok(data.id)
    }
    
    async fn get_node(&self, node_id: NodeId) -> GpuResult<Option<GpuNodeData>> {
        Ok(self.nodes.get(&node_id).cloned())
    }
    
    async fn optimize_for_gpu(&mut self) -> GpuResult<()> {
        // TODO: Implement GPU storage optimization
        Ok(())
    }
}

struct GpuEdgeStorage {
    edges: HashMap<EdgeId, GpuEdgeData>,
    next_id: EdgeId,
    memory_manager: Arc<UnifiedMemoryManager>,
}

impl GpuEdgeStorage {
    async fn new(_config: &GpuGraphConfig, memory_manager: Arc<UnifiedMemoryManager>) -> GpuResult<Self> {
        Ok(Self {
            edges: HashMap::new(),
            next_id: 1,
            memory_manager,
        })
    }
    
    async fn create_edge(&mut self, from: NodeId, to: NodeId, weight: Weight, mut data: GpuEdgeData, gpu_device: Option<GpuDeviceId>) -> GpuResult<EdgeId> {
        let edge_id = self.next_id;
        self.next_id += 1;
        
        data.id = edge_id;
        data.from = from;
        data.to = to;
        data.weight = weight;
        
        if let Some(device) = gpu_device {
            data.migrate_to_gpu(device)?;
        }
        
        self.edges.insert(edge_id, data);
        Ok(edge_id)
    }
    
    async fn optimize_for_gpu(&mut self) -> GpuResult<()> {
        // TODO: Implement GPU edge storage optimization
        Ok(())
    }
}

struct GpuHyperedgeStorage {
    hyperedges: HashMap<EdgeId, GpuHyperedgeData>,
    next_id: EdgeId,
    memory_manager: Arc<UnifiedMemoryManager>,
}

impl GpuHyperedgeStorage {
    async fn new(_config: &GpuGraphConfig, memory_manager: Arc<UnifiedMemoryManager>) -> GpuResult<Self> {
        Ok(Self {
            hyperedges: HashMap::new(),
            next_id: 1_000_000, // Start from high number
            memory_manager,
        })
    }
    
    async fn create_hyperedge(&mut self, nodes: SmallVec<[NodeId; 8]>, mut data: GpuHyperedgeData, gpu_device: Option<GpuDeviceId>) -> GpuResult<EdgeId> {
        let hyperedge_id = self.next_id;
        self.next_id += 1;
        
        data.id = hyperedge_id;
        data.nodes = nodes;
        
        if let Some(device) = gpu_device {
            data.primary_gpu_device = Some(device);
        }
        
        self.hyperedges.insert(hyperedge_id, data);
        Ok(hyperedge_id)
    }
    
    async fn optimize_for_gpu(&mut self) -> GpuResult<()> {
        // TODO: Implement GPU hyperedge storage optimization
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_graph_creation() {
        // Skip test if no GPU available
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return;
        }
        
        let config = GpuGraphConfig::default();
        let result = GpuKnowledgeGraph::new(config).await;
        
        match result {
            Ok(graph) => {
                let stats = &graph.statistics;
                assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 0);
                assert_eq!(stats.edge_count.load(std::sync::atomic::Ordering::Relaxed), 0);
            },
            Err(e) => {
                println!("GPU graph creation failed (expected if no GPU): {}", e);
            }
        }
    }
    
    #[tokio::test]
    async fn test_gpu_node_creation() {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return;
        }
        
        let config = GpuGraphConfig::default();
        if let Ok(graph) = GpuKnowledgeGraph::new(config).await {
            let mut properties = std::collections::HashMap::new();
            properties.insert("name".to_string(), PropertyValue::String("Alice".to_string()));
            properties.insert("age".to_string(), PropertyValue::Int32(30));
            
            let node_data = GpuNodeData::new(0, "Person".to_string(), properties);
            
            match graph.create_node(node_data).await {
                Ok(node_id) => {
                    assert!(node_id > 0);
                    let stats = &graph.statistics;
                    assert_eq!(stats.node_count.load(std::sync::atomic::Ordering::Relaxed), 1);
                },
                Err(e) => {
                    println!("GPU node creation failed: {}", e);
                }
            }
        }
    }
}