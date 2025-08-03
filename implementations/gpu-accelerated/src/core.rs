//! Core types and traits for GPU-accelerated knowledge graph
//!
//! This module defines the fundamental types, traits, and constants used
//! throughout the GPU-accelerated implementation. It extends the basic types
//! with GPU-specific optimizations and memory layouts.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use bytemuck::{Pod, Zeroable};

// Re-export commonly used types
pub use crate::error::{GpuKnowledgeGraphError, GpuResult};

/// Unique identifier for graph nodes (64-bit for maximum range)
pub type NodeId = u64;

/// Unique identifier for graph edges (64-bit for maximum range)
pub type EdgeId = u64;

/// Edge weight type optimized for GPU SIMD operations (32-bit for vectorization)
pub type Weight = f32;

/// Hash type for fast lookups and comparisons
pub type Hash = u64;

/// GPU device identifier
pub type GpuDeviceId = i32;

/// CUDA stream identifier
pub type CudaStream = usize;

/// Property value optimized for GPU memory layout and processing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[repr(C)] // Ensure C-compatible layout for GPU kernels
pub enum PropertyValue {
    /// Null/empty value
    Null,
    /// Boolean value
    Bool(bool),
    /// 32-bit signed integer (GPU-optimized)
    Int32(i32),
    /// 64-bit signed integer
    Int64(i64),
    /// 32-bit floating point (GPU SIMD-friendly)
    Float32(f32),
    /// 64-bit floating point
    Float64(f64),
    /// UTF-8 string (stored as GPU buffer reference)
    String(String),
    /// Binary data (stored as GPU buffer reference)
    Bytes(Vec<u8>),
    /// Array of 32-bit integers (GPU vectorized)
    Int32Array(SmallVec<[i32; 8]>),
    /// Array of 64-bit integers
    Int64Array(SmallVec<[i64; 4]>),
    /// Array of 32-bit floats (GPU vectorized, CUDA-friendly)
    Float32Array(SmallVec<[f32; 8]>),
    /// Array of 64-bit floats
    Float64Array(SmallVec<[f64; 4]>),
    /// Array of strings
    StringArray(SmallVec<[String; 4]>),
    /// GPU buffer reference for large data
    GpuBuffer(GpuBufferRef),
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
            PropertyValue::GpuBuffer(buf_ref) => buf_ref.size,
        }
    }
    
    /// Check if this is a numeric type suitable for GPU SIMD operations
    pub fn is_gpu_simd_compatible(&self) -> bool {
        matches!(
            self,
            PropertyValue::Float32(_) 
                | PropertyValue::Float32Array(_)
                | PropertyValue::Int32(_)
                | PropertyValue::Int32Array(_)
        )
    }
    
    /// Check if this value is stored on GPU
    pub fn is_gpu_resident(&self) -> bool {
        matches!(self, PropertyValue::GpuBuffer(_))
    }
    
    /// Get GPU buffer reference if this value is GPU-resident
    pub fn as_gpu_buffer(&self) -> Option<&GpuBufferRef> {
        match self {
            PropertyValue::GpuBuffer(buf_ref) => Some(buf_ref),
            _ => None,
        }
    }
}

/// Reference to data stored in GPU memory
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuBufferRef {
    /// GPU device ID where buffer is located
    pub device_id: GpuDeviceId,
    
    /// GPU memory pointer (as u64 for serialization)
    pub gpu_ptr: u64,
    
    /// Size of buffer in bytes
    pub size: usize,
    
    /// Data type stored in buffer
    pub data_type: GpuDataType,
    
    /// Whether buffer uses unified memory
    pub is_unified: bool,
}

/// Data types optimized for GPU processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuDataType {
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Raw bytes
    Bytes,
    /// String data (UTF-8)
    String,
}

/// Property map for flexible node/edge attributes
pub type PropertyMap = HashMap<String, PropertyValue>;

/// GPU-optimized node data with aligned memory layout
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(32))] // 32-byte alignment for GPU coalesced access
pub struct GpuNodeData {
    /// Node ID
    pub id: NodeId,
    
    /// Human-readable label for the node
    pub label: String,
    
    /// Key-value properties (may reference GPU buffers)
    pub properties: PropertyMap,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last modification timestamp
    pub updated_at: SystemTime,
    
    /// GPU device where node data is primarily stored
    pub primary_gpu_device: Option<GpuDeviceId>,
    
    /// Memory usage on GPU in bytes
    pub gpu_memory_usage: usize,
}

impl GpuNodeData {
    /// Create new GPU-optimized node data
    pub fn new(id: NodeId, label: String, properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        
        // Calculate GPU memory usage
        let gpu_memory_usage = properties.values()
            .filter(|v| v.is_gpu_resident())
            .map(|v| v.memory_size())
            .sum();
        
        Self {
            id,
            label,
            properties,
            created_at: now,
            updated_at: now,
            primary_gpu_device: None,
            gpu_memory_usage,
        }
    }
    
    /// Get the total memory size of this node
    pub fn memory_size(&self) -> usize {
        self.label.len() 
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 64 // Timestamps, GPU fields, and overhead
    }
    
    /// Update properties and modification time
    pub fn update_properties(&mut self, new_properties: PropertyMap) {
        // Recalculate GPU memory usage
        self.gpu_memory_usage = new_properties.values()
            .filter(|v| v.is_gpu_resident())
            .map(|v| v.memory_size())
            .sum();
        
        self.properties = new_properties;
        self.updated_at = SystemTime::now();
    }
    
    /// Check if node data is primarily stored on GPU
    pub fn is_gpu_resident(&self) -> bool {
        self.primary_gpu_device.is_some()
    }
    
    /// Migrate node data to specified GPU device
    pub fn migrate_to_gpu(&mut self, device_id: GpuDeviceId) -> GpuResult<()> {
        // TODO: Implement GPU migration logic
        self.primary_gpu_device = Some(device_id);
        tracing::debug!("Migrated node {} to GPU device {}", self.id, device_id);
        Ok(())
    }
}

/// GPU-optimized edge data with aligned memory layout
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(32))] // 32-byte alignment for GPU coalesced access
pub struct GpuEdgeData {
    /// Edge ID
    pub id: EdgeId,
    
    /// Source node ID
    pub from: NodeId,
    
    /// Target node ID
    pub to: NodeId,
    
    /// Edge weight
    pub weight: Weight,
    
    /// Key-value properties (may reference GPU buffers)
    pub properties: PropertyMap,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last modification timestamp
    pub updated_at: SystemTime,
    
    /// GPU device where edge data is primarily stored
    pub primary_gpu_device: Option<GpuDeviceId>,
    
    /// Memory usage on GPU in bytes
    pub gpu_memory_usage: usize,
}

impl GpuEdgeData {
    /// Create new GPU-optimized edge data
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, weight: Weight, properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        
        // Calculate GPU memory usage
        let gpu_memory_usage = properties.values()
            .filter(|v| v.is_gpu_resident())
            .map(|v| v.memory_size())
            .sum();
        
        Self {
            id,
            from,
            to,
            weight,
            properties,
            created_at: now,
            updated_at: now,
            primary_gpu_device: None,
            gpu_memory_usage,
        }
    }
    
    /// Get the total memory size of this edge
    pub fn memory_size(&self) -> usize {
        self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 64 // IDs, weight, timestamps, GPU fields, and overhead
    }
    
    /// Check if edge data is primarily stored on GPU
    pub fn is_gpu_resident(&self) -> bool {
        self.primary_gpu_device.is_some()
    }
    
    /// Migrate edge data to specified GPU device
    pub fn migrate_to_gpu(&mut self, device_id: GpuDeviceId) -> GpuResult<()> {
        // TODO: Implement GPU migration logic
        self.primary_gpu_device = Some(device_id);
        tracing::debug!("Migrated edge {} to GPU device {}", self.id, device_id);
        Ok(())
    }
}

/// GPU-optimized hyperedge data for N-ary relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(32))] // 32-byte alignment for GPU coalesced access  
pub struct GpuHyperedgeData {
    /// Hyperedge ID
    pub id: EdgeId,
    
    /// Nodes connected by this hyperedge
    pub nodes: SmallVec<[NodeId; 8]>,
    
    /// Key-value properties (may reference GPU buffers)
    pub properties: PropertyMap,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last modification timestamp
    pub updated_at: SystemTime,
    
    /// GPU device where hyperedge data is primarily stored
    pub primary_gpu_device: Option<GpuDeviceId>,
    
    /// Memory usage on GPU in bytes
    pub gpu_memory_usage: usize,
}

impl GpuHyperedgeData {
    /// Create new GPU-optimized hyperedge data
    pub fn new(id: EdgeId, nodes: SmallVec<[NodeId; 8]>, properties: PropertyMap) -> Self {
        let now = SystemTime::now();
        
        // Calculate GPU memory usage
        let gpu_memory_usage = properties.values()
            .filter(|v| v.is_gpu_resident())
            .map(|v| v.memory_size())
            .sum();
        
        Self {
            id,
            nodes,
            properties,
            created_at: now,
            updated_at: now,
            primary_gpu_device: None,
            gpu_memory_usage,
        }
    }
    
    /// Get the total memory size of this hyperedge
    pub fn memory_size(&self) -> usize {
        self.nodes.len() * 8 
            + self.properties.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            + 64 // Timestamps, GPU fields, and overhead
    }
    
    /// Check if hyperedge data is primarily stored on GPU
    pub fn is_gpu_resident(&self) -> bool {
        self.primary_gpu_device.is_some()
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

/// GPU-optimized path representation with aligned data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(32))] // GPU-friendly alignment
pub struct GpuPath {
    /// Sequence of nodes in the path
    pub nodes: Vec<NodeId>,
    
    /// Sequence of edges in the path
    pub edges: Vec<EdgeId>,
    
    /// Edge weights along the path (GPU-optimized f32)
    pub weights: Vec<Weight>,
    
    /// Total path weight
    pub total_weight: Weight,
    
    /// Path length (number of edges)
    pub length: usize,
    
    /// Time taken to compute this path
    pub computation_time: Duration,
    
    /// Confidence score for approximate algorithms
    pub confidence: f32,
    
    /// GPU device used for computation
    pub computed_on_gpu: Option<GpuDeviceId>,
    
    /// GPU memory used for computation (bytes)
    pub gpu_memory_used: usize,
}

impl GpuPath {
    /// Create a new empty GPU path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            weights: Vec::new(),
            total_weight: 0.0,
            length: 0,
            computation_time: Duration::ZERO,
            confidence: 1.0,
            computed_on_gpu: None,
            gpu_memory_used: 0,
        }
    }
    
    /// Check if this path is valid (nodes and edges match)
    pub fn is_valid(&self) -> bool {
        self.nodes.len() > 0 
            && (self.nodes.len() == self.edges.len() + 1)
            && (self.weights.len() == self.edges.len())
            && (self.length == self.edges.len())
    }
    
    /// Check if path was computed on GPU
    pub fn was_computed_on_gpu(&self) -> bool {
        self.computed_on_gpu.is_some()
    }
}

impl Default for GpuPath {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-optimized traversal result with performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C, align(32))] // GPU-friendly alignment
pub struct GpuTraversalResult {
    /// Nodes visited during traversal
    pub nodes: Vec<NodeId>,
    
    /// Edges traversed
    pub edges: Vec<EdgeId>,
    
    /// Depth of each visited node
    pub depths: Vec<u32>,
    
    /// Distances for weighted traversals (GPU-optimized f32)
    pub distances: Vec<Weight>,
    
    /// Total number of nodes visited
    pub nodes_visited: usize,
    
    /// Total number of edges traversed
    pub edges_traversed: usize,
    
    /// Time taken for traversal
    pub duration: Duration,
    
    /// Memory used during traversal (host)
    pub host_memory_used: usize,
    
    /// Memory used during traversal (GPU)
    pub gpu_memory_used: usize,
    
    /// Confidence score
    pub confidence: f32,
    
    /// GPU performance metrics
    pub gpu_kernel_time: Duration,
    pub memory_transfer_time: Duration,
    pub cuda_streams_used: u32,
    pub gpu_device_used: Option<GpuDeviceId>,
    pub gpu_compute_utilization: f32,
    pub gpu_memory_utilization: f32,
}

impl GpuTraversalResult {
    /// Create a new empty GPU traversal result
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            depths: Vec::new(),
            distances: Vec::new(),
            nodes_visited: 0,
            edges_traversed: 0,
            duration: Duration::ZERO,
            host_memory_used: 0,
            gpu_memory_used: 0,
            confidence: 1.0,
            gpu_kernel_time: Duration::ZERO,
            memory_transfer_time: Duration::ZERO,
            cuda_streams_used: 0,
            gpu_device_used: None,
            gpu_compute_utilization: 0.0,
            gpu_memory_utilization: 0.0,
        }
    }
    
    /// Check if traversal was executed on GPU
    pub fn was_executed_on_gpu(&self) -> bool {
        self.gpu_device_used.is_some()
    }
    
    /// Get total GPU time (kernel + transfer)
    pub fn total_gpu_time(&self) -> Duration {
        self.gpu_kernel_time + self.memory_transfer_time
    }
    
    /// Get GPU speedup ratio (if available)
    pub fn gpu_speedup_ratio(&self) -> Option<f32> {
        if self.gpu_kernel_time.as_nanos() > 0 {
            Some(self.duration.as_nanos() as f32 / self.gpu_kernel_time.as_nanos() as f32)
        } else {
            None
        }
    }
}

impl Default for GpuTraversalResult {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-optimized graph statistics with atomic counters
#[derive(Debug)]
pub struct GpuGraphStatistics {
    /// Total number of nodes
    pub node_count: AtomicU64,
    
    /// Total number of edges
    pub edge_count: AtomicU64,
    
    /// Total number of hyperedges
    pub hyperedge_count: AtomicU64,
    
    /// Total host memory usage in bytes
    pub host_memory_usage: AtomicUsize,
    
    /// Total GPU memory usage in bytes (across all devices)
    pub gpu_memory_usage: AtomicUsize,
    
    /// Number of operations performed on CPU
    pub cpu_operations_performed: AtomicU64,
    
    /// Number of operations performed on GPU
    pub gpu_operations_performed: AtomicU64,
    
    /// Number of queries executed on CPU
    pub cpu_queries_executed: AtomicU64,
    
    /// Number of queries executed on GPU
    pub gpu_queries_executed: AtomicU64,
    
    /// Average CPU query time in nanoseconds
    pub average_cpu_query_time_ns: AtomicU64,
    
    /// Average GPU query time in nanoseconds
    pub average_gpu_query_time_ns: AtomicU64,
    
    /// GPU cache hit ratio (0.0 to 1.0, stored as u32)
    pub gpu_cache_hit_ratio_x1000: AtomicU64,
    
    /// Number of CUDA kernel launches
    pub cuda_kernel_launches: AtomicU64,
    
    /// Number of GPU memory transfers
    pub gpu_memory_transfers: AtomicU64,
    
    /// Total GPU memory transfer volume (bytes)
    pub gpu_transfer_volume: AtomicU64,
    
    /// Start time for uptime calculation
    pub start_time: SystemTime,
}

impl GpuGraphStatistics {
    /// Create new GPU statistics with all counters at zero
    pub fn new() -> Self {
        Self {
            node_count: AtomicU64::new(0),
            edge_count: AtomicU64::new(0),
            hyperedge_count: AtomicU64::new(0),
            host_memory_usage: AtomicUsize::new(0),
            gpu_memory_usage: AtomicUsize::new(0),
            cpu_operations_performed: AtomicU64::new(0),
            gpu_operations_performed: AtomicU64::new(0),
            cpu_queries_executed: AtomicU64::new(0),
            gpu_queries_executed: AtomicU64::new(0),
            average_cpu_query_time_ns: AtomicU64::new(0),
            average_gpu_query_time_ns: AtomicU64::new(0),
            gpu_cache_hit_ratio_x1000: AtomicU64::new(0),
            cuda_kernel_launches: AtomicU64::new(0),
            gpu_memory_transfers: AtomicU64::new(0),
            gpu_transfer_volume: AtomicU64::new(0),
            start_time: SystemTime::now(),
        }
    }
    
    /// Get GPU cache hit ratio as a float [0.0, 1.0]
    pub fn gpu_cache_hit_ratio(&self) -> f32 {
        self.gpu_cache_hit_ratio_x1000.load(Ordering::Relaxed) as f32 / 1000.0
    }
    
    /// Set GPU cache hit ratio from a float [0.0, 1.0]
    pub fn set_gpu_cache_hit_ratio(&self, ratio: f32) {
        let ratio_x1000 = (ratio.clamp(0.0, 1.0) * 1000.0) as u64;
        self.gpu_cache_hit_ratio_x1000.store(ratio_x1000, Ordering::Relaxed);
    }
    
    /// Get uptime since creation
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed().unwrap_or(Duration::ZERO)
    }
    
    /// Get GPU utilization percentage
    pub fn gpu_utilization(&self) -> f32 {
        let total_ops = self.cpu_operations_performed.load(Ordering::Relaxed) 
            + self.gpu_operations_performed.load(Ordering::Relaxed);
        
        if total_ops > 0 {
            (self.gpu_operations_performed.load(Ordering::Relaxed) as f32 / total_ops as f32) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get average GPU speedup ratio
    pub fn average_gpu_speedup(&self) -> f32 {
        let cpu_time = self.average_cpu_query_time_ns.load(Ordering::Relaxed);
        let gpu_time = self.average_gpu_query_time_ns.load(Ordering::Relaxed);
        
        if gpu_time > 0 {
            cpu_time as f32 / gpu_time as f32
        } else {
            1.0
        }
    }
    
    /// Reset all counters
    pub fn reset(&self) {
        self.node_count.store(0, Ordering::Relaxed);
        self.edge_count.store(0, Ordering::Relaxed);
        self.hyperedge_count.store(0, Ordering::Relaxed);
        self.host_memory_usage.store(0, Ordering::Relaxed);
        self.gpu_memory_usage.store(0, Ordering::Relaxed);
        self.cpu_operations_performed.store(0, Ordering::Relaxed);
        self.gpu_operations_performed.store(0, Ordering::Relaxed);
        self.cpu_queries_executed.store(0, Ordering::Relaxed);
        self.gpu_queries_executed.store(0, Ordering::Relaxed);
        self.average_cpu_query_time_ns.store(0, Ordering::Relaxed);
        self.average_gpu_query_time_ns.store(0, Ordering::Relaxed);
        self.gpu_cache_hit_ratio_x1000.store(0, Ordering::Relaxed);
        self.cuda_kernel_launches.store(0, Ordering::Relaxed);
        self.gpu_memory_transfers.store(0, Ordering::Relaxed);
        self.gpu_transfer_volume.store(0, Ordering::Relaxed);
    }
}

impl Default for GpuGraphStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU-optimized pattern matching structures
pub use crate::hybrid_ultra_fast::core::{
    Pattern, PatternNode, PatternEdge, PatternConstraints, PatternMatch,
    CentralityAlgorithm, TraversalAlgorithm, CommunityAlgorithm
};

/// Trait for GPU-optimizable components
pub trait GpuOptimizable {
    /// Optimize component for GPU execution
    type GpuOptimizationResult;
    
    /// Perform GPU optimization
    fn optimize_for_gpu(&mut self) -> GpuResult<Self::GpuOptimizationResult>;
    
    /// Check if GPU optimization is needed
    fn needs_gpu_optimization(&self) -> bool;
    
    /// Get preferred GPU device for this component
    fn preferred_gpu_device(&self) -> Option<GpuDeviceId>;
}

/// Trait for components with GPU memory management
pub trait GpuMemoryManaged {
    /// Allocate GPU memory for this component
    fn allocate_gpu_memory(&mut self, device_id: GpuDeviceId) -> GpuResult<()>;
    
    /// Free GPU memory used by this component
    fn free_gpu_memory(&mut self) -> GpuResult<()>;
    
    /// Get GPU memory usage
    fn gpu_memory_usage(&self) -> usize;
    
    /// Check if component has GPU memory allocated
    fn has_gpu_memory(&self) -> bool;
    
    /// Migrate to different GPU device
    fn migrate_gpu_memory(&mut self, target_device: GpuDeviceId) -> GpuResult<()>;
}

/// Constants for GPU-optimized operations
pub mod gpu_constants {
    /// Preferred GPU memory alignment (bytes)
    pub const GPU_MEMORY_ALIGNMENT: usize = 256;
    
    /// CUDA warp size
    pub const CUDA_WARP_SIZE: u32 = 32;
    
    /// Preferred CUDA block size for graph operations
    pub const PREFERRED_BLOCK_SIZE: u32 = 256;
    
    /// Maximum CUDA grid size
    pub const MAX_GRID_SIZE: u32 = 65535;
    
    /// GPU L2 cache line size (bytes)
    pub const GPU_CACHE_LINE_SIZE: usize = 128;
    
    /// Minimum GPU memory for optimal performance (bytes)
    pub const MIN_GPU_MEMORY: usize = 8 * 1024 * 1024 * 1024; // 8GB
}

/// GPU-specific error types
pub mod gpu_errors {
    use super::*;
    
    /// CUDA-specific error
    #[derive(Debug, Clone)]
    pub struct CudaError {
        pub code: i32,
        pub message: String,
    }
    
    /// GPU memory error
    #[derive(Debug, Clone)]
    pub struct GpuMemoryError {
        pub device_id: GpuDeviceId,
        pub requested_bytes: usize,
        pub available_bytes: usize,
        pub message: String,
    }
    
    /// GPU kernel launch error
    #[derive(Debug, Clone)]
    pub struct KernelLaunchError {
        pub kernel_name: String,
        pub grid_size: (u32, u32, u32),
        pub block_size: (u32, u32, u32),
        pub message: String,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_property_value() {
        let prop = PropertyValue::Float32Array(vec![1.0, 2.0, 3.0, 4.0].into());
        assert!(prop.is_gpu_simd_compatible());
        assert_eq!(prop.memory_size(), 16); // 4 * 4 bytes
        assert!(!prop.is_gpu_resident());
    }
    
    #[test]
    fn test_gpu_buffer_ref() {
        let buf_ref = GpuBufferRef {
            device_id: 0,
            gpu_ptr: 0x1000,
            size: 1024,
            data_type: GpuDataType::Float32,
            is_unified: false,
        };
        
        let prop = PropertyValue::GpuBuffer(buf_ref);
        assert!(prop.is_gpu_resident());
        assert_eq!(prop.memory_size(), 1024);
    }
    
    #[test]
    fn test_gpu_node_data() {
        let mut properties = PropertyMap::new();
        properties.insert("x".to_string(), PropertyValue::Float32(1.0));
        properties.insert("y".to_string(), PropertyValue::Float32(2.0));
        
        let mut node = GpuNodeData::new(1, "TestNode".to_string(), properties);
        assert!(!node.is_gpu_resident());
        
        node.migrate_to_gpu(0).expect("Failed to migrate to GPU");
        assert!(node.is_gpu_resident());
        assert_eq!(node.primary_gpu_device, Some(0));
    }
    
    #[test]
    fn test_gpu_path_validation() {
        let mut path = GpuPath::new();
        assert!(!path.is_valid()); // Empty path is invalid
        
        path.nodes = vec![1, 2, 3];
        path.edges = vec![10, 20];
        path.weights = vec![1.0, 2.0];
        path.length = 2;
        path.total_weight = 3.0;
        
        assert!(path.is_valid());
        assert!(!path.was_computed_on_gpu());
        
        path.computed_on_gpu = Some(0);
        assert!(path.was_computed_on_gpu());
    }
    
    #[test]
    fn test_gpu_statistics() {
        let stats = GpuGraphStatistics::new();
        
        stats.node_count.store(1000, Ordering::Relaxed);
        stats.gpu_operations_performed.store(750, Ordering::Relaxed);
        stats.cpu_operations_performed.store(250, Ordering::Relaxed);
        
        assert_eq!(stats.node_count.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.gpu_utilization(), 75.0); // 750 / 1000 * 100
        
        stats.set_gpu_cache_hit_ratio(0.95);
        assert!((stats.gpu_cache_hit_ratio() - 0.95).abs() < 0.001);
        
        stats.reset();
        assert_eq!(stats.node_count.load(Ordering::Relaxed), 0);
        assert_eq!(stats.gpu_utilization(), 0.0);
    }
    
    #[test]
    fn test_gpu_traversal_result() {
        let mut result = GpuTraversalResult::new();
        assert!(!result.was_executed_on_gpu());
        
        result.gpu_device_used = Some(0);
        result.gpu_kernel_time = Duration::from_millis(10);
        result.memory_transfer_time = Duration::from_millis(5);
        result.duration = Duration::from_millis(100);
        
        assert!(result.was_executed_on_gpu());
        assert_eq!(result.total_gpu_time(), Duration::from_millis(15));
        assert_eq!(result.gpu_speedup_ratio(), Some(10.0)); // 100ms / 10ms
    }
}