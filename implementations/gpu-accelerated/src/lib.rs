//! # GPU-Accelerated Ultra-Fast Knowledge Graph Database
//!
//! This is the ultimate performance knowledge graph implementation leveraging:
//! - **GPU Acceleration**: CUDA kernels for massive parallel processing
//! - **cuGraph Integration**: RAPIDS cuGraph for optimized graph algorithms
//! - **Multi-GPU Support**: Distributed processing across multiple GPUs
//! - **CPU-GPU Hybrid**: Intelligent workload distribution
//! - **Zero-Copy Memory**: Unified memory and optimized transfers
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   CPU Layer     │    │   GPU Layer     │    │  Multi-GPU      │
//! │   (Orchestration)│◄──►│   (Computation) │◄──►│  (Distribution) │
//! │                 │    │                 │    │                 │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!        ▲                       ▲                       ▲
//!        │                       │                       │
//!        ▼                       ▼                       ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ Rust Host Code  │    │ CUDA Kernels    │    │ NCCL Multi-GPU  │
//! │ Memory Management│    │ cuGraph/cuBLAS  │    │ Communication   │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```
//!
//! ## Performance Goals
//!
//! - **10,000x+** speedup over traditional graph databases
//! - **Sub-nanosecond** graph operations on large datasets
//! - **Trillions** of nodes and edges support
//! - **Real-time** analytics on massive graphs
//! - **Petascale** distributed processing

#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Re-export core types for convenience
pub use crate::core::*;
pub use crate::graph::GpuKnowledgeGraph;

// Core modules
pub mod core;
pub mod graph;
pub mod gpu;
pub mod algorithms;
pub mod memory;
pub mod kernels;

// Utility modules
pub mod error;
pub mod metrics;
pub mod cuda_kernels;

// Benchmarking module
#[cfg(feature = "benchmarks")]
#[cfg_attr(docsrs, doc(cfg(feature = "benchmarks")))]
pub mod benchmarks;

// Optional feature modules
#[cfg(feature = "distributed")]
#[cfg_attr(docsrs, doc(cfg(feature = "distributed")))]
pub mod distributed;

// SIMD optimization module
#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;

#[cfg(feature = "profiling")]
#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
pub mod profiling;

/// Core result type for GPU knowledge graph operations
pub type GpuResult<T> = std::result::Result<T, error::GpuKnowledgeGraphError>;

/// Core trait for GPU-accelerated components
pub trait GpuAccelerated {
    /// Get GPU performance metrics for this component
    fn get_gpu_metrics(&self) -> metrics::GpuPerformanceMetrics;
    
    /// Reset GPU performance counters
    fn reset_gpu_metrics(&mut self);
    
    /// Optimize component for current GPU workload
    fn optimize_for_gpu(&mut self) -> GpuResult<()>;
    
    /// Check if component can run on GPU
    fn supports_gpu(&self) -> bool;
    
    /// Get preferred GPU device for this component
    fn preferred_gpu_device(&self) -> Option<i32>;
}

/// GPU device information and capabilities
pub mod device_info {
    //! GPU device detection and capability information
    
    /// CUDA compute capability
    pub const MIN_COMPUTE_CAPABILITY: (i32, i32) = (8, 0); // RTX 3000+ series
    
    /// Recommended compute capability for optimal performance
    pub const RECOMMENDED_COMPUTE_CAPABILITY: (i32, i32) = (8, 9); // RTX 4090+
    
    /// Minimum GPU memory required (in bytes)
    pub const MIN_GPU_MEMORY: usize = 8 * 1024 * 1024 * 1024; // 8GB
    
    /// Recommended GPU memory for large graphs (in bytes)
    pub const RECOMMENDED_GPU_MEMORY: usize = 24 * 1024 * 1024 * 1024; // 24GB
    
    /// Maximum concurrent CUDA streams
    pub const MAX_CUDA_STREAMS: usize = 32;
    
    /// Default CUDA stream count
    pub const DEFAULT_CUDA_STREAMS: usize = 8;
    
    /// Check if GPU supports required features
    pub fn is_gpu_compatible(device_id: i32) -> crate::GpuResult<bool> {
        // TODO: Implement GPU compatibility checking
        Ok(true) // Placeholder
    }
    
    /// Get optimal block size for CUDA kernels
    pub fn get_optimal_block_size(device_id: i32) -> crate::GpuResult<(u32, u32, u32)> {
        // TODO: Query device properties and calculate optimal block size
        Ok((256, 1, 1)) // Placeholder: 256 threads per block
    }
    
    /// Get optimal grid size for given problem size
    pub fn get_optimal_grid_size(device_id: i32, problem_size: usize) -> crate::GpuResult<(u32, u32, u32)> {
        let (block_x, _, _) = get_optimal_block_size(device_id)?;
        let grid_x = ((problem_size as u32 + block_x - 1) / block_x).min(65535); // Max grid size
        Ok((grid_x, 1, 1))
    }
    
    /// Get human-readable GPU description
    pub fn get_gpu_description(device_id: i32) -> crate::GpuResult<String> {
        // TODO: Query device properties
        Ok(format!("CUDA Device {}", device_id)) // Placeholder
    }
}

/// Global GPU initialization for all components
pub fn init_gpu() -> GpuResult<()> {
    tracing::info!("🚀 Initializing GPU-accelerated knowledge graph");
    
    // Initialize CUDA runtime
    #[cfg(feature = "cuda")]
    {
        gpu::init_cuda_runtime()?;
        tracing::info!("✅ CUDA runtime initialized");
    }
    
    // Initialize GPU memory pools
    memory::init_gpu_memory_pools()?;
    tracing::info!("✅ GPU memory pools initialized");
    
    // Initialize CUDA kernels
    kernels::init_cuda_kernels()?;
    tracing::info!("✅ CUDA kernels initialized");
    
    // Initialize performance monitoring
    metrics::init_gpu_metrics()?;
    tracing::info!("✅ GPU metrics initialized");
    
    // Initialize multi-GPU support if available
    #[cfg(feature = "distributed")]
    {
        if gpu::get_gpu_count()? > 1 {
            distributed::init_multi_gpu()?;
            tracing::info!("✅ Multi-GPU support initialized");
        }
    }
    
    // Warm up GPU kernels
    warm_up_gpu_kernels()?;
    
    tracing::info!(
        "🚀 GPU Knowledge Graph initialized with {} device(s)",
        gpu::get_gpu_count().unwrap_or(0)
    );
    
    Ok(())
}

/// Warm up GPU kernels for optimal performance
fn warm_up_gpu_kernels() -> GpuResult<()> {
    tracing::debug!("🔥 Warming up GPU kernels");
    
    // Warm up basic GPU operations
    gpu::warm_up_basic_ops()?;
    
    // Warm up graph algorithm kernels
    algorithms::warm_up_algorithm_kernels()?;
    
    // Warm up memory transfer operations
    memory::warm_up_memory_transfers()?;
    
    tracing::debug!("✅ GPU kernels warmed up");
    Ok(())
}

/// Get global GPU statistics
pub fn get_gpu_stats() -> metrics::GlobalGpuStats {
    metrics::get_global_gpu_stats()
}

/// Run comprehensive GPU benchmarks
#[cfg(feature = "benchmarks")]
pub fn run_gpu_benchmarks() -> GpuResult<metrics::GpuBenchmarkResults> {
    crate::benchmarks::run_comprehensive_gpu_benchmarks()
}

/// Convenience macro for GPU memory allocation with error handling
#[macro_export]
macro_rules! gpu_alloc {
    ($size:expr) => {{
        $crate::memory::allocate_gpu_memory($size)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::memory_allocation(
                format!("Failed to allocate {} bytes on GPU: {}", $size, e)
            ))
    }};
    ($ty:ty, $count:expr) => {{
        let size = std::mem::size_of::<$ty>() * $count;
        $crate::memory::allocate_gpu_memory_typed::<$ty>($count)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::memory_allocation(
                format!("Failed to allocate {} elements of type {} on GPU: {}", $count, stringify!($ty), e)
            ))
    }};
}

/// Convenience macro for launching CUDA kernels with error handling
#[macro_export]
macro_rules! launch_kernel {
    ($kernel:ident, $grid:expr, $block:expr, $args:expr) => {{
        $crate::kernels::launch_cuda_kernel(
            stringify!($kernel),
            $grid,
            $block,
            $args
        ).map_err(|e| $crate::error::GpuKnowledgeGraphError::kernel_launch(
            format!("Failed to launch kernel {}: {}", stringify!($kernel), e)
        ))
    }};
}

/// Convenience macro for GPU-CPU memory transfers
#[macro_export]
macro_rules! gpu_transfer {
    (host_to_device, $host_ptr:expr, $device_ptr:expr, $size:expr) => {{
        $crate::memory::copy_host_to_device($host_ptr, $device_ptr, $size)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::memory_transfer(
                format!("Failed to transfer {} bytes to GPU: {}", $size, e)
            ))
    }};
    (device_to_host, $device_ptr:expr, $host_ptr:expr, $size:expr) => {{
        $crate::memory::copy_device_to_host($device_ptr, $host_ptr, $size)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::memory_transfer(
                format!("Failed to transfer {} bytes from GPU: {}", $size, e)
            ))
    }};
    (device_to_device, $src_ptr:expr, $dst_ptr:expr, $size:expr) => {{
        $crate::memory::copy_device_to_device($src_ptr, $dst_ptr, $size)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::memory_transfer(
                format!("Failed to transfer {} bytes on GPU: {}", $size, e)
            ))
    }};
}

/// Convenience macro for GPU kernel synchronization
#[macro_export]
macro_rules! gpu_sync {
    () => {{
        $crate::gpu::synchronize_device()
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::synchronization(
                format!("Failed to synchronize GPU: {}", e)
            ))
    }};
    ($stream:expr) => {{
        $crate::gpu::synchronize_stream($stream)
            .map_err(|e| $crate::error::GpuKnowledgeGraphError::synchronization(
                format!("Failed to synchronize GPU stream: {}", e)
            ))
    }};
}

/// Fast GPU-based hash function for node IDs
#[inline]
pub fn gpu_fast_hash(value: u64) -> u64 {
    // GPU-optimized hash function that works well on both CPU and GPU
    value.wrapping_mul(0x9e3779b97f4a7c15_u64).rotate_left(31)
}

/// GPU memory alignment for optimal performance
pub const GPU_MEMORY_ALIGNMENT: usize = 256; // 256-byte alignment for GPU

/// Compile-time assertion for GPU memory alignment
#[macro_export]
macro_rules! assert_gpu_aligned {
    ($ptr:expr) => {{
        debug_assert_eq!(
            ($ptr as usize) % $crate::GPU_MEMORY_ALIGNMENT,
            0,
            "Pointer not aligned for GPU operations"
        );
    }};
}

/// GPU-specific constants
pub mod gpu_constants {
    /// Maximum threads per CUDA block
    pub const MAX_THREADS_PER_BLOCK: u32 = 1024;
    
    /// Warp size for NVIDIA GPUs
    pub const WARP_SIZE: u32 = 32;
    
    /// Maximum blocks per grid dimension
    pub const MAX_BLOCKS_PER_GRID: u32 = 65535;
    
    /// Shared memory per block (bytes)
    pub const SHARED_MEMORY_PER_BLOCK: usize = 48 * 1024; // 48KB
    
    /// Maximum registers per thread
    pub const MAX_REGISTERS_PER_THREAD: u32 = 255;
    
    /// L2 cache size (typical)
    pub const L2_CACHE_SIZE: usize = 6 * 1024 * 1024; // 6MB
    
    /// Memory bandwidth (GB/s) - RTX 4090
    pub const MEMORY_BANDWIDTH_GBPS: f32 = 1008.0;
    
    /// Peak compute throughput (TFLOPS) - RTX 4090 FP32
    pub const PEAK_COMPUTE_TFLOPS: f32 = 83.0;
}

// Re-exports for convenience
pub use cudarc::driver::CudaDevice;
pub use ahash::AHashMap as FastHashMap;
pub use ahash::AHashSet as FastHashSet;
pub use dashmap::DashMap as ConcurrentHashMap;
pub use parking_lot::{Mutex, RwLock};
pub use smallvec::SmallVec;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_initialization() {
        // Note: This test requires a CUDA-capable GPU
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            init_gpu().expect("Failed to initialize GPU components");
        } else {
            eprintln!("Skipping GPU test - no CUDA devices available");
        }
    }
    
    #[test]
    fn test_gpu_hash_function() {
        let hash1 = gpu_fast_hash(12345);
        let hash2 = gpu_fast_hash(12346);
        assert_ne!(hash1, hash2);
        
        // Test hash distribution
        let mut hashes = std::collections::HashSet::new();
        for i in 0..1000 {
            hashes.insert(gpu_fast_hash(i));
        }
        // Should have good distribution (most values unique)
        assert!(hashes.len() > 990);
    }
    
    #[test]
    fn test_device_info() {
        // Test device compatibility checking
        if let Ok(compatible) = device_info::is_gpu_compatible(0) {
            println!("GPU 0 compatible: {}", compatible);
        }
        
        if let Ok(description) = device_info::get_gpu_description(0) {
            println!("GPU 0 description: {}", description);
        }
    }
    
    #[test]
    fn test_gpu_constants() {
        // Verify GPU constants are reasonable
        assert!(gpu_constants::MAX_THREADS_PER_BLOCK <= 1024);
        assert_eq!(gpu_constants::WARP_SIZE, 32);
        assert!(gpu_constants::SHARED_MEMORY_PER_BLOCK > 0);
    }
    
    #[test]
    fn test_memory_alignment() {
        let alignment = GPU_MEMORY_ALIGNMENT;
        assert!(alignment.is_power_of_two());
        assert!(alignment >= 128); // Minimum reasonable alignment
    }
}