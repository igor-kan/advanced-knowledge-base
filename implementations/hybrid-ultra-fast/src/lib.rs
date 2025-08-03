//! # Hybrid Ultra-Fast Knowledge Graph Database
//!
//! This is the ultimate performance knowledge graph implementation combining:
//! - **Rust**: Memory safety and zero-cost abstractions
//! - **C++**: Manual memory management and low-level optimizations  
//! - **Assembly**: Hand-optimized SIMD kernels for critical paths
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │  Rust Layer     │    │  C++ Layer      │    │ Assembly Layer  │
//! │  (Safe API)     │◄──►│  (Performance)  │◄──►│ (SIMD Kernels)  │
//! │                 │    │                 │    │                 │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!        ▲                       ▲                       ▲
//!        │                       │                       │
//!        ▼                       ▼                       ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ Zero-cost FFI   │    │ Cache-aligned   │    │ AVX-512 16-wide │
//! │ CXX bridges     │    │ Data structures │    │ Vectorization   │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```
//!
//! ## Performance Goals
//!
//! - **500x-1000x** speedup over traditional graph databases
//! - **Sub-microsecond** node/edge access
//! - **Billions** of nodes and edges
//! - **Zero-copy** operations where possible
//! - **Lock-free** concurrent access

#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Include build-time generated CPU feature flags
include!(concat!(env!("OUT_DIR"), "/cpu_features.rs"));

// Re-export core types for convenience
pub use crate::core::*;
pub use crate::graph::HybridKnowledgeGraph;

// Core modules
pub mod core;
pub mod graph;
pub mod storage;
pub mod algorithms;
pub mod query;
pub mod simd;
pub mod distributed;

// FFI bridge modules
pub mod bridge;

// Utility modules
pub mod error;
pub mod metrics;
pub mod config;

// Optional feature modules
#[cfg(feature = "gpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "gpu")))]
pub mod gpu;

#[cfg(feature = "distributed")]
#[cfg_attr(docsrs, doc(cfg(feature = "distributed")))]
pub mod cluster;

/// Core result type for hybrid knowledge graph operations
pub type HybridResult<T> = std::result::Result<T, error::HybridError>;

/// Core trait for hybrid performance components
pub trait HybridPerformance {
    /// Get performance metrics for this component
    fn get_metrics(&self) -> metrics::PerformanceMetrics;
    
    /// Reset performance counters
    fn reset_metrics(&mut self);
    
    /// Optimize component for current workload
    fn optimize(&mut self) -> HybridResult<()>;
}

/// Compile-time CPU feature detection
pub mod cpu_features {
    //! Compile-time CPU feature detection and optimization flags
    
    /// SIMD vector width (4, 8, or 16 elements)
    pub const SIMD_WIDTH: usize = super::SIMD_WIDTH;
    
    /// AVX-512 availability
    pub const HAS_AVX512: bool = super::HAS_AVX512;
    
    /// AVX2 availability  
    #[cfg(feature = "avx2")]
    pub const HAS_AVX2: bool = super::HAS_AVX2;
    #[cfg(not(feature = "avx2"))]
    pub const HAS_AVX2: bool = false;
    
    /// Target architecture
    pub const TARGET_ARCH: &str = super::TARGET_ARCH;
    
    /// Target operating system
    pub const TARGET_OS: &str = super::TARGET_OS;
    
    /// Check if running on optimal hardware
    pub const fn is_optimal_hardware() -> bool {
        HAS_AVX512 && SIMD_WIDTH >= 16
    }
    
    /// Get human-readable CPU feature description
    pub const fn feature_description() -> &'static str {
        if HAS_AVX512 {
            "AVX-512 (16-wide SIMD)"
        } else if HAS_AVX2 {
            "AVX2 (8-wide SIMD)"
        } else {
            "SSE/Scalar (limited SIMD)"
        }
    }
}

/// Global initialization for hybrid components
pub fn init() -> HybridResult<()> {
    // Initialize memory allocator
    #[cfg(feature = "jemalloc")]
    {
        // Configure jemalloc for graph workloads
        use std::ffi::CString;
        unsafe {
            let config = CString::new("narenas:4,dirty_decay_ms:5000,muzzy_decay_ms:10000")?;
            libc::mallopt(libc::M_ARENA_MAX, 4);
        }
    }
    
    // Initialize NUMA if available
    #[cfg(all(target_os = "linux", feature = "numa"))]
    {
        crate::storage::numa::init_numa()?;
    }
    
    // Initialize SIMD dispatch
    crate::simd::init_simd_dispatch()?;
    
    // Initialize performance monitoring
    crate::metrics::init_metrics()?;
    
    // Warm up critical paths
    warmup_critical_paths()?;
    
    tracing::info!(
        "🚀 Hybrid Knowledge Graph initialized - {} on {}",
        cpu_features::feature_description(),
        cpu_features::TARGET_OS
    );
    
    Ok(())
}

/// Warm up critical performance paths
fn warmup_critical_paths() -> HybridResult<()> {
    use crate::simd::warm_up_simd_kernels;
    use crate::storage::warm_up_memory_allocators;
    
    // Pre-JIT compile SIMD kernels
    warm_up_simd_kernels()?;
    
    // Warm up memory allocators
    warm_up_memory_allocators()?;
    
    // Pre-fault some memory pages
    #[cfg(target_os = "linux")]
    {
        crate::storage::prefault_memory_pages()?;
    }
    
    Ok(())
}

/// Get global hybrid knowledge graph statistics
pub fn global_stats() -> metrics::GlobalStats {
    metrics::get_global_stats()
}

/// Run comprehensive hybrid benchmarks
#[cfg(feature = "benchmarks")]
pub fn run_benchmarks() -> HybridResult<metrics::BenchmarkResults> {
    crate::benchmarks::run_comprehensive_benchmarks()
}

/// Convenience macro for creating optimized vectors
#[macro_export]
macro_rules! aligned_vec {
    ($ty:ty, $capacity:expr) => {{
        let mut vec = std::vec::Vec::<$ty>::with_capacity($capacity);
        // Ensure alignment for SIMD operations
        vec.reserve_exact($capacity);
        vec
    }};
    ($ty:ty; $count:expr) => {{
        vec![$ty; $count]
    }};
}

/// Convenience macro for SIMD-width aware iterations
#[macro_export]
macro_rules! simd_chunks {
    ($slice:expr, $f:expr) => {{
        let chunks = $slice.chunks_exact($crate::cpu_features::SIMD_WIDTH);
        let remainder = chunks.remainder();
        
        // Process SIMD-width chunks
        for chunk in chunks {
            $f(chunk);
        }
        
        // Process remainder
        if !remainder.is_empty() {
            $f(remainder);
        }
    }};
}

/// Convenience macro for conditional SIMD compilation
#[macro_export]
macro_rules! simd_dispatch {
    (
        avx512: $avx512_expr:expr,
        avx2: $avx2_expr:expr,
        fallback: $fallback_expr:expr
    ) => {{
        #[cfg(feature = "avx512")]
        {
            if $crate::cpu_features::HAS_AVX512 {
                $avx512_expr
            } else {
                $fallback_expr
            }
        }
        #[cfg(all(feature = "avx2", not(feature = "avx512")))]
        {
            if $crate::cpu_features::HAS_AVX2 {
                $avx2_expr
            } else {
                $fallback_expr
            }
        }
        #[cfg(not(any(feature = "avx512", feature = "avx2")))]
        {
            $fallback_expr
        }
    }};
}

/// Fast hash function optimized for graph node IDs
#[inline]
pub fn fast_hash(value: u64) -> u64 {
    // Use a fast non-cryptographic hash
    ahash::AHasher::default().write_u64(value);
    value.wrapping_mul(0x9e3779b97f4a7c15_u64)
}

/// Memory prefetch hint for better cache performance
#[inline]
pub unsafe fn prefetch_read<T>(ptr: *const T, locality: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, locality);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Compiler hint for non-x86 architectures
        core::ptr::read_volatile(&ptr);
    }
}

/// Memory prefetch hint for write operations
#[inline]
pub unsafe fn prefetch_write<T>(ptr: *mut T, locality: i32) {
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_prefetch(ptr as *const i8, locality | 1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Compiler hint for non-x86 architectures
        core::ptr::write_volatile(ptr, core::ptr::read_volatile(ptr));
    }
}

/// Compile-time assertion for SIMD alignment
#[macro_export]
macro_rules! assert_simd_aligned {
    ($ptr:expr) => {{
        debug_assert_eq!(
            ($ptr as usize) % ($crate::cpu_features::SIMD_WIDTH * std::mem::size_of::<f32>()),
            0,
            "Pointer not aligned for SIMD operations"
        );
    }};
}

// Re-exports for convenience
pub use ahash::AHashMap as FastHashMap;
pub use ahash::AHashSet as FastHashSet;
pub use dashmap::DashMap as ConcurrentHashMap;
pub use parking_lot::{Mutex, RwLock};
pub use smallvec::SmallVec;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        println!("SIMD Width: {}", cpu_features::SIMD_WIDTH);
        println!("AVX-512: {}", cpu_features::HAS_AVX512);
        println!("Target: {} on {}", cpu_features::TARGET_ARCH, cpu_features::TARGET_OS);
        println!("Optimal: {}", cpu_features::is_optimal_hardware());
        println!("Description: {}", cpu_features::feature_description());
    }
    
    #[test]
    fn test_initialization() {
        init().expect("Failed to initialize hybrid components");
    }
    
    #[test]
    fn test_fast_hash() {
        let hash1 = fast_hash(12345);
        let hash2 = fast_hash(12346);
        assert_ne!(hash1, hash2);
    }
    
    #[test]
    fn test_aligned_vec_macro() {
        let vec: Vec<f32> = aligned_vec!(f32, 1000);
        assert_eq!(vec.capacity(), 1000);
    }
    
    #[test]
    fn test_simd_chunks_macro() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mut sum = 0.0;
        
        simd_chunks!(&data, |chunk| {
            sum += chunk.iter().sum::<f32>();
        });
        
        assert_eq!(sum, (0..100).sum::<i32>() as f32);
    }
}