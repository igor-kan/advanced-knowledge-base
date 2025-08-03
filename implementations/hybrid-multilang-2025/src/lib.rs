/*!
# Hybrid Multi-Language Knowledge Graph Engine

This crate provides an ultra-high performance knowledge graph engine that combines
the strengths of multiple programming languages:

- **Rust**: Memory safety, concurrency, and modern systems programming
- **C++**: SIMD optimization, mature algorithms, and GPU programming  
- **Fortran**: Unmatched numerical computing performance
- **C**: Direct system integration and minimal overhead

## Architecture

The engine is designed as a layered architecture where Rust provides the safe
coordination layer, while specialized components are implemented in the most
appropriate language for their specific performance requirements.

## Quick Start

```rust
use hybrid_kg::{HybridGraph, GraphConfig, Algorithm, AnalysisRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = HybridGraph::new(GraphConfig::default()).await?;
    
    // Load data
    graph.load_from_file("knowledge_graph.kg").await?;
    
    // Run analysis
    let results = graph.execute_analysis(AnalysisRequest {
        algorithm: Algorithm::PageRank,
        iterations: 100,
        use_gpu_acceleration: true,
        ..Default::default()
    }).await?;
    
    println!("Processed {} nodes in {:?}", 
             results.nodes_processed, 
             results.execution_time);
    
    Ok(())
}
```

## Features

- **Massive Scale**: Handle billions of nodes and edges
- **High Performance**: Optimized algorithms in multiple languages
- **Memory Safety**: Rust's safety guarantees across language boundaries
- **GPU Acceleration**: CUDA kernels for parallel graph algorithms
- **SIMD Optimization**: Vectorized operations for maximum throughput
- **Numerical Excellence**: Fortran-based mathematical computations
*/

pub mod config;
pub mod core;
pub mod algorithms;
pub mod query;
pub mod storage;
pub mod ffi;
pub mod metrics;
pub mod error;

// Re-export main types
pub use config::{GraphConfig, AlgorithmConfig};
pub use core::{HybridGraph, NodeId, EdgeId};
pub use algorithms::{Algorithm, AnalysisRequest, AnalysisResults};
pub use query::{Query, QueryResult, QueryEngine};
pub use storage::{GraphStorage, StorageBackend};
pub use error::{HybridError, Result};
pub use metrics::{PerformanceMetrics, MetricsCollector};

use tracing::info;
use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize the hybrid knowledge graph engine
/// 
/// This function sets up logging, initializes foreign function interfaces,
/// and prepares all language backends for operation.
pub fn init() -> Result<()> {
    INIT.call_once(|| {
        // Initialize tracing
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::from_default_env()
                    .add_directive("hybrid_kg=info".parse().unwrap())
            )
            .init();
        
        info!("Initializing Hybrid Knowledge Graph Engine");
        
        // Initialize FFI components
        #[cfg(feature = "cpp-backend")]
        ffi::cpp::initialize().expect("Failed to initialize C++ backend");
        
        #[cfg(feature = "fortran-numerics")]
        ffi::fortran::initialize().expect("Failed to initialize Fortran numerics");
        
        #[cfg(feature = "c-system")]
        ffi::c_system::initialize().expect("Failed to initialize C system layer");
        
        #[cfg(feature = "gpu")]
        ffi::gpu::initialize().expect("Failed to initialize GPU support");
        
        info!("Hybrid Knowledge Graph Engine initialized successfully");
    });
    
    Ok(())
}

/// Get version information for all language components
pub fn version_info() -> std::collections::HashMap<String, String> {
    let mut versions = std::collections::HashMap::new();
    
    versions.insert("rust".to_string(), env!("CARGO_PKG_VERSION").to_string());
    
    #[cfg(feature = "cpp-backend")]
    versions.insert("cpp".to_string(), ffi::cpp::version());
    
    #[cfg(feature = "fortran-numerics")]
    versions.insert("fortran".to_string(), ffi::fortran::version());
    
    #[cfg(feature = "c-system")]
    versions.insert("c-system".to_string(), ffi::c_system::version());
    
    versions
}

/// Performance benchmark suite
#[cfg(feature = "profiling")]
pub mod benchmark {
    use super::*;
    use std::time::Instant;
    
    /// Run comprehensive performance benchmarks
    pub async fn run_benchmarks() -> Result<BenchmarkResults> {
        let config = GraphConfig {
            enable_gpu: true,
            enable_simd: true,
            enable_parallel: true,
            ..Default::default()
        };
        
        let graph = HybridGraph::new(config).await?;
        
        // Generate test data
        let start = Instant::now();
        graph.generate_test_data(1_000_000, 10_000_000).await?;
        let data_generation_time = start.elapsed();
        
        // Benchmark different algorithms
        let mut results = BenchmarkResults::new();
        results.data_generation_time = data_generation_time;
        
        // PageRank benchmark
        let start = Instant::now();
        let _pr_results = graph.execute_analysis(AnalysisRequest {
            algorithm: Algorithm::PageRank,
            iterations: 50,
            use_gpu_acceleration: true,
            ..Default::default()
        }).await?;
        results.pagerank_time = start.elapsed();
        
        // Graph traversal benchmark
        let start = Instant::now();
        let _traversal_results = graph.execute_analysis(AnalysisRequest {
            algorithm: Algorithm::BreadthFirstSearch,
            start_nodes: vec![0],
            use_simd_optimization: true,
            ..Default::default()
        }).await?;
        results.traversal_time = start.elapsed();
        
        // Numerical computation benchmark
        let start = Instant::now();
        let _numerical_results = graph.execute_analysis(AnalysisRequest {
            algorithm: Algorithm::SpectralClustering,
            clustering_dimensions: 128,
            use_fortran_numerics: true,
            ..Default::default()
        }).await?;
        results.numerical_time = start.elapsed();
        
        Ok(results)
    }
    
    #[derive(Debug)]
    pub struct BenchmarkResults {
        pub data_generation_time: std::time::Duration,
        pub pagerank_time: std::time::Duration,
        pub traversal_time: std::time::Duration,  
        pub numerical_time: std::time::Duration,
    }
    
    impl BenchmarkResults {
        pub fn new() -> Self {
            Self {
                data_generation_time: std::time::Duration::ZERO,
                pagerank_time: std::time::Duration::ZERO,
                traversal_time: std::time::Duration::ZERO,
                numerical_time: std::time::Duration::ZERO,
            }
        }
        
        pub fn total_time(&self) -> std::time::Duration {
            self.data_generation_time + 
            self.pagerank_time + 
            self.traversal_time + 
            self.numerical_time
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_initialization() {
        assert!(init().is_ok());
    }
    
    #[tokio::test]
    async fn test_version_info() {
        init().unwrap();
        let versions = version_info();
        assert!(versions.contains_key("rust"));
    }
    
    #[tokio::test]
    async fn test_basic_graph_operations() {
        init().unwrap();
        
        let graph = HybridGraph::new(GraphConfig::default()).await.unwrap();
        
        // Add some test data
        let node1 = graph.add_node("person".to_string(), 
                                  std::collections::HashMap::new()).await.unwrap();
        let node2 = graph.add_node("company".to_string(), 
                                  std::collections::HashMap::new()).await.unwrap();
        
        let _edge = graph.add_edge(node1, node2, "works_at".to_string(), 
                                  std::collections::HashMap::new()).await.unwrap();
        
        let node_count = graph.node_count().await.unwrap();
        let edge_count = graph.edge_count().await.unwrap();
        
        assert_eq!(node_count, 2);
        assert_eq!(edge_count, 1);
    }
}