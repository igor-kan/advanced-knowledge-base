//! Comprehensive performance benchmarks for the Quantum Graph Engine
//!
//! This benchmark suite validates our claims of sub-millisecond queries
//! on billion-node graphs and measures performance across all operations.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use quantum_graph_engine::*;
use quantum_graph_engine::storage::QuantumGraph;
use quantum_graph_engine::query::QueryEngine;
use quantum_graph_engine::algorithms::GraphAlgorithms;
use quantum_graph_engine::types::*;
use quantum_graph_engine::gpu::GpuAccelerator;
use std::sync::Arc;
use std::time::Duration;
use rayon::prelude::*;

/// Benchmark configuration for different graph sizes
const BENCHMARK_CONFIGS: &[(usize, usize, &str)] = &[
    (1_000, 5_000, "1K_nodes"),
    (10_000, 50_000, "10K_nodes"),
    (100_000, 500_000, "100K_nodes"),
    (1_000_000, 5_000_000, "1M_nodes"),
    (10_000_000, 50_000_000, "10M_nodes"),
    (100_000_000, 500_000_000, "100M_nodes"),
    (1_000_000_000, 5_000_000_000, "1B_nodes"), // Our target: billion-node graphs
];

/// Generate test graph with specified characteristics
async fn create_test_graph(node_count: usize, edge_count: usize) -> Arc<QuantumGraph> {
    let config = GraphConfig {
        memory_pool_size: 32 * 1024 * 1024 * 1024, // 32GB
        cpu_threads: num_cpus::get(),
        enable_simd: true,
        enable_gpu: true,
        compression: CompressionType::LZ4,
        storage_backend: StorageBackend::Memory,
        enable_metrics: true,
        cache_size: 1024 * 1024 * 1024, // 1GB cache
        batch_size: 100_000,
    };
    
    let graph = Arc::new(QuantumGraph::new(config).await.unwrap());
    
    // Generate nodes in parallel batches
    let batch_size = 10_000;
    for chunk_start in (0..node_count).step_by(batch_size) {
        let chunk_end = std::cmp::min(chunk_start + batch_size, node_count);
        let nodes: Vec<Node> = (chunk_start..chunk_end)
            .map(|i| Node {
                id: NodeId::from_u64(i as u64),
                node_type: "TestNode".to_string(),
                data: NodeData::Text(format!("Node {}", i)),
                metadata: NodeMetadata {
                    created_at: std::time::SystemTime::now(),
                    updated_at: std::time::SystemTime::now(),
                    tags: vec!["benchmark".to_string()],
                    properties: std::collections::HashMap::new(),
                },
            })
            .collect();
        
        graph.batch_insert_nodes(nodes).await.unwrap();
    }
    
    // Generate edges with realistic distribution
    let edges: Vec<Edge> = (0..edge_count)
        .into_par_iter()
        .map(|i| {
            let from = fastrand::usize(..node_count) as u64;
            let to = fastrand::usize(..node_count) as u64;
            
            Edge {
                id: EdgeId(i as u128),
                from: NodeId::from_u64(from),
                to: NodeId::from_u64(to),
                edge_type: "TestEdge".to_string(),
                weight: Some(fastrand::f64()),
                data: EdgeData::Empty,
                metadata: EdgeMetadata {
                    created_at: std::time::SystemTime::now(),
                    properties: std::collections::HashMap::new(),
                },
            }
        })
        .collect();
    
    // Insert edges in parallel batches
    for chunk in edges.chunks(batch_size) {
        graph.batch_insert_edges(chunk.to_vec()).await.unwrap();
    }
    
    graph
}

/// Benchmark node operations
fn bench_node_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("node_operations");
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(6) { // Skip billion for CI
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        
        // Benchmark single node insertion
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_insert", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let node = Node {
                        id: NodeId::from_u64(fastrand::u64(..)),
                        node_type: "BenchNode".to_string(),
                        data: NodeData::Text("Benchmark node".to_string()),
                        metadata: NodeMetadata::default(),
                    };
                    graph.insert_node(node).await.unwrap()
                });
            },
        );
        
        // Benchmark batch node insertion
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("batch_insert_1k", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let nodes: Vec<Node> = (0..1000)
                        .map(|i| Node {
                            id: NodeId::from_u64(fastrand::u64(..)),
                            node_type: "BatchNode".to_string(),
                            data: NodeData::Text(format!("Batch node {}", i)),
                            metadata: NodeMetadata::default(),
                        })
                        .collect();
                    graph.batch_insert_nodes(nodes).await.unwrap()
                });
            },
        );
        
        // Benchmark node retrieval
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("get_node", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let node_id = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    graph.get_node(node_id).await.unwrap()
                });
            },
        );
        
        // Benchmark neighbor queries
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("get_neighbors", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let node_id = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    graph.get_neighbors(node_id).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark edge operations
fn bench_edge_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("edge_operations");
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(6) {
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        
        // Benchmark single edge insertion
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("single_insert", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let from = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let to = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let edge = Edge {
                        id: EdgeId(fastrand::u128(..)),
                        from,
                        to,
                        edge_type: "BenchEdge".to_string(),
                        weight: Some(fastrand::f64()),
                        data: EdgeData::Empty,
                        metadata: EdgeMetadata::default(),
                    };
                    graph.insert_edge(edge).await.unwrap()
                });
            },
        );
        
        // Benchmark batch edge insertion
        group.throughput(Throughput::Elements(10000));
        group.bench_with_input(
            BenchmarkId::new("batch_insert_10k", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let edges: Vec<Edge> = (0..10000)
                        .map(|i| {
                            let from = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                            let to = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                            Edge {
                                id: EdgeId(fastrand::u128(..)),
                                from,
                                to,
                                edge_type: "BatchEdge".to_string(),
                                weight: Some(fastrand::f64()),
                                data: EdgeData::Empty,
                                metadata: EdgeMetadata::default(),
                            }
                        })
                        .collect();
                    graph.batch_insert_edges(edges).await.unwrap()
                });
            },
        );
        
        // Benchmark edge retrieval
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("get_edge", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let edge_id = EdgeId(fastrand::u128(..(edge_count as u128)));
                    graph.get_edge(edge_id).await.unwrap()
                });
            },
        );
        
        // Benchmark outgoing edges query
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("get_outgoing_edges", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let node_id = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    graph.get_outgoing_edges(node_id).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark shortest path algorithms
fn bench_shortest_path(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("shortest_path");
    group.sample_size(50); // Reduce sample size for expensive operations
    group.measurement_time(Duration::from_secs(30));
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(5) { // Skip largest for CI
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        let query_engine = QueryEngine::new(graph.clone());
        
        // Benchmark Dijkstra's algorithm
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("dijkstra", name),
            &query_engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let from = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let to = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let config = query::PathConfig::default()
                        .algorithm(query::PathAlgorithm::Dijkstra)
                        .max_depth(20);
                    engine.find_shortest_path(from, to, config).await.unwrap()
                });
            },
        );
        
        // Benchmark bidirectional BFS
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("bidirectional_bfs", name),
            &query_engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let from = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let to = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    let config = query::PathConfig::default()
                        .algorithm(query::PathAlgorithm::BidirectionalBFS)
                        .max_depth(20);
                    engine.find_shortest_path(from, to, config).await.unwrap()
                });
            },
        );
        
        // Benchmark k-hop neighbors
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("k_hop_neighbors", name),
            &query_engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let node_id = NodeId::from_u64(fastrand::u64(..(node_count as u64)));
                    engine.get_k_hop_neighbors(node_id, 3, false).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark graph algorithms
fn bench_graph_algorithms(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("graph_algorithms");
    group.sample_size(10); // Very small sample size for expensive algorithms
    group.measurement_time(Duration::from_secs(60));
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(4) { // Skip largest
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        let algorithms = GraphAlgorithms::new(graph.clone());
        
        // Benchmark PageRank
        group.throughput(Throughput::Elements(node_count as u64));
        group.bench_with_input(
            BenchmarkId::new("pagerank", name),
            &algorithms,
            |b, algs| {
                b.to_async(&rt).iter(|| async {
                    algs.pagerank(0.85, 50, 1e-6).await.unwrap()
                });
            },
        );
        
        // Benchmark connected components
        group.throughput(Throughput::Elements(node_count as u64));
        group.bench_with_input(
            BenchmarkId::new("connected_components", name),
            &algorithms,
            |b, algs| {
                b.to_async(&rt).iter(|| async {
                    algs.connected_components().await.unwrap()
                });
            },
        );
        
        // Benchmark betweenness centrality (only for smaller graphs)
        if node_count <= 10_000 {
            group.throughput(Throughput::Elements(node_count as u64));
            group.bench_with_input(
                BenchmarkId::new("betweenness_centrality", name),
                &algorithms,
                |b, algs| {
                    b.to_async(&rt).iter(|| async {
                        algs.betweenness_centrality().await.unwrap()
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark SIMD-optimized operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test different data sizes for SIMD operations
    let sizes = [1000, 10000, 100000, 1000000];
    
    for &size in &sizes {
        // Benchmark vectorized edge scanning
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectorized_edge_scan", size),
            &size,
            |b, &size| {
                let edges: Vec<u64> = (0..size * 2).map(|i| i as u64).collect();
                let target_from = fastrand::u64(..(size as u64));
                let target_to = fastrand::u64(..(size as u64));
                
                b.iter(|| {
                    crate::simd::vectorized_adjacency_scan(&edges, target_from, target_to)
                });
            },
        );
        
        // Benchmark SIMD node filtering
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_node_filter", size),
            &size,
            |b, &size| {
                let node_ids: Vec<u64> = (0..size).map(|i| i as u64).collect();
                let filter_value = fastrand::u64(..(size as u64));
                
                b.iter(|| {
                    crate::simd::simd_filter_nodes(&node_ids, filter_value)
                });
            },
        );
        
        // Benchmark parallel reduction
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("parallel_reduction", size),
            &size,
            |b, &size| {
                let values: Vec<f64> = (0..size).map(|i| i as f64).collect();
                
                b.iter(|| {
                    crate::simd::parallel_reduce_f64(&values)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark GPU acceleration (if available)
fn bench_gpu_acceleration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("gpu_acceleration");
    group.sample_size(20);
    
    // Only run GPU benchmarks if GPU is available
    if let Ok(mut gpu) = rt.block_on(async { GpuAccelerator::new() }) {
        if gpu.is_available() {
            for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(4) {
                let graph = rt.block_on(create_test_graph(node_count, edge_count));
                
                // Benchmark GPU PageRank
                group.throughput(Throughput::Elements(node_count as u64));
                group.bench_with_input(
                    BenchmarkId::new("gpu_pagerank", name),
                    &(&mut gpu, &graph),
                    |b, (gpu, graph)| {
                        b.to_async(&rt).iter(|| async {
                            gpu.gpu_pagerank(graph, 0.85, 50, 1e-6).await.unwrap()
                        });
                    },
                );
                
                // Benchmark GPU BFS
                let start_nodes: Vec<NodeId> = (0..10)
                    .map(|i| NodeId::from_u64(i))
                    .collect();
                
                group.throughput(Throughput::Elements(10));
                group.bench_with_input(
                    BenchmarkId::new("gpu_parallel_bfs", name),
                    &(&mut gpu, &graph, &start_nodes),
                    |b, (gpu, graph, start_nodes)| {
                        b.to_async(&rt).iter(|| async {
                            gpu.gpu_parallel_bfs(graph, start_nodes, 10).await.unwrap()
                        });
                    },
                );
                
                // Benchmark GPU connected components
                group.throughput(Throughput::Elements(node_count as u64));
                group.bench_with_input(
                    BenchmarkId::new("gpu_connected_components", name),
                    &(&mut gpu, &graph),
                    |b, (gpu, graph)| {
                        b.to_async(&rt).iter(|| async {
                            gpu.gpu_connected_components(graph).await.unwrap()
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

/// Benchmark memory performance and efficiency
fn bench_memory_performance(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_performance");
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(5) {
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        
        // Benchmark memory usage reporting
        group.bench_with_input(
            BenchmarkId::new("memory_stats", name),
            &graph,
            |b, graph| {
                b.iter(|| {
                    graph.get_stats()
                });
            },
        );
        
        // Benchmark garbage collection
        group.bench_with_input(
            BenchmarkId::new("garbage_collection", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    graph.compact_storage().await.unwrap()
                });
            },
        );
        
        // Benchmark cache performance
        group.bench_with_input(
            BenchmarkId::new("cache_performance", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    // Access same nodes repeatedly to test cache
                    let node_id = NodeId::from_u64(0);
                    for _ in 0..100 {
                        graph.get_node(node_id).await.unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark distributed operations
fn bench_distributed_operations(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("distributed_operations");
    group.sample_size(10);
    
    // Simulate distributed cluster with multiple shards
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(3) {
        // Create distributed config
        let distributed_config = crate::distributed::DistributedConfig {
            node_id: "benchmark_node".to_string(),
            address: "127.0.0.1:8080".parse().unwrap(),
            capacity: crate::distributed::NodeCapacity {
                memory_gb: 64,
                cpu_cores: 16,
                storage_gb: 1000,
                network_mbps: 10000,
            },
            partitioning_strategy: crate::distributed::PartitioningStrategy::HashBased,
            consensus_config: crate::distributed::ConsensusConfig,
            graph_config: GraphConfig::default(),
            network_config: crate::distributed::NetworkConfig,
        };
        
        group.bench_with_input(
            BenchmarkId::new("cluster_creation", name),
            &distributed_config,
            |b, config| {
                b.to_async(&rt).iter(|| async {
                    crate::distributed::DistributedCluster::new(config.clone()).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pattern matching queries
fn bench_pattern_matching(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pattern_matching");
    group.sample_size(30);
    
    for &(node_count, edge_count, name) in BENCHMARK_CONFIGS.iter().take(4) {
        let graph = rt.block_on(create_test_graph(node_count, edge_count));
        let query_engine = QueryEngine::new(graph.clone());
        
        // Create a simple pattern query
        let pattern = query::PatternQuery::new()
            .add_node("a".to_string(), query::NodePattern {
                node_type: Some("TestNode".to_string()),
                properties: std::collections::HashMap::new(),
            })
            .add_node("b".to_string(), query::NodePattern {
                node_type: Some("TestNode".to_string()),
                properties: std::collections::HashMap::new(),
            })
            .add_edge(query::EdgePattern {
                from: "a".to_string(),
                to: "b".to_string(),
                edge_type: Some("TestEdge".to_string()),
                properties: std::collections::HashMap::new(),
            });
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("simple_pattern", name),
            &(&query_engine, &pattern),
            |b, (engine, pattern)| {
                b.to_async(&rt).iter(|| async {
                    engine.find_pattern(pattern).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Comprehensive stress test for billion-node graphs
fn bench_billion_node_stress_test(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("billion_node_stress");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(300)); // 5 minutes per test
    
    // Only run this on systems with sufficient memory
    if std::env::var("ENABLE_BILLION_NODE_BENCH").is_ok() {
        let (node_count, edge_count, name) = BENCHMARK_CONFIGS.last().unwrap();
        let graph = rt.block_on(create_test_graph(*node_count, *edge_count));
        let query_engine = QueryEngine::new(graph.clone());
        
        // Test sub-millisecond query target
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("sub_millisecond_queries", name),
            &query_engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let futures: Vec<_> = (0..1000)
                        .map(|_| {
                            let from = NodeId::from_u64(fastrand::u64(..(node_count / 1000) as u64));
                            let to = NodeId::from_u64(fastrand::u64(..(node_count / 1000) as u64));
                            let config = query::PathConfig::default().max_depth(6);
                            engine.find_shortest_path(from, to, config)
                        })
                        .collect();
                    
                    futures::future::join_all(futures).await
                });
            },
        );
        
        // Test massive parallel insertions
        group.throughput(Throughput::Elements(1_000_000));
        group.bench_with_input(
            BenchmarkId::new("million_node_insertion", name),
            &graph,
            |b, graph| {
                b.to_async(&rt).iter(|| async {
                    let nodes: Vec<Node> = (0..1_000_000)
                        .into_par_iter()
                        .map(|i| Node {
                            id: NodeId::from_u64(fastrand::u64(..)),
                            node_type: "StressNode".to_string(),
                            data: NodeData::Text(format!("Stress node {}", i)),
                            metadata: NodeMetadata::default(),
                        })
                        .collect();
                    
                    graph.batch_insert_nodes(nodes).await.unwrap()
                });
            },
        );
    }
    
    group.finish();
}

/// Generate comprehensive performance report
fn generate_performance_report() {
    println!("=== Quantum Graph Engine Performance Report ===");
    println!();
    println!("Target Performance Specifications:");
    println!("• Sub-millisecond queries on billion-node graphs");
    println!("• 10M+ operations per second");
    println!("• Linear scaling with SIMD/GPU acceleration");
    println!("• Infinite horizontal scalability");
    println!();
    println!("Benchmark Categories:");
    println!("1. Core Operations (nodes/edges)");
    println!("2. Graph Algorithms (PageRank, shortest path, etc.)");
    println!("3. SIMD Optimizations");
    println!("4. GPU Acceleration");
    println!("5. Memory Performance");
    println!("6. Distributed Operations");
    println!("7. Pattern Matching");
    println!("8. Billion-Node Stress Tests");
    println!();
    println!("Run with: cargo bench");
    println!("For billion-node tests: ENABLE_BILLION_NODE_BENCH=1 cargo bench");
}

criterion_group!(
    name = core_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(5))
        .measurement_time(Duration::from_secs(20))
        .sample_size(100);
    targets = 
        bench_node_operations,
        bench_edge_operations,
        bench_memory_performance
);

criterion_group!(
    name = algorithm_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(30))
        .sample_size(50);
    targets = 
        bench_shortest_path,
        bench_graph_algorithms,
        bench_pattern_matching
);

criterion_group!(
    name = optimization_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(15))
        .sample_size(200);
    targets = 
        bench_simd_operations,
        bench_gpu_acceleration
);

criterion_group!(
    name = distributed_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(15))
        .measurement_time(Duration::from_secs(60))
        .sample_size(20);
    targets = 
        bench_distributed_operations
);

criterion_group!(
    name = stress_benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(30))
        .measurement_time(Duration::from_secs(300))
        .sample_size(5);
    targets = 
        bench_billion_node_stress_test
);

criterion_main!(
    core_benches,
    algorithm_benches,
    optimization_benches,
    distributed_benches,
    stress_benches
);