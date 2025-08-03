//! Comprehensive benchmarks for ultra-fast knowledge graph operations
//!
//! This benchmark suite tests:
//! - Node and edge creation performance
//! - Query execution speed
//! - SIMD operation efficiency
//! - Memory usage patterns
//! - Scalability characteristics

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ultra_fast_knowledge_graph::*;
use std::time::Duration;
use rayon::prelude::*;

const SMALL_GRAPH_NODES: usize = 1_000;
const MEDIUM_GRAPH_NODES: usize = 100_000;
const LARGE_GRAPH_NODES: usize = 1_000_000;
const HUGE_GRAPH_NODES: usize = 10_000_000;

/// Benchmark node creation performance
fn bench_node_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_creation");
    
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES, LARGE_GRAPH_NODES] {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("single_threaded", size), &size, |b, &size| {
            let config = GraphConfig::default();
            let graph = UltraFastKnowledgeGraph::new(config).unwrap();
            
            b.iter(|| {
                for i in 0..size {
                    let data = NodeData::new(
                        format!("node_{}", i),
                        serde_json::json!({"type": "benchmark", "index": i})
                    );
                    black_box(graph.create_node(data).unwrap());
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("batch_parallel", size), &size, |b, &size| {
            let config = GraphConfig::default();
            let graph = UltraFastKnowledgeGraph::new(config).unwrap();
            
            b.iter(|| {
                let nodes: Vec<NodeData> = (0..size).map(|i| {
                    NodeData::new(
                        format!("node_{}", i),
                        serde_json::json!({"type": "benchmark", "index": i})
                    )
                }).collect();
                
                black_box(graph.batch_create_nodes(nodes).unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark edge creation performance
fn bench_edge_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_creation");
    
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES, LARGE_GRAPH_NODES] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Setup: Create nodes first
        let config = GraphConfig::default();
        let graph = UltraFastKnowledgeGraph::new(config).unwrap();
        
        let nodes: Vec<NodeData> = (0..size).map(|i| {
            NodeData::new(
                format!("node_{}", i),
                serde_json::json!({"type": "benchmark"})
            )
        }).collect();
        
        let node_ids = graph.batch_create_nodes(nodes).unwrap();
        
        group.bench_with_input(BenchmarkId::new("single_threaded", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size.min(10000) { // Limit edges for reasonable test time
                    let from = node_ids[i % node_ids.len()];
                    let to = node_ids[(i + 1) % node_ids.len()];
                    let weight = Weight(1.0 + (i as f32) * 0.1);
                    let data = EdgeData::new(serde_json::json!({"type": "benchmark"}));
                    
                    black_box(graph.create_edge(from, to, weight, data).unwrap());
                }
            });
        });
        
        group.bench_with_input(BenchmarkId::new("batch_parallel", size), &size, |b, &size| {
            b.iter(|| {
                let edges: Vec<(NodeId, NodeId, Weight, EdgeData)> = (0..size.min(10000))
                    .map(|i| {
                        let from = node_ids[i % node_ids.len()];
                        let to = node_ids[(i + 1) % node_ids.len()];
                        let weight = Weight(1.0 + (i as f32) * 0.1);
                        let data = EdgeData::new(serde_json::json!({"type": "benchmark"}));
                        (from, to, weight, data)
                    }).collect();
                
                black_box(graph.batch_create_edges(edges).unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark graph traversal algorithms
fn bench_traversal_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal_algorithms");
    group.measurement_time(Duration::from_secs(10));
    
    // Create test graphs of different sizes
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES] {
        let graph = create_test_graph(size, size * 2);
        
        group.bench_with_input(BenchmarkId::new("bfs", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.traverse_bfs(0, Some(5)).unwrap());
            });
        });
        
        group.bench_with_input(BenchmarkId::new("shortest_path", size), &size, |b, _| {
            b.iter(|| {
                let target = (size / 2) as NodeId;
                black_box(graph.shortest_path(0, target).unwrap());
            });
        });
        
        group.bench_with_input(BenchmarkId::new("neighborhood", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.get_neighborhood(0, 3).unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark centrality algorithms
fn bench_centrality_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("centrality_algorithms");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10); // Fewer samples for expensive algorithms
    
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES] {
        let graph = create_test_graph(size, size * 2);
        
        group.bench_with_input(BenchmarkId::new("degree_centrality", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.compute_centrality(CentralityAlgorithm::Degree).unwrap());
            });
        });
        
        group.bench_with_input(BenchmarkId::new("pagerank", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.compute_centrality(CentralityAlgorithm::PageRank).unwrap());
            });
        });
        
        if size <= SMALL_GRAPH_NODES {
            group.bench_with_input(BenchmarkId::new("betweenness_centrality", size), &size, |b, _| {
                b.iter(|| {
                    black_box(graph.compute_centrality(CentralityAlgorithm::Betweenness).unwrap());
                });
            });
        }
    }
    
    group.finish();
}

/// Benchmark pattern matching performance
fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    group.measurement_time(Duration::from_secs(10));
    
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES] {
        let graph = create_test_graph(size, size * 2);
        
        // Simple pattern: A -> B
        let simple_pattern = create_simple_pattern();
        
        group.bench_with_input(BenchmarkId::new("simple_pattern", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.find_pattern(&simple_pattern).unwrap());
            });
        });
        
        // Complex pattern: A -> B -> C with constraints
        let complex_pattern = create_complex_pattern();
        
        group.bench_with_input(BenchmarkId::new("complex_pattern", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.find_pattern(&complex_pattern).unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark SIMD operations
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    let simd_processor = simd::SimdProcessor::new();
    
    for &size in &[1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(size as u64));
        
        // Distance update benchmark
        group.bench_with_input(BenchmarkId::new("distance_update", size), &size, |b, &size| {
            let mut distances = vec![f32::INFINITY; size];
            let new_distances: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let mask = vec![true; size];
            
            b.iter(|| {
                black_box(simd_processor.simd_distance_update(
                    &mut distances, &new_distances, &mask
                ).unwrap());
            });
        });
        
        // Neighbor counting benchmark
        group.bench_with_input(BenchmarkId::new("neighbor_counting", size), &size, |b, &size| {
            let adjacency_chunks: Vec<u64> = (0..size/8).map(|i| i as u64 | 0xFF00000000000000).collect();
            let node_masks: Vec<u64> = (0..size/8).map(|i| if i % 2 == 0 { u64::MAX } else { 0 }).collect();
            
            b.iter(|| {
                black_box(simd_processor.simd_count_neighbors(
                    &adjacency_chunks, &node_masks
                ).unwrap());
            });
        });
        
        // Pattern matching benchmark
        group.bench_with_input(BenchmarkId::new("pattern_matching", size), &size, |b, &size| {
            let candidates: Vec<NodeId> = (0..size as NodeId).collect();
            let pattern_constraints = vec![0.5f32; 16]; // 16 constraints
            let node_features: Vec<f32> = (0..size * 16).map(|i| (i % 100) as f32 / 100.0).collect();
            
            b.iter(|| {
                black_box(simd_processor.simd_pattern_match(
                    &candidates, &pattern_constraints, &node_features, 16
                ).unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory operations
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");
    
    for &size in &[1024, 4096, 16384, 65536, 262144] {
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(BenchmarkId::new("simd_memcpy", size), &size, |b, &size| {
            let src = vec![42u8; size];
            let mut dst = vec![0u8; size];
            
            b.iter(|| {
                simd::SimdMemory::simd_memcpy(&mut dst, &src).unwrap();
                black_box(&dst);
            });
        });
        
        group.bench_with_input(BenchmarkId::new("simd_memcmp", size), &size, |b, &size| {
            let a = vec![42u8; size];
            let b_vec = vec![42u8; size];
            
            b.iter(|| {
                black_box(simd::SimdMemory::simd_memcmp(&a, &b_vec));
            });
        });
        
        group.bench_with_input(BenchmarkId::new("simd_memset", size), &size, |b, &size| {
            let mut data = vec![0u8; size];
            
            b.iter(|| {
                simd::SimdMemory::simd_memset(&mut data, 255);
                black_box(&data);
            });
        });
    }
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(10));
    
    for &thread_count in &[1, 2, 4, 8, 16] {
        for &operations_per_thread in &[1000, 10000] {
            let total_ops = thread_count * operations_per_thread;
            
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}threads_{}ops", thread_count, total_ops)), 
                &(thread_count, operations_per_thread), 
                |b, &(threads, ops_per_thread)| {
                    let config = GraphConfig::default();
                    let graph = UltraFastKnowledgeGraph::new(config).unwrap();
                    
                    b.iter(|| {
                        // Create nodes concurrently
                        (0..threads).into_par_iter().for_each(|thread_id| {
                            for i in 0..ops_per_thread {
                                let node_id = thread_id * ops_per_thread + i;
                                let data = NodeData::new(
                                    format!("node_{}_{}", thread_id, i),
                                    serde_json::json!({"thread": thread_id, "index": i})
                                );
                                black_box(graph.create_node(data).unwrap());
                            }
                        });
                    });
                }
            );
        }
    }
    
    group.finish();
}

/// Benchmark query performance under different loads
fn bench_query_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_performance");
    group.measurement_time(Duration::from_secs(10));
    
    // Create a medium-sized graph for testing
    let graph = create_test_graph(MEDIUM_GRAPH_NODES, MEDIUM_GRAPH_NODES * 2);
    
    for &concurrent_queries in &[1, 4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_bfs", concurrent_queries), 
            &concurrent_queries, 
            |b, &query_count| {
                b.iter(|| {
                    (0..query_count).into_par_iter().for_each(|i| {
                        let start_node = (i * 1000) as NodeId % (MEDIUM_GRAPH_NODES as NodeId);
                        black_box(graph.traverse_bfs(start_node, Some(4)).unwrap());
                    });
                });
            }
        );
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_shortest_path", concurrent_queries), 
            &concurrent_queries, 
            |b, &query_count| {
                b.iter(|| {
                    (0..query_count).into_par_iter().for_each(|i| {
                        let start = (i * 1000) as NodeId % (MEDIUM_GRAPH_NODES as NodeId);
                        let end = ((i + 1) * 1000) as NodeId % (MEDIUM_GRAPH_NODES as NodeId);
                        black_box(graph.shortest_path(start, end).unwrap());
                    });
                });
            }
        );
    }
    
    group.finish();
}

/// Benchmark memory usage and scalability
fn bench_memory_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scalability");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
    
    for &node_count in &[1000, 10000, 100000, 1000000] {
        group.bench_with_input(BenchmarkId::new("memory_usage", node_count), &node_count, |b, &nodes| {
            b.iter_with_setup(
                || {
                    let config = GraphConfig {
                        initial_node_capacity: nodes,
                        initial_edge_capacity: nodes * 2,
                        ..Default::default()
                    };
                    UltraFastKnowledgeGraph::new(config).unwrap()
                },
                |graph| {
                    // Create nodes and measure memory
                    let nodes_data: Vec<NodeData> = (0..nodes).map(|i| {
                        NodeData::new(
                            format!("node_{}", i),
                            serde_json::json!({"index": i, "type": "test"})
                        )
                    }).collect();
                    
                    let node_ids = graph.batch_create_nodes(nodes_data).unwrap();
                    
                    // Create edges
                    let edges: Vec<(NodeId, NodeId, Weight, EdgeData)> = (0..nodes)
                        .map(|i| {
                            let from = node_ids[i];
                            let to = node_ids[(i + 1) % nodes];
                            let weight = Weight(1.0);
                            let data = EdgeData::new(serde_json::json!({"edge": i}));
                            (from, to, weight, data)
                        })
                        .collect();
                    
                    graph.batch_create_edges(edges).unwrap();
                    
                    black_box(graph.get_statistics());
                }
            );
        });
    }
    
    group.finish();
}

/// Benchmark storage optimization operations
fn bench_storage_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("storage_optimization");
    
    for &size in &[SMALL_GRAPH_NODES, MEDIUM_GRAPH_NODES] {
        let graph = create_test_graph(size, size * 2);
        
        group.bench_with_input(BenchmarkId::new("optimize_storage", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.optimize_storage().unwrap());
            });
        });
        
        group.bench_with_input(BenchmarkId::new("get_statistics", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.get_statistics());
            });
        });
        
        group.bench_with_input(BenchmarkId::new("get_memory_usage", size), &size, |b, _| {
            b.iter(|| {
                black_box(graph.get_memory_usage());
            });
        });
    }
    
    group.finish();
}

// Helper functions

/// Create a test graph with specified number of nodes and edges
fn create_test_graph(node_count: usize, edge_count: usize) -> UltraFastKnowledgeGraph {
    let config = GraphConfig {
        initial_node_capacity: node_count,
        initial_edge_capacity: edge_count,
        ..Default::default()
    };
    
    let graph = UltraFastKnowledgeGraph::new(config).unwrap();
    
    // Create nodes
    let nodes: Vec<NodeData> = (0..node_count).map(|i| {
        NodeData::new(
            format!("node_{}", i),
            serde_json::json!({
                "type": "test",
                "index": i,
                "category": i % 10,
                "value": i as f64 / node_count as f64
            })
        )
    }).collect();
    
    let node_ids = graph.batch_create_nodes(nodes).unwrap();
    
    // Create edges with various patterns
    let mut edges = Vec::new();
    
    // Ring structure
    for i in 0..node_count {
        let from = node_ids[i];
        let to = node_ids[(i + 1) % node_count];
        let weight = Weight(1.0 + (i as f32 % 10.0) * 0.1);
        let data = EdgeData::new(serde_json::json!({"type": "ring", "index": i}));
        edges.push((from, to, weight, data));
    }
    
    // Random edges to reach target edge count
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    while edges.len() < edge_count {
        let mut hasher = DefaultHasher::new();
        edges.len().hash(&mut hasher);
        let hash = hasher.finish();
        
        let from_idx = (hash % node_count as u64) as usize;
        let to_idx = ((hash >> 16) % node_count as u64) as usize;
        
        if from_idx != to_idx {
            let from = node_ids[from_idx];
            let to = node_ids[to_idx];
            let weight = Weight(((hash >> 32) % 100) as f32 / 100.0);
            let data = EdgeData::new(serde_json::json!({"type": "random"}));
            edges.push((from, to, weight, data));
        }
    }
    
    // Create edges in batches for better performance
    graph.batch_create_edges(edges).unwrap();
    
    graph
}

/// Create a simple pattern for testing
fn create_simple_pattern() -> Pattern {
    Pattern {
        nodes: vec![
            PatternNode {
                id: "A".to_string(),
                type_filter: Some("test".to_string()),
                property_filters: std::collections::HashMap::new(),
            },
            PatternNode {
                id: "B".to_string(),
                type_filter: Some("test".to_string()),
                property_filters: std::collections::HashMap::new(),
            },
        ],
        edges: vec![
            PatternEdge {
                from: "A".to_string(),
                to: "B".to_string(),
                type_filter: None,
                direction: EdgeDirection::Outgoing,
                weight_range: None,
            },
        ],
        constraints: PatternConstraints {
            max_results: Some(100),
            timeout: Some(Duration::from_secs(1)),
            min_confidence: Some(0.5),
        },
    }
}

/// Create a complex pattern for testing
fn create_complex_pattern() -> Pattern {
    let mut property_filters = std::collections::HashMap::new();
    property_filters.insert("category".to_string(), serde_json::json!(5));
    
    Pattern {
        nodes: vec![
            PatternNode {
                id: "A".to_string(),
                type_filter: Some("test".to_string()),
                property_filters: property_filters.clone(),
            },
            PatternNode {
                id: "B".to_string(),
                type_filter: Some("test".to_string()),
                property_filters: std::collections::HashMap::new(),
            },
            PatternNode {
                id: "C".to_string(),
                type_filter: Some("test".to_string()),
                property_filters: std::collections::HashMap::new(),
            },
        ],
        edges: vec![
            PatternEdge {
                from: "A".to_string(),
                to: "B".to_string(),
                type_filter: Some("ring".to_string()),
                direction: EdgeDirection::Outgoing,
                weight_range: Some((0.0, 2.0)),
            },
            PatternEdge {
                from: "B".to_string(),
                to: "C".to_string(),
                type_filter: None,
                direction: EdgeDirection::Outgoing,
                weight_range: None,
            },
        ],
        constraints: PatternConstraints {
            max_results: Some(50),
            timeout: Some(Duration::from_millis(500)),
            min_confidence: Some(0.7),
        },
    }
}

// Benchmark groups
criterion_group!(
    benches,
    bench_node_creation,
    bench_edge_creation,
    bench_traversal_algorithms,
    bench_centrality_algorithms,
    bench_pattern_matching,
    bench_simd_operations,
    bench_memory_operations,
    bench_concurrent_operations,
    bench_query_performance,
    bench_memory_scalability,
    bench_storage_optimization
);

criterion_main!(benches);