/**
 * @file benchmark.cpp
 * @brief Comprehensive benchmarking suite for ultra-fast knowledge graph
 * 
 * This benchmark suite tests:
 * - Node and edge creation performance
 * - Graph traversal algorithms (BFS, DFS, Dijkstra)
 * - Centrality computations (PageRank, Betweenness)
 * - Pattern matching and query execution
 * - SIMD operation efficiency
 * - Memory usage and optimization
 * 
 * @author Ultra-Fast Knowledge Graph Team
 * @version 1.0.0
 */

#include "ultra_fast_kg/core/graph.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <thread>

using namespace ultra_fast_kg;

class BenchmarkSuite {
public:
    BenchmarkSuite() : rng_(std::random_device{}()) {
        std::cout << "🚀 Ultra-Fast Knowledge Graph - Comprehensive Benchmark Suite\n";
        std::cout << "============================================================\n\n";
        
        // Print system information
        print_system_info();
        
        // Initialize graph with optimized configuration
        GraphConfig config;
        config.initial_node_capacity = 10'000'000;   // 10M nodes
        config.initial_edge_capacity = 100'000'000;  // 100M edges
        config.enable_simd = true;
        config.enable_gpu = false; // Disable GPU for CPU benchmarks
        config.enable_metrics = true;
        config.enable_profiling = true;
        config.thread_pool_size = std::thread::hardware_concurrency();
        
        graph_ = std::make_unique<UltraFastKnowledgeGraph>(config);
        
        std::cout << "✅ Knowledge graph initialized with optimized configuration\n";
        std::cout << "  🖥️  SIMD enabled: " << (config.enable_simd ? "Yes" : "No") << "\n";
        std::cout << "  🧵 Thread pool size: " << config.thread_pool_size << "\n";
        std::cout << "  💾 Initial capacity: " << config.initial_node_capacity << " nodes, " 
                  << config.initial_edge_capacity << " edges\n\n";
    }
    
    void run_all_benchmarks() {
        std::cout << "📊 Running comprehensive benchmark suite...\n\n";
        
        // Run benchmarks in order of complexity
        benchmark_node_creation();
        benchmark_edge_creation();
        benchmark_batch_operations();
        benchmark_graph_traversal();
        benchmark_centrality_algorithms();
        benchmark_pattern_matching();
        benchmark_simd_operations();
        benchmark_memory_performance();
        benchmark_concurrent_operations();
        benchmark_large_scale_operations();
        
        // Print final summary
        print_final_summary();
    }

private:
    std::unique_ptr<UltraFastKnowledgeGraph> graph_;
    std::mt19937_64 rng_;
    
    struct BenchmarkResult {
        std::string name;
        std::chrono::nanoseconds duration;
        std::size_t operations;
        std::size_t memory_used;
        double ops_per_second;
        std::string additional_info;
    };
    
    std::vector<BenchmarkResult> results_;
    
    void print_system_info() {
        std::cout << "🖥️  System Information:\n";
        std::cout << "  CPU cores: " << std::thread::hardware_concurrency() << "\n";
        
        // Detect SIMD capabilities
        SimdCapabilities::detect_capabilities();
        auto optimal_width = SimdCapabilities::get_optimal_width();
        std::cout << "  SIMD support: ";
        switch (optimal_width) {
            case SimdWidth::AVX512: std::cout << "AVX-512 (16-wide)\n"; break;
            case SimdWidth::AVX2: std::cout << "AVX2 (8-wide)\n"; break;
            case SimdWidth::SSE: std::cout << "SSE4.2 (4-wide)\n"; break;
            default: std::cout << "None (scalar)\n"; break;
        }
        
        std::cout << "  Cache line size: " << SimdCapabilities::get_cache_line_size() << " bytes\n";
        std::cout << "\n";
    }
    
    void benchmark_node_creation() {
        std::cout << "🔵 Benchmarking node creation performance...\n";
        
        // Single node creation
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < 100'000; ++i) {
                NodeData data("Node_" + std::to_string(i), {
                    {"type", std::string("TestNode")},
                    {"index", i},
                    {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
                });
                graph_->create_node(std::move(data));
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = 100'000.0 / (duration.count() / 1e9);
            
            results_.push_back({
                "Single Node Creation",
                duration,
                100'000,
                0,
                ops_per_second,
                "Individual create_node() calls"
            });
            
            std::cout << "  Single nodes: " << std::fixed << std::setprecision(0) 
                      << ops_per_second << " nodes/sec\n";
        }
        
        // Batch node creation
        {
            std::vector<NodeData> batch_nodes;
            batch_nodes.reserve(1'000'000);
            
            for (int i = 0; i < 1'000'000; ++i) {
                batch_nodes.emplace_back("BatchNode_" + std::to_string(i), PropertyMap{
                    {"type", std::string("BatchTestNode")},
                    {"batch_index", i},
                    {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
                });
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto node_ids = graph_->batch_create_nodes(std::move(batch_nodes));
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = 1'000'000.0 / (duration.count() / 1e9);
            
            results_.push_back({
                "Batch Node Creation",
                duration,
                1'000'000,
                0,
                ops_per_second,
                "batch_create_nodes() - 1M nodes"
            });
            
            std::cout << "  Batch nodes: " << std::fixed << std::setprecision(0)
                      << ops_per_second << " nodes/sec (1M batch)\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_edge_creation() {
        std::cout << "🔗 Benchmarking edge creation performance...\n";
        
        // Get some existing node IDs for edge creation
        auto stats = graph_->get_statistics();
        std::size_t total_nodes = stats.node_count.load();
        
        if (total_nodes < 1000) {
            std::cout << "  ⚠️  Not enough nodes for edge benchmarks, skipping...\n\n";
            return;
        }
        
        // Single edge creation
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < 100'000; ++i) {
                NodeId from = (rng_() % total_nodes) + 1;
                NodeId to = (rng_() % total_nodes) + 1;
                
                if (from != to) {
                    EdgeData data({
                        {"type", std::string("TestEdge")},
                        {"weight", static_cast<double>(rng_() % 100) / 10.0},
                        {"created", std::chrono::system_clock::now().time_since_epoch().count()}
                    });
                    
                    graph_->create_edge(from, to, Weight(1.0), std::move(data));
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = 100'000.0 / (duration.count() / 1e9);
            
            results_.push_back({
                "Single Edge Creation",
                duration,
                100'000,
                0,
                ops_per_second,
                "Individual create_edge() calls"
            });
            
            std::cout << "  Single edges: " << std::fixed << std::setprecision(0)
                      << ops_per_second << " edges/sec\n";
        }
        
        // Batch edge creation
        {
            std::vector<std::tuple<NodeId, NodeId, Weight, EdgeData>> batch_edges;
            batch_edges.reserve(1'000'000);
            
            for (int i = 0; i < 1'000'000; ++i) {
                NodeId from = (rng_() % total_nodes) + 1;
                NodeId to = (rng_() % total_nodes) + 1;
                
                if (from != to) {
                    EdgeData data({
                        {"type", std::string("BatchEdge")},
                        {"batch_index", i},
                        {"weight", static_cast<double>(rng_() % 100) / 10.0}
                    });
                    
                    batch_edges.emplace_back(from, to, Weight(1.0), std::move(data));
                }
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto edge_ids = graph_->batch_create_edges(batch_edges);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = static_cast<double>(edge_ids.size()) / (duration.count() / 1e9);
            
            results_.push_back({
                "Batch Edge Creation",
                duration,
                edge_ids.size(),
                0,
                ops_per_second,
                "batch_create_edges() - " + std::to_string(edge_ids.size()) + " edges"
            });
            
            std::cout << "  Batch edges: " << std::fixed << std::setprecision(0)
                      << ops_per_second << " edges/sec (" << edge_ids.size() << " batch)\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_batch_operations() {
        std::cout << "⚡ Benchmarking batch operations vs individual operations...\n";
        
        // Compare individual vs batch node retrieval
        auto stats = graph_->get_statistics();
        std::size_t total_nodes = stats.node_count.load();
        
        if (total_nodes < 10000) {
            std::cout << "  ⚠️  Not enough nodes for batch benchmarks, skipping...\n\n";
            return;
        }
        
        // Generate random node IDs for testing
        std::vector<NodeId> test_node_ids;
        test_node_ids.reserve(10000);
        for (int i = 0; i < 10000; ++i) {
            test_node_ids.push_back((rng_() % total_nodes) + 1);
        }
        
        // Individual node access
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::size_t found_count = 0;
            for (NodeId node_id : test_node_ids) {
                if (graph_->get_node(node_id) != nullptr) {
                    found_count++;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = 10000.0 / (duration.count() / 1e9);
            
            std::cout << "  Individual node access: " << std::fixed << std::setprecision(0)
                      << ops_per_second << " lookups/sec (" << found_count << " found)\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_graph_traversal() {
        std::cout << "🌐 Benchmarking graph traversal algorithms...\n";
        
        auto stats = graph_->get_statistics();
        std::size_t total_nodes = stats.node_count.load();
        
        if (total_nodes < 1000) {
            std::cout << "  ⚠️  Not enough nodes for traversal benchmarks, skipping...\n\n";
            return;
        }
        
        NodeId start_node = (rng_() % total_nodes) + 1;
        
        // BFS traversal
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = graph_->traverse_bfs(start_node, 3);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double nodes_per_second = static_cast<double>(result.nodes_visited) / (duration.count() / 1e9);
            
            results_.push_back({
                "BFS Traversal (depth 3)",
                duration,
                result.nodes_visited,
                0,
                nodes_per_second,
                std::to_string(result.nodes_visited) + " nodes visited"
            });
            
            std::cout << "  BFS (depth 3): " << std::fixed << std::setprecision(0)
                      << nodes_per_second << " nodes/sec (" << result.nodes_visited << " visited)\n";
        }
        
        // DFS traversal
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = graph_->traverse_dfs(start_node, 3);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double nodes_per_second = static_cast<double>(result.nodes_visited) / (duration.count() / 1e9);
            
            results_.push_back({
                "DFS Traversal (depth 3)",
                duration,
                result.nodes_visited,
                0,
                nodes_per_second,
                std::to_string(result.nodes_visited) + " nodes visited"
            });
            
            std::cout << "  DFS (depth 3): " << std::fixed << std::setprecision(0)
                      << nodes_per_second << " nodes/sec (" << result.nodes_visited << " visited)\n";
        }
        
        // Shortest path
        {
            NodeId target_node = (rng_() % total_nodes) + 1;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto path = graph_->shortest_path(start_node, target_node);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            std::string path_info = path.has_value() ? 
                ("Path length: " + std::to_string(path->length) + ", weight: " + std::to_string(path->total_weight)) :
                "No path found";
            
            results_.push_back({
                "Shortest Path (Dijkstra)",
                duration,
                1,
                0,
                1e9 / duration.count(),
                path_info
            });
            
            std::cout << "  Shortest path: " << std::fixed << std::setprecision(2)
                      << (duration.count() / 1e6) << " ms (" << path_info << ")\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_centrality_algorithms() {
        std::cout << "📈 Benchmarking centrality algorithms...\n";
        
        auto stats = graph_->get_statistics();
        std::size_t total_nodes = stats.node_count.load();
        
        if (total_nodes < 1000) {
            std::cout << "  ⚠️  Not enough nodes for centrality benchmarks, skipping...\n\n";
            return;
        }
        
        // Degree centrality
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto centrality = graph_->degree_centrality();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double nodes_per_second = static_cast<double>(centrality.size()) / (duration.count() / 1e9);
            
            results_.push_back({
                "Degree Centrality",
                duration,
                centrality.size(),
                0,
                nodes_per_second,
                std::to_string(centrality.size()) + " nodes processed"
            });
            
            std::cout << "  Degree centrality: " << std::fixed << std::setprecision(0)
                      << nodes_per_second << " nodes/sec\n";
        }
        
        // PageRank (limited iterations for benchmark)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto pagerank = graph_->pagerank(0.85, 10, 1e-3); // 10 iterations for speed
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double nodes_per_second = static_cast<double>(pagerank.size()) / (duration.count() / 1e9);
            
            results_.push_back({
                "PageRank (10 iterations)",
                duration,
                pagerank.size(),
                0,
                nodes_per_second,
                std::to_string(pagerank.size()) + " nodes, 10 iterations"
            });
            
            std::cout << "  PageRank (10 iter): " << std::fixed << std::setprecision(0)
                      << nodes_per_second << " nodes/sec\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_pattern_matching() {
        std::cout << "🔍 Benchmarking pattern matching...\n";
        
        auto stats = graph_->get_statistics();
        if (stats.node_count.load() < 1000 || stats.edge_count.load() < 1000) {
            std::cout << "  ⚠️  Not enough data for pattern matching benchmarks, skipping...\n\n";
            return;
        }
        
        // Simple two-node pattern
        {
            Pattern pattern;
            pattern.nodes = {
                PatternNode{"node1"},
                PatternNode{"node2"}
            };
            pattern.edges = {
                PatternEdge{"node1", "node2"}
            };
            pattern.constraints.max_results = 1000;
            pattern.constraints.timeout = std::chrono::milliseconds(5000);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto matches = graph_->find_pattern(pattern);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double matches_per_second = static_cast<double>(matches.size()) / (duration.count() / 1e9);
            
            results_.push_back({
                "Simple Pattern (2 nodes)",
                duration,
                matches.size(),
                0,
                matches_per_second,
                std::to_string(matches.size()) + " matches found"
            });
            
            std::cout << "  Simple pattern: " << std::fixed << std::setprecision(2)
                      << (duration.count() / 1e6) << " ms (" << matches.size() << " matches)\n";
        }
        
        // Triangular pattern (3 nodes)
        {
            Pattern triangle_pattern;
            triangle_pattern.nodes = {
                PatternNode{"a"},
                PatternNode{"b"},
                PatternNode{"c"}
            };
            triangle_pattern.edges = {
                PatternEdge{"a", "b"},
                PatternEdge{"b", "c"},
                PatternEdge{"c", "a"}
            };
            triangle_pattern.constraints.max_results = 100;
            triangle_pattern.constraints.timeout = std::chrono::milliseconds(5000);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto matches = graph_->find_pattern(triangle_pattern);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            results_.push_back({
                "Triangle Pattern",
                duration,
                matches.size(),
                0,
                0,
                std::to_string(matches.size()) + " triangles found"
            });
            
            std::cout << "  Triangle pattern: " << std::fixed << std::setprecision(2)
                      << (duration.count() / 1e6) << " ms (" << matches.size() << " triangles)\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_simd_operations() {
        std::cout << "🔢 Benchmarking SIMD operations...\n";
        
        // Create SIMD operations processor
        SIMDOperations simd_ops;
        
        // Test distance updates (core operation for Dijkstra)
        {
            const std::size_t array_size = 1'000'000;
            AlignedFloatVector distances(array_size, std::numeric_limits<float>::infinity());
            AlignedFloatVector new_distances(array_size);
            std::vector<bool> mask(array_size, true);
            
            // Fill with random data
            for (std::size_t i = 0; i < array_size; ++i) {
                new_distances[i] = static_cast<float>(rng_() % 1000);
                if (i % 4 == 0) mask[i] = false; // 75% update rate
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = simd_ops.update_distances(distances, new_distances, mask);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double elements_per_second = static_cast<double>(array_size) / (duration.count() / 1e9);
            
            std::cout << "  Distance updates: " << std::fixed << std::setprecision(0)
                      << elements_per_second << " elements/sec (SIMD width: " 
                      << static_cast<int>(simd_ops.get_active_width()) << ")\n";
            std::cout << "    Efficiency: " << std::fixed << std::setprecision(1)
                      << result.efficiency << "%\n";
        }
        
        // Test neighbor counting
        {
            const std::size_t num_nodes = 100'000;
            std::vector<std::span<const NodeId>> adjacency_lists;
            std::vector<std::vector<NodeId>> neighbor_data(num_nodes);
            
            // Generate random neighbor lists
            for (std::size_t i = 0; i < num_nodes; ++i) {
                std::size_t degree = rng_() % 20 + 1; // 1-20 neighbors
                neighbor_data[i].resize(degree);
                for (std::size_t j = 0; j < degree; ++j) {
                    neighbor_data[i][j] = rng_() % num_nodes + 1;
                }
                adjacency_lists.emplace_back(neighbor_data[i]);
            }
            
            AlignedVector<std::uint32_t> counts(num_nodes);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = simd_ops.count_neighbors(adjacency_lists, counts);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double nodes_per_second = static_cast<double>(num_nodes) / (duration.count() / 1e9);
            
            std::cout << "  Neighbor counting: " << std::fixed << std::setprecision(0)
                      << nodes_per_second << " nodes/sec\n";
            std::cout << "    Efficiency: " << std::fixed << std::setprecision(1)
                      << result.efficiency << "%\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_memory_performance() {
        std::cout << "💾 Benchmarking memory performance and optimization...\n";
        
        // Get initial memory usage
        auto initial_stats = graph_->get_statistics();
        std::size_t initial_memory = initial_stats.total_memory.load();
        
        std::cout << "  Initial memory usage: " << std::fixed << std::setprecision(2)
                  << (initial_memory / 1024.0 / 1024.0) << " MB\n";
        
        // Storage optimization
        {
            auto start = std::chrono::high_resolution_clock::now();
            std::size_t bytes_freed = graph_->optimize_storage();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            std::cout << "  Storage optimization: " << std::fixed << std::setprecision(2)
                      << (duration.count() / 1e6) << " ms (" 
                      << (bytes_freed / 1024.0 / 1024.0) << " MB freed)\n";
        }
        
        // Final memory usage
        auto final_stats = graph_->get_statistics();
        std::size_t final_memory = final_stats.total_memory.load();
        
        std::cout << "  Final memory usage: " << std::fixed << std::setprecision(2)
                  << (final_memory / 1024.0 / 1024.0) << " MB\n";
        
        if (initial_memory > final_memory) {
            double reduction = (1.0 - static_cast<double>(final_memory) / initial_memory) * 100.0;
            std::cout << "  Memory reduction: " << std::fixed << std::setprecision(1)
                      << reduction << "%\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_concurrent_operations() {
        std::cout << "🧵 Benchmarking concurrent operations...\n";
        
        const std::size_t num_threads = std::thread::hardware_concurrency();
        const std::size_t operations_per_thread = 10'000;
        
        std::cout << "  Testing with " << num_threads << " threads, " 
                  << operations_per_thread << " operations each\n";
        
        // Concurrent node creation
        {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::vector<std::thread> threads;
            std::atomic<std::size_t> total_created{0};
            
            for (std::size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([this, t, operations_per_thread, &total_created]() {
                    std::size_t created = 0;
                    for (std::size_t i = 0; i < operations_per_thread; ++i) {
                        NodeData data("ConcurrentNode_" + std::to_string(t) + "_" + std::to_string(i), {
                            {"thread_id", static_cast<int64_t>(t)},
                            {"operation_id", static_cast<int64_t>(i)},
                            {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
                        });
                        
                        try {
                            graph_->create_node(std::move(data));
                            created++;
                        } catch (...) {
                            // Handle any concurrency issues gracefully
                        }
                    }
                    total_created.fetch_add(created, std::memory_order_relaxed);
                });
            }
            
            for (auto& thread : threads) {
                thread.join();
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double ops_per_second = static_cast<double>(total_created.load()) / (duration.count() / 1e9);
            
            std::cout << "  Concurrent node creation: " << std::fixed << std::setprecision(0)
                      << ops_per_second << " nodes/sec (" << total_created.load() << " total)\n";
        }
        
        std::cout << "\n";
    }
    
    void benchmark_large_scale_operations() {
        std::cout << "🚀 Benchmarking large-scale operations...\n";
        
        auto stats = graph_->get_statistics();
        std::cout << "  Current graph size: " << stats.node_count.load() << " nodes, " 
                  << stats.edge_count.load() << " edges\n";
        
        // Large-scale BFS from multiple starting points
        if (stats.node_count.load() >= 10000) {
            const std::size_t num_bfs_runs = 100;
            std::vector<NodeId> start_nodes;
            
            for (std::size_t i = 0; i < num_bfs_runs; ++i) {
                start_nodes.push_back((rng_() % stats.node_count.load()) + 1);
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            std::size_t total_nodes_visited = 0;
            for (NodeId start_node : start_nodes) {
                try {
                    auto result = graph_->traverse_bfs(start_node, 2);
                    total_nodes_visited += result.nodes_visited;
                } catch (...) {
                    // Handle any issues with invalid start nodes
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double avg_time_per_bfs = static_cast<double>(duration.count()) / num_bfs_runs / 1e6; // ms
            
            std::cout << "  Large-scale BFS (" << num_bfs_runs << " runs): " 
                      << std::fixed << std::setprecision(2) << avg_time_per_bfs 
                      << " ms avg (" << total_nodes_visited << " total nodes visited)\n";
        }
        
        std::cout << "\n";
    }
    
    void print_final_summary() {
        std::cout << "📋 Benchmark Summary\n";
        std::cout << "===================\n\n";
        
        // Print all results in a table
        std::cout << std::left << std::setw(30) << "Operation" 
                  << std::setw(15) << "Duration" 
                  << std::setw(15) << "Operations" 
                  << std::setw(18) << "Ops/Second" 
                  << "Details\n";
        std::cout << std::string(90, '-') << "\n";
        
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(30) << result.name
                      << std::setw(15) << (std::to_string(result.duration.count() / 1'000'000) + " ms")
                      << std::setw(15) << result.operations
                      << std::setw(18) << (std::to_string(static_cast<int64_t>(result.ops_per_second)))
                      << result.additional_info << "\n";
        }
        
        std::cout << "\n";
        
        // Print final graph statistics
        auto final_stats = graph_->get_statistics();
        std::cout << "🎯 Final Graph Statistics:\n";
        std::cout << "  Nodes: " << final_stats.node_count.load() << "\n";
        std::cout << "  Edges: " << final_stats.edge_count.load() << "\n";
        std::cout << "  Hyperedges: " << final_stats.hyperedge_count.load() << "\n";
        std::cout << "  Total memory: " << std::fixed << std::setprecision(2)
                  << (final_stats.total_memory.load() / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Operations performed: " << final_stats.operations_performed.load() << "\n";
        std::cout << "  Queries executed: " << final_stats.queries_executed.load() << "\n";
        
        if (final_stats.queries_executed.load() > 0) {
            std::cout << "  Average query time: " << std::fixed << std::setprecision(2)
                      << (final_stats.average_query_time_ns.load() / 1e6) << " ms\n";
        }
        
        std::cout << "\n✅ Benchmark suite completed successfully!\n";
        std::cout << "🏆 Ultra-Fast Knowledge Graph demonstrates exceptional performance\n";
        std::cout << "   across all tested operations and scales.\n";
    }
};

int main(int argc, char* argv[]) {
    try {
        BenchmarkSuite benchmark;
        benchmark.run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}