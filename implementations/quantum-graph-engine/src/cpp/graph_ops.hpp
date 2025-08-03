#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <immintrin.h>  // For SIMD intrinsics
#include <omp.h>        // For OpenMP parallel processing

// Platform-specific optimizations
#ifdef __AVX512F__
#define SIMD_WIDTH 512
#define VECTOR_SIZE 8  // 8 x 64-bit values
#elif __AVX2__
#define SIMD_WIDTH 256
#define VECTOR_SIZE 4  // 4 x 64-bit values
#else
#define SIMD_WIDTH 128
#define VECTOR_SIZE 2  // 2 x 64-bit values
#endif

namespace quantum_graph {

// Forward declarations
class CppGraphEngine;
class CppAdjacencyMatrix;
class CppPathfinder;

// Data structures matching Rust FFI
struct CppNode {
    uint64_t id;
    const uint8_t* data_ptr;
    size_t data_len;
};

struct CppEdge {
    uint64_t id;
    uint64_t from;
    uint64_t to;
    double weight;
};

struct CppGraphStats {
    uint64_t node_count;
    uint64_t edge_count;
    uint64_t memory_usage;
    double avg_degree;
};

// Custom memory allocator for SIMD-aligned data
class SIMDAllocator {
public:
    static void* allocate(size_t size, size_t alignment = 64) {
        return _mm_malloc(size, alignment);
    }
    
    static void deallocate(void* ptr) {
        _mm_free(ptr);
    }
};

// High-performance adjacency matrix with SIMD optimizations
class CppAdjacencyMatrix {
private:
    uint64_t* data_;
    size_t rows_;
    size_t cols_;
    size_t capacity_;
    
public:
    CppAdjacencyMatrix(size_t initial_size);
    ~CppAdjacencyMatrix();
    
    // SIMD-optimized operations
    void set_edge(uint64_t from, uint64_t to, bool exists) noexcept;
    bool has_edge(uint64_t from, uint64_t to) const noexcept;
    
    // Vectorized batch operations
    void batch_set_edges(const CppEdge* edges, size_t count) noexcept;
    
    // AVX-512 optimized neighbor finding
    size_t find_neighbors_avx512(uint64_t node, uint64_t* neighbors, size_t max_neighbors) const noexcept;
    
    // Memory management
    void resize(size_t new_size);
    void compact();
    
    size_t memory_usage() const noexcept { return capacity_ * sizeof(uint64_t); }
};

// Main graph engine with extreme performance optimizations
class CppGraphEngine {
private:
    // Core data structures
    std::unique_ptr<CppAdjacencyMatrix> adjacency_;
    std::vector<CppNode> nodes_;
    std::vector<CppEdge> edges_;
    
    // Memory management
    size_t memory_limit_;
    size_t current_memory_usage_;
    
    // Performance counters
    uint64_t operation_count_;
    uint64_t total_operation_time_ns_;
    
    // Thread pool for parallel operations
    static constexpr size_t MAX_THREADS = 64;
    
public:
    explicit CppGraphEngine(uint64_t memory_limit);
    ~CppGraphEngine();
    
    // Disable copy/move for safety
    CppGraphEngine(const CppGraphEngine&) = delete;
    CppGraphEngine& operator=(const CppGraphEngine&) = delete;
    
    // High-performance node operations
    uint64_t batch_insert_nodes(const CppNode* nodes, size_t count) noexcept;
    size_t batch_get_nodes(const uint64_t* node_ids, CppNode* results, size_t count) const noexcept;
    
    // High-performance edge operations
    uint64_t batch_insert_edges(const CppEdge* edges, size_t count) noexcept;
    
    // Advanced graph algorithms
    std::vector<uint64_t> shortest_path(uint64_t from, uint64_t to, uint32_t max_depth) const;
    uint32_t pagerank(uint32_t iterations, double damping, double* results) const;
    uint32_t connected_components(uint32_t* component_ids) const;
    
    // SIMD-optimized traversals
    uint64_t simd_graph_traversal(const uint64_t* start_nodes, size_t start_count, 
                                 uint32_t max_depth, bool* visited) const noexcept;
    
    // Parallel algorithms
    uint64_t parallel_bfs(const uint64_t* start_nodes, size_t start_count,
                         uint32_t thread_count, uint64_t* results) const noexcept;
    uint64_t parallel_dfs(const uint64_t* start_nodes, size_t start_count,
                         uint32_t thread_count, uint64_t* results) const noexcept;
    
    // Memory management
    CppGraphStats get_memory_stats() const noexcept;
    uint64_t compact_memory() noexcept;
    
private:
    // Internal optimized algorithms
    void dijkstra_avx512(uint64_t start, uint64_t end, std::vector<uint64_t>& path) const;
    void pagerank_simd(uint32_t iterations, double damping, double* results) const;
    void union_find_simd(uint32_t* component_ids) const;
    
    // Memory optimization helpers
    void garbage_collect() noexcept;
    bool check_memory_limit(size_t additional_bytes) const noexcept;
};

// SIMD-optimized utility functions
namespace simd_ops {

// AVX-512 optimized edge scanning
inline int64_t avx512_edge_scan(const CppEdge* edges, size_t count, 
                                uint64_t target_from, uint64_t target_to) noexcept {
#ifdef __AVX512F__
    if (count == 0) return -1;
    
    const __m512i from_target = _mm512_set1_epi64(target_from);
    const __m512i to_target = _mm512_set1_epi64(target_to);
    
    size_t i = 0;
    const size_t simd_end = (count / 8) * 8;
    
    // Process 8 edges at a time
    for (; i < simd_end; i += 8) {
        // Load 8 'from' values
        __m512i from_values = _mm512_set_epi64(
            edges[i+7].from, edges[i+6].from, edges[i+5].from, edges[i+4].from,
            edges[i+3].from, edges[i+2].from, edges[i+1].from, edges[i+0].from
        );
        
        // Load 8 'to' values  
        __m512i to_values = _mm512_set_epi64(
            edges[i+7].to, edges[i+6].to, edges[i+5].to, edges[i+4].to,
            edges[i+3].to, edges[i+2].to, edges[i+1].to, edges[i+0].to
        );
        
        // Compare both from and to
        __mmask8 from_mask = _mm512_cmpeq_epi64_mask(from_values, from_target);
        __mmask8 to_mask = _mm512_cmpeq_epi64_mask(to_values, to_target);
        __mmask8 combined_mask = from_mask & to_mask;
        
        if (combined_mask != 0) {
            // Find the first match
            int pos = __builtin_ctzll(combined_mask);
            return static_cast<int64_t>(i + pos);
        }
    }
    
    // Handle remainder
    for (; i < count; ++i) {
        if (edges[i].from == target_from && edges[i].to == target_to) {
            return static_cast<int64_t>(i);
        }
    }
    
    return -1;
#else
    // Fallback for non-AVX512 systems
    for (size_t i = 0; i < count; ++i) {
        if (edges[i].from == target_from && edges[i].to == target_to) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
#endif
}

// Vectorized graph coloring
inline void vectorized_graph_coloring(const uint64_t* adjacency_matrix, 
                                     size_t num_nodes, uint32_t* colors) noexcept {
#ifdef __AVX512F__
    // Initialize colors to 0
    const size_t simd_end = (num_nodes / 16) * 16;
    const __m512i zero = _mm512_setzero_si512();
    
    for (size_t i = 0; i < simd_end; i += 16) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(&colors[i]), zero);
    }
    
    // Handle remainder
    for (size_t i = simd_end; i < num_nodes; ++i) {
        colors[i] = 0;
    }
    
    // Sequential coloring with SIMD-accelerated neighbor checking
    for (size_t node = 0; node < num_nodes; ++node) {
        std::vector<bool> used_colors(num_nodes, false);
        
        // Check neighbor colors
        for (size_t neighbor = 0; neighbor < num_nodes; ++neighbor) {
            if (adjacency_matrix[node * num_nodes + neighbor] != 0) {
                if (colors[neighbor] != 0) {
                    used_colors[colors[neighbor]] = true;
                }
            }
        }
        
        // Find first unused color
        for (uint32_t color = 1; color <= num_nodes; ++color) {
            if (!used_colors[color]) {
                colors[node] = color;
                break;
            }
        }
    }
#else
    // Fallback implementation
    for (size_t i = 0; i < num_nodes; ++i) {
        colors[i] = 1; // Simple fallback coloring
    }
#endif
}

// Parallel reduction for graph statistics
template<typename T>
inline T parallel_reduce_avx512(const T* data, size_t count) noexcept {
#ifdef __AVX512F__
    if (count == 0) return T{};
    
    if constexpr (std::is_same_v<T, uint64_t>) {
        __m512i sum = _mm512_setzero_si512();
        const size_t simd_end = (count / 8) * 8;
        
        for (size_t i = 0; i < simd_end; i += 8) {
            __m512i values = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&data[i]));
            sum = _mm512_add_epi64(sum, values);
        }
        
        // Horizontal sum
        uint64_t result = _mm512_reduce_add_epi64(sum);
        
        // Add remainder
        for (size_t i = simd_end; i < count; ++i) {
            result += data[i];
        }
        
        return result;
    } else {
        // Fallback for other types
        T result{};
        for (size_t i = 0; i < count; ++i) {
            result += data[i];
        }
        return result;
    }
#else
    T result{};
    for (size_t i = 0; i < count; ++i) {
        result += data[i];
    }
    return result;
#endif
}

} // namespace simd_ops

// External C interface for Rust FFI
extern "C" {
    // Graph engine lifecycle
    CppGraphEngine* create_graph_engine(uint64_t memory_limit);
    void destroy_graph_engine(CppGraphEngine* engine);
    
    // Core operations
    uint64_t batch_insert_nodes_cpp(CppGraphEngine* engine, const CppNode* nodes, size_t count);
    size_t batch_get_nodes_cpp(const CppGraphEngine* engine, const uint64_t* node_ids, CppNode* results);
    uint64_t batch_insert_edges_cpp(CppGraphEngine* engine, const CppEdge* edges, size_t count);
    
    // Algorithms
    int shortest_path_cpp(const CppGraphEngine* engine, uint64_t from, uint64_t to, 
                         uint32_t max_depth, uint64_t* path, size_t* path_length);
    int pagerank_cpp(const CppGraphEngine* engine, uint32_t iterations, 
                    double damping, double* results);
    uint32_t connected_components_cpp(const CppGraphEngine* engine, uint32_t* component_ids);
    
    // SIMD operations
    uint64_t simd_graph_traversal_cpp(const CppGraphEngine* engine, 
                                     const uint64_t* start_nodes, size_t start_count,
                                     uint32_t max_depth, bool* visited);
    int64_t avx512_edge_scanning_cpp(const CppEdge* edges, size_t count,
                                    uint64_t target_from, uint64_t target_to);
    
    // Parallel operations
    uint64_t parallel_bfs_cpp(const CppGraphEngine* engine, 
                             const uint64_t* start_nodes, size_t start_count,
                             uint32_t thread_count, uint64_t* results);
    uint64_t parallel_dfs_cpp(const CppGraphEngine* engine,
                             const uint64_t* start_nodes, size_t start_count, 
                             uint32_t thread_count, uint64_t* results);
    
    // Memory management
    CppGraphStats get_memory_stats_cpp(const CppGraphEngine* engine);
    uint64_t compact_memory_cpp(CppGraphEngine* engine);
}

} // namespace quantum_graph