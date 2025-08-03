/**
 * @file types.hpp
 * @brief Core type definitions for ultra-fast knowledge graph
 * 
 * This file contains fundamental types, enums, and data structures
 * used throughout the knowledge graph implementation.
 * 
 * @author Ultra-Fast Knowledge Graph Team
 * @version 1.0.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>
#include <chrono>
#include <variant>

namespace ultra_fast_kg {

// ==================== FUNDAMENTAL TYPES ====================

/// Unique identifier for nodes
using NodeId = std::uint64_t;

/// Unique identifier for edges  
using EdgeId = std::uint64_t;

/// Edge weight type for algorithms
using Weight = double;

/// Hash type for fast comparisons
using Hash = std::uint64_t;

/// Size type for memory calculations
using Size = std::size_t;

/// Index type for array access
using Index = std::uint32_t;

// ==================== PROPERTY SYSTEM ====================

/// Property value variant type
using PropertyValue = std::variant<
    std::nullptr_t,
    bool,
    std::int32_t,
    std::int64_t,
    double,
    std::string,
    std::vector<std::int32_t>,
    std::vector<std::int64_t>,
    std::vector<double>,
    std::vector<std::string>
>;

/// Property map for flexible node/edge attributes
using PropertyMap = std::unordered_map<std::string, PropertyValue>;

// ==================== CORE DATA STRUCTURES ====================

/**
 * @brief Node data structure with properties and metadata
 */
struct NodeData {
    std::string label;
    PropertyMap properties;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    NodeData() = default;
    
    NodeData(std::string label, PropertyMap properties = {})
        : label(std::move(label))
        , properties(std::move(properties))
        , created_at(std::chrono::system_clock::now())
        , updated_at(created_at) {}
    
    // Efficient move operations
    NodeData(NodeData&&) noexcept = default;
    NodeData& operator=(NodeData&&) noexcept = default;
    
    // Deleted copy to prevent accidental copies
    NodeData(const NodeData&) = delete;
    NodeData& operator=(const NodeData&) = delete;
};

/**
 * @brief Edge data structure with type, properties, and metadata
 */
struct EdgeData {
    PropertyMap properties;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    EdgeData() = default;
    
    explicit EdgeData(PropertyMap properties = {})
        : properties(std::move(properties))
        , created_at(std::chrono::system_clock::now())
        , updated_at(created_at) {}
    
    // Efficient move operations
    EdgeData(EdgeData&&) noexcept = default;
    EdgeData& operator=(EdgeData&&) noexcept = default;
    
    // Deleted copy to prevent accidental copies
    EdgeData(const EdgeData&) = delete;
    EdgeData& operator=(const EdgeData&) = delete;
};

/**
 * @brief Hyperedge data for N-ary relationships
 */
struct HyperedgeData {
    std::vector<NodeId> nodes;
    PropertyMap properties;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    HyperedgeData() = default;
    
    HyperedgeData(std::vector<NodeId> nodes, PropertyMap properties = {})
        : nodes(std::move(nodes))
        , properties(std::move(properties))
        , created_at(std::chrono::system_clock::now())
        , updated_at(created_at) {}
    
    // Move operations
    HyperedgeData(HyperedgeData&&) noexcept = default;
    HyperedgeData& operator=(HyperedgeData&&) noexcept = default;
    
    // Deleted copy operations
    HyperedgeData(const HyperedgeData&) = delete;
    HyperedgeData& operator=(const HyperedgeData&) = delete;
};

// ==================== PATH AND TRAVERSAL ====================

/**
 * @brief Path representation for shortest path algorithms
 */
struct Path {
    std::vector<NodeId> nodes;
    std::vector<EdgeId> edges;
    std::vector<Weight> weights;
    Weight total_weight = 0.0;
    std::size_t length = 0;
    std::chrono::nanoseconds computation_time{0};
};

/**
 * @brief Direction for edge traversal
 */
enum class EdgeDirection : std::uint8_t {
    Outgoing = 0,
    Incoming = 1,
    Both = 2
};

// ==================== PATTERN MATCHING ====================

/**
 * @brief Pattern node for graph pattern matching
 */
struct PatternNode {
    std::string id;
    std::optional<std::string> type_filter;
    PropertyMap property_filters;
    
    PatternNode() = default;
    PatternNode(std::string id) : id(std::move(id)) {}
};

/**
 * @brief Pattern edge for graph pattern matching
 */
struct PatternEdge {
    std::string from;
    std::string to;
    std::optional<std::string> type_filter;
    EdgeDirection direction = EdgeDirection::Outgoing;
    std::optional<std::pair<Weight, Weight>> weight_range;
    
    PatternEdge() = default;
    PatternEdge(std::string from, std::string to) 
        : from(std::move(from)), to(std::move(to)) {}
};

/**
 * @brief Complete pattern for subgraph matching
 */
struct Pattern {
    std::vector<PatternNode> nodes;
    std::vector<PatternEdge> edges;
    struct Constraints {
        std::optional<std::size_t> max_results;
        std::optional<std::chrono::milliseconds> timeout;
        std::optional<double> min_confidence;
        std::optional<std::chrono::system_clock::time_point> temporal_start;
        std::optional<std::chrono::system_clock::time_point> temporal_end;
    } constraints;
};

/**
 * @brief Pattern match result with confidence score
 */
struct PatternMatch {
    std::unordered_map<std::string, NodeId> node_bindings;
    std::unordered_map<std::string, EdgeId> edge_bindings;
    double score = 0.0;
    std::chrono::nanoseconds computation_time{0};
};

// ==================== ALGORITHM TYPES ====================

/**
 * @brief Centrality algorithm types
 */
enum class CentralityAlgorithm : std::uint8_t {
    Degree = 0,
    Betweenness = 1,
    Closeness = 2,
    Eigenvector = 3,
    PageRank = 4,
    Katz = 5
};

/**
 * @brief Traversal algorithm types
 */
enum class TraversalAlgorithm : std::uint8_t {
    BreadthFirst = 0,
    DepthFirst = 1,
    Dijkstra = 2,
    AStar = 3
};

/**
 * @brief Community detection algorithm types
 */
enum class CommunityAlgorithm : std::uint8_t {
    Louvain = 0,
    LabelPropagation = 1,
    FastGreedy = 2,
    WalkTrap = 3
};

// ==================== STORAGE TYPES ====================

/**
 * @brief Compression algorithm types
 */
enum class CompressionType : std::uint8_t {
    None = 0,
    LZ4 = 1,
    Zstd = 2,
    Snappy = 3
};

/**
 * @brief Storage backend types
 */
enum class StorageBackend : std::uint8_t {
    Memory = 0,
    RocksDB = 1,
    LMDB = 2,
    Custom = 3
};

/**
 * @brief Export format types
 */
enum class ExportFormat : std::uint8_t {
    GraphML = 0,
    GEXF = 1,
    JSON = 2,
    CSV = 3,
    Parquet = 4,
    Binary = 5
};

// ==================== MEMORY MANAGEMENT ====================

/**
 * @brief Memory usage breakdown
 */
struct MemoryUsage {
    std::size_t nodes = 0;
    std::size_t edges = 0;
    std::size_t outgoing_csr = 0;
    std::size_t incoming_csr = 0;
    std::size_t indices = 0;
    std::size_t metadata = 0;
    std::size_t total = 0;
    double compression_ratio = 1.0;
};

/**
 * @brief Performance metrics
 */
struct PerformanceMetrics {
    std::chrono::nanoseconds last_query_time{0};
    std::size_t queries_executed = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t simd_operations = 0;
    double cache_hit_ratio = 0.0;
    double simd_efficiency = 0.0;
};

/**
 * @brief Profiling data for performance analysis
 */
struct ProfilingData {
    std::unordered_map<std::string, std::chrono::nanoseconds> operation_times;
    std::unordered_map<std::string, std::size_t> operation_counts;
    std::size_t memory_allocations = 0;
    std::size_t memory_deallocations = 0;
    std::size_t peak_memory_usage = 0;
    std::chrono::steady_clock::time_point profile_start;
    std::chrono::steady_clock::time_point profile_end;
};

// ==================== TRANSACTION TYPES ====================

/**
 * @brief Read transaction for consistent queries
 */
class ReadTransaction {
public:
    ReadTransaction() = default;
    virtual ~ReadTransaction() = default;
    
    // Non-copyable, movable
    ReadTransaction(const ReadTransaction&) = delete;
    ReadTransaction& operator=(const ReadTransaction&) = delete;
    ReadTransaction(ReadTransaction&&) noexcept = default;
    ReadTransaction& operator=(ReadTransaction&&) noexcept = default;
    
    virtual void commit() = 0;
    virtual void abort() = 0;
    virtual bool is_active() const = 0;
};

/**
 * @brief Write transaction for atomic updates
 */
class WriteTransaction {
public:
    WriteTransaction() = default;
    virtual ~WriteTransaction() = default;
    
    // Non-copyable, movable
    WriteTransaction(const WriteTransaction&) = delete;
    WriteTransaction& operator=(const WriteTransaction&) = delete;
    WriteTransaction(WriteTransaction&&) noexcept = default;
    WriteTransaction& operator=(WriteTransaction&&) noexcept = default;
    
    virtual void commit() = 0;
    virtual void abort() = 0;
    virtual bool is_active() const = 0;
};

// ==================== ERROR HANDLING ====================

/**
 * @brief Graph operation error types
 */
enum class GraphErrorType : std::uint8_t {
    Success = 0,
    NodeNotFound = 1,
    EdgeNotFound = 2,
    InvalidOperation = 3,
    MemoryAllocation = 4,
    StorageError = 5,
    TimeoutError = 6,
    ConcurrencyError = 7,
    ValidationError = 8,
    NetworkError = 9,
    SystemError = 10
};

/**
 * @brief Graph exception class
 */
class GraphException : public std::exception {
private:
    GraphErrorType error_type_;
    std::string message_;
    
public:
    GraphException(GraphErrorType type, std::string message)
        : error_type_(type), message_(std::move(message)) {}
    
    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }
    
    [[nodiscard]] GraphErrorType error_type() const noexcept {
        return error_type_;
    }
    
    [[nodiscard]] const std::string& message() const noexcept {
        return message_;
    }
};

// ==================== SIMD TYPES ====================

/**
 * @brief SIMD width detection
 */
enum class SimdWidth : std::uint8_t {
    None = 0,
    SSE = 4,
    AVX = 8,
    AVX2 = 8,
    AVX512 = 16
};

/**
 * @brief SIMD operation result
 */
struct SimdResult {
    std::size_t operations_performed = 0;
    std::chrono::nanoseconds execution_time{0};
    double efficiency = 0.0;
    SimdWidth width_used = SimdWidth::None;
};

// ==================== UTILITY FUNCTIONS ====================

/**
 * @brief Hash combine function for efficient hash computation
 */
inline Hash hash_combine(Hash seed, Hash value) noexcept {
    return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

/**
 * @brief Hash function for NodeId pairs
 */
inline Hash hash_edge(NodeId from, NodeId to) noexcept {
    return hash_combine(std::hash<NodeId>{}(from), std::hash<NodeId>{}(to));
}

/**
 * @brief Check if SIMD width is supported
 */
[[nodiscard]] bool is_simd_supported(SimdWidth width) noexcept;

/**
 * @brief Get optimal SIMD width for current CPU
 */
[[nodiscard]] SimdWidth get_optimal_simd_width() noexcept;

/**
 * @brief Convert property value to string for debugging
 */
[[nodiscard]] std::string property_to_string(const PropertyValue& value);

/**
 * @brief Calculate memory alignment for SIMD operations
 */
[[nodiscard]] constexpr std::size_t simd_alignment(SimdWidth width) noexcept {
    switch (width) {
        case SimdWidth::SSE: return 16;
        case SimdWidth::AVX:
        case SimdWidth::AVX2: return 32;
        case SimdWidth::AVX512: return 64;
        default: return sizeof(void*);
    }
}

// ==================== CACHE-ALIGNED ALLOCATOR ====================

/**
 * @brief Cache-aligned allocator for optimal memory access
 */
template<typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
    
    AlignedAllocator() noexcept = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    [[nodiscard]] pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) noexcept {
        free(p);
    }
    
    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }
    
    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

// ==================== ALIGNED VECTOR TYPES ====================

/// Cache-aligned vector for SIMD operations
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

/// Commonly used aligned vector types
using AlignedNodeIdVector = AlignedVector<NodeId>;
using AlignedWeightVector = AlignedVector<Weight>;
using AlignedIndexVector = AlignedVector<Index>;
using AlignedFloatVector = AlignedVector<float>;
using AlignedDoubleVector = AlignedVector<double>;

} // namespace ultra_fast_kg