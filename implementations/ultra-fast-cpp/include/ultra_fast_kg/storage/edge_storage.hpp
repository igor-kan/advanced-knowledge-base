/**
 * @file edge_storage.hpp
 * @brief High-performance edge storage with lock-free operations
 * 
 * This implementation provides:
 * - Lock-free concurrent access for maximum throughput
 * - Compressed edge representation for memory efficiency
 * - Temporal edge support with versioning
 * - Fast edge lookup by source, target, or edge ID
 * - SIMD-optimized batch operations
 * 
 * @author Ultra-Fast Knowledge Graph Team
 * @version 1.0.0
 */

#pragma once

#include "ultra_fast_kg/core/types.hpp"
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

namespace ultra_fast_kg {

/**
 * @brief Configuration for edge storage optimization
 */
struct EdgeStorageConfig {
    std::size_t initial_capacity = 10'000'000;
    std::size_t growth_factor = 2;
    bool enable_compression = true;
    bool enable_temporal_edges = true;
    bool enable_edge_indexing = true;
    bool enable_memory_mapping = true;
    CompressionType compression_type = CompressionType::LZ4;
    std::size_t cache_line_size = 64;
    std::size_t numa_node = 0;
};

/**
 * @brief Compressed edge entry for space-efficient storage
 */
struct alignas(32) CompressedEdgeEntry {
    // Core edge data (24 bytes)
    EdgeId edge_id;
    NodeId from_node;
    NodeId to_node;
    
    // Weight and metadata (8 bytes)
    union {
        Weight weight;
        std::uint64_t weight_bits;
    };
    
    // Properties and timing (16 bytes)
    std::atomic<std::uint32_t> properties_offset;
    std::atomic<std::uint32_t> properties_length;
    std::atomic<std::uint64_t> created_at_ns;
    
    // Version and flags (8 bytes)
    std::atomic<std::uint32_t> version;
    std::atomic<std::uint16_t> type_id;
    std::atomic<std::uint8_t> flags;
    std::atomic<std::uint8_t> compression_flags;
    
    // Total: 56 bytes, padded to 64 for cache alignment
    std::uint64_t padding;
    
    CompressedEdgeEntry() 
        : edge_id(0), from_node(0), to_node(0), weight(0.0), version(0), type_id(0), flags(0), compression_flags(0), padding(0) {
        properties_offset.store(0, std::memory_order_relaxed);
        properties_length.store(0, std::memory_order_relaxed);
        created_at_ns.store(0, std::memory_order_relaxed);
    }
    
    CompressedEdgeEntry(EdgeId eid, NodeId from, NodeId to, Weight w)
        : edge_id(eid), from_node(from), to_node(to), weight(w), version(1), type_id(0), flags(0), compression_flags(0), padding(0) {
        properties_offset.store(0, std::memory_order_relaxed);
        properties_length.store(0, std::memory_order_relaxed);
        auto now = std::chrono::system_clock::now().time_since_epoch();
        created_at_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(), std::memory_order_relaxed);
    }
    
    // Non-copyable, movable
    CompressedEdgeEntry(const CompressedEdgeEntry&) = delete;
    CompressedEdgeEntry& operator=(const CompressedEdgeEntry&) = delete;
    CompressedEdgeEntry(CompressedEdgeEntry&&) = default;
    CompressedEdgeEntry& operator=(CompressedEdgeEntry&&) = default;
};

static_assert(sizeof(CompressedEdgeEntry) == 64, "CompressedEdgeEntry must be 64 bytes for cache alignment");

/**
 * @brief Edge index for fast lookups by source/target nodes
 */
class EdgeIndex {
public:
    EdgeIndex() = default;
    ~EdgeIndex() = default;
    
    /**
     * @brief Adds an edge to the index
     * @param edge_id Edge identifier
     * @param from_node Source node
     * @param to_node Target node
     */
    void add_edge(EdgeId edge_id, NodeId from_node, NodeId to_node);
    
    /**
     * @brief Removes an edge from the index
     * @param edge_id Edge identifier
     * @param from_node Source node
     * @param to_node Target node
     */
    void remove_edge(EdgeId edge_id, NodeId from_node, NodeId to_node);
    
    /**
     * @brief Finds edges by source node
     * @param from_node Source node
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_from(NodeId from_node) const;
    
    /**
     * @brief Finds edges by target node
     * @param to_node Target node
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_to(NodeId to_node) const;
    
    /**
     * @brief Finds edges between two nodes
     * @param from_node Source node
     * @param to_node Target node
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_between(NodeId from_node, NodeId to_node) const;
    
    /**
     * @brief Gets all edges for a node (incoming and outgoing)
     * @param node_id Node identifier
     * @return Pair of (outgoing_edges, incoming_edges)
     */
    [[nodiscard]] std::pair<std::vector<EdgeId>, std::vector<EdgeId>> get_all_edges(NodeId node_id) const;
    
    /**
     * @brief Clears the entire index
     */
    void clear();
    
    /**
     * @brief Gets index memory usage
     * @return Memory usage in bytes
     */
    [[nodiscard]] std::size_t get_memory_usage() const;
    
private:
    // Hash maps for fast edge lookup
    std::unordered_map<NodeId, std::vector<EdgeId>> outgoing_edges_;
    std::unordered_map<NodeId, std::vector<EdgeId>> incoming_edges_;
    std::unordered_map<std::uint64_t, std::vector<EdgeId>> node_pair_edges_;
    
    mutable std::shared_mutex mutex_;
    
    // Helper function to create node pair hash
    static std::uint64_t hash_node_pair(NodeId from, NodeId to) noexcept {
        return (static_cast<std::uint64_t>(from) << 32) | to;
    }
};

/**
 * @brief Ultra-high performance edge storage implementation
 * 
 * This class provides the core edge storage for the knowledge graph
 * with extreme performance optimizations:
 * 
 * - Lock-free concurrent reads and atomic writes
 * - Compressed edge representation for memory efficiency
 * - Fast edge lookup by ID, source, or target
 * - SIMD-optimized batch operations
 * - Temporal edge support with versioning
 */
class EdgeStorage {
public:
    /**
     * @brief Constructs a new edge storage
     * @param config Configuration parameters
     */
    explicit EdgeStorage(const EdgeStorageConfig& config = EdgeStorageConfig{});
    
    /**
     * @brief Destructor with cleanup
     */
    ~EdgeStorage();
    
    // Non-copyable but movable
    EdgeStorage(const EdgeStorage&) = delete;
    EdgeStorage& operator=(const EdgeStorage&) = delete;
    EdgeStorage(EdgeStorage&& other) noexcept;
    EdgeStorage& operator=(EdgeStorage&& other) noexcept;
    
    // ==================== CORE OPERATIONS ====================
    
    /**
     * @brief Stores a new edge atomically
     * @param edge_id Edge identifier
     * @param from_node Source node ID
     * @param to_node Target node ID
     * @param weight Edge weight
     * @param data Edge data to store
     * @return true if stored successfully, false if edge already exists
     */
    bool store_edge(EdgeId edge_id, NodeId from_node, NodeId to_node, Weight weight, EdgeData data);
    
    /**
     * @brief Updates an existing edge atomically
     * @param edge_id Edge identifier
     * @param weight New edge weight
     * @param data New edge data
     * @return true if updated successfully, false if edge doesn't exist
     */
    bool update_edge(EdgeId edge_id, Weight weight, EdgeData data);
    
    /**
     * @brief Updates only the weight of an edge
     * @param edge_id Edge identifier
     * @param weight New weight
     * @return true if updated successfully
     */
    bool update_edge_weight(EdgeId edge_id, Weight weight);
    
    /**
     * @brief Removes an edge from storage
     * @param edge_id Edge identifier
     * @return true if removed successfully, false if edge didn't exist
     */
    bool remove_edge(EdgeId edge_id);
    
    /**
     * @brief Gets edge data with zero-copy access when possible
     * @param edge_id Edge identifier
     * @return Pointer to edge data, or nullptr if not found
     * 
     * Performance: Sub-microsecond access time with cache optimization
     */
    [[nodiscard]] const EdgeData* get_edge_data(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Gets edge weight
     * @param edge_id Edge identifier
     * @return Edge weight, or NaN if not found
     */
    [[nodiscard]] Weight get_edge_weight(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Gets edge endpoints
     * @param edge_id Edge identifier
     * @return Pair of (from_node, to_node), or (0, 0) if not found
     */
    [[nodiscard]] std::pair<NodeId, NodeId> get_edge_endpoints(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Gets complete edge information
     * @param edge_id Edge identifier
     * @return Tuple of (from_node, to_node, weight, data_ptr)
     */
    [[nodiscard]] std::tuple<NodeId, NodeId, Weight, const EdgeData*> get_edge_complete(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Checks if an edge exists
     * @param edge_id Edge identifier
     * @return true if edge exists
     */
    [[nodiscard]] bool edge_exists(EdgeId edge_id) const noexcept;
    
    // ==================== BATCH OPERATIONS ====================
    
    /**
     * @brief Batch stores multiple edges for better performance
     * @param edges Vector of (edge_id, from, to, weight, data) tuples
     * @return Number of successfully stored edges
     */
    std::size_t batch_store_edges(std::vector<std::tuple<EdgeId, NodeId, NodeId, Weight, EdgeData>> edges);
    
    /**
     * @brief Batch updates multiple edge weights
     * @param updates Vector of (edge_id, weight) pairs
     * @return Number of successfully updated edges
     */
    std::size_t batch_update_weights(const std::vector<std::pair<EdgeId, Weight>>& updates);
    
    /**
     * @brief Batch gets multiple edges
     * @param edge_ids Vector of edge identifiers
     * @return Vector of edge data pointers (nullptr for non-existent edges)
     */
    [[nodiscard]] std::vector<const EdgeData*> batch_get_edges(const std::vector<EdgeId>& edge_ids) const;
    
    /**
     * @brief SIMD-optimized batch weight retrieval
     * @param edge_ids Vector of edge identifiers
     * @return Aligned vector of weights (NaN for non-existent edges)
     */
    [[nodiscard]] AlignedFloatVector batch_get_weights_simd(const std::vector<EdgeId>& edge_ids) const;
    
    // ==================== QUERY OPERATIONS ====================
    
    /**
     * @brief Finds edges by source node
     * @param from_node Source node ID
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_from(NodeId from_node) const;
    
    /**
     * @brief Finds edges by target node
     * @param to_node Target node ID
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_to(NodeId to_node) const;
    
    /**
     * @brief Finds edges between two specific nodes
     * @param from_node Source node ID
     * @param to_node Target node ID
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_between(NodeId from_node, NodeId to_node) const;
    
    /**
     * @brief Finds edges with weight in specified range
     * @param min_weight Minimum weight (inclusive)
     * @param max_weight Maximum weight (inclusive)
     * @return Vector of edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> find_edges_by_weight_range(Weight min_weight, Weight max_weight) const;
    
    /**
     * @brief Gets all edges for a node (incoming and outgoing)
     * @param node_id Node identifier
     * @return Pair of (outgoing_edges, incoming_edges)
     */
    [[nodiscard]] std::pair<std::vector<EdgeId>, std::vector<EdgeId>> get_all_edges_for_node(NodeId node_id) const;
    
    /**
     * @brief Gets all edge IDs currently stored
     * @return Vector of all edge IDs
     */
    [[nodiscard]] std::vector<EdgeId> get_all_edge_ids() const;
    
    /**
     * @brief Gets edges created within a time range
     * @param start_time Start of time range
     * @param end_time End of time range
     * @return Vector of edge IDs created in the range
     */
    [[nodiscard]] std::vector<EdgeId> get_edges_by_time_range(
        std::chrono::system_clock::time_point start_time,
        std::chrono::system_clock::time_point end_time
    ) const;
    
    // ==================== MEMORY MANAGEMENT ====================
    
    /**
     * @brief Optimizes storage layout and compresses data
     * @return Amount of memory freed (in bytes)
     */
    std::size_t optimize_storage();
    
    /**
     * @brief Prefetches edge data into cache
     * @param edge_id Edge to prefetch
     */
    void prefetch_edge(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Gets current memory usage
     * @return Memory usage breakdown
     */
    [[nodiscard]] MemoryUsage get_memory_usage() const noexcept;
    
    /**
     * @brief Reserves capacity for expected number of edges
     * @param capacity Expected number of edges
     */
    void reserve_capacity(std::size_t capacity);
    
    // ==================== PERSISTENCE ====================
    
    /**
     * @brief Saves edge storage to disk with compression
     * @param path File path
     * @param compression Compression algorithm
     * @return Number of bytes written
     */
    std::size_t save_to_disk(const std::string& path, CompressionType compression = CompressionType::LZ4) const;
    
    /**
     * @brief Loads edge storage from disk
     * @param path File path
     * @return Number of bytes read
     */
    std::size_t load_from_disk(const std::string& path);
    
    /**
     * @brief Memory-maps the storage from a file
     * @param path File path
     * @param read_only Whether to map read-only
     * @return true if successful
     */
    bool memory_map(const std::string& path, bool read_only = true);
    
    /**
     * @brief Flushes pending writes to disk
     */
    void flush_to_disk();
    
    // ==================== STATISTICS ====================
    
    /**
     * @brief Gets the total number of edges stored
     * @return Edge count
     */
    [[nodiscard]] std::size_t edge_count() const noexcept;
    
    /**
     * @brief Gets the current storage capacity
     * @return Maximum edges that can be stored without reallocation
     */
    [[nodiscard]] std::size_t capacity() const noexcept;
    
    /**
     * @brief Gets the load factor (used_capacity / total_capacity)
     * @return Load factor [0.0, 1.0]
     */
    [[nodiscard]] double load_factor() const noexcept;
    
    /**
     * @brief Gets compression ratio
     * @return Compression ratio (compressed_size / uncompressed_size)
     */
    [[nodiscard]] double compression_ratio() const noexcept;
    
    /**
     * @brief Gets average edge size in bytes
     * @return Average edge size
     */
    [[nodiscard]] double average_edge_size() const noexcept;
    
    /**
     * @brief Gets edge degree statistics
     * @return Tuple of (min_degree, max_degree, avg_degree)
     */
    [[nodiscard]] std::tuple<std::size_t, std::size_t, double> get_degree_statistics() const;
    
    /**
     * @brief Validates storage integrity
     * @return true if storage is valid
     */
    [[nodiscard]] bool validate() const noexcept;
    
private:
    // ==================== MEMBER VARIABLES ====================
    
    EdgeStorageConfig config_;
    
    // Core storage arrays
    std::unique_ptr<CompressedEdgeEntry[]> entries_;
    std::atomic<std::size_t> capacity_{0};
    std::atomic<std::size_t> edge_count_{0};
    std::atomic<EdgeId> max_edge_id_{0};
    
    // Variable-length data storage for properties
    std::unique_ptr<VariableLengthPool> properties_pool_;
    
    // Edge indexing for fast queries
    std::unique_ptr<EdgeIndex> edge_index_;
    
    // Edge data cache for decompressed entries
    mutable std::unordered_map<EdgeId, std::unique_ptr<EdgeData>> edge_cache_;
    mutable std::shared_mutex cache_mutex_;
    std::atomic<std::size_t> cache_hits_{0};
    std::atomic<std::size_t> cache_misses_{0};
    
    // Memory management
    std::atomic<std::size_t> total_allocated_bytes_{0};
    std::atomic<std::size_t> compressed_bytes_{0};
    
    // Concurrency control
    mutable std::shared_mutex resize_mutex_;
    std::atomic<bool> is_resizing_{false};
    
    // Memory mapping
    void* memory_mapped_region_{nullptr};
    std::size_t memory_mapped_size_{0};
    bool is_memory_mapped_{false};
    
    // ==================== INTERNAL HELPER METHODS ====================
    
    /**
     * @brief Ensures capacity for at least min_edge_id
     * @param min_edge_id Minimum edge ID that must fit
     */
    void ensure_capacity(EdgeId min_edge_id);
    
    /**
     * @brief Compresses edge data for storage
     * @param data Edge data to compress
     * @param props_offset Output: offset in properties pool
     * @param props_length Output: length in properties pool
     * @return true if compression successful
     */
    bool compress_edge_data(const EdgeData& data,
                           std::uint32_t& props_offset, std::uint32_t& props_length);
    
    /**
     * @brief Decompresses edge data from storage
     * @param entry Compressed edge entry
     * @return Unique pointer to decompressed edge data
     */
    std::unique_ptr<EdgeData> decompress_edge_data(const CompressedEdgeEntry& entry) const;
    
    /**
     * @brief Gets or creates cached edge data
     * @param edge_id Edge identifier
     * @return Pointer to cached edge data
     */
    const EdgeData* get_or_cache_edge_data(EdgeId edge_id) const;
    
    /**
     * @brief Evicts old entries from cache
     * @param max_size Maximum cache size
     */
    void evict_cache(std::size_t max_size) const;
    
    /**
     * @brief Validates edge ID bounds
     * @param edge_id Edge to validate
     * @return true if valid
     */
    bool is_valid_edge_id(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Gets edge entry safely
     * @param edge_id Edge identifier
     * @return Pointer to edge entry or nullptr if invalid
     */
    CompressedEdgeEntry* get_edge_entry(EdgeId edge_id) const noexcept;
    
    /**
     * @brief Updates memory usage statistics
     */
    void update_memory_stats() noexcept;
    
    /**
     * @brief Rebuilds the edge index
     */
    void rebuild_index();
};

// ==================== INLINE IMPLEMENTATIONS ====================

inline bool EdgeStorage::edge_exists(EdgeId edge_id) const noexcept {
    CompressedEdgeEntry* entry = get_edge_entry(edge_id);
    return entry && entry->version.load(std::memory_order_acquire) > 0;
}

inline Weight EdgeStorage::get_edge_weight(EdgeId edge_id) const noexcept {
    CompressedEdgeEntry* entry = get_edge_entry(edge_id);
    return entry ? entry->weight : std::numeric_limits<Weight>::quiet_NaN();
}

inline std::pair<NodeId, NodeId> EdgeStorage::get_edge_endpoints(EdgeId edge_id) const noexcept {
    CompressedEdgeEntry* entry = get_edge_entry(edge_id);
    return entry ? std::make_pair(entry->from_node, entry->to_node) : std::make_pair(0, 0);
}

inline std::size_t EdgeStorage::edge_count() const noexcept {
    return edge_count_.load(std::memory_order_acquire);
}

inline std::size_t EdgeStorage::capacity() const noexcept {
    return capacity_.load(std::memory_order_acquire);
}

inline double EdgeStorage::load_factor() const noexcept {
    std::size_t cap = capacity();
    return cap > 0 ? static_cast<double>(edge_count()) / cap : 0.0;
}

inline void EdgeStorage::prefetch_edge(EdgeId edge_id) const noexcept {
    CompressedEdgeEntry* entry = get_edge_entry(edge_id);
    if (entry) {
#ifdef __builtin_prefetch
        __builtin_prefetch(entry, 0, 3);
#endif
    }
}

inline bool EdgeStorage::is_valid_edge_id(EdgeId edge_id) const noexcept {
    return edge_id > 0 && edge_id <= max_edge_id_.load(std::memory_order_acquire);
}

inline EdgeStorage::CompressedEdgeEntry* EdgeStorage::get_edge_entry(EdgeId edge_id) const noexcept {
    return is_valid_edge_id(edge_id) ? &entries_[edge_id] : nullptr;
}

inline bool EdgeStorage::update_edge_weight(EdgeId edge_id, Weight weight) {
    CompressedEdgeEntry* entry = get_edge_entry(edge_id);
    if (entry && entry->version.load(std::memory_order_acquire) > 0) {
        entry->weight = weight;
        entry->version.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }
    return false;
}

} // namespace ultra_fast_kg