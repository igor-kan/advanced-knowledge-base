/**
 * @file node_storage.hpp
 * @brief High-performance node storage with lock-free operations
 * 
 * This implementation provides:
 * - Lock-free concurrent access using atomic operations
 * - Memory-mapped persistence for large datasets
 * - Cache-aligned storage for optimal CPU performance
 * - Compression for space efficiency
 * - NUMA-aware memory allocation
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
 * @brief Configuration for node storage optimization
 */
struct NodeStorageConfig {
    std::size_t initial_capacity = 1'000'000;
    std::size_t growth_factor = 2;
    bool enable_compression = true;
    bool enable_memory_mapping = true;
    bool enable_numa_allocation = true;
    std::size_t cache_line_size = 64;
    CompressionType compression_type = CompressionType::LZ4;
    std::size_t numa_node = 0;
};

/**
 * @brief Compressed node entry for space efficiency
 */
struct alignas(64) CompressedNodeEntry {
    // Core data (32 bytes)
    NodeId node_id;
    std::atomic<std::uint32_t> label_offset;
    std::atomic<std::uint32_t> label_length;
    std::atomic<std::uint32_t> properties_offset;
    std::atomic<std::uint32_t> properties_length;
    std::atomic<std::uint64_t> created_at_ns;
    std::atomic<std::uint64_t> updated_at_ns;
    
    // Metadata (16 bytes)
    std::atomic<std::uint32_t> version;
    std::atomic<std::uint16_t> type_id;
    std::atomic<std::uint8_t> flags;
    std::atomic<std::uint8_t> compression_flags;
    std::uint32_t reserved;
    
    // Padding to cache line (16 bytes)
    std::uint32_t padding[4];
    
    CompressedNodeEntry() : node_id(0), version(0), type_id(0), flags(0), compression_flags(0), reserved(0) {
        label_offset.store(0, std::memory_order_relaxed);
        label_length.store(0, std::memory_order_relaxed);
        properties_offset.store(0, std::memory_order_relaxed);
        properties_length.store(0, std::memory_order_relaxed);
        created_at_ns.store(0, std::memory_order_relaxed);
        updated_at_ns.store(0, std::memory_order_relaxed);
    }
    
    // Non-copyable, movable
    CompressedNodeEntry(const CompressedNodeEntry&) = delete;
    CompressedNodeEntry& operator=(const CompressedNodeEntry&) = delete;
    CompressedNodeEntry(CompressedNodeEntry&&) = default;
    CompressedNodeEntry& operator=(CompressedNodeEntry&&) = default;
};

static_assert(sizeof(CompressedNodeEntry) == 64, "CompressedNodeEntry must be exactly one cache line");

/**
 * @brief Memory pool for variable-length data storage
 */
class VariableLengthPool {
public:
    explicit VariableLengthPool(std::size_t initial_size = 64 * 1024 * 1024); // 64MB
    ~VariableLengthPool();
    
    /**
     * @brief Allocates space for variable-length data
     * @param size Number of bytes to allocate
     * @return Offset in the pool, or 0 if allocation failed
     */
    std::uint32_t allocate(std::size_t size);
    
    /**
     * @brief Deallocates previously allocated space
     * @param offset Offset returned by allocate()
     * @param size Size that was allocated
     */
    void deallocate(std::uint32_t offset, std::size_t size);
    
    /**
     * @brief Gets pointer to data at offset
     * @param offset Offset in the pool
     * @return Pointer to data, or nullptr if invalid offset
     */
    void* get_pointer(std::uint32_t offset) const noexcept;
    
    /**
     * @brief Gets read-only pointer to data at offset
     * @param offset Offset in the pool
     * @return Const pointer to data, or nullptr if invalid offset
     */
    const void* get_const_pointer(std::uint32_t offset) const noexcept;
    
    /**
     * @brief Gets current pool usage
     * @return Number of bytes used
     */
    std::size_t get_usage() const noexcept;
    
    /**
     * @brief Gets total pool capacity
     * @return Total capacity in bytes
     */
    std::size_t get_capacity() const noexcept;
    
    /**
     * @brief Compacts the pool to reduce fragmentation
     * @return Number of bytes freed
     */
    std::size_t compact();
    
private:
    std::unique_ptr<std::uint8_t[]> data_;
    std::atomic<std::size_t> capacity_;
    std::atomic<std::size_t> used_;
    std::atomic<std::uint32_t> next_offset_;
    mutable std::shared_mutex mutex_;
    
    // Free list for deallocated blocks
    struct FreeBlock {
        std::uint32_t offset;
        std::uint32_t size;
        FreeBlock* next;
    };
    std::atomic<FreeBlock*> free_list_{nullptr};
    
    void expand_pool(std::size_t min_additional_size);
    FreeBlock* find_free_block(std::size_t size);
    void insert_free_block(std::uint32_t offset, std::uint32_t size);
};

/**
 * @brief Ultra-high performance node storage implementation
 * 
 * This class provides the core node storage for the knowledge graph
 * with extreme performance optimizations:
 * 
 * - Lock-free concurrent reads and atomic writes
 * - Cache-aligned node entries for optimal CPU performance
 * - Compressed storage for space efficiency
 * - Memory-mapped persistence for large datasets
 * - NUMA-aware allocation for multi-socket systems
 */
class NodeStorage {
public:
    /**
     * @brief Constructs a new node storage
     * @param config Configuration parameters
     */
    explicit NodeStorage(const NodeStorageConfig& config = NodeStorageConfig{});
    
    /**
     * @brief Destructor with cleanup
     */
    ~NodeStorage();
    
    // Non-copyable but movable
    NodeStorage(const NodeStorage&) = delete;
    NodeStorage& operator=(const NodeStorage&) = delete;
    NodeStorage(NodeStorage&& other) noexcept;
    NodeStorage& operator=(NodeStorage&& other) noexcept;
    
    // ==================== CORE OPERATIONS ====================
    
    /**
     * @brief Stores a new node atomically
     * @param node_id Node identifier
     * @param data Node data to store
     * @return true if stored successfully, false if node already exists
     */
    bool store_node(NodeId node_id, NodeData data);
    
    /**
     * @brief Updates an existing node atomically
     * @param node_id Node identifier
     * @param data New node data
     * @return true if updated successfully, false if node doesn't exist
     */
    bool update_node(NodeId node_id, NodeData data);
    
    /**
     * @brief Removes a node from storage
     * @param node_id Node identifier
     * @return true if removed successfully, false if node didn't exist
     */
    bool remove_node(NodeId node_id);
    
    /**
     * @brief Gets node data with zero-copy access when possible
     * @param node_id Node identifier
     * @return Pointer to node data, or nullptr if not found
     * 
     * Performance: Sub-microsecond access time with cache optimization
     */
    [[nodiscard]] const NodeData* get_node_data(NodeId node_id) const noexcept;
    
    /**
     * @brief Gets node label with zero-copy access
     * @param node_id Node identifier
     * @return String view of the label, or empty if not found
     */
    [[nodiscard]] std::string_view get_node_label(NodeId node_id) const noexcept;
    
    /**
     * @brief Gets node properties with zero-copy access
     * @param node_id Node identifier
     * @return Pointer to property map, or nullptr if not found
     */
    [[nodiscard]] const PropertyMap* get_node_properties(NodeId node_id) const noexcept;
    
    /**
     * @brief Checks if a node exists
     * @param node_id Node identifier
     * @return true if node exists
     */
    [[nodiscard]] bool node_exists(NodeId node_id) const noexcept;
    
    // ==================== BATCH OPERATIONS ====================
    
    /**
     * @brief Batch stores multiple nodes for better performance
     * @param nodes Vector of (node_id, node_data) pairs
     * @return Number of successfully stored nodes
     */
    std::size_t batch_store_nodes(std::vector<std::pair<NodeId, NodeData>> nodes);
    
    /**
     * @brief Batch updates multiple nodes
     * @param nodes Vector of (node_id, node_data) pairs
     * @return Number of successfully updated nodes
     */
    std::size_t batch_update_nodes(std::vector<std::pair<NodeId, NodeData>> nodes);
    
    /**
     * @brief Batch gets multiple nodes
     * @param node_ids Vector of node identifiers
     * @return Vector of node data pointers (nullptr for non-existent nodes)
     */
    [[nodiscard]] std::vector<const NodeData*> batch_get_nodes(const std::vector<NodeId>& node_ids) const;
    
    // ==================== QUERY OPERATIONS ====================
    
    /**
     * @brief Finds nodes by label pattern
     * @param pattern Label pattern (supports wildcards)
     * @param max_results Maximum number of results
     * @return Vector of matching node IDs
     */
    [[nodiscard]] std::vector<NodeId> find_nodes_by_label(const std::string& pattern, 
                                                         std::size_t max_results = 1000) const;
    
    /**
     * @brief Finds nodes by property filter
     * @param property_key Property name
     * @param property_value Property value to match
     * @param max_results Maximum number of results
     * @return Vector of matching node IDs
     */
    [[nodiscard]] std::vector<NodeId> find_nodes_by_property(const std::string& property_key,
                                                            const PropertyValue& property_value,
                                                            std::size_t max_results = 1000) const;
    
    /**
     * @brief Gets all node IDs currently stored
     * @return Vector of all node IDs
     */
    [[nodiscard]] std::vector<NodeId> get_all_node_ids() const;
    
    /**
     * @brief Gets nodes created within a time range
     * @param start_time Start of time range
     * @param end_time End of time range
     * @return Vector of node IDs created in the range
     */
    [[nodiscard]] std::vector<NodeId> get_nodes_by_time_range(
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
     * @brief Prefetches node data into cache
     * @param node_id Node to prefetch
     */
    void prefetch_node(NodeId node_id) const noexcept;
    
    /**
     * @brief Gets current memory usage
     * @return Memory usage breakdown
     */
    [[nodiscard]] MemoryUsage get_memory_usage() const noexcept;
    
    /**
     * @brief Reserves capacity for expected number of nodes
     * @param capacity Expected number of nodes
     */
    void reserve_capacity(std::size_t capacity);
    
    // ==================== PERSISTENCE ====================
    
    /**
     * @brief Saves node storage to disk with compression
     * @param path File path
     * @param compression Compression algorithm
     * @return Number of bytes written
     */
    std::size_t save_to_disk(const std::string& path, CompressionType compression = CompressionType::LZ4) const;
    
    /**
     * @brief Loads node storage from disk
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
     * @brief Gets the total number of nodes stored
     * @return Node count
     */
    [[nodiscard]] std::size_t node_count() const noexcept;
    
    /**
     * @brief Gets the current storage capacity
     * @return Maximum nodes that can be stored without reallocation
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
     * @brief Gets average node size in bytes
     * @return Average node size
     */
    [[nodiscard]] double average_node_size() const noexcept;
    
    /**
     * @brief Validates storage integrity
     * @return true if storage is valid
     */
    [[nodiscard]] bool validate() const noexcept;
    
private:
    // ==================== MEMBER VARIABLES ====================
    
    NodeStorageConfig config_;
    
    // Core storage arrays
    std::unique_ptr<CompressedNodeEntry[]> entries_;
    std::atomic<std::size_t> capacity_{0};
    std::atomic<std::size_t> node_count_{0};
    std::atomic<NodeId> max_node_id_{0};
    
    // Variable-length data storage
    std::unique_ptr<VariableLengthPool> label_pool_;
    std::unique_ptr<VariableLengthPool> properties_pool_;
    
    // Node data cache for decompressed entries
    mutable std::unordered_map<NodeId, std::unique_ptr<NodeData>> node_cache_;
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
    
    // Property indexing for fast queries
    std::unordered_map<std::string, std::unordered_map<Hash, std::vector<NodeId>>> property_index_;
    mutable std::shared_mutex index_mutex_;
    
    // ==================== INTERNAL HELPER METHODS ====================
    
    /**
     * @brief Ensures capacity for at least min_node_id
     * @param min_node_id Minimum node ID that must fit
     */
    void ensure_capacity(NodeId min_node_id);
    
    /**
     * @brief Compresses node data for storage
     * @param data Node data to compress
     * @param label_offset Output: offset in label pool
     * @param label_length Output: length in label pool
     * @param props_offset Output: offset in properties pool
     * @param props_length Output: length in properties pool
     * @return true if compression successful
     */
    bool compress_node_data(const NodeData& data,
                           std::uint32_t& label_offset, std::uint32_t& label_length,
                           std::uint32_t& props_offset, std::uint32_t& props_length);
    
    /**
     * @brief Decompresses node data from storage
     * @param entry Compressed node entry
     * @return Unique pointer to decompressed node data
     */
    std::unique_ptr<NodeData> decompress_node_data(const CompressedNodeEntry& entry) const;
    
    /**
     * @brief Updates property index for a node
     * @param node_id Node identifier
     * @param properties Node properties
     */
    void update_property_index(NodeId node_id, const PropertyMap& properties);
    
    /**
     * @brief Removes node from property index
     * @param node_id Node identifier
     * @param properties Node properties to remove
     */
    void remove_from_property_index(NodeId node_id, const PropertyMap& properties);
    
    /**
     * @brief Gets or creates cached node data
     * @param node_id Node identifier
     * @return Pointer to cached node data
     */
    const NodeData* get_or_cache_node_data(NodeId node_id) const;
    
    /**
     * @brief Evicts old entries from cache
     * @param max_size Maximum cache size
     */
    void evict_cache(std::size_t max_size) const;
    
    /**
     * @brief Validates node ID bounds
     * @param node_id Node to validate
     * @return true if valid
     */
    bool is_valid_node_id(NodeId node_id) const noexcept;
    
    /**
     * @brief Gets node entry safely
     * @param node_id Node identifier
     * @return Pointer to node entry or nullptr if invalid
     */
    CompressedNodeEntry* get_node_entry(NodeId node_id) const noexcept;
    
    /**
     * @brief Updates memory usage statistics
     */
    void update_memory_stats() noexcept;
    
    /**
     * @brief Hashes property value for indexing
     * @param value Property value to hash
     * @return Hash value
     */
    Hash hash_property_value(const PropertyValue& value) const noexcept;
};

// ==================== INLINE IMPLEMENTATIONS ====================

inline bool NodeStorage::node_exists(NodeId node_id) const noexcept {
    CompressedNodeEntry* entry = get_node_entry(node_id);
    return entry && entry->version.load(std::memory_order_acquire) > 0;
}

inline std::size_t NodeStorage::node_count() const noexcept {
    return node_count_.load(std::memory_order_acquire);
}

inline std::size_t NodeStorage::capacity() const noexcept {
    return capacity_.load(std::memory_order_acquire);
}

inline double NodeStorage::load_factor() const noexcept {
    std::size_t cap = capacity();
    return cap > 0 ? static_cast<double>(node_count()) / cap : 0.0;
}

inline void NodeStorage::prefetch_node(NodeId node_id) const noexcept {
    CompressedNodeEntry* entry = get_node_entry(node_id);
    if (entry) {
#ifdef __builtin_prefetch
        __builtin_prefetch(entry, 0, 3);
#endif
    }
}

inline bool NodeStorage::is_valid_node_id(NodeId node_id) const noexcept {
    return node_id > 0 && node_id <= max_node_id_.load(std::memory_order_acquire);
}

inline NodeStorage::CompressedNodeEntry* NodeStorage::get_node_entry(NodeId node_id) const noexcept {
    return is_valid_node_id(node_id) ? &entries_[node_id] : nullptr;
}

} // namespace ultra_fast_kg