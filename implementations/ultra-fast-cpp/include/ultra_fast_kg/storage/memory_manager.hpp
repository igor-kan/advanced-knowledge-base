/**
 * @file memory_manager.hpp
 * @brief Ultra-high performance memory management for knowledge graphs
 * 
 * This implementation provides:
 * - Custom allocators optimized for graph workloads
 * - NUMA-aware memory allocation
 * - Memory-mapped file support
 * - Lock-free memory pools
 * - Cache-aligned allocation
 * - Memory usage tracking and optimization
 * 
 * @author Ultra-Fast Knowledge Graph Team
 * @version 1.0.0
 */

#pragma once

#include "ultra_fast_kg/core/types.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// Platform-specific includes
#ifdef __linux__
#include <numa.h>
#include <sys/mman.h>
#endif

namespace ultra_fast_kg {

/**
 * @brief Configuration for memory management
 */
struct MemoryManagerConfig {
    // Basic allocation settings
    std::size_t initial_pool_size = 1ULL * 1024 * 1024 * 1024; // 1GB
    std::size_t max_pool_size = 64ULL * 1024 * 1024 * 1024;    // 64GB
    std::size_t page_size = 4096;
    bool enable_huge_pages = true;
    
    // NUMA settings
    bool enable_numa = true;
    std::size_t numa_node = 0;
    bool numa_interleave = false;
    
    // Memory mapping
    bool enable_memory_mapping = true;
    std::string memory_map_path = "./kg_memory";
    bool prefault_pages = true;
    
    // Pool management
    std::size_t small_block_size = 64;
    std::size_t medium_block_size = 1024;
    std::size_t large_block_size = 65536;
    std::size_t alignment = 64; // Cache line alignment
    
    // Garbage collection
    bool enable_gc = true;
    std::size_t gc_threshold_mb = 1024; // 1GB
    std::chrono::milliseconds gc_interval{10000}; // 10 seconds
    
    // Monitoring
    bool enable_monitoring = true;
    bool track_allocations = false; // Expensive, only for debugging
};

/**
 * @brief Memory allocation statistics
 */
struct MemoryStats {
    // Basic counters
    std::atomic<std::size_t> total_allocated{0};
    std::atomic<std::size_t> total_freed{0};
    std::atomic<std::size_t> current_usage{0};
    std::atomic<std::size_t> peak_usage{0};
    
    // Allocation counters
    std::atomic<std::size_t> allocation_count{0};
    std::atomic<std::size_t> deallocation_count{0};
    std::atomic<std::size_t> reallocation_count{0};
    
    // Pool statistics
    std::atomic<std::size_t> small_pool_usage{0};
    std::atomic<std::size_t> medium_pool_usage{0};
    std::atomic<std::size_t> large_pool_usage{0};
    std::atomic<std::size_t> huge_pool_usage{0};
    
    // Performance metrics
    std::atomic<std::uint64_t> allocation_time_ns{0};
    std::atomic<std::uint64_t> deallocation_time_ns{0};
    std::atomic<std::size_t> cache_hits{0};
    std::atomic<std::size_t> cache_misses{0};
    
    // NUMA statistics
    std::array<std::atomic<std::size_t>, 8> numa_usage{}; // Up to 8 NUMA nodes
    
    void reset() noexcept {
        total_allocated.store(0);
        total_freed.store(0);
        current_usage.store(0);
        peak_usage.store(0);
        allocation_count.store(0);
        deallocation_count.store(0);
        reallocation_count.store(0);
        small_pool_usage.store(0);
        medium_pool_usage.store(0);
        large_pool_usage.store(0);
        huge_pool_usage.store(0);
        allocation_time_ns.store(0);
        deallocation_time_ns.store(0);
        cache_hits.store(0);
        cache_misses.store(0);
        for (auto& usage : numa_usage) {
            usage.store(0);
        }
    }
};

/**
 * @brief Lock-free memory pool for fixed-size allocations
 */
template<std::size_t BlockSize, std::size_t Alignment = 64>
class LockFreePool {
public:
    static constexpr std::size_t BLOCK_SIZE = (BlockSize + Alignment - 1) & ~(Alignment - 1);
    
    explicit LockFreePool(std::size_t initial_blocks = 1024)
        : block_count_(initial_blocks) {
        allocate_chunk();
    }
    
    ~LockFreePool() {
        cleanup();
    }
    
    /**
     * @brief Allocates a block from the pool
     * @return Pointer to allocated block, or nullptr if allocation failed
     */
    void* allocate() noexcept {
        FreeBlock* block = free_list_.load(std::memory_order_acquire);
        
        while (block) {
            FreeBlock* next = block->next.load(std::memory_order_relaxed);
            if (free_list_.compare_exchange_weak(block, next, std::memory_order_release, std::memory_order_acquire)) {
                allocated_count_.fetch_add(1, std::memory_order_relaxed);
                return block;
            }
        }
        
        // No free blocks available, try to allocate new chunk
        if (allocate_chunk()) {
            return allocate(); // Retry
        }
        
        return nullptr;
    }
    
    /**
     * @brief Deallocates a block back to the pool
     * @param ptr Pointer to block to deallocate
     */
    void deallocate(void* ptr) noexcept {
        if (!ptr) return;
        
        FreeBlock* block = static_cast<FreeBlock*>(ptr);
        FreeBlock* current_head = free_list_.load(std::memory_order_relaxed);
        
        do {
            block->next.store(current_head, std::memory_order_relaxed);
        } while (!free_list_.compare_exchange_weak(current_head, block, std::memory_order_release, std::memory_order_relaxed));
        
        allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Gets current allocation statistics
     * @return Pair of (allocated_blocks, total_capacity)
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t> get_stats() const noexcept {
        return {allocated_count_.load(), block_count_.load()};
    }
    
private:
    struct FreeBlock {
        std::atomic<FreeBlock*> next{nullptr};
    };
    
    std::atomic<FreeBlock*> free_list_{nullptr};
    std::atomic<std::size_t> allocated_count_{0};
    std::atomic<std::size_t> block_count_{0};
    std::vector<std::unique_ptr<std::uint8_t[]>> chunks_;
    std::mutex chunk_mutex_;
    
    bool allocate_chunk() {
        std::lock_guard<std::mutex> lock(chunk_mutex_);
        
        std::size_t chunk_size = block_count_.load() * BLOCK_SIZE;
        auto chunk = std::make_unique<std::uint8_t[]>(chunk_size);
        
        if (!chunk) return false;
        
        // Initialize free list for this chunk
        std::uint8_t* ptr = chunk.get();
        FreeBlock* prev = nullptr;
        
        for (std::size_t i = 0; i < block_count_.load(); ++i) {
            FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr + i * BLOCK_SIZE);
            if (prev) {
                prev->next.store(block, std::memory_order_relaxed);
            } else {
                // First block becomes the new head
                FreeBlock* current_head = free_list_.load(std::memory_order_relaxed);
                do {
                    block->next.store(current_head, std::memory_order_relaxed);
                } while (!free_list_.compare_exchange_weak(current_head, block, std::memory_order_release, std::memory_order_relaxed));
            }
            prev = block;
        }
        
        chunks_.push_back(std::move(chunk));
        block_count_.fetch_add(block_count_.load(), std::memory_order_relaxed);
        
        return true;
    }
    
    void cleanup() {
        chunks_.clear();
        free_list_.store(nullptr);
        allocated_count_.store(0);
        block_count_.store(0);
    }
};

/**
 * @brief Memory-mapped file manager
 */
class MemoryMappedFile {
public:
    MemoryMappedFile() = default;
    ~MemoryMappedFile();
    
    /**
     * @brief Maps a file into memory
     * @param path File path
     * @param size File size (0 = use file size)
     * @param read_only Whether to map read-only
     * @param prefault Whether to prefault pages
     * @return true if successful
     */
    bool map_file(const std::string& path, std::size_t size = 0, bool read_only = false, bool prefault = true);
    
    /**
     * @brief Creates and maps a new file
     * @param path File path
     * @param size File size
     * @param prefault Whether to prefault pages
     * @return true if successful
     */
    bool create_file(const std::string& path, std::size_t size, bool prefault = true);
    
    /**
     * @brief Unmaps the file
     */
    void unmap();
    
    /**
     * @brief Gets the mapped memory pointer
     * @return Pointer to mapped memory, or nullptr if not mapped
     */
    [[nodiscard]] void* data() const noexcept { return data_; }
    
    /**
     * @brief Gets the mapped size
     * @return Size of mapped region
     */
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    
    /**
     * @brief Checks if file is mapped
     * @return true if mapped
     */
    [[nodiscard]] bool is_mapped() const noexcept { return data_ != nullptr; }
    
    /**
     * @brief Syncs changes to disk
     * @param async Whether to sync asynchronously
     * @return true if successful
     */
    bool sync(bool async = false);
    
    /**
     * @brief Advises the kernel about memory usage patterns
     * @param advice Memory advice (e.g., MADV_SEQUENTIAL)
     */
    void advise(int advice);
    
private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
    int fd_ = -1;
    bool read_only_ = false;
};

/**
 * @brief Ultra-high performance memory manager
 * 
 * This class provides comprehensive memory management for the knowledge graph
 * with extreme performance optimizations:
 * 
 * - Lock-free memory pools for different allocation sizes
 * - NUMA-aware allocation for multi-socket systems
 * - Memory-mapped file support for persistence
 * - Cache-aligned allocation for optimal CPU performance
 * - Comprehensive memory usage tracking
 */
class MemoryManager {
public:
    /**
     * @brief Constructs a new memory manager
     * @param config Configuration parameters
     */
    explicit MemoryManager(const MemoryManagerConfig& config = MemoryManagerConfig{});
    
    /**
     * @brief Destructor with cleanup
     */
    ~MemoryManager();
    
    // Non-copyable, non-movable (singleton-like)
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) = delete;
    MemoryManager& operator=(MemoryManager&&) = delete;
    
    // ==================== ALLOCATION OPERATIONS ====================
    
    /**
     * @brief Allocates memory with specified alignment
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment (must be power of 2)
     * @return Pointer to allocated memory, or nullptr if failed
     */
    [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = 64);
    
    /**
     * @brief Deallocates previously allocated memory
     * @param ptr Pointer to memory to deallocate
     * @param size Size that was allocated (for pool selection)
     */
    void deallocate(void* ptr, std::size_t size) noexcept;
    
    /**
     * @brief Reallocates memory with new size
     * @param ptr Existing pointer (can be nullptr)
     * @param old_size Previous size
     * @param new_size New size
     * @param alignment Memory alignment
     * @return Pointer to reallocated memory
     */
    [[nodiscard]] void* reallocate(void* ptr, std::size_t old_size, std::size_t new_size, std::size_t alignment = 64);
    
    /**
     * @brief Allocates cache-aligned memory
     * @param size Number of bytes to allocate
     * @return Pointer to cache-aligned memory
     */
    [[nodiscard]] void* allocate_aligned(std::size_t size) {
        return allocate(size, config_.alignment);
    }
    
    /**
     * @brief Allocates NUMA-local memory
     * @param size Number of bytes to allocate
     * @param numa_node NUMA node preference
     * @param alignment Memory alignment
     * @return Pointer to allocated memory
     */
    [[nodiscard]] void* allocate_numa(std::size_t size, std::size_t numa_node, std::size_t alignment = 64);
    
    // ==================== SPECIALIZED ALLOCATORS ====================
    
    /**
     * @brief Gets the small block pool (< 1KB)
     * @return Reference to small block pool
     */
    [[nodiscard]] LockFreePool<64>& get_small_pool() { return small_pool_; }
    
    /**
     * @brief Gets the medium block pool (1KB - 64KB)
     * @return Reference to medium block pool
     */
    [[nodiscard]] LockFreePool<1024>& get_medium_pool() { return medium_pool_; }
    
    /**
     * @brief Gets the large block pool (> 64KB)
     * @return Reference to large block pool
     */
    [[nodiscard]] LockFreePool<65536>& get_large_pool() { return large_pool_; }
    
    // ==================== MEMORY MAPPING ====================
    
    /**
     * @brief Creates a memory-mapped region
     * @param name Unique name for the mapping
     * @param path File path
     * @param size Size of mapping
     * @param read_only Whether to map read-only
     * @return Pointer to mapped memory, or nullptr if failed
     */
    [[nodiscard]] void* create_memory_mapping(const std::string& name, const std::string& path, 
                                             std::size_t size, bool read_only = false);
    
    /**
     * @brief Gets an existing memory mapping
     * @param name Name of the mapping
     * @return Pointer to mapped memory, or nullptr if not found
     */
    [[nodiscard]] void* get_memory_mapping(const std::string& name);
    
    /**
     * @brief Removes a memory mapping
     * @param name Name of the mapping
     * @return true if removed successfully
     */
    bool remove_memory_mapping(const std::string& name);
    
    /**
     * @brief Syncs all memory mappings to disk
     * @param async Whether to sync asynchronously
     */
    void sync_all_mappings(bool async = false);
    
    // ==================== MEMORY MONITORING ====================
    
    /**
     * @brief Gets current memory statistics
     * @return Memory usage statistics
     */
    [[nodiscard]] const MemoryStats& get_stats() const noexcept { return stats_; }
    
    /**
     * @brief Gets detailed memory usage breakdown
     * @return Memory usage breakdown
     */
    [[nodiscard]] MemoryUsage get_memory_usage() const;
    
    /**
     * @brief Gets memory usage for specific NUMA node
     * @param numa_node NUMA node ID
     * @return Memory usage in bytes
     */
    [[nodiscard]] std::size_t get_numa_usage(std::size_t numa_node) const;
    
    /**
     * @brief Checks if memory usage is above threshold
     * @param threshold_bytes Memory threshold in bytes
     * @return true if above threshold
     */
    [[nodiscard]] bool is_memory_pressure(std::size_t threshold_bytes) const;
    
    // ==================== OPTIMIZATION ====================
    
    /**
     * @brief Performs garbage collection and memory optimization
     * @return Amount of memory freed (in bytes)
     */
    std::size_t garbage_collect();
    
    /**
     * @brief Compacts memory pools to reduce fragmentation
     * @return Amount of memory defragmented
     */
    std::size_t compact_pools();
    
    /**
     * @brief Prefaults memory pages for better performance
     * @param ptr Memory pointer
     * @param size Memory size
     */
    void prefault_memory(void* ptr, std::size_t size);
    
    /**
     * @brief Advises the kernel about memory usage patterns
     * @param ptr Memory pointer
     * @param size Memory size
     * @param advice Memory advice
     */
    void memory_advise(void* ptr, std::size_t size, int advice);
    
    // ==================== CONFIGURATION ====================
    
    /**
     * @brief Gets the current configuration
     * @return Configuration reference
     */
    [[nodiscard]] const MemoryManagerConfig& get_config() const noexcept { return config_; }
    
    /**
     * @brief Updates configuration (some settings may require restart)
     * @param config New configuration
     */
    void update_config(const MemoryManagerConfig& config);
    
    /**
     * @brief Enables or disables memory tracking
     * @param enable Whether to enable tracking
     */
    void set_tracking_enabled(bool enable) { config_.track_allocations = enable; }
    
    /**
     * @brief Gets system memory information
     * @return Tuple of (total_memory, available_memory, numa_nodes)
     */
    [[nodiscard]] std::tuple<std::size_t, std::size_t, std::size_t> get_system_memory_info() const;
    
private:
    // ==================== MEMBER VARIABLES ====================
    
    MemoryManagerConfig config_;
    mutable MemoryStats stats_;
    
    // Memory pools for different sizes
    LockFreePool<64> small_pool_;      // < 1KB allocations
    LockFreePool<1024> medium_pool_;   // 1KB - 64KB allocations  
    LockFreePool<65536> large_pool_;   // > 64KB allocations
    
    // Memory mappings
    std::unordered_map<std::string, std::unique_ptr<MemoryMappedFile>> memory_mappings_;
    std::mutex mappings_mutex_;
    
    // Allocation tracking (when enabled)
    std::unordered_map<void*, std::pair<std::size_t, std::chrono::steady_clock::time_point>> allocations_;
    std::mutex tracking_mutex_;
    
    // Garbage collection
    std::atomic<bool> gc_running_{false};
    std::chrono::steady_clock::time_point last_gc_;
    
    // NUMA support
    bool numa_available_ = false;
    std::size_t numa_nodes_ = 1;
    
    // ==================== INTERNAL HELPER METHODS ====================
    
    /**
     * @brief Initializes NUMA support
     */
    void initialize_numa();
    
    /**
     * @brief Selects appropriate pool for allocation size
     * @param size Allocation size
     * @return Pool type (0=small, 1=medium, 2=large, 3=huge)
     */
    int select_pool(std::size_t size) const noexcept;
    
    /**
     * @brief Allocates from system (fallback)
     * @param size Number of bytes to allocate
     * @param alignment Memory alignment
     * @return Pointer to allocated memory
     */
    void* system_allocate(std::size_t size, std::size_t alignment);
    
    /**
     * @brief Deallocates to system
     * @param ptr Pointer to deallocate
     * @param size Size that was allocated
     */
    void system_deallocate(void* ptr, std::size_t size) noexcept;
    
    /**
     * @brief Updates memory statistics
     * @param allocated Bytes allocated (can be negative for deallocation)
     * @param numa_node NUMA node (if applicable)
     */
    void update_stats(std::ptrdiff_t allocated, std::size_t numa_node = 0);
    
    /**
     * @brief Records allocation for tracking
     * @param ptr Allocated pointer
     * @param size Allocation size
     */
    void record_allocation(void* ptr, std::size_t size);
    
    /**
     * @brief Removes allocation record
     * @param ptr Pointer to remove
     * @return Size that was allocated
     */
    std::size_t remove_allocation_record(void* ptr);
    
    /**
     * @brief Checks if garbage collection should run
     * @return true if GC should run
     */
    bool should_run_gc() const;
    
    /**
     * @brief Internal garbage collection implementation
     * @return Bytes freed
     */
    std::size_t gc_internal();
};

// ==================== GLOBAL MEMORY MANAGER ====================

/**
 * @brief Gets the global memory manager instance
 * @return Reference to global memory manager
 */
MemoryManager& get_global_memory_manager();

/**
 * @brief Initializes the global memory manager with custom configuration
 * @param config Memory manager configuration
 */
void initialize_global_memory_manager(const MemoryManagerConfig& config);

// ==================== CONVENIENCE FUNCTIONS ====================

/**
 * @brief Allocates cache-aligned memory using global manager
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory
 */
[[nodiscard]] inline void* aligned_alloc(std::size_t size) {
    return get_global_memory_manager().allocate_aligned(size);
}

/**
 * @brief Deallocates memory using global manager
 * @param ptr Pointer to deallocate
 * @param size Size that was allocated
 */
inline void aligned_free(void* ptr, std::size_t size) {
    get_global_memory_manager().deallocate(ptr, size);
}

/**
 * @brief RAII wrapper for aligned memory allocation
 */
template<typename T>
class AlignedPtr {
public:
    explicit AlignedPtr(std::size_t count = 1) 
        : ptr_(static_cast<T*>(aligned_alloc(sizeof(T) * count)))
        , count_(count) {}
    
    ~AlignedPtr() {
        if (ptr_) {
            aligned_free(ptr_, sizeof(T) * count_);
        }
    }
    
    // Non-copyable, movable
    AlignedPtr(const AlignedPtr&) = delete;
    AlignedPtr& operator=(const AlignedPtr&) = delete;
    
    AlignedPtr(AlignedPtr&& other) noexcept 
        : ptr_(std::exchange(other.ptr_, nullptr))
        , count_(std::exchange(other.count_, 0)) {}
    
    AlignedPtr& operator=(AlignedPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                aligned_free(ptr_, sizeof(T) * count_);
            }
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    
    [[nodiscard]] T* get() const noexcept { return ptr_; }
    [[nodiscard]] T* operator->() const noexcept { return ptr_; }
    [[nodiscard]] T& operator*() const noexcept { return *ptr_; }
    [[nodiscard]] T& operator[](std::size_t index) const noexcept { return ptr_[index]; }
    [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }
    
    [[nodiscard]] T* release() noexcept {
        count_ = 0;
        return std::exchange(ptr_, nullptr);
    }
    
    void reset(T* new_ptr = nullptr, std::size_t new_count = 0) {
        if (ptr_) {
            aligned_free(ptr_, sizeof(T) * count_);
        }
        ptr_ = new_ptr;
        count_ = new_count;
    }
    
private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace ultra_fast_kg