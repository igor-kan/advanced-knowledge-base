/**
 * @file query_engine.hpp
 * @brief Ultra-high performance query engine for knowledge graphs
 * 
 * This implementation provides:
 * - Cost-based query optimization and planning
 * - Parallel query execution with work stealing
 * - Advanced pattern matching with subgraph isomorphism
 * - Real-time query processing with caching
 * - SIMD-optimized query operators
 * 
 * @author Ultra-Fast Knowledge Graph Team
 * @version 1.0.0
 */

#pragma once

#include "ultra_fast_kg/core/types.hpp"
#include "ultra_fast_kg/storage/csr_matrix.hpp"
#include "ultra_fast_kg/storage/node_storage.hpp"
#include "ultra_fast_kg/storage/edge_storage.hpp"
#include "ultra_fast_kg/algorithms/algorithms.hpp"
#include "ultra_fast_kg/simd/simd_operations.hpp"
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <memory>

namespace ultra_fast_kg {

/**
 * @brief Query execution statistics
 */
struct QueryStats {
    std::chrono::nanoseconds compilation_time{0};
    std::chrono::nanoseconds execution_time{0};
    std::chrono::nanoseconds total_time{0};
    
    std::size_t nodes_scanned = 0;
    std::size_t edges_scanned = 0;
    std::size_t results_produced = 0;
    std::size_t intermediate_results = 0;
    
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t memory_allocated = 0;
    std::size_t simd_operations = 0;
    
    bool used_index = false;
    bool used_parallel = false;
    bool used_simd = false;
    double selectivity = 0.0;
    double cost_estimate = 0.0;
    
    void reset() {
        *this = QueryStats{};
    }
};

/**
 * @brief Query configuration and optimization hints
 */
struct QueryConfig {
    // Execution settings
    std::size_t max_results = 10000;
    std::chrono::milliseconds timeout{30000}; // 30 seconds
    bool enable_parallel = true;
    bool enable_simd = true;
    std::size_t max_threads = 0; // 0 = auto-detect
    
    // Optimization settings
    bool enable_cost_optimization = true;
    bool enable_caching = true;
    bool enable_index_usage = true;
    bool enable_pushdown = true; // Predicate pushdown
    
    // Memory settings
    std::size_t memory_limit_mb = 1024; // 1GB
    bool enable_spilling = true;
    bool prefer_memory_over_cpu = false;
    
    // Quality settings
    double min_confidence = 0.0;
    bool exact_matching = true;
    bool enable_approximation = false;
    double approximation_factor = 0.95;
};

/**
 * @brief Query compilation and execution plan
 */
struct QueryPlan {
    enum class OperatorType {
        Scan,           // Table/index scan
        Filter,         // Predicate filtering
        Join,           // Join between relations
        Project,        // Column projection
        Sort,           // Sorting
        Limit,          // Result limiting
        Aggregate,      // Aggregation
        Union,          // Set union
        Intersect,      // Set intersection
        Traversal,      // Graph traversal
        PatternMatch    // Pattern matching
    };
    
    struct Operator {
        OperatorType type;
        std::string description;
        double estimated_cost;
        double estimated_selectivity;
        std::size_t estimated_cardinality;
        std::unordered_map<std::string, std::string> parameters;
        std::vector<std::unique_ptr<Operator>> children;
        
        Operator(OperatorType t, std::string desc, double cost = 0.0)
            : type(t), description(std::move(desc)), estimated_cost(cost)
            , estimated_selectivity(1.0), estimated_cardinality(0) {}
    };
    
    std::unique_ptr<Operator> root;
    double total_estimated_cost = 0.0;
    std::size_t estimated_memory_usage = 0;
    bool can_use_parallel = false;
    bool can_use_simd = false;
    bool uses_indices = false;
    
    QueryPlan() = default;
    QueryPlan(QueryPlan&&) = default;
    QueryPlan& operator=(QueryPlan&&) = default;
    
    // Non-copyable
    QueryPlan(const QueryPlan&) = delete;
    QueryPlan& operator=(const QueryPlan&) = delete;
};

/**
 * @brief Query result with metadata
 */
struct QueryResult {
    std::vector<PatternMatch> matches;
    std::vector<NodeId> nodes;
    std::vector<EdgeId> edges;
    std::unordered_map<std::string, std::vector<PropertyValue>> projected_properties;
    
    QueryStats statistics;
    bool is_complete = true;
    bool is_approximate = false;
    double confidence = 1.0;
    std::string execution_plan;
    
    // Pagination support
    std::size_t total_count = 0;
    std::size_t offset = 0;
    std::size_t limit = 0;
    bool has_more = false;
    
    QueryResult() = default;
    QueryResult(QueryResult&&) = default;
    QueryResult& operator=(QueryResult&&) = default;
    
    // Non-copyable
    QueryResult(const QueryResult&) = delete;
    QueryResult& operator=(const QueryResult&) = delete;
};

/**
 * @brief Advanced query with multiple clauses and optimization
 */
struct Query {
    // Pattern matching
    std::vector<Pattern> patterns;
    
    // Filtering conditions
    struct FilterCondition {
        std::string property_name;
        std::string operator_type; // "=", "!=", "<", ">", "<=", ">=", "contains", "regex"
        PropertyValue value;
        bool negated = false;
    };
    std::vector<FilterCondition> node_filters;
    std::vector<FilterCondition> edge_filters;
    
    // Traversal specifications
    struct TraversalSpec {
        NodeId start_node;
        std::optional<NodeId> target_node;
        TraversalAlgorithm algorithm = TraversalAlgorithm::BreadthFirst;
        std::optional<std::uint32_t> max_depth;
        std::vector<FilterCondition> path_filters;
    };
    std::vector<TraversalSpec> traversals;
    
    // Projection (which properties to return)
    std::vector<std::string> projected_node_properties;
    std::vector<std::string> projected_edge_properties;
    
    // Sorting and limiting
    struct SortSpec {
        std::string property_name;
        bool ascending = true;
        bool nulls_first = false;
    };
    std::vector<SortSpec> sort_order;
    std::optional<std::size_t> limit;
    std::optional<std::size_t> offset;
    
    // Aggregation
    struct AggregateSpec {
        std::string function; // "count", "sum", "avg", "min", "max"
        std::string property_name;
        std::string alias;
    };
    std::vector<AggregateSpec> aggregates;
    
    // Grouping
    std::vector<std::string> group_by_properties;
    
    // Query configuration
    QueryConfig config;
    
    Query() = default;
    Query(Query&&) = default;
    Query& operator=(Query&&) = default;
    
    // Non-copyable
    Query(const Query&) = delete;
    Query& operator=(const Query&) = delete;
};

/**
 * @brief Query result cache for performance optimization
 */
class QueryCache {
public:
    explicit QueryCache(std::size_t max_size_mb = 256);
    ~QueryCache() = default;
    
    /**
     * @brief Gets cached result for query
     * @param query_hash Query hash
     * @return Cached result or nullptr if not found
     */
    [[nodiscard]] const QueryResult* get(std::uint64_t query_hash) const;
    
    /**
     * @brief Stores query result in cache
     * @param query_hash Query hash
     * @param result Query result to cache
     */
    void put(std::uint64_t query_hash, std::shared_ptr<QueryResult> result);
    
    /**
     * @brief Clears the cache
     */
    void clear();
    
    /**
     * @brief Gets cache statistics
     * @return Tuple of (hits, misses, size_mb, entries)
     */
    [[nodiscard]] std::tuple<std::size_t, std::size_t, double, std::size_t> get_stats() const;
    
private:
    struct CacheEntry {
        std::shared_ptr<QueryResult> result;
        std::chrono::steady_clock::time_point last_accessed;
        std::size_t access_count = 1;
        std::size_t size_bytes = 0;
    };
    
    mutable std::unordered_map<std::uint64_t, CacheEntry> cache_;
    mutable std::mutex cache_mutex_;
    std::size_t max_size_bytes_;
    mutable std::size_t current_size_bytes_ = 0;
    mutable std::size_t cache_hits_ = 0;
    mutable std::size_t cache_misses_ = 0;
    
    void evict_lru();
    std::size_t estimate_result_size(const QueryResult& result) const;
};

/**
 * @brief Ultra-high performance query engine
 * 
 * This class provides comprehensive query processing capabilities
 * with extreme performance optimizations:
 * 
 * - Cost-based query optimization and planning
 * - Parallel execution with SIMD acceleration
 * - Advanced caching and memoization
 * - Real-time query processing
 * - Adaptive algorithm selection
 */
class QueryEngine {
public:
    /**
     * @brief Constructs the query engine
     * @param csr_matrix Pointer to CSR adjacency matrix
     * @param node_storage Pointer to node storage
     * @param edge_storage Pointer to edge storage
     * @param algorithm_engine Pointer to algorithm engine
     * @param simd_ops Pointer to SIMD operations
     */
    QueryEngine(const CSRMatrix* csr_matrix,
               const NodeStorage* node_storage,
               const EdgeStorage* edge_storage,
               const AlgorithmEngine* algorithm_engine,
               const SIMDOperations* simd_ops);
    
    /**
     * @brief Destructor
     */
    ~QueryEngine();
    
    // Non-copyable, movable
    QueryEngine(const QueryEngine&) = delete;
    QueryEngine& operator=(const QueryEngine&) = delete;
    QueryEngine(QueryEngine&&) noexcept;
    QueryEngine& operator=(QueryEngine&&) noexcept;
    
    // ==================== QUERY EXECUTION ====================
    
    /**
     * @brief Executes a complex query with optimization
     * @param query Query to execute
     * @return Query result with matches and statistics
     * 
     * Performance: Sub-millisecond execution for most queries
     */
    [[nodiscard]] QueryResult execute_query(Query query);
    
    /**
     * @brief Executes a simple pattern match query
     * @param pattern Pattern to search for
     * @param config Query configuration
     * @return Pattern matches
     */
    [[nodiscard]] std::vector<PatternMatch> execute_pattern_query(
        const Pattern& pattern,
        const QueryConfig& config = QueryConfig{}
    );
    
    /**
     * @brief Executes a traversal query
     * @param start_node Starting node
     * @param algorithm Traversal algorithm
     * @param max_depth Maximum depth
     * @param config Query configuration
     * @return Traversal result
     */
    [[nodiscard]] TraversalResult execute_traversal_query(
        NodeId start_node,
        TraversalAlgorithm algorithm,
        std::optional<std::uint32_t> max_depth = std::nullopt,
        const QueryConfig& config = QueryConfig{}
    );
    
    /**
     * @brief Executes multiple queries in parallel
     * @param queries Vector of queries to execute
     * @return Vector of query results
     */
    [[nodiscard]] std::vector<QueryResult> execute_batch_queries(std::vector<Query> queries);
    
    /**
     * @brief Executes streaming query with result callback
     * @param query Query to execute
     * @param callback Function called for each result batch
     * @param batch_size Size of result batches
     * @return Final query statistics
     */
    [[nodiscard]] QueryStats execute_streaming_query(
        Query query,
        std::function<void(const std::vector<PatternMatch>&)> callback,
        std::size_t batch_size = 1000
    );
    
    // ==================== QUERY OPTIMIZATION ====================
    
    /**
     * @brief Compiles and optimizes a query
     * @param query Query to compile
     * @return Optimized query execution plan
     */
    [[nodiscard]] QueryPlan compile_query(const Query& query);
    
    /**
     * @brief Estimates query execution cost
     * @param query Query to analyze
     * @return Estimated cost and selectivity
     */
    [[nodiscard]] std::pair<double, double> estimate_query_cost(const Query& query);
    
    /**
     * @brief Optimizes query execution plan
     * @param plan Original execution plan
     * @return Optimized execution plan
     */
    [[nodiscard]] QueryPlan optimize_plan(QueryPlan plan);
    
    /**
     * @brief Explains query execution plan
     * @param query Query to explain
     * @return Human-readable execution plan description
     */
    [[nodiscard]] std::string explain_query(const Query& query);
    
    // ==================== ADVANCED FEATURES ====================
    
    /**
     * @brief Executes approximate query with quality guarantees
     * @param query Query to execute
     * @param quality_threshold Minimum quality [0.0, 1.0]
     * @return Approximate query result
     */
    [[nodiscard]] QueryResult execute_approximate_query(
        Query query,
        double quality_threshold = 0.95
    );
    
    /**
     * @brief Executes temporal query with time constraints
     * @param query Query with temporal constraints
     * @param time_range Time range for filtering
     * @return Query result filtered by time
     */
    [[nodiscard]] QueryResult execute_temporal_query(
        Query query,
        std::pair<std::chrono::system_clock::time_point, std::chrono::system_clock::time_point> time_range
    );
    
    /**
     * @brief Executes fuzzy pattern matching
     * @param pattern Pattern with relaxed constraints
     * @param similarity_threshold Minimum similarity [0.0, 1.0]
     * @param config Query configuration
     * @return Fuzzy pattern matches
     */
    [[nodiscard]] std::vector<PatternMatch> execute_fuzzy_pattern_query(
        const Pattern& pattern,
        double similarity_threshold = 0.8,
        const QueryConfig& config = QueryConfig{}
    );
    
    // ==================== CACHING AND PERFORMANCE ====================
    
    /**
     * @brief Enables or disables query result caching
     * @param enable Whether to enable caching
     * @param cache_size_mb Cache size in megabytes
     */
    void set_caching_enabled(bool enable, std::size_t cache_size_mb = 256);
    
    /**
     * @brief Clears query result cache
     */
    void clear_cache();
    
    /**
     * @brief Gets query engine performance statistics
     * @return Performance metrics
     */
    [[nodiscard]] PerformanceMetrics get_performance_metrics() const;
    
    /**
     * @brief Resets performance counters
     */
    void reset_performance_metrics();
    
    /**
     * @brief Gets cache statistics
     * @return Cache hit/miss ratio and size information
     */
    [[nodiscard]] std::tuple<double, std::size_t, std::size_t> get_cache_stats() const;
    
    // ==================== CONFIGURATION ====================
    
    /**
     * @brief Sets default query configuration
     * @param config Default configuration for queries
     */
    void set_default_config(const QueryConfig& config);
    
    /**
     * @brief Gets current default configuration
     * @return Default query configuration
     */
    [[nodiscard]] const QueryConfig& get_default_config() const noexcept;
    
    /**
     * @brief Enables or disables parallel query execution
     * @param enable Whether to enable parallel execution
     * @param max_threads Maximum number of threads (0 = auto)
     */
    void set_parallel_execution(bool enable, std::size_t max_threads = 0);
    
    /**
     * @brief Enables or disables SIMD optimizations
     * @param enable Whether to enable SIMD
     */
    void set_simd_enabled(bool enable);
    
private:
    // ==================== MEMBER VARIABLES ====================
    
    const CSRMatrix* csr_matrix_;
    const NodeStorage* node_storage_;
    const EdgeStorage* edge_storage_;
    const AlgorithmEngine* algorithm_engine_;
    const SIMDOperations* simd_ops_;
    
    // Query processing components
    std::unique_ptr<QueryCache> cache_;
    QueryConfig default_config_;
    
    // Performance tracking
    mutable PerformanceMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    
    // Thread pool for parallel execution
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // ==================== INTERNAL QUERY PROCESSING ====================
    
    /**
     * @brief Executes a compiled query plan
     * @param plan Compiled execution plan
     * @param query Original query
     * @return Query result
     */
    QueryResult execute_plan(const QueryPlan& plan, const Query& query);
    
    /**
     * @brief Executes a single query operator
     * @param op Operator to execute
     * @param input Input data
     * @param context Execution context
     * @return Operator result
     */
    struct OperatorResult;
    OperatorResult execute_operator(
        const QueryPlan::Operator& op,
        const OperatorResult& input,
        const Query& context
    );
    
    /**
     * @brief Optimizes filter predicates
     * @param filters Filter conditions to optimize
     * @return Optimized filters with cost estimates
     */
    std::vector<Query::FilterCondition> optimize_filters(
        const std::vector<Query::FilterCondition>& filters
    );
    
    /**
     * @brief Selects optimal join algorithm
     * @param left_cardinality Left relation cardinality
     * @param right_cardinality Right relation cardinality
     * @param join_selectivity Join selectivity estimate
     * @return Best join algorithm
     */
    std::string select_join_algorithm(
        std::size_t left_cardinality,
        std::size_t right_cardinality,
        double join_selectivity
    );
    
    // ==================== PATTERN MATCHING INTERNALS ====================
    
    /**
     * @brief Compiles pattern for efficient execution
     * @param pattern Pattern to compile
     * @return Compiled pattern representation
     */
    struct CompiledPattern;
    std::unique_ptr<CompiledPattern> compile_pattern(const Pattern& pattern);
    
    /**
     * @brief Executes pattern matching with optimization
     * @param compiled_pattern Compiled pattern
     * @param config Execution configuration
     * @return Pattern matches
     */
    std::vector<PatternMatch> execute_pattern_matching(
        const CompiledPattern& compiled_pattern,
        const QueryConfig& config
    );
    
    /**
     * @brief Optimizes pattern matching order
     * @param pattern Original pattern
     * @return Optimized pattern with reordered constraints
     */
    Pattern optimize_pattern_order(const Pattern& pattern);
    
    // ==================== COST ESTIMATION ====================
    
    /**
     * @brief Estimates scan operation cost
     * @param table_name Table to scan
     * @param filters Applied filters
     * @return Cost estimate
     */
    double estimate_scan_cost(
        const std::string& table_name,
        const std::vector<Query::FilterCondition>& filters
    );
    
    /**
     * @brief Estimates join operation cost
     * @param left_cardinality Left input cardinality
     * @param right_cardinality Right input cardinality
     * @param join_selectivity Join selectivity
     * @return Cost estimate
     */
    double estimate_join_cost(
        std::size_t left_cardinality,
        std::size_t right_cardinality,
        double join_selectivity
    );
    
    /**
     * @brief Estimates filter selectivity
     * @param filter Filter condition
     * @return Selectivity estimate [0.0, 1.0]
     */
    double estimate_filter_selectivity(const Query::FilterCondition& filter);
    
    // ==================== UTILITY METHODS ====================
    
    /**
     * @brief Computes hash for query caching
     * @param query Query to hash
     * @return Query hash value
     */
    std::uint64_t compute_query_hash(const Query& query);
    
    /**
     * @brief Updates performance metrics
     * @param operation Operation name
     * @param stats Query statistics
     */
    void update_metrics(const std::string& operation, const QueryStats& stats);
    
    /**
     * @brief Validates query structure
     * @param query Query to validate
     * @return true if query is valid
     */
    bool validate_query(const Query& query);
    
    /**
     * @brief Converts pattern match to query result
     * @param matches Pattern matches
     * @param stats Query statistics
     * @return Query result
     */
    QueryResult convert_matches_to_result(
        std::vector<PatternMatch> matches,
        QueryStats stats
    );
};

// ==================== QUERY BUILDER UTILITY ====================

/**
 * @brief Fluent query builder for convenient query construction
 */
class QueryBuilder {
public:
    QueryBuilder() = default;
    
    // Pattern matching
    QueryBuilder& pattern(const Pattern& p) { query_.patterns.push_back(p); return *this; }
    
    // Filtering
    QueryBuilder& where_node(const std::string& property, const std::string& op, const PropertyValue& value) {
        query_.node_filters.push_back({property, op, value, false});
        return *this;
    }
    
    QueryBuilder& where_edge(const std::string& property, const std::string& op, const PropertyValue& value) {
        query_.edge_filters.push_back({property, op, value, false});
        return *this;
    }
    
    // Traversal
    QueryBuilder& traverse_from(NodeId start, TraversalAlgorithm algo = TraversalAlgorithm::BreadthFirst) {
        Query::TraversalSpec spec;
        spec.start_node = start;
        spec.algorithm = algo;
        query_.traversals.push_back(std::move(spec));
        return *this;
    }
    
    // Projection
    QueryBuilder& select_node_properties(const std::vector<std::string>& props) {
        query_.projected_node_properties = props;
        return *this;
    }
    
    QueryBuilder& select_edge_properties(const std::vector<std::string>& props) {
        query_.projected_edge_properties = props;
        return *this;
    }
    
    // Sorting and limiting
    QueryBuilder& order_by(const std::string& property, bool ascending = true) {
        query_.sort_order.push_back({property, ascending, false});
        return *this;
    }
    
    QueryBuilder& limit(std::size_t n) { query_.limit = n; return *this; }
    QueryBuilder& offset(std::size_t n) { query_.offset = n; return *this; }
    
    // Configuration
    QueryBuilder& config(const QueryConfig& cfg) { query_.config = cfg; return *this; }
    QueryBuilder& timeout(std::chrono::milliseconds ms) { query_.config.timeout = ms; return *this; }
    QueryBuilder& max_results(std::size_t n) { query_.config.max_results = n; return *this; }
    
    // Build final query
    [[nodiscard]] Query build() && { return std::move(query_); }
    [[nodiscard]] const Query& build() const & { return query_; }
    
private:
    Query query_;
};

} // namespace ultra_fast_kg