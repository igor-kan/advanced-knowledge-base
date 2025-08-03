//! Configuration management for hybrid knowledge graph
//!
//! This module provides comprehensive configuration management for all
//! hybrid components including performance tuning, feature flags, and
//! environment-specific settings.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{HybridError, HybridResult};

/// Main configuration for the hybrid knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Graph-specific configuration
    pub graph: GraphConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Algorithm configuration
    pub algorithms: AlgorithmConfig,
    
    /// Query engine configuration
    pub query: QueryConfig,
    
    /// SIMD configuration
    pub simd: SIMDConfig,
    
    /// Memory management configuration
    pub memory: MemoryConfig,
    
    /// Performance tuning configuration
    pub performance: PerformanceConfig,
    
    /// Network configuration (for distributed mode)
    #[cfg(feature = "distributed")]
    pub network: NetworkConfig,
    
    /// GPU configuration (if enabled)
    #[cfg(feature = "gpu")]
    pub gpu: GpuConfig,
    
    /// Monitoring and metrics configuration
    pub monitoring: MonitoringConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            graph: GraphConfig::default(),
            storage: StorageConfig::default(),
            algorithms: AlgorithmConfig::default(),
            query: QueryConfig::default(),
            simd: SIMDConfig::default(),
            memory: MemoryConfig::default(),
            performance: PerformanceConfig::default(),
            #[cfg(feature = "distributed")]
            network: NetworkConfig::default(),
            #[cfg(feature = "gpu")]
            gpu: GpuConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Graph-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Initial capacity for nodes
    pub initial_node_capacity: usize,
    
    /// Initial capacity for edges
    pub initial_edge_capacity: usize,
    
    /// Initial capacity for hyperedges
    pub initial_hyperedge_capacity: usize,
    
    /// Maximum number of nodes (0 = unlimited)
    pub max_nodes: usize,
    
    /// Maximum number of edges (0 = unlimited)
    pub max_edges: usize,
    
    /// Enable hypergraph support
    pub enable_hypergraphs: bool,
    
    /// Enable temporal features
    pub enable_temporal: bool,
    
    /// Default edge weight
    pub default_edge_weight: f32,
    
    /// Enable automatic optimization
    pub auto_optimize: bool,
    
    /// Optimization interval
    pub optimization_interval: Duration,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            initial_node_capacity: 1_000_000,
            initial_edge_capacity: 10_000_000,
            initial_hyperedge_capacity: 100_000,
            max_nodes: 0,
            max_edges: 0,
            enable_hypergraphs: true,
            enable_temporal: false,
            default_edge_weight: 1.0,
            auto_optimize: true,
            optimization_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Storage layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Data directory path
    pub data_dir: PathBuf,
    
    /// Enable memory-mapped files
    pub enable_mmap: bool,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    
    /// Compression level (0-9)
    pub compression_level: u32,
    
    /// Enable checksums for data integrity
    pub enable_checksums: bool,
    
    /// Page size for memory mapping
    pub page_size: usize,
    
    /// Cache size for frequently accessed data
    pub cache_size: usize,
    
    /// Enable write-ahead logging
    pub enable_wal: bool,
    
    /// WAL flush interval
    pub wal_flush_interval: Duration,
    
    /// Enable periodic backups
    pub enable_backups: bool,
    
    /// Backup interval
    pub backup_interval: Duration,
    
    /// Number of backup copies to keep
    pub backup_retention: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            enable_mmap: true,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Lz4,
            compression_level: 4,
            enable_checksums: true,
            page_size: 64 * 1024, // 64KB
            cache_size: 1024 * 1024 * 1024, // 1GB
            enable_wal: true,
            wal_flush_interval: Duration::from_millis(100),
            enable_backups: false,
            backup_interval: Duration::from_secs(3600), // 1 hour
            backup_retention: 24, // 1 day of hourly backups
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// LZ4 - fast compression
    Lz4,
    /// Zstandard - balanced compression
    Zstd,
    /// LZMA - high compression ratio
    Lzma,
}

/// Algorithm implementation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// Enable parallel algorithm execution
    pub enable_parallel: bool,
    
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    
    /// Work stealing queue size
    pub work_queue_size: usize,
    
    /// Enable algorithm caching
    pub enable_caching: bool,
    
    /// Cache size for algorithm results
    pub cache_size: usize,
    
    /// Cache TTL for algorithm results
    pub cache_ttl: Duration,
    
    /// BFS configuration
    pub bfs: BfsConfig,
    
    /// DFS configuration
    pub dfs: DfsConfig,
    
    /// Shortest path configuration
    pub shortest_path: ShortestPathConfig,
    
    /// PageRank configuration
    pub pagerank: PageRankConfig,
    
    /// Centrality configuration
    pub centrality: CentralityConfig,
    
    /// Community detection configuration
    pub community: CommunityConfig,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            worker_threads: 0,
            work_queue_size: 1024,
            enable_caching: true,
            cache_size: 128 * 1024 * 1024, // 128MB
            cache_ttl: Duration::from_secs(3600), // 1 hour
            bfs: BfsConfig::default(),
            dfs: DfsConfig::default(),
            shortest_path: ShortestPathConfig::default(),
            pagerank: PageRankConfig::default(),
            centrality: CentralityConfig::default(),
            community: CommunityConfig::default(),
        }
    }
}

/// BFS algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfsConfig {
    /// Maximum depth to explore
    pub max_depth: u32,
    
    /// Enable bidirectional search
    pub enable_bidirectional: bool,
    
    /// Batch size for parallel processing
    pub batch_size: usize,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            enable_bidirectional: true,
            batch_size: 1000,
        }
    }
}

/// DFS algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DfsConfig {
    /// Maximum depth to explore
    pub max_depth: u32,
    
    /// Stack size for DFS
    pub stack_size: usize,
}

impl Default for DfsConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            stack_size: 10000,
        }
    }
}

/// Shortest path algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathConfig {
    /// Algorithm to use
    pub algorithm: ShortestPathAlgorithm,
    
    /// Enable A* heuristic
    pub enable_astar_heuristic: bool,
    
    /// Heap size for Dijkstra's algorithm
    pub heap_size: usize,
    
    /// Maximum path length to consider
    pub max_path_length: usize,
}

impl Default for ShortestPathConfig {
    fn default() -> Self {
        Self {
            algorithm: ShortestPathAlgorithm::Dijkstra,
            enable_astar_heuristic: false,
            heap_size: 100000,
            max_path_length: 1000,
        }
    }
}

/// Shortest path algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ShortestPathAlgorithm {
    /// Dijkstra's algorithm
    Dijkstra,
    /// A* algorithm
    AStar,
    /// Bellman-Ford algorithm
    BellmanFord,
    /// Floyd-Warshall algorithm
    FloydWarshall,
}

/// PageRank algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankConfig {
    /// Damping factor
    pub damping_factor: f64,
    
    /// Convergence tolerance
    pub tolerance: f64,
    
    /// Maximum iterations
    pub max_iterations: usize,
    
    /// Enable personalized PageRank
    pub enable_personalized: bool,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            tolerance: 1e-6,
            max_iterations: 100,
            enable_personalized: false,
        }
    }
}

/// Centrality algorithms configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityConfig {
    /// Enable approximate algorithms for large graphs
    pub enable_approximate: bool,
    
    /// Sample size for approximate algorithms
    pub sample_size: usize,
    
    /// Confidence level for approximate results
    pub confidence_level: f64,
}

impl Default for CentralityConfig {
    fn default() -> Self {
        Self {
            enable_approximate: true,
            sample_size: 1000,
            confidence_level: 0.95,
        }
    }
}

/// Community detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityConfig {
    /// Algorithm to use
    pub algorithm: CommunityAlgorithm,
    
    /// Resolution parameter for modularity optimization
    pub resolution: f64,
    
    /// Maximum number of iterations
    pub max_iterations: usize,
    
    /// Minimum community size
    pub min_community_size: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            algorithm: CommunityAlgorithm::Louvain,
            resolution: 1.0,
            max_iterations: 100,
            min_community_size: 3,
        }
    }
}

/// Community detection algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm
    Louvain,
    /// Label propagation
    LabelPropagation,
    /// Fast greedy modularity
    FastGreedy,
    /// Walktrap algorithm
    WalkTrap,
}

/// Query engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryConfig {
    /// Enable query optimization
    pub enable_optimization: bool,
    
    /// Query timeout
    pub timeout: Duration,
    
    /// Maximum results per query
    pub max_results: usize,
    
    /// Enable query caching
    pub enable_caching: bool,
    
    /// Query cache size
    pub cache_size: usize,
    
    /// Cache TTL
    pub cache_ttl: Duration,
    
    /// Enable query profiling
    pub enable_profiling: bool,
    
    /// Pattern matching configuration
    pub pattern_matching: PatternMatchingConfig,
}

impl Default for QueryConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            timeout: Duration::from_secs(30),
            max_results: 10000,
            enable_caching: true,
            cache_size: 256 * 1024 * 1024, // 256MB
            cache_ttl: Duration::from_secs(1800), // 30 minutes
            enable_profiling: false,
            pattern_matching: PatternMatchingConfig::default(),
        }
    }
}

/// Pattern matching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchingConfig {
    /// Enable approximate matching
    pub enable_approximate: bool,
    
    /// Minimum confidence for approximate matches
    pub min_confidence: f64,
    
    /// Maximum pattern size
    pub max_pattern_size: usize,
    
    /// Enable subgraph isomorphism
    pub enable_isomorphism: bool,
}

impl Default for PatternMatchingConfig {
    fn default() -> Self {
        Self {
            enable_approximate: true,
            min_confidence: 0.8,
            max_pattern_size: 20,
            enable_isomorphism: true,
        }
    }
}

/// SIMD optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIMDConfig {
    /// Enable SIMD operations
    pub enabled: bool,
    
    /// Force specific SIMD width (0 = auto-detect)
    pub force_width: usize,
    
    /// Enable automatic fallback to scalar operations
    pub enable_fallback: bool,
    
    /// SIMD efficiency threshold for fallback
    pub efficiency_threshold: f32,
    
    /// Enable SIMD for specific operations
    pub enable_for_algorithms: bool,
    pub enable_for_memory_ops: bool,
    pub enable_for_pattern_matching: bool,
    
    /// SIMD warmup iterations
    pub warmup_iterations: usize,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            force_width: 0,
            enable_fallback: true,
            efficiency_threshold: 0.7,
            enable_for_algorithms: true,
            enable_for_memory_ops: true,
            enable_for_pattern_matching: true,
            warmup_iterations: 1000,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Memory allocator to use
    pub allocator: MemoryAllocator,
    
    /// Enable huge pages
    pub enable_huge_pages: bool,
    
    /// Huge page size
    pub huge_page_size: usize,
    
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: usize,
    
    /// Enable NUMA awareness
    pub enable_numa: bool,
    
    /// NUMA node to bind to (-1 = auto)
    pub numa_node: i32,
    
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    
    /// Prefetch distance
    pub prefetch_distance: usize,
    
    /// Enable memory alignment
    pub enable_alignment: bool,
    
    /// Memory alignment size
    pub alignment_size: usize,
    
    /// Buffer pool configuration
    pub buffer_pool: BufferPoolConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            allocator: MemoryAllocator::System,
            enable_huge_pages: true,
            huge_page_size: 2 * 1024 * 1024, // 2MB
            memory_limit: 0,
            enable_numa: true,
            numa_node: -1,
            enable_prefetching: true,
            prefetch_distance: 8,
            enable_alignment: true,
            alignment_size: 64, // Cache line size
            buffer_pool: BufferPoolConfig::default(),
        }
    }
}

/// Memory allocators
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryAllocator {
    /// System default allocator
    System,
    /// jemalloc
    Jemalloc,
    /// mimalloc
    Mimalloc,
    /// tcmalloc
    Tcmalloc,
}

/// Buffer pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferPoolConfig {
    /// Enable buffer pooling
    pub enabled: bool,
    
    /// Pool size in bytes
    pub pool_size: usize,
    
    /// Buffer sizes to pool
    pub buffer_sizes: Vec<usize>,
    
    /// Maximum buffers per size
    pub max_buffers_per_size: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_size: 512 * 1024 * 1024, // 512MB
            buffer_sizes: vec![64, 256, 1024, 4096, 16384, 65536],
            max_buffers_per_size: 1000,
        }
    }
}

/// Performance tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Thread pool configuration
    pub thread_pool: ThreadPoolConfig,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// I/O configuration
    pub io: IoConfig,
    
    /// Enable performance profiling
    pub enable_profiling: bool,
    
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    
    /// Monitoring interval
    pub monitoring_interval: Duration,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            thread_pool: ThreadPoolConfig::default(),
            cache: CacheConfig::default(),
            io: IoConfig::default(),
            enable_profiling: false,
            enable_monitoring: true,
            monitoring_interval: Duration::from_secs(60),
        }
    }
}

/// Thread pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    
    /// Stack size per thread
    pub stack_size: usize,
    
    /// Enable work stealing
    pub enable_work_stealing: bool,
    
    /// Pin threads to CPU cores
    pub pin_to_cores: bool,
    
    /// Thread priority
    pub thread_priority: ThreadPriority,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0,
            stack_size: 2 * 1024 * 1024, // 2MB
            enable_work_stealing: true,
            pin_to_cores: false,
            thread_priority: ThreadPriority::Normal,
        }
    }
}

/// Thread priorities
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThreadPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    Realtime,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 cache size (per thread)
    pub l1_cache_size: usize,
    
    /// L2 cache size (shared)
    pub l2_cache_size: usize,
    
    /// Cache line size
    pub cache_line_size: usize,
    
    /// Enable cache optimization
    pub enable_optimization: bool,
    
    /// Cache replacement policy
    pub replacement_policy: CacheReplacementPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_cache_size: 32 * 1024, // 32KB
            l2_cache_size: 512 * 1024, // 512KB
            cache_line_size: 64,
            enable_optimization: true,
            replacement_policy: CacheReplacementPolicy::Lru,
        }
    }
}

/// Cache replacement policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheReplacementPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// Random replacement
    Random,
    /// First In, First Out
    Fifo,
}

/// I/O configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoConfig {
    /// Enable asynchronous I/O
    pub enable_async: bool,
    
    /// I/O buffer size
    pub buffer_size: usize,
    
    /// Number of I/O threads
    pub io_threads: usize,
    
    /// Enable direct I/O
    pub enable_direct_io: bool,
    
    /// Read-ahead size
    pub read_ahead_size: usize,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            enable_async: true,
            buffer_size: 64 * 1024, // 64KB
            io_threads: 4,
            enable_direct_io: false,
            read_ahead_size: 256 * 1024, // 256KB
        }
    }
}

/// Network configuration for distributed mode
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Bind address
    pub bind_address: String,
    
    /// Port number
    pub port: u16,
    
    /// Enable TLS
    pub enable_tls: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<PathBuf>,
    
    /// TLS private key path
    pub tls_key_path: Option<PathBuf>,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Buffer sizes
    pub send_buffer_size: usize,
    pub recv_buffer_size: usize,
}

#[cfg(feature = "distributed")]
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0".to_string(),
            port: 8080,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            connection_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(60),
            max_connections: 1000,
            enable_compression: true,
            send_buffer_size: 64 * 1024,
            recv_buffer_size: 64 * 1024,
        }
    }
}

/// GPU configuration
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    
    /// GPU device ID to use
    pub device_id: i32,
    
    /// Memory limit on GPU
    pub memory_limit: usize,
    
    /// Enable unified memory
    pub enable_unified_memory: bool,
    
    /// CUDA stream count
    pub stream_count: usize,
    
    /// Enable GPU-CPU memory transfers
    pub enable_memory_transfers: bool,
    
    /// Preferred GPU operations
    pub preferred_operations: Vec<GpuOperation>,
}

#[cfg(feature = "gpu")]
impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
            memory_limit: 8 * 1024 * 1024 * 1024, // 8GB
            enable_unified_memory: true,
            stream_count: 4,
            enable_memory_transfers: true,
            preferred_operations: vec![
                GpuOperation::MatrixMultiplication,
                GpuOperation::PageRank,
                GpuOperation::ShortestPath,
            ],
        }
    }
}

/// GPU operations that can be accelerated
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GpuOperation {
    /// Matrix multiplication
    MatrixMultiplication,
    /// PageRank algorithm
    PageRank,
    /// Shortest path algorithms
    ShortestPath,
    /// Graph traversal
    Traversal,
    /// Pattern matching
    PatternMatching,
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Metrics collection interval
    pub collection_interval: Duration,
    
    /// Enable detailed profiling
    pub enable_profiling: bool,
    
    /// Enable performance alerts
    pub enable_alerts: bool,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Metrics export configuration
    pub export: MetricsExportConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            collection_interval: Duration::from_secs(60),
            enable_profiling: false,
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            export: MetricsExportConfig::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold (percentage)
    pub memory_usage_percent: f32,
    
    /// CPU usage threshold (percentage)
    pub cpu_usage_percent: f32,
    
    /// Average query time threshold (milliseconds)
    pub query_time_ms: u64,
    
    /// Cache hit ratio threshold
    pub cache_hit_ratio: f32,
    
    /// Error rate threshold (errors per minute)
    pub error_rate_per_minute: u64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_usage_percent: 85.0,
            cpu_usage_percent: 90.0,
            query_time_ms: 1000,
            cache_hit_ratio: 0.7,
            error_rate_per_minute: 10,
        }
    }
}

/// Metrics export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportConfig {
    /// Enable Prometheus export
    pub enable_prometheus: bool,
    
    /// Prometheus endpoint port
    pub prometheus_port: u16,
    
    /// Enable JSON export
    pub enable_json: bool,
    
    /// JSON export file path
    pub json_export_path: Option<PathBuf>,
    
    /// Export interval
    pub export_interval: Duration,
}

impl Default for MetricsExportConfig {
    fn default() -> Self {
        Self {
            enable_prometheus: false,
            prometheus_port: 9090,
            enable_json: false,
            json_export_path: None,
            export_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable authentication
    pub enable_authentication: bool,
    
    /// Enable authorization
    pub enable_authorization: bool,
    
    /// Enable encryption at rest
    pub enable_encryption_at_rest: bool,
    
    /// Enable encryption in transit
    pub enable_encryption_in_transit: bool,
    
    /// Encryption key path
    pub encryption_key_path: Option<PathBuf>,
    
    /// Enable audit logging
    pub enable_audit_logging: bool,
    
    /// Audit log path
    pub audit_log_path: Option<PathBuf>,
    
    /// Maximum login attempts
    pub max_login_attempts: u32,
    
    /// Login timeout
    pub login_timeout: Duration,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_authentication: false,
            enable_authorization: false,
            enable_encryption_at_rest: false,
            enable_encryption_in_transit: false,
            encryption_key_path: None,
            enable_audit_logging: false,
            audit_log_path: None,
            max_login_attempts: 3,
            login_timeout: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Enable console logging
    pub enable_console: bool,
    
    /// Enable file logging
    pub enable_file: bool,
    
    /// Log file path
    pub file_path: Option<PathBuf>,
    
    /// Maximum log file size
    pub max_file_size: usize,
    
    /// Number of log files to keep
    pub max_files: usize,
    
    /// Enable structured logging
    pub enable_structured: bool,
    
    /// Log format
    pub format: LogFormat,
    
    /// Enable performance logging
    pub enable_performance: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            enable_console: true,
            enable_file: false,
            file_path: None,
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 10,
            enable_structured: true,
            format: LogFormat::Json,
            enable_performance: false,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogLevel {
    /// Trace level
    Trace,
    /// Debug level
    Debug,
    /// Info level
    Info,
    /// Warning level
    Warn,
    /// Error level
    Error,
}

/// Log formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LogFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Structured format
    Structured,
}

impl HybridConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> HybridResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| HybridError::configuration(format!("Failed to read config file: {}", e)))?;
        
        let config: HybridConfig = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&content)
                .map_err(|e| HybridError::configuration(format!("Failed to parse TOML config: {}", e)))?,
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| HybridError::configuration(format!("Failed to parse JSON config: {}", e)))?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .map_err(|e| HybridError::configuration(format!("Failed to parse YAML config: {}", e)))?,
            _ => return Err(HybridError::configuration("Unsupported config file format")),
        };
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file<P: AsRef<std::path::Path>>(&self, path: P) -> HybridResult<()> {
        self.validate()?;
        
        let content = match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::to_string_pretty(self)
                .map_err(|e| HybridError::configuration(format!("Failed to serialize TOML config: {}", e)))?,
            Some("json") => serde_json::to_string_pretty(self)
                .map_err(|e| HybridError::configuration(format!("Failed to serialize JSON config: {}", e)))?,
            Some("yaml") | Some("yml") => serde_yaml::to_string(self)
                .map_err(|e| HybridError::configuration(format!("Failed to serialize YAML config: {}", e)))?,
            _ => return Err(HybridError::configuration("Unsupported config file format")),
        };
        
        std::fs::write(path, content)
            .map_err(|e| HybridError::configuration(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> HybridResult<()> {
        // Validate graph configuration
        if self.graph.initial_node_capacity == 0 {
            return Err(HybridError::configuration("Initial node capacity must be greater than 0"));
        }
        
        if self.graph.initial_edge_capacity == 0 {
            return Err(HybridError::configuration("Initial edge capacity must be greater than 0"));
        }
        
        // Validate memory configuration
        if self.memory.alignment_size == 0 || !self.memory.alignment_size.is_power_of_two() {
            return Err(HybridError::configuration("Memory alignment size must be a power of 2"));
        }
        
        // Validate SIMD configuration
        if self.simd.force_width != 0 && !matches!(self.simd.force_width, 4 | 8 | 16) {
            return Err(HybridError::configuration("SIMD width must be 0 (auto), 4, 8, or 16"));
        }
        
        // Validate performance configuration
        if self.performance.thread_pool.stack_size < 64 * 1024 {
            return Err(HybridError::configuration("Thread stack size must be at least 64KB"));
        }
        
        tracing::debug!("Configuration validation completed successfully");
        Ok(())
    }
    
    /// Merge with environment variables
    pub fn merge_with_env(&mut self) -> HybridResult<()> {
        // Override config values with environment variables
        if let Ok(val) = std::env::var("HYBRID_KG_WORKER_THREADS") {
            self.performance.thread_pool.worker_threads = val.parse()
                .map_err(|_| HybridError::configuration("Invalid HYBRID_KG_WORKER_THREADS"))?;
        }
        
        if let Ok(val) = std::env::var("HYBRID_KG_MEMORY_LIMIT") {
            self.memory.memory_limit = val.parse()
                .map_err(|_| HybridError::configuration("Invalid HYBRID_KG_MEMORY_LIMIT"))?;
        }
        
        if let Ok(val) = std::env::var("HYBRID_KG_ENABLE_SIMD") {
            self.simd.enabled = val.parse()
                .map_err(|_| HybridError::configuration("Invalid HYBRID_KG_ENABLE_SIMD"))?;
        }
        
        if let Ok(val) = std::env::var("HYBRID_KG_LOG_LEVEL") {
            self.logging.level = match val.to_lowercase().as_str() {
                "trace" => LogLevel::Trace,
                "debug" => LogLevel::Debug,
                "info" => LogLevel::Info,
                "warn" => LogLevel::Warn,
                "error" => LogLevel::Error,
                _ => return Err(HybridError::configuration("Invalid HYBRID_KG_LOG_LEVEL")),
            };
        }
        
        Ok(())
    }
    
    /// Get optimized configuration for the current hardware
    pub fn optimize_for_hardware(&mut self) -> HybridResult<()> {
        // Auto-detect CPU features
        if self.simd.force_width == 0 {
            if crate::cpu_features::HAS_AVX512 {
                self.simd.force_width = 16;
            } else if crate::cpu_features::HAS_AVX2 {
                self.simd.force_width = 8;
            } else {
                self.simd.force_width = 4;
            }
        }
        
        // Auto-detect thread count
        if self.performance.thread_pool.worker_threads == 0 {
            self.performance.thread_pool.worker_threads = 
                std::thread::available_parallelism()?.get();
        }
        
        // Auto-detect memory configuration
        if self.memory.memory_limit == 0 {
            // Use 80% of available system memory
            if let Ok(mem_info) = sys_info::mem_info() {
                self.memory.memory_limit = (mem_info.total as usize * 1024 * 80) / 100;
            }
        }
        
        tracing::info!("Configuration optimized for current hardware");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = HybridConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = HybridConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: HybridConfig = serde_json::from_str(&json).unwrap();
        assert!(deserialized.validate().is_ok());
    }
    
    #[test]
    fn test_config_file_operations() {
        let config = HybridConfig::default();
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("test.json");
        
        // Save and load config
        config.to_file(&config_path).unwrap();
        let loaded_config = HybridConfig::from_file(&config_path).unwrap();
        
        assert!(loaded_config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = HybridConfig::default();
        
        // Test invalid configuration
        config.graph.initial_node_capacity = 0;
        assert!(config.validate().is_err());
        
        // Fix and test valid configuration
        config.graph.initial_node_capacity = 1000;
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_hardware_optimization() {
        let mut config = HybridConfig::default();
        config.performance.thread_pool.worker_threads = 0;
        config.simd.force_width = 0;
        
        config.optimize_for_hardware().unwrap();
        
        assert!(config.performance.thread_pool.worker_threads > 0);
        assert!(config.simd.force_width > 0);
    }
}