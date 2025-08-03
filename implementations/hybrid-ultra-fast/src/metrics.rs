//! Performance metrics and monitoring for hybrid knowledge graph
//!
//! This module provides comprehensive performance monitoring, profiling,
//! and metrics collection across all hybrid components.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{HybridError, HybridResult};

/// Comprehensive performance metrics for the hybrid knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total number of nodes
    pub node_count: u64,
    
    /// Total number of edges
    pub edge_count: u64,
    
    /// Total number of hyperedges
    pub hyperedge_count: u64,
    
    /// Total memory usage in bytes
    pub memory_usage: usize,
    
    /// Total operations performed
    pub operations_performed: u64,
    
    /// Total queries executed
    pub queries_executed: u64,
    
    /// Average query execution time
    pub average_query_time: Duration,
    
    /// Cache hit ratio (0.0 to 1.0)
    pub cache_hit_ratio: f32,
    
    /// Total SIMD operations performed
    pub simd_operations: u64,
    
    /// System uptime
    pub uptime: Duration,
    
    /// SIMD vector width being used
    pub simd_width: usize,
    
    /// Whether AVX-512 is available
    pub has_avx512: bool,
    
    /// Number of worker threads
    pub thread_count: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            hyperedge_count: 0,
            memory_usage: 0,
            operations_performed: 0,
            queries_executed: 0,
            average_query_time: Duration::ZERO,
            cache_hit_ratio: 0.0,
            simd_operations: 0,
            uptime: Duration::ZERO,
            simd_width: crate::cpu_features::SIMD_WIDTH,
            has_avx512: crate::cpu_features::HAS_AVX512,
            thread_count: rayon::current_num_threads(),
        }
    }
}

/// Detailed memory usage breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total allocated memory
    pub total: usize,
    
    /// Memory used by nodes
    pub nodes: usize,
    
    /// Memory used by edges
    pub edges: usize,
    
    /// Memory used by hyperedges
    pub hyperedges: usize,
    
    /// Memory used by CSR matrices
    pub csr_matrices: usize,
    
    /// Memory used by indexes
    pub indexes: usize,
    
    /// Memory used by caches
    pub caches: usize,
    
    /// Memory used by thread pools
    pub thread_pools: usize,
    
    /// Memory used by other components
    pub other: usize,
    
    /// Memory overhead percentage
    pub overhead_percentage: f32,
    
    /// Compression ratio achieved
    pub compression_ratio: f32,
}

impl MemoryUsage {
    /// Calculate overhead percentage
    pub fn calculate_overhead(&mut self) {
        let active_memory = self.nodes + self.edges + self.hyperedges;
        if active_memory > 0 {
            self.overhead_percentage = 
                ((self.total - active_memory) as f32 / active_memory as f32) * 100.0;
        }
    }
}

/// Operation-specific timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTiming {
    /// Operation name
    pub name: String,
    
    /// Number of times executed
    pub count: u64,
    
    /// Total time spent
    pub total_time: Duration,
    
    /// Average execution time
    pub average_time: Duration,
    
    /// Minimum execution time
    pub min_time: Duration,
    
    /// Maximum execution time
    pub max_time: Duration,
    
    /// 95th percentile execution time
    pub p95_time: Duration,
    
    /// 99th percentile execution time
    pub p99_time: Duration,
    
    /// Last execution time
    pub last_time: Duration,
}

impl OperationTiming {
    /// Create new operation timing
    pub fn new(name: String) -> Self {
        Self {
            name,
            count: 0,
            total_time: Duration::ZERO,
            average_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            p95_time: Duration::ZERO,
            p99_time: Duration::ZERO,
            last_time: Duration::ZERO,
        }
    }
    
    /// Record a new timing measurement
    pub fn record(&mut self, duration: Duration) {
        self.count += 1;
        self.total_time += duration;
        self.average_time = self.total_time / self.count as u32;
        self.last_time = duration;
        
        if duration < self.min_time {
            self.min_time = duration;
        }
        
        if duration > self.max_time {
            self.max_time = duration;
        }
        
        // TODO: Implement proper percentile calculation with histograms
        // For now, use approximations
        self.p95_time = self.max_time.mul_f64(0.95);
        self.p99_time = self.max_time.mul_f64(0.99);
    }
}

/// SIMD performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIMDMetrics {
    /// Total SIMD operations performed
    pub total_operations: u64,
    
    /// Operations performed with different vector widths
    pub operations_by_width: HashMap<usize, u64>,
    
    /// SIMD efficiency (0.0 to 1.0)
    pub efficiency: f32,
    
    /// Average SIMD utilization percentage
    pub average_utilization: f32,
    
    /// Number of fallbacks to scalar operations
    pub scalar_fallbacks: u64,
    
    /// Vectorization ratio
    pub vectorization_ratio: f32,
}

impl Default for SIMDMetrics {
    fn default() -> Self {
        let mut operations_by_width = HashMap::new();
        operations_by_width.insert(4, 0);   // SSE
        operations_by_width.insert(8, 0);   // AVX2
        operations_by_width.insert(16, 0);  // AVX-512
        
        Self {
            total_operations: 0,
            operations_by_width,
            efficiency: 0.0,
            average_utilization: 0.0,
            scalar_fallbacks: 0,
            vectorization_ratio: 0.0,
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Total cache accesses
    pub total_accesses: u64,
    
    /// Cache hits
    pub hits: u64,
    
    /// Cache misses
    pub misses: u64,
    
    /// Hit ratio (0.0 to 1.0)
    pub hit_ratio: f32,
    
    /// Cache evictions
    pub evictions: u64,
    
    /// Cache size in bytes
    pub size_bytes: usize,
    
    /// Maximum cache size
    pub max_size_bytes: usize,
    
    /// Average access time
    pub average_access_time: Duration,
}

impl CacheMetrics {
    /// Update hit ratio
    pub fn update_hit_ratio(&mut self) {
        if self.total_accesses > 0 {
            self.hit_ratio = self.hits as f32 / self.total_accesses as f32;
        }
    }
}

/// Global metrics collector
static GLOBAL_METRICS: once_cell::sync::Lazy<Arc<RwLock<GlobalMetrics>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(GlobalMetrics::new())));

/// Global metrics aggregator
#[derive(Debug)]
pub struct GlobalMetrics {
    /// Start time for uptime calculation
    start_time: SystemTime,
    
    /// Operation timings by name
    operation_timings: HashMap<String, OperationTiming>,
    
    /// SIMD performance metrics
    simd_metrics: SIMDMetrics,
    
    /// Cache performance metrics
    cache_metrics: HashMap<String, CacheMetrics>,
    
    /// Memory usage over time
    memory_timeline: Vec<(SystemTime, usize)>,
    
    /// Query throughput over time
    query_throughput: Vec<(SystemTime, u64)>,
    
    /// Last metrics snapshot
    last_snapshot: Option<PerformanceMetrics>,
}

impl GlobalMetrics {
    /// Create new global metrics
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            operation_timings: HashMap::new(),
            simd_metrics: SIMDMetrics::default(),
            cache_metrics: HashMap::new(),
            memory_timeline: Vec::new(),
            query_throughput: Vec::new(),
            last_snapshot: None,
        }
    }
    
    /// Record operation timing
    pub fn record_operation(&mut self, name: String, duration: Duration) {
        self.operation_timings
            .entry(name.clone())
            .or_insert_with(|| OperationTiming::new(name))
            .record(duration);
    }
    
    /// Record SIMD operation
    pub fn record_simd_operation(&mut self, width: usize, efficiency: f32) {
        self.simd_metrics.total_operations += 1;
        *self.simd_metrics.operations_by_width.entry(width).or_insert(0) += 1;
        
        // Update efficiency with exponential moving average
        let alpha = 0.1;
        self.simd_metrics.efficiency = 
            self.simd_metrics.efficiency * (1.0 - alpha) + efficiency * alpha;
        
        // Update vectorization ratio
        let total_ops = self.simd_metrics.total_operations + self.simd_metrics.scalar_fallbacks;
        if total_ops > 0 {
            self.simd_metrics.vectorization_ratio = 
                self.simd_metrics.total_operations as f32 / total_ops as f32;
        }
    }
    
    /// Record scalar fallback
    pub fn record_scalar_fallback(&mut self) {
        self.simd_metrics.scalar_fallbacks += 1;
        
        // Update vectorization ratio
        let total_ops = self.simd_metrics.total_operations + self.simd_metrics.scalar_fallbacks;
        if total_ops > 0 {
            self.simd_metrics.vectorization_ratio = 
                self.simd_metrics.total_operations as f32 / total_ops as f32;
        }
    }
    
    /// Record cache access
    pub fn record_cache_access(&mut self, cache_name: String, hit: bool, access_time: Duration) {
        let cache_metrics = self.cache_metrics
            .entry(cache_name)
            .or_insert_with(|| CacheMetrics {
                total_accesses: 0,
                hits: 0,
                misses: 0,
                hit_ratio: 0.0,
                evictions: 0,
                size_bytes: 0,
                max_size_bytes: 0,
                average_access_time: Duration::ZERO,
            });
        
        cache_metrics.total_accesses += 1;
        if hit {
            cache_metrics.hits += 1;
        } else {
            cache_metrics.misses += 1;
        }
        cache_metrics.update_hit_ratio();
        
        // Update average access time with exponential moving average
        let alpha = 0.1;
        cache_metrics.average_access_time = Duration::from_nanos(
            (cache_metrics.average_access_time.as_nanos() as f64 * (1.0 - alpha) +
             access_time.as_nanos() as f64 * alpha) as u64
        );
    }
    
    /// Record memory usage
    pub fn record_memory_usage(&mut self, usage: usize) {
        self.memory_timeline.push((SystemTime::now(), usage));
        
        // Keep only recent entries (last 1000)
        if self.memory_timeline.len() > 1000 {
            self.memory_timeline.drain(0..100);
        }
    }
    
    /// Record query throughput
    pub fn record_query_throughput(&mut self, queries_per_second: u64) {
        self.query_throughput.push((SystemTime::now(), queries_per_second));
        
        // Keep only recent entries (last 1000)
        if self.query_throughput.len() > 1000 {
            self.query_throughput.drain(0..100);
        }
    }
    
    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed().unwrap_or(Duration::ZERO)
    }
    
    /// Get operation timings
    pub fn get_operation_timings(&self) -> &HashMap<String, OperationTiming> {
        &self.operation_timings
    }
    
    /// Get SIMD metrics
    pub fn get_simd_metrics(&self) -> &SIMDMetrics {
        &self.simd_metrics
    }
    
    /// Get cache metrics
    pub fn get_cache_metrics(&self) -> &HashMap<String, CacheMetrics> {
        &self.cache_metrics
    }
    
    /// Get memory timeline
    pub fn get_memory_timeline(&self) -> &[(SystemTime, usize)] {
        &self.memory_timeline
    }
    
    /// Get query throughput timeline
    pub fn get_query_throughput(&self) -> &[(SystemTime, u64)] {
        &self.query_throughput
    }
}

/// Initialize global metrics system
pub fn init_metrics() -> HybridResult<()> {
    tracing::info!("📊 Initializing metrics system");
    
    // Initialize global metrics
    let _metrics = GLOBAL_METRICS.read();
    
    tracing::info!("✅ Metrics system initialized");
    Ok(())
}

/// Get global statistics
pub fn get_global_stats() -> GlobalStats {
    let metrics = GLOBAL_METRICS.read();
    
    GlobalStats {
        uptime: metrics.uptime(),
        total_operations: metrics.operation_timings.values()
            .map(|timing| timing.count)
            .sum(),
        simd_operations: metrics.simd_metrics.total_operations,
        scalar_fallbacks: metrics.simd_metrics.scalar_fallbacks,
        vectorization_ratio: metrics.simd_metrics.vectorization_ratio,
        cache_hit_ratio: metrics.cache_metrics.values()
            .map(|cache| cache.hit_ratio)
            .sum::<f32>() / metrics.cache_metrics.len().max(1) as f32,
        memory_peak: metrics.memory_timeline.iter()
            .map(|(_, usage)| *usage)
            .max()
            .unwrap_or(0),
        query_peak_throughput: metrics.query_throughput.iter()
            .map(|(_, throughput)| *throughput)
            .max()
            .unwrap_or(0),
    }
}

/// Global performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalStats {
    /// System uptime
    pub uptime: Duration,
    
    /// Total operations performed across all components
    pub total_operations: u64,
    
    /// Total SIMD operations
    pub simd_operations: u64,
    
    /// Total scalar fallbacks
    pub scalar_fallbacks: u64,
    
    /// Overall vectorization ratio
    pub vectorization_ratio: f32,
    
    /// Average cache hit ratio across all caches
    pub cache_hit_ratio: f32,
    
    /// Peak memory usage
    pub memory_peak: usize,
    
    /// Peak query throughput
    pub query_peak_throughput: u64,
}

/// Record operation timing globally
pub fn record_operation_timing(name: impl Into<String>, duration: Duration) {
    GLOBAL_METRICS.write().record_operation(name.into(), duration);
}

/// Record SIMD operation globally
pub fn record_simd_operation(width: usize, efficiency: f32) {
    GLOBAL_METRICS.write().record_simd_operation(width, efficiency);
}

/// Record scalar fallback globally
pub fn record_scalar_fallback() {
    GLOBAL_METRICS.write().record_scalar_fallback();
}

/// Record cache access globally
pub fn record_cache_access(cache_name: impl Into<String>, hit: bool, access_time: Duration) {
    GLOBAL_METRICS.write().record_cache_access(cache_name.into(), hit, access_time);
}

/// Record memory usage globally
pub fn record_memory_usage(usage: usize) {
    GLOBAL_METRICS.write().record_memory_usage(usage);
}

/// Record query throughput globally
pub fn record_query_throughput(queries_per_second: u64) {
    GLOBAL_METRICS.write().record_query_throughput(queries_per_second);
}

/// Timer for measuring operation durations
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    /// Start a new timer
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
        }
    }
    
    /// Finish timing and record result
    pub fn finish(self) -> Duration {
        let duration = self.start.elapsed();
        record_operation_timing(&self.name, duration);
        duration
    }
    
    /// Get elapsed time without finishing
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Macro for easy timing measurements
#[macro_export]
macro_rules! time_operation {
    ($name:expr, $code:block) => {{
        let timer = $crate::metrics::Timer::new($name);
        let result = $code;
        timer.finish();
        result
    }};
}

/// Performance profiler for detailed analysis
pub struct Profiler {
    enabled: bool,
    samples: Vec<ProfileSample>,
}

/// Individual profiling sample
#[derive(Debug, Clone)]
pub struct ProfileSample {
    /// Function or operation name
    pub name: String,
    
    /// Execution duration
    pub duration: Duration,
    
    /// Timestamp when sample was taken
    pub timestamp: SystemTime,
    
    /// Thread ID
    pub thread_id: std::thread::ThreadId,
    
    /// Memory usage at time of sample
    pub memory_usage: Option<usize>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Profiler {
    /// Create new profiler
    pub fn new() -> Self {
        Self {
            enabled: false,
            samples: Vec::new(),
        }
    }
    
    /// Enable profiling
    pub fn enable(&mut self) {
        self.enabled = true;
        tracing::info!("🔍 Profiler enabled");
    }
    
    /// Disable profiling
    pub fn disable(&mut self) {
        self.enabled = false;
        tracing::info!("🔍 Profiler disabled");
    }
    
    /// Record a profiling sample
    pub fn record_sample(&mut self, sample: ProfileSample) {
        if self.enabled {
            self.samples.push(sample);
            
            // Limit sample count to prevent memory growth
            if self.samples.len() > 10000 {
                self.samples.drain(0..1000);
            }
        }
    }
    
    /// Get all samples
    pub fn get_samples(&self) -> &[ProfileSample] {
        &self.samples
    }
    
    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }
    
    /// Generate profiling report
    pub fn generate_report(&self) -> ProfilingReport {
        let mut function_stats = HashMap::new();
        
        for sample in &self.samples {
            let stats = function_stats
                .entry(sample.name.clone())
                .or_insert_with(|| FunctionStats {
                    name: sample.name.clone(),
                    call_count: 0,
                    total_time: Duration::ZERO,
                    average_time: Duration::ZERO,
                    min_time: Duration::MAX,
                    max_time: Duration::ZERO,
                });
            
            stats.call_count += 1;
            stats.total_time += sample.duration;
            stats.average_time = stats.total_time / stats.call_count as u32;
            
            if sample.duration < stats.min_time {
                stats.min_time = sample.duration;
            }
            
            if sample.duration > stats.max_time {
                stats.max_time = sample.duration;
            }
        }
        
        ProfilingReport {
            total_samples: self.samples.len(),
            function_stats,
            start_time: self.samples.first().map(|s| s.timestamp),
            end_time: self.samples.last().map(|s| s.timestamp),
        }
    }
}

/// Function-level performance statistics
#[derive(Debug, Clone)]
pub struct FunctionStats {
    /// Function name
    pub name: String,
    
    /// Number of calls
    pub call_count: u64,
    
    /// Total execution time
    pub total_time: Duration,
    
    /// Average execution time
    pub average_time: Duration,
    
    /// Minimum execution time
    pub min_time: Duration,
    
    /// Maximum execution time
    pub max_time: Duration,
}

/// Comprehensive profiling report
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Total number of samples
    pub total_samples: usize,
    
    /// Per-function statistics
    pub function_stats: HashMap<String, FunctionStats>,
    
    /// Profiling start time
    pub start_time: Option<SystemTime>,
    
    /// Profiling end time
    pub end_time: Option<SystemTime>,
}

impl ProfilingReport {
    /// Get top functions by total time
    pub fn top_functions_by_total_time(&self, limit: usize) -> Vec<&FunctionStats> {
        let mut functions: Vec<_> = self.function_stats.values().collect();
        functions.sort_by(|a, b| b.total_time.cmp(&a.total_time));
        functions.into_iter().take(limit).collect()
    }
    
    /// Get top functions by average time
    pub fn top_functions_by_average_time(&self, limit: usize) -> Vec<&FunctionStats> {
        let mut functions: Vec<_> = self.function_stats.values().collect();
        functions.sort_by(|a, b| b.average_time.cmp(&a.average_time));
        functions.into_iter().take(limit).collect()
    }
    
    /// Get top functions by call count
    pub fn top_functions_by_call_count(&self, limit: usize) -> Vec<&FunctionStats> {
        let mut functions: Vec<_> = self.function_stats.values().collect();
        functions.sort_by(|a, b| b.call_count.cmp(&a.call_count));
        functions.into_iter().take(limit).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_operation_timing() {
        let mut timing = OperationTiming::new("test_op".to_string());
        
        timing.record(Duration::from_millis(100));
        timing.record(Duration::from_millis(200));
        timing.record(Duration::from_millis(150));
        
        assert_eq!(timing.count, 3);
        assert_eq!(timing.average_time, Duration::from_millis(150));
        assert_eq!(timing.min_time, Duration::from_millis(100));
        assert_eq!(timing.max_time, Duration::from_millis(200));
    }
    
    #[test]
    fn test_global_metrics() {
        record_operation_timing("test_operation", Duration::from_micros(500));
        record_simd_operation(16, 0.95);
        record_scalar_fallback();
        record_cache_access("test_cache", true, Duration::from_nanos(100));
        
        let stats = get_global_stats();
        assert!(stats.total_operations > 0);
        assert!(stats.simd_operations > 0);
        assert!(stats.scalar_fallbacks > 0);
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new("test_timer");
        thread::sleep(Duration::from_millis(10));
        let duration = timer.finish();
        
        assert!(duration >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_profiler() {
        let mut profiler = Profiler::new();
        profiler.enable();
        
        let sample = ProfileSample {
            name: "test_function".to_string(),
            duration: Duration::from_millis(50),
            timestamp: SystemTime::now(),
            thread_id: thread::current().id(),
            memory_usage: Some(1024 * 1024),
            metadata: HashMap::new(),
        };
        
        profiler.record_sample(sample);
        
        let report = profiler.generate_report();
        assert_eq!(report.total_samples, 1);
        assert!(report.function_stats.contains_key("test_function"));
    }
    
    #[test]
    fn test_simd_metrics() {
        let mut metrics = SIMDMetrics::default();
        
        // Simulate some SIMD operations
        for _ in 0..100 {
            metrics.total_operations += 1;
            *metrics.operations_by_width.get_mut(&16).unwrap() += 1;
        }
        
        for _ in 0..50 {
            metrics.total_operations += 1;
            *metrics.operations_by_width.get_mut(&8).unwrap() += 1;
        }
        
        // Add some scalar fallbacks
        metrics.scalar_fallbacks = 25;
        
        // Update vectorization ratio
        let total_ops = metrics.total_operations + metrics.scalar_fallbacks;
        metrics.vectorization_ratio = metrics.total_operations as f32 / total_ops as f32;
        
        assert_eq!(metrics.total_operations, 150);
        assert_eq!(metrics.scalar_fallbacks, 25);
        assert!((metrics.vectorization_ratio - (150.0 / 175.0)).abs() < 0.001);
    }
}