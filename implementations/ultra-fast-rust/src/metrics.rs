//! Performance metrics collection and monitoring
//!
//! This module implements:
//! - Real-time performance monitoring
//! - SIMD operation timing
//! - Memory usage tracking
//! - Throughput measurement
//! - Latency profiling

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use dashmap::DashMap;
use serde::{Serialize, Deserialize};

/// Comprehensive metrics collector for ultra-fast knowledge graph
#[derive(Debug)]
pub struct MetricsCollector {
    /// Operation counters
    operations: DashMap<String, AtomicU64>,
    
    /// Timing measurements
    timings: DashMap<String, TimingMetrics>,
    
    /// Memory usage tracking
    memory_stats: MemoryStats,
    
    /// Throughput measurements
    throughput: ThroughputMetrics,
    
    /// SIMD operation performance
    simd_metrics: SimdMetrics,
    
    /// Start time for uptime calculation
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            operations: DashMap::new(),
            timings: DashMap::new(),
            memory_stats: MemoryStats::new(),
            throughput: ThroughputMetrics::new(),
            simd_metrics: SimdMetrics::new(),
            start_time: Instant::now(),
        }
    }

    /// Record a single operation
    #[inline]
    pub fn record_operation(&self, operation: &str) {
        self.operations
            .entry(operation.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch operation
    #[inline]
    pub fn record_batch_operation(&self, operation: &str, count: usize) {
        self.operations
            .entry(operation.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(count as u64, Ordering::Relaxed);
            
        // Update throughput metrics
        self.throughput.record_batch(count, Instant::now());
    }

    /// Record operation duration
    #[inline]
    pub fn record_operation_duration(&self, operation: &str, duration: Duration) {
        self.timings
            .entry(operation.to_string())
            .or_insert_with(|| TimingMetrics::new())
            .record(duration);
    }

    /// Record traversal performance
    pub fn record_traversal(&self, algorithm: &str, duration: Duration, nodes_visited: usize) {
        let operation = format!("traversal_{}", algorithm);
        self.record_operation_duration(&operation, duration);
        
        // Calculate traversal rate (nodes per second)
        let rate = nodes_visited as f64 / duration.as_secs_f64();
        self.throughput.record_traversal_rate(rate);
    }

    /// Record pattern search performance
    pub fn record_pattern_search(&self, duration: Duration, matches_found: usize) {
        self.record_operation_duration("pattern_search", duration);
        
        // Record pattern matching efficiency
        let efficiency = matches_found as f64 / duration.as_secs_f64();
        self.throughput.record_pattern_efficiency(efficiency);
    }

    /// Record SIMD operation performance
    pub fn record_simd_operation(&self, operation: &str, elements_processed: usize, duration: Duration) {
        self.simd_metrics.record_operation(operation, elements_processed, duration);
    }

    /// Update memory usage statistics
    pub fn update_memory_usage(&self, category: &str, bytes: usize) {
        self.memory_stats.update(category, bytes);
    }

    /// Get comprehensive metrics summary
    pub fn get_summary(&self) -> MetricsSummary {
        let operations: HashMap<String, u64> = self.operations
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        let timings: HashMap<String, TimingSummary> = self.timings
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_summary()))
            .collect();

        MetricsSummary {
            uptime: self.start_time.elapsed(),
            operations,
            timings,
            memory: self.memory_stats.get_summary(),
            throughput: self.throughput.get_summary(),
            simd: self.simd_metrics.get_summary(),
        }
    }

    /// Get real-time performance statistics
    pub fn get_realtime_stats(&self) -> RealtimeStats {
        RealtimeStats {
            current_memory_mb: self.memory_stats.get_current_usage_mb(),
            operations_per_second: self.throughput.get_current_ops_per_second(),
            average_latency_us: self.get_average_latency_microseconds(),
            simd_efficiency: self.simd_metrics.get_efficiency_percent(),
        }
    }

    /// Calculate average latency across all operations
    fn get_average_latency_microseconds(&self) -> f64 {
        let mut total_time = Duration::default();
        let mut total_operations = 0u64;

        for entry in self.timings.iter() {
            let timing = entry.value();
            total_time += timing.get_total_time();
            total_operations += timing.get_count();
        }

        if total_operations > 0 {
            total_time.as_micros() as f64 / total_operations as f64
        } else {
            0.0
        }
    }
}

/// Timing metrics for individual operations
#[derive(Debug)]
pub struct TimingMetrics {
    count: AtomicU64,
    total_time: AtomicU64, // in nanoseconds
    min_time: AtomicU64,   // in nanoseconds
    max_time: AtomicU64,   // in nanoseconds
    recent_times: parking_lot::RwLock<Vec<u64>>, // Ring buffer for recent times
}

impl TimingMetrics {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_time: AtomicU64::new(0),
            min_time: AtomicU64::new(u64::MAX),
            max_time: AtomicU64::new(0),
            recent_times: parking_lot::RwLock::new(Vec::with_capacity(1000)),
        }
    }

    fn record(&self, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_time.fetch_add(nanos, Ordering::Relaxed);
        
        // Update min/max with atomic compare-and-swap
        let mut current_min = self.min_time.load(Ordering::Relaxed);
        while nanos < current_min {
            match self.min_time.compare_exchange_weak(
                current_min, nanos, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
        
        let mut current_max = self.max_time.load(Ordering::Relaxed);
        while nanos > current_max {
            match self.max_time.compare_exchange_weak(
                current_max, nanos, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
        
        // Add to recent times (with circular buffer)
        let mut recent = self.recent_times.write();
        if recent.len() >= 1000 {
            recent.remove(0);
        }
        recent.push(nanos);
    }

    fn get_summary(&self) -> TimingSummary {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_time.load(Ordering::Relaxed);
        let min = self.min_time.load(Ordering::Relaxed);
        let max = self.max_time.load(Ordering::Relaxed);
        
        let average = if count > 0 { total / count } else { 0 };
        
        // Calculate percentiles from recent times
        let recent = self.recent_times.read();
        let mut sorted_times = recent.clone();
        sorted_times.sort_unstable();
        
        let p50 = percentile(&sorted_times, 0.5);
        let p95 = percentile(&sorted_times, 0.95);
        let p99 = percentile(&sorted_times, 0.99);

        TimingSummary {
            count,
            average_ns: average,
            min_ns: if min == u64::MAX { 0 } else { min },
            max_ns: max,
            p50_ns: p50,
            p95_ns: p95,
            p99_ns: p99,
        }
    }

    fn get_total_time(&self) -> Duration {
        Duration::from_nanos(self.total_time.load(Ordering::Relaxed))
    }

    fn get_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

/// Memory usage statistics
#[derive(Debug)]
pub struct MemoryStats {
    categories: DashMap<String, AtomicUsize>,
    peak_usage: AtomicUsize,
    last_update: parking_lot::RwLock<Instant>,
}

impl MemoryStats {
    fn new() -> Self {
        Self {
            categories: DashMap::new(),
            peak_usage: AtomicUsize::new(0),
            last_update: parking_lot::RwLock::new(Instant::now()),
        }
    }

    fn update(&self, category: &str, bytes: usize) {
        self.categories
            .entry(category.to_string())
            .or_insert_with(|| AtomicUsize::new(0))
            .store(bytes, Ordering::Relaxed);
        
        // Update peak usage
        let total_usage = self.get_total_usage();
        let mut current_peak = self.peak_usage.load(Ordering::Relaxed);
        while total_usage > current_peak {
            match self.peak_usage.compare_exchange_weak(
                current_peak, total_usage, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
        
        *self.last_update.write() = Instant::now();
    }

    fn get_total_usage(&self) -> usize {
        self.categories
            .iter()
            .map(|entry| entry.value().load(Ordering::Relaxed))
            .sum()
    }

    fn get_current_usage_mb(&self) -> f64 {
        self.get_total_usage() as f64 / (1024.0 * 1024.0)
    }

    fn get_summary(&self) -> MemorySummary {
        let categories: HashMap<String, usize> = self.categories
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load(Ordering::Relaxed)))
            .collect();

        MemorySummary {
            total_bytes: self.get_total_usage(),
            peak_bytes: self.peak_usage.load(Ordering::Relaxed),
            categories,
            last_update: self.last_update.read().elapsed(),
        }
    }
}

/// Throughput metrics tracking
#[derive(Debug)]
pub struct ThroughputMetrics {
    operations_window: parking_lot::RwLock<Vec<(Instant, usize)>>,
    traversal_rates: parking_lot::RwLock<Vec<f64>>,
    pattern_efficiencies: parking_lot::RwLock<Vec<f64>>,
}

impl ThroughputMetrics {
    fn new() -> Self {
        Self {
            operations_window: parking_lot::RwLock::new(Vec::new()),
            traversal_rates: parking_lot::RwLock::new(Vec::new()),
            pattern_efficiencies: parking_lot::RwLock::new(Vec::new()),
        }
    }

    fn record_batch(&self, count: usize, timestamp: Instant) {
        let mut window = self.operations_window.write();
        window.push((timestamp, count));
        
        // Keep only last 60 seconds of data
        let cutoff = timestamp - Duration::from_secs(60);
        window.retain(|(ts, _)| *ts > cutoff);
    }

    fn record_traversal_rate(&self, rate: f64) {
        let mut rates = self.traversal_rates.write();
        rates.push(rate);
        
        // Keep only last 1000 measurements
        if rates.len() > 1000 {
            rates.remove(0);
        }
    }

    fn record_pattern_efficiency(&self, efficiency: f64) {
        let mut efficiencies = self.pattern_efficiencies.write();
        efficiencies.push(efficiency);
        
        // Keep only last 1000 measurements
        if efficiencies.len() > 1000 {
            efficiencies.remove(0);
        }
    }

    fn get_current_ops_per_second(&self) -> f64 {
        let window = self.operations_window.read();
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(1);
        
        let recent_ops: usize = window
            .iter()
            .filter(|(ts, _)| *ts > cutoff)
            .map(|(_, count)| count)
            .sum();
        
        recent_ops as f64
    }

    fn get_summary(&self) -> ThroughputSummary {
        let traversal_rates = self.traversal_rates.read();
        let pattern_efficiencies = self.pattern_efficiencies.read();
        
        ThroughputSummary {
            current_ops_per_second: self.get_current_ops_per_second(),
            average_traversal_rate: traversal_rates.iter().sum::<f64>() / traversal_rates.len().max(1) as f64,
            average_pattern_efficiency: pattern_efficiencies.iter().sum::<f64>() / pattern_efficiencies.len().max(1) as f64,
        }
    }
}

/// SIMD operation performance metrics
#[derive(Debug)]
pub struct SimdMetrics {
    operations: DashMap<String, SimdOperationStats>,
}

impl SimdMetrics {
    fn new() -> Self {
        Self {
            operations: DashMap::new(),
        }
    }

    fn record_operation(&self, operation: &str, elements: usize, duration: Duration) {
        self.operations
            .entry(operation.to_string())
            .or_insert_with(|| SimdOperationStats::new())
            .record(elements, duration);
    }

    fn get_efficiency_percent(&self) -> f64 {
        let mut total_efficiency = 0.0;
        let mut count = 0;

        for entry in self.operations.iter() {
            total_efficiency += entry.value().get_efficiency();
            count += 1;
        }

        if count > 0 {
            (total_efficiency / count as f64) * 100.0
        } else {
            0.0
        }
    }

    fn get_summary(&self) -> SimdSummary {
        let operations: HashMap<String, SimdOperationSummary> = self.operations
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_summary()))
            .collect();

        SimdSummary {
            operations,
            overall_efficiency_percent: self.get_efficiency_percent(),
        }
    }
}

/// Statistics for individual SIMD operations
#[derive(Debug)]
pub struct SimdOperationStats {
    total_elements: AtomicU64,
    total_time: AtomicU64, // in nanoseconds
    operation_count: AtomicU64,
}

impl SimdOperationStats {
    fn new() -> Self {
        Self {
            total_elements: AtomicU64::new(0),
            total_time: AtomicU64::new(0),
            operation_count: AtomicU64::new(0),
        }
    }

    fn record(&self, elements: usize, duration: Duration) {
        self.total_elements.fetch_add(elements as u64, Ordering::Relaxed);
        self.total_time.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }

    fn get_efficiency(&self) -> f64 {
        let elements = self.total_elements.load(Ordering::Relaxed);
        let time_ns = self.total_time.load(Ordering::Relaxed);
        
        if time_ns > 0 {
            // Calculate elements per nanosecond, normalized
            (elements as f64) / (time_ns as f64) * 1_000_000.0
        } else {
            0.0
        }
    }

    fn get_summary(&self) -> SimdOperationSummary {
        let elements = self.total_elements.load(Ordering::Relaxed);
        let time_ns = self.total_time.load(Ordering::Relaxed);
        let count = self.operation_count.load(Ordering::Relaxed);

        SimdOperationSummary {
            total_elements: elements,
            total_operations: count,
            average_elements_per_op: if count > 0 { elements / count } else { 0 },
            efficiency_score: self.get_efficiency(),
        }
    }
}

/// Helper function to calculate percentiles
fn percentile(sorted_data: &[u64], p: f64) -> u64 {
    if sorted_data.is_empty() {
        return 0;
    }

    let index = (p * (sorted_data.len() - 1) as f64) as usize;
    sorted_data.get(index).copied().unwrap_or(0)
}

/// Summary types for metrics reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub uptime: Duration,
    pub operations: HashMap<String, u64>,
    pub timings: HashMap<String, TimingSummary>,
    pub memory: MemorySummary,
    pub throughput: ThroughputSummary,
    pub simd: SimdSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingSummary {
    pub count: u64,
    pub average_ns: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    pub total_bytes: usize,
    pub peak_bytes: usize,
    pub categories: HashMap<String, usize>,
    pub last_update: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputSummary {
    pub current_ops_per_second: f64,
    pub average_traversal_rate: f64,
    pub average_pattern_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdSummary {
    pub operations: HashMap<String, SimdOperationSummary>,
    pub overall_efficiency_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdOperationSummary {
    pub total_elements: u64,
    pub total_operations: u64,
    pub average_elements_per_op: u64,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeStats {
    pub current_memory_mb: f64,
    pub operations_per_second: f64,
    pub average_latency_us: f64,
    pub simd_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        collector.record_operation("test_op");
        collector.record_operation("test_op");
        collector.record_batch_operation("batch_op", 100);
        
        let summary = collector.get_summary();
        
        assert_eq!(summary.operations.get("test_op"), Some(&2));
        assert_eq!(summary.operations.get("batch_op"), Some(&100));
    }

    #[test]
    fn test_timing_metrics() {
        let timing = TimingMetrics::new();
        
        timing.record(Duration::from_millis(10));
        timing.record(Duration::from_millis(20));
        timing.record(Duration::from_millis(15));
        
        let summary = timing.get_summary();
        
        assert_eq!(summary.count, 3);
        assert!(summary.average_ns > 0);
        assert!(summary.min_ns <= summary.max_ns);
    }

    #[test]
    fn test_memory_stats() {
        let memory = MemoryStats::new();
        
        memory.update("nodes", 1024);
        memory.update("edges", 2048);
        
        let summary = memory.get_summary();
        
        assert_eq!(summary.total_bytes, 3072);
        assert_eq!(summary.categories.get("nodes"), Some(&1024));
        assert_eq!(summary.categories.get("edges"), Some(&2048));
    }
}