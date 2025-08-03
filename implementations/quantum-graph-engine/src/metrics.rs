//! Performance metrics and monitoring for the Quantum Graph Engine
//!
//! This module provides:
//! - Real-time performance monitoring
//! - Prometheus metrics integration
//! - Performance profiling and analysis
//! - Health checks and alerting

use crate::{Error, Result};
use prometheus::{
    Counter, Gauge, Histogram, IntCounter, IntGauge, 
    Registry, Opts, HistogramOpts, GaugeOpts, CounterOpts
};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Global metrics collector
pub static GLOBAL_METRICS: once_cell::sync::Lazy<Arc<Mutex<MetricsCollector>>> = 
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(MetricsCollector::new())));

/// Initialize global metrics system
pub fn init_global_metrics() -> Result<()> {
    let _metrics = GLOBAL_METRICS.lock().map_err(|_| Error::Internal("Failed to lock metrics".to_string()))?;
    tracing::info!("Global metrics system initialized");
    Ok(())
}

/// Comprehensive metrics collector
pub struct MetricsCollector {
    /// Prometheus registry
    registry: Registry,
    
    // Core operation metrics
    node_operations: IntCounter,
    edge_operations: IntCounter,
    query_operations: IntCounter,
    
    // Performance metrics
    operation_duration: Histogram,
    memory_usage: Gauge,
    cpu_usage: Gauge,
    
    // Graph statistics
    total_nodes: IntGauge,
    total_edges: IntGauge,
    graph_density: Gauge,
    
    // Query performance
    query_latency: Histogram,
    query_throughput: Counter,
    cache_hit_rate: Gauge,
    
    // SIMD performance
    simd_operations: IntCounter,
    simd_speedup: Gauge,
    
    // Distributed metrics
    network_latency: Histogram,
    shard_balance: Gauge,
    consensus_operations: IntCounter,
    
    // Error tracking
    error_count: IntCounter,
    
    // Custom metrics
    custom_metrics: HashMap<String, Box<dyn CustomMetric + Send>>,
    
    // Performance history
    performance_history: PerformanceHistory,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let registry = Registry::new();
        
        // Initialize core metrics
        let node_operations = IntCounter::with_opts(
            CounterOpts::new("quantum_node_operations_total", "Total node operations")
        ).unwrap();
        registry.register(Box::new(node_operations.clone())).unwrap();
        
        let edge_operations = IntCounter::with_opts(
            CounterOpts::new("quantum_edge_operations_total", "Total edge operations")
        ).unwrap();
        registry.register(Box::new(edge_operations.clone())).unwrap();
        
        let query_operations = IntCounter::with_opts(
            CounterOpts::new("quantum_query_operations_total", "Total query operations")
        ).unwrap();
        registry.register(Box::new(query_operations.clone())).unwrap();
        
        let operation_duration = Histogram::with_opts(
            HistogramOpts::new("quantum_operation_duration_seconds", "Operation duration in seconds")
                .buckets(vec![0.001, 0.01, 0.1, 1.0, 10.0])
        ).unwrap();
        registry.register(Box::new(operation_duration.clone())).unwrap();
        
        let memory_usage = Gauge::with_opts(
            GaugeOpts::new("quantum_memory_usage_bytes", "Memory usage in bytes")
        ).unwrap();
        registry.register(Box::new(memory_usage.clone())).unwrap();
        
        let cpu_usage = Gauge::with_opts(
            GaugeOpts::new("quantum_cpu_usage_percent", "CPU usage percentage")
        ).unwrap();
        registry.register(Box::new(cpu_usage.clone())).unwrap();
        
        let total_nodes = IntGauge::with_opts(
            GaugeOpts::new("quantum_total_nodes", "Total number of nodes in graph")
        ).unwrap();
        registry.register(Box::new(total_nodes.clone())).unwrap();
        
        let total_edges = IntGauge::with_opts(
            GaugeOpts::new("quantum_total_edges", "Total number of edges in graph")
        ).unwrap();
        registry.register(Box::new(total_edges.clone())).unwrap();
        
        let graph_density = Gauge::with_opts(
            GaugeOpts::new("quantum_graph_density", "Graph density (edges/max_possible_edges)")
        ).unwrap();
        registry.register(Box::new(graph_density.clone())).unwrap();
        
        let query_latency = Histogram::with_opts(
            HistogramOpts::new("quantum_query_latency_seconds", "Query latency in seconds")
                .buckets(vec![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
        ).unwrap();
        registry.register(Box::new(query_latency.clone())).unwrap();
        
        let query_throughput = Counter::with_opts(
            CounterOpts::new("quantum_query_throughput_total", "Total queries processed")
        ).unwrap();
        registry.register(Box::new(query_throughput.clone())).unwrap();
        
        let cache_hit_rate = Gauge::with_opts(
            GaugeOpts::new("quantum_cache_hit_rate", "Cache hit rate (0-1)")
        ).unwrap();
        registry.register(Box::new(cache_hit_rate.clone())).unwrap();
        
        let simd_operations = IntCounter::with_opts(
            CounterOpts::new("quantum_simd_operations_total", "Total SIMD operations")
        ).unwrap();
        registry.register(Box::new(simd_operations.clone())).unwrap();
        
        let simd_speedup = Gauge::with_opts(
            GaugeOpts::new("quantum_simd_speedup_factor", "SIMD speedup factor vs scalar")
        ).unwrap();
        registry.register(Box::new(simd_speedup.clone())).unwrap();
        
        let network_latency = Histogram::with_opts(
            HistogramOpts::new("quantum_network_latency_seconds", "Network latency in seconds")
                .buckets(vec![0.001, 0.01, 0.1, 1.0])
        ).unwrap();
        registry.register(Box::new(network_latency.clone())).unwrap();
        
        let shard_balance = Gauge::with_opts(
            GaugeOpts::new("quantum_shard_balance", "Shard balance factor (0-1, 1=perfect)")
        ).unwrap();
        registry.register(Box::new(shard_balance.clone())).unwrap();
        
        let consensus_operations = IntCounter::with_opts(
            CounterOpts::new("quantum_consensus_operations_total", "Total consensus operations")
        ).unwrap();
        registry.register(Box::new(consensus_operations.clone())).unwrap();
        
        let error_count = IntCounter::with_opts(
            CounterOpts::new("quantum_errors_total", "Total errors encountered")
        ).unwrap();
        registry.register(Box::new(error_count.clone())).unwrap();
        
        Self {
            registry,
            node_operations,
            edge_operations,
            query_operations,
            operation_duration,
            memory_usage,
            cpu_usage,
            total_nodes,
            total_edges,
            graph_density,
            query_latency,
            query_throughput,
            cache_hit_rate,
            simd_operations,
            simd_speedup,
            network_latency,
            shard_balance,
            consensus_operations,
            error_count,
            custom_metrics: HashMap::new(),
            performance_history: PerformanceHistory::new(),
        }
    }
    
    /// Record a node operation
    pub fn record_node_operation(&self, operation_type: OperationType, duration: Duration) {
        self.node_operations.inc();
        self.operation_duration.observe(duration.as_secs_f64());
        self.performance_history.record_operation(operation_type, duration);
    }
    
    /// Record an edge operation
    pub fn record_edge_operation(&self, operation_type: OperationType, duration: Duration) {
        self.edge_operations.inc();
        self.operation_duration.observe(duration.as_secs_f64());
        self.performance_history.record_operation(operation_type, duration);
    }
    
    /// Record a query operation
    pub fn record_query(&self, query_type: QueryType, duration: Duration, results_count: usize) {
        self.query_operations.inc();
        self.query_latency.observe(duration.as_secs_f64());
        self.query_throughput.inc();
        self.performance_history.record_query(query_type, duration, results_count);
    }
    
    /// Update graph statistics
    pub fn update_graph_stats(&self, node_count: u64, edge_count: u64, density: f64) {
        self.total_nodes.set(node_count as i64);
        self.total_edges.set(edge_count as i64);
        self.graph_density.set(density);
    }
    
    /// Record SIMD operation performance
    pub fn record_simd_performance(&self, simd_time: Duration, scalar_time: Duration) {
        self.simd_operations.inc();
        let speedup = scalar_time.as_secs_f64() / simd_time.as_secs_f64();
        self.simd_speedup.set(speedup);
    }
    
    /// Record network operation
    pub fn record_network_operation(&self, latency: Duration) {
        self.network_latency.observe(latency.as_secs_f64());
    }
    
    /// Update distributed metrics
    pub fn update_distributed_metrics(&self, shard_balance: f64) {
        self.shard_balance.set(shard_balance);
    }
    
    /// Record consensus operation
    pub fn record_consensus_operation(&self) {
        self.consensus_operations.inc();
    }
    
    /// Record an error
    pub fn record_error(&self, error_type: &str) {
        self.error_count.inc();
        tracing::error!("Recorded error: {}", error_type);
    }
    
    /// Update memory usage
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage.set(bytes as f64);
    }
    
    /// Update CPU usage
    pub fn update_cpu_usage(&self, percentage: f64) {
        self.cpu_usage.set(percentage);
    }
    
    /// Update cache hit rate
    pub fn update_cache_hit_rate(&self, rate: f64) {
        self.cache_hit_rate.set(rate);
    }
    
    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: String, metric: Box<dyn CustomMetric + Send>) {
        self.custom_metrics.insert(name, metric);
    }
    
    /// Get performance summary
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_operations: self.node_operations.get() + self.edge_operations.get(),
            total_queries: self.query_operations.get(),
            average_query_latency_ms: self.performance_history.average_query_latency().as_millis() as f64,
            current_memory_usage_mb: self.memory_usage.get() / (1024.0 * 1024.0),
            current_cpu_usage_percent: self.cpu_usage.get(),
            graph_node_count: self.total_nodes.get() as u64,
            graph_edge_count: self.total_edges.get() as u64,
            graph_density: self.graph_density.get(),
            cache_hit_rate: self.cache_hit_rate.get(),
            simd_speedup_factor: self.simd_speedup.get(),
            error_count: self.error_count.get(),
            uptime_seconds: self.performance_history.uptime().as_secs(),
        }
    }
    
    /// Get Prometheus metrics
    pub fn get_prometheus_metrics(&self) -> Result<String> {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families)
            .map_err(|e| Error::Internal(format!("Failed to encode metrics: {}", e)))
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> PerformanceReport {
        let summary = self.get_performance_summary();
        let bottlenecks = self.identify_bottlenecks();
        let recommendations = self.generate_recommendations(&bottlenecks);
        
        PerformanceReport {
            timestamp: SystemTime::now(),
            summary,
            bottlenecks,
            recommendations,
            detailed_metrics: self.performance_history.get_detailed_metrics(),
        }
    }
    
    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Check query latency
        let avg_latency = self.performance_history.average_query_latency();
        if avg_latency > Duration::from_millis(100) {
            bottlenecks.push(PerformanceBottleneck {
                category: BottleneckCategory::QueryLatency,
                severity: if avg_latency > Duration::from_secs(1) {
                    Severity::High
                } else {
                    Severity::Medium
                },
                description: format!("Average query latency is {:.2}ms", avg_latency.as_millis()),
                impact: "Queries are taking longer than expected".to_string(),
            });
        }
        
        // Check memory usage
        let memory_mb = self.memory_usage.get() / (1024.0 * 1024.0);
        if memory_mb > 16384.0 { // 16GB
            bottlenecks.push(PerformanceBottleneck {
                category: BottleneckCategory::Memory,
                severity: Severity::Medium,
                description: format!("Memory usage is {:.1}MB", memory_mb),
                impact: "High memory usage may affect performance".to_string(),
            });
        }
        
        // Check CPU usage
        let cpu_usage = self.cpu_usage.get();
        if cpu_usage > 80.0 {
            bottlenecks.push(PerformanceBottleneck {
                category: BottleneckCategory::CPU,
                severity: Severity::High,
                description: format!("CPU usage is {:.1}%", cpu_usage),
                impact: "High CPU usage may cause query delays".to_string(),
            });
        }
        
        // Check cache hit rate
        let cache_rate = self.cache_hit_rate.get();
        if cache_rate < 0.8 {
            bottlenecks.push(PerformanceBottleneck {
                category: BottleneckCategory::Cache,
                severity: Severity::Medium,
                description: format!("Cache hit rate is {:.1}%", cache_rate * 100.0),
                impact: "Low cache hit rate increases query latency".to_string(),
            });
        }
        
        bottlenecks
    }
    
    /// Generate optimization recommendations
    fn generate_recommendations(&self, bottlenecks: &[PerformanceBottleneck]) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        for bottleneck in bottlenecks {
            match bottleneck.category {
                BottleneckCategory::QueryLatency => {
                    recommendations.push(OptimizationRecommendation {
                        category: RecommendationCategory::Indexing,
                        priority: RecommendationPriority::High,
                        title: "Optimize Query Performance".to_string(),
                        description: "Consider adding indexes or optimizing query patterns".to_string(),
                        estimated_impact: "50-80% latency reduction".to_string(),
                    });
                }
                BottleneckCategory::Memory => {
                    recommendations.push(OptimizationRecommendation {
                        category: RecommendationCategory::Memory,
                        priority: RecommendationPriority::Medium,
                        title: "Optimize Memory Usage".to_string(),
                        description: "Enable compression or increase memory pool size".to_string(),
                        estimated_impact: "20-40% memory reduction".to_string(),
                    });
                }
                BottleneckCategory::CPU => {
                    recommendations.push(OptimizationRecommendation {
                        category: RecommendationCategory::Parallelization,
                        priority: RecommendationPriority::High,
                        title: "Increase Parallelization".to_string(),
                        description: "Enable SIMD optimizations or add more CPU threads".to_string(),
                        estimated_impact: "30-60% performance improvement".to_string(),
                    });
                }
                BottleneckCategory::Cache => {
                    recommendations.push(OptimizationRecommendation {
                        category: RecommendationCategory::Caching,
                        priority: RecommendationPriority::Medium,
                        title: "Improve Cache Strategy".to_string(),
                        description: "Increase cache size or optimize cache eviction policy".to_string(),
                        estimated_impact: "15-30% latency reduction".to_string(),
                    });
                }
                _ => {}
            }
        }
        
        recommendations
    }
}

/// Performance history tracking
pub struct PerformanceHistory {
    start_time: Instant,
    operation_history: Vec<OperationRecord>,
    query_history: Vec<QueryRecord>,
    max_history_size: usize,
}

impl PerformanceHistory {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            operation_history: Vec::new(),
            query_history: Vec::new(),
            max_history_size: 10000,
        }
    }
    
    fn record_operation(&mut self, operation_type: OperationType, duration: Duration) {
        let record = OperationRecord {
            timestamp: Instant::now(),
            operation_type,
            duration,
        };
        
        self.operation_history.push(record);
        
        // Keep history size manageable
        if self.operation_history.len() > self.max_history_size {
            self.operation_history.remove(0);
        }
    }
    
    fn record_query(&mut self, query_type: QueryType, duration: Duration, results_count: usize) {
        let record = QueryRecord {
            timestamp: Instant::now(),
            query_type,
            duration,
            results_count,
        };
        
        self.query_history.push(record);
        
        if self.query_history.len() > self.max_history_size {
            self.query_history.remove(0);
        }
    }
    
    fn average_query_latency(&self) -> Duration {
        if self.query_history.is_empty() {
            return Duration::from_millis(0);
        }
        
        let total_nanos: u64 = self.query_history.iter()
            .map(|record| record.duration.as_nanos() as u64)
            .sum();
        
        Duration::from_nanos(total_nanos / self.query_history.len() as u64)
    }
    
    fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    fn get_detailed_metrics(&self) -> DetailedMetrics {
        DetailedMetrics {
            recent_operations: self.operation_history.iter().rev().take(100).cloned().collect(),
            recent_queries: self.query_history.iter().rev().take(100).cloned().collect(),
            operation_stats: self.calculate_operation_stats(),
            query_stats: self.calculate_query_stats(),
        }
    }
    
    fn calculate_operation_stats(&self) -> HashMap<OperationType, OperationStats> {
        let mut stats = HashMap::new();
        
        for record in &self.operation_history {
            let entry = stats.entry(record.operation_type).or_insert_with(OperationStats::new);
            entry.count += 1;
            entry.total_duration += record.duration;
            entry.min_duration = entry.min_duration.min(record.duration);
            entry.max_duration = entry.max_duration.max(record.duration);
        }
        
        // Calculate averages
        for stat in stats.values_mut() {
            if stat.count > 0 {
                stat.avg_duration = Duration::from_nanos(stat.total_duration.as_nanos() as u64 / stat.count);
            }
        }
        
        stats
    }
    
    fn calculate_query_stats(&self) -> HashMap<QueryType, QueryStats> {
        let mut stats = HashMap::new();
        
        for record in &self.query_history {
            let entry = stats.entry(record.query_type).or_insert_with(QueryStats::new);
            entry.count += 1;
            entry.total_duration += record.duration;
            entry.total_results += record.results_count;
            entry.min_duration = entry.min_duration.min(record.duration);
            entry.max_duration = entry.max_duration.max(record.duration);
        }
        
        // Calculate averages
        for stat in stats.values_mut() {
            if stat.count > 0 {
                stat.avg_duration = Duration::from_nanos(stat.total_duration.as_nanos() as u64 / stat.count);
                stat.avg_results = stat.total_results as f64 / stat.count as f64;
            }
        }
        
        stats
    }
}

/// Operation types for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    NodeInsert,
    NodeUpdate,
    NodeDelete,
    NodeGet,
    EdgeInsert,
    EdgeUpdate,
    EdgeDelete,
    EdgeGet,
    BatchInsert,
}

/// Query types for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryType {
    ShortestPath,
    PatternMatch,
    Traversal,
    NeighborQuery,
    Aggregation,
}

/// Custom metric trait
pub trait CustomMetric {
    fn record_value(&mut self, value: f64);
    fn get_current_value(&self) -> f64;
    fn get_name(&self) -> &str;
}

/// Performance summary snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_operations: u64,
    pub total_queries: u64,
    pub average_query_latency_ms: f64,
    pub current_memory_usage_mb: f64,
    pub current_cpu_usage_percent: f64,
    pub graph_node_count: u64,
    pub graph_edge_count: u64,
    pub graph_density: f64,
    pub cache_hit_rate: f64,
    pub simd_speedup_factor: f64,
    pub error_count: u64,
    pub uptime_seconds: u64,
}

/// Detailed performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub timestamp: SystemTime,
    pub summary: PerformanceSummary,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub detailed_metrics: DetailedMetrics,
}

/// Performance bottleneck identification
#[derive(Debug, Clone)]
pub struct PerformanceBottleneck {
    pub category: BottleneckCategory,
    pub severity: Severity,
    pub description: String,
    pub impact: String,
}

#[derive(Debug, Clone, Copy)]
pub enum BottleneckCategory {
    QueryLatency,
    Memory,
    CPU,
    Network,
    Storage,
    Cache,
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization recommendations
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub estimated_impact: String,
}

#[derive(Debug, Clone, Copy)]
pub enum RecommendationCategory {
    Indexing,
    Memory,
    Parallelization,
    Caching,
    Networking,
    Hardware,
}

#[derive(Debug, Clone, Copy)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Detailed metrics for analysis
#[derive(Debug, Clone)]
pub struct DetailedMetrics {
    pub recent_operations: Vec<OperationRecord>,
    pub recent_queries: Vec<QueryRecord>,
    pub operation_stats: HashMap<OperationType, OperationStats>,
    pub query_stats: HashMap<QueryType, QueryStats>,
}

/// Operation record for history
#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub timestamp: Instant,
    pub operation_type: OperationType,
    pub duration: Duration,
}

/// Query record for history
#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub timestamp: Instant,
    pub query_type: QueryType,
    pub duration: Duration,
    pub results_count: usize,
}

/// Operation statistics
#[derive(Debug, Clone)]
pub struct OperationStats {
    pub count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
}

impl OperationStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::from_nanos(0),
            avg_duration: Duration::from_nanos(0),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::from_nanos(0),
        }
    }
}

/// Query statistics
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub total_results: usize,
    pub avg_results: f64,
}

impl QueryStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::from_nanos(0),
            avg_duration: Duration::from_nanos(0),
            min_duration: Duration::from_secs(u64::MAX),
            max_duration: Duration::from_nanos(0),
            total_results: 0,
            avg_results: 0.0,
        }
    }
}

/// Utility functions for metrics
pub fn record_operation_metric(operation_type: OperationType, duration: Duration) -> Result<()> {
    GLOBAL_METRICS.lock()
        .map_err(|_| Error::Internal("Failed to lock metrics".to_string()))?
        .record_node_operation(operation_type, duration);
    Ok(())
}

pub fn record_query_metric(query_type: QueryType, duration: Duration, results_count: usize) -> Result<()> {
    GLOBAL_METRICS.lock()
        .map_err(|_| Error::Internal("Failed to lock metrics".to_string()))?
        .record_query(query_type, duration, results_count);
    Ok(())
}

pub fn get_performance_summary() -> Result<PerformanceSummary> {
    Ok(GLOBAL_METRICS.lock()
        .map_err(|_| Error::Internal("Failed to lock metrics".to_string()))?
        .get_performance_summary())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        
        // Verify initial state
        assert_eq!(collector.node_operations.get(), 0);
        assert_eq!(collector.edge_operations.get(), 0);
        assert_eq!(collector.query_operations.get(), 0);
    }
    
    #[test]
    fn test_operation_recording() {
        let collector = MetricsCollector::new();
        
        collector.record_node_operation(OperationType::NodeInsert, Duration::from_millis(10));
        collector.record_edge_operation(OperationType::EdgeInsert, Duration::from_millis(5));
        
        assert_eq!(collector.node_operations.get(), 1);
        assert_eq!(collector.edge_operations.get(), 1);
    }
    
    #[test]
    fn test_performance_summary() {
        let collector = MetricsCollector::new();
        
        // Record some operations
        collector.record_query(QueryType::ShortestPath, Duration::from_millis(50), 10);
        collector.update_graph_stats(1000, 5000, 0.01);
        collector.update_memory_usage(1024 * 1024 * 1024); // 1GB
        
        let summary = collector.get_performance_summary();
        
        assert_eq!(summary.total_queries, 1);
        assert_eq!(summary.graph_node_count, 1000);
        assert_eq!(summary.graph_edge_count, 5000);
        assert_eq!(summary.graph_density, 0.01);
        assert_eq!(summary.current_memory_usage_mb, 1024.0);
    }
    
    #[test]
    fn test_bottleneck_identification() {
        let collector = MetricsCollector::new();
        
        // Simulate high CPU usage
        collector.update_cpu_usage(90.0);
        
        let bottlenecks = collector.identify_bottlenecks();
        
        assert!(!bottlenecks.is_empty());
        assert!(bottlenecks.iter().any(|b| matches!(b.category, BottleneckCategory::CPU)));
    }
}