//! GPU performance metrics and monitoring
//!
//! This module provides comprehensive GPU performance monitoring including
//! kernel execution metrics, memory usage tracking, and device utilization.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::{GpuDeviceId, GpuResult};
use crate::error::GpuKnowledgeGraphError;
use crate::gpu::GpuManager;

/// GPU performance metrics collector
pub struct GpuMetricsCollector {
    /// GPU manager reference
    gpu_manager: Arc<GpuManager>,
    
    /// Per-device metrics
    device_metrics: Arc<RwLock<HashMap<GpuDeviceId, Arc<RwLock<GpuDeviceMetrics>>>>>,
    
    /// Global GPU metrics
    global_metrics: Arc<RwLock<GlobalGpuMetrics>>,
    
    /// Metrics collection interval
    collection_interval: Duration,
    
    /// Last collection time
    last_collection: Arc<RwLock<Instant>>,
    
    /// Metrics history for trend analysis
    metrics_history: Arc<RwLock<Vec<MetricsSnapshot>>>,
}

impl GpuMetricsCollector {
    /// Create new GPU metrics collector
    pub async fn new(
        gpu_manager: Arc<GpuManager>,
        collection_interval: Duration
    ) -> GpuResult<Self> {
        tracing::info!("📊 Initializing GPU metrics collector");
        
        let device_metrics = Arc::new(RwLock::new(HashMap::new()));
        let global_metrics = Arc::new(RwLock::new(GlobalGpuMetrics::new()));
        let last_collection = Arc::new(RwLock::new(Instant::now()));
        let metrics_history = Arc::new(RwLock::new(Vec::new()));
        
        // Initialize per-device metrics
        {
            let mut metrics_map = device_metrics.write();
            for device_id in gpu_manager.get_available_devices() {
                metrics_map.insert(
                    device_id,
                    Arc::new(RwLock::new(GpuDeviceMetrics::new(device_id)))
                );
            }
        }
        
        Ok(Self {
            gpu_manager,
            device_metrics,
            global_metrics,
            collection_interval,
            last_collection,
            metrics_history,
        })
    }
    
    /// Start metrics collection background task
    pub async fn start_collection(&self) -> GpuResult<()> {
        tracing::info!("🚀 Starting GPU metrics collection");
        
        // TODO: Implement background metrics collection task
        // This would typically run in a separate thread/task
        
        Ok(())
    }
    
    /// Collect current GPU metrics
    pub async fn collect_metrics(&self) -> GpuResult<MetricsSnapshot> {
        let start_time = Instant::now();
        
        // Collect device-specific metrics
        let mut device_snapshots = HashMap::new();
        {
            let device_metrics = self.device_metrics.read();
            for (&device_id, device_metric) in device_metrics.iter() {
                let device = self.gpu_manager.get_device(device_id)?;
                let metrics = device_metric.read();
                
                // Get current device state
                let utilization = device.get_utilization().await?;
                let (used_memory, total_memory) = device.get_memory_usage().await?;
                
                let snapshot = DeviceMetricsSnapshot {
                    device_id,
                    timestamp: SystemTime::now(),
                    utilization,
                    memory_used: used_memory,
                    memory_total: total_memory,
                    memory_utilization: used_memory as f32 / total_memory as f32,
                    kernel_launches: metrics.kernel_launches,
                    total_kernel_time: metrics.total_kernel_time,
                    average_kernel_time: if metrics.kernel_launches > 0 {
                        metrics.total_kernel_time / metrics.kernel_launches as u32
                    } else {
                        Duration::ZERO
                    },
                    memory_transfers: metrics.memory_transfers,
                    total_transfer_time: metrics.total_transfer_time,
                    average_transfer_time: if metrics.memory_transfers > 0 {
                        metrics.total_transfer_time / metrics.memory_transfers as u32
                    } else {
                        Duration::ZERO
                    },
                    errors: metrics.error_count,
                    temperature: metrics.temperature,
                    power_usage: metrics.power_usage,
                    clock_speed: metrics.clock_speed,
                    memory_clock: metrics.memory_clock,
                };
                
                device_snapshots.insert(device_id, snapshot);
            }
        }
        
        // Update global metrics
        let global_snapshot = {
            let mut global = self.global_metrics.write();
            global.total_operations += device_snapshots.len() as u64;
            global.last_update = SystemTime::now();
            
            // Aggregate across all devices
            let total_memory_used: usize = device_snapshots.values().map(|s| s.memory_used).sum();
            let total_memory: usize = device_snapshots.values().map(|s| s.memory_total).sum();
            let average_utilization: f32 = device_snapshots.values().map(|s| s.utilization).sum::<f32>() / device_snapshots.len() as f32;
            let total_kernel_launches: u64 = device_snapshots.values().map(|s| s.kernel_launches).sum();
            let total_errors: u64 = device_snapshots.values().map(|s| s.errors).sum();
            
            global.total_memory_used = total_memory_used;
            global.total_memory = total_memory;
            global.average_utilization = average_utilization;
            global.total_kernel_launches = total_kernel_launches;
            global.total_errors = total_errors;
            
            GlobalMetricsSnapshot {
                timestamp: SystemTime::now(),
                device_count: device_snapshots.len(),
                total_memory_used,
                total_memory,
                memory_utilization: total_memory_used as f32 / total_memory as f32,
                average_utilization,
                total_kernel_launches,
                total_memory_transfers: device_snapshots.values().map(|s| s.memory_transfers).sum(),
                total_errors,
                collection_time: start_time.elapsed(),
            }
        };
        
        let snapshot = MetricsSnapshot {
            timestamp: SystemTime::now(),
            devices: device_snapshots,
            global: global_snapshot,
        };
        
        // Store in history
        {
            let mut history = self.metrics_history.write();
            history.push(snapshot.clone());
            
            // Keep only last 1000 snapshots
            if history.len() > 1000 {
                history.remove(0);
            }
        }
        
        *self.last_collection.write() = Instant::now();
        
        Ok(snapshot)
    }
    
    /// Record kernel execution
    pub fn record_kernel_execution(
        &self,
        device_id: GpuDeviceId,
        kernel_name: &str,
        execution_time: Duration,
        success: bool
    ) {
        if let Some(device_metrics) = self.device_metrics.read().get(&device_id) {
            let mut metrics = device_metrics.write();
            metrics.record_kernel_execution(kernel_name, execution_time, success);
        }
        
        // Update global metrics
        let mut global = self.global_metrics.write();
        global.total_operations += 1;
        if !success {
            global.total_errors += 1;
        }
    }
    
    /// Record memory transfer
    pub fn record_memory_transfer(
        &self,
        device_id: GpuDeviceId,
        transfer_type: MemoryTransferType,
        bytes: usize,
        duration: Duration,
        success: bool
    ) {
        if let Some(device_metrics) = self.device_metrics.read().get(&device_id) {
            let mut metrics = device_metrics.write();
            metrics.record_memory_transfer(transfer_type, bytes, duration, success);
        }
    }
    
    /// Get metrics summary
    pub async fn get_metrics_summary(&self) -> GpuResult<MetricsSummary> {
        let snapshot = self.collect_metrics().await?;
        
        Ok(MetricsSummary {
            collection_time: snapshot.timestamp,
            device_count: snapshot.global.device_count,
            total_memory_gb: snapshot.global.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            memory_utilization: snapshot.global.memory_utilization,
            average_gpu_utilization: snapshot.global.average_utilization,
            total_kernel_launches: snapshot.global.total_kernel_launches,
            total_memory_transfers: snapshot.global.total_memory_transfers,
            error_rate: if snapshot.global.total_kernel_launches > 0 {
                snapshot.global.total_errors as f64 / snapshot.global.total_kernel_launches as f64
            } else {
                0.0
            },
            collection_overhead: snapshot.global.collection_time,
        })
    }
    
    /// Export metrics to JSON
    pub async fn export_metrics(&self) -> GpuResult<String> {
        let snapshot = self.collect_metrics().await?;
        serde_json::to_string_pretty(&snapshot)
            .map_err(|e| GpuKnowledgeGraphError::Serialization(e))
    }
    
    /// Get performance trends
    pub fn get_performance_trends(&self, duration: Duration) -> GpuResult<PerformanceTrends> {
        let history = self.metrics_history.read();
        let cutoff_time = SystemTime::now() - duration;
        
        let recent_snapshots: Vec<_> = history.iter()
            .filter(|s| s.timestamp >= cutoff_time)
            .collect();
        
        if recent_snapshots.is_empty() {
            return Ok(PerformanceTrends::default());
        }
        
        // Calculate trends
        let utilizations: Vec<f32> = recent_snapshots.iter()
            .map(|s| s.global.average_utilization)
            .collect();
        
        let memory_utilizations: Vec<f32> = recent_snapshots.iter()
            .map(|s| s.global.memory_utilization)
            .collect();
        
        let error_rates: Vec<f64> = recent_snapshots.iter()
            .map(|s| {
                if s.global.total_kernel_launches > 0 {
                    s.global.total_errors as f64 / s.global.total_kernel_launches as f64
                } else {
                    0.0
                }
            })
            .collect();
        
        Ok(PerformanceTrends {
            timeframe: duration,
            sample_count: recent_snapshots.len(),
            utilization_trend: calculate_trend(&utilizations),
            memory_trend: calculate_trend(&memory_utilizations),
            error_trend: calculate_trend(&error_rates),
            peak_utilization: utilizations.iter().fold(0.0f32, |a, &b| a.max(b)),
            peak_memory_utilization: memory_utilizations.iter().fold(0.0f32, |a, &b| a.max(b)),
            average_utilization: utilizations.iter().sum::<f32>() / utilizations.len() as f32,
            average_memory_utilization: memory_utilizations.iter().sum::<f32>() / memory_utilizations.len() as f32,
        })
    }
}

/// Per-device GPU metrics
#[derive(Debug)]
pub struct GpuDeviceMetrics {
    /// Device ID
    pub device_id: GpuDeviceId,
    
    /// Kernel execution metrics
    pub kernel_launches: u64,
    pub total_kernel_time: Duration,
    pub kernel_errors: u64,
    pub kernel_history: HashMap<String, KernelMetrics>,
    
    /// Memory transfer metrics
    pub memory_transfers: u64,
    pub total_transfer_time: Duration,
    pub transfer_errors: u64,
    pub bytes_transferred: u64,
    
    /// Hardware metrics
    pub temperature: f32,
    pub power_usage: f32,
    pub clock_speed: u32,
    pub memory_clock: u32,
    
    /// Error tracking
    pub error_count: u64,
    pub last_error_time: Option<SystemTime>,
    
    /// Performance tracking
    pub utilization_history: Vec<(SystemTime, f32)>,
    pub memory_usage_history: Vec<(SystemTime, usize, usize)>,
    
    /// Creation time
    pub created_at: SystemTime,
}

impl GpuDeviceMetrics {
    pub fn new(device_id: GpuDeviceId) -> Self {
        Self {
            device_id,
            kernel_launches: 0,
            total_kernel_time: Duration::ZERO,
            kernel_errors: 0,
            kernel_history: HashMap::new(),
            memory_transfers: 0,
            total_transfer_time: Duration::ZERO,
            transfer_errors: 0,
            bytes_transferred: 0,
            temperature: 0.0,
            power_usage: 0.0,
            clock_speed: 0,
            memory_clock: 0,
            error_count: 0,
            last_error_time: None,
            utilization_history: Vec::new(),
            memory_usage_history: Vec::new(),
            created_at: SystemTime::now(),
        }
    }
    
    pub fn record_kernel_execution(&mut self, kernel_name: &str, execution_time: Duration, success: bool) {
        self.kernel_launches += 1;
        self.total_kernel_time += execution_time;
        
        if !success {
            self.kernel_errors += 1;
            self.error_count += 1;
            self.last_error_time = Some(SystemTime::now());
        }
        
        // Update kernel-specific metrics
        let kernel_metrics = self.kernel_history.entry(kernel_name.to_string())
            .or_insert_with(|| KernelMetrics::new(kernel_name));
        kernel_metrics.record_execution(execution_time, success);
    }
    
    pub fn record_memory_transfer(&mut self, transfer_type: MemoryTransferType, bytes: usize, duration: Duration, success: bool) {
        self.memory_transfers += 1;
        self.total_transfer_time += duration;
        self.bytes_transferred += bytes as u64;
        
        if !success {
            self.transfer_errors += 1;
            self.error_count += 1;
            self.last_error_time = Some(SystemTime::now());
        }
    }
    
    pub fn update_hardware_metrics(&mut self, temperature: f32, power_usage: f32, clock_speed: u32, memory_clock: u32) {
        self.temperature = temperature;
        self.power_usage = power_usage;
        self.clock_speed = clock_speed;
        self.memory_clock = memory_clock;
    }
    
    pub fn record_utilization(&mut self, utilization: f32) {
        self.utilization_history.push((SystemTime::now(), utilization));
        
        // Keep only last 100 entries
        if self.utilization_history.len() > 100 {
            self.utilization_history.remove(0);
        }
    }
    
    pub fn record_memory_usage(&mut self, used: usize, total: usize) {
        self.memory_usage_history.push((SystemTime::now(), used, total));
        
        // Keep only last 100 entries
        if self.memory_usage_history.len() > 100 {
            self.memory_usage_history.remove(0);
        }
    }
}

/// Kernel-specific metrics
#[derive(Debug, Clone)]
pub struct KernelMetrics {
    pub name: String,
    pub executions: u64,
    pub total_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub successes: u64,
    pub failures: u64,
    pub last_execution: Option<SystemTime>,
}

impl KernelMetrics {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            executions: 0,
            total_time: Duration::ZERO,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            successes: 0,
            failures: 0,
            last_execution: None,
        }
    }
    
    pub fn record_execution(&mut self, duration: Duration, success: bool) {
        self.executions += 1;
        self.total_time += duration;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        self.last_execution = Some(SystemTime::now());
        
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
    }
    
    pub fn average_time(&self) -> Duration {
        if self.executions > 0 {
            self.total_time / self.executions as u32
        } else {
            Duration::ZERO
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.executions > 0 {
            self.successes as f64 / self.executions as f64
        } else {
            0.0
        }
    }
}

/// Global GPU metrics
#[derive(Debug)]
pub struct GlobalGpuMetrics {
    pub total_operations: u64,
    pub total_errors: u64,
    pub total_memory_used: usize,
    pub total_memory: usize,
    pub average_utilization: f32,
    pub total_kernel_launches: u64,
    pub created_at: SystemTime,
    pub last_update: SystemTime,
}

impl GlobalGpuMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            total_errors: 0,
            total_memory_used: 0,
            total_memory: 0,
            average_utilization: 0.0,
            total_kernel_launches: 0,
            created_at: SystemTime::now(),
            last_update: SystemTime::now(),
        }
    }
}

/// Memory transfer types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTransferType {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    UnifiedMemory,
}

/// Metrics snapshot for a specific point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: SystemTime,
    pub devices: HashMap<GpuDeviceId, DeviceMetricsSnapshot>,
    pub global: GlobalMetricsSnapshot,
}

/// Device metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetricsSnapshot {
    pub device_id: GpuDeviceId,
    pub timestamp: SystemTime,
    pub utilization: f32,
    pub memory_used: usize,
    pub memory_total: usize,
    pub memory_utilization: f32,
    pub kernel_launches: u64,
    pub total_kernel_time: Duration,
    pub average_kernel_time: Duration,
    pub memory_transfers: u64,
    pub total_transfer_time: Duration,
    pub average_transfer_time: Duration,
    pub errors: u64,
    pub temperature: f32,
    pub power_usage: f32,
    pub clock_speed: u32,
    pub memory_clock: u32,
}

/// Global metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalMetricsSnapshot {
    pub timestamp: SystemTime,
    pub device_count: usize,
    pub total_memory_used: usize,
    pub total_memory: usize,
    pub memory_utilization: f32,
    pub average_utilization: f32,
    pub total_kernel_launches: u64,
    pub total_memory_transfers: u64,
    pub total_errors: u64,
    pub collection_time: Duration,
}

/// High-level metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub collection_time: SystemTime,
    pub device_count: usize,
    pub total_memory_gb: f64,
    pub memory_utilization: f32,
    pub average_gpu_utilization: f32,
    pub total_kernel_launches: u64,
    pub total_memory_transfers: u64,
    pub error_rate: f64,
    pub collection_overhead: Duration,
}

/// Performance trends analysis
#[derive(Debug, Clone, Default)]
pub struct PerformanceTrends {
    pub timeframe: Duration,
    pub sample_count: usize,
    pub utilization_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub error_trend: TrendDirection,
    pub peak_utilization: f32,
    pub peak_memory_utilization: f32,
    pub average_utilization: f32,
    pub average_memory_utilization: f32,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrendDirection {
    #[default]
    Stable,
    Increasing,
    Decreasing,
    Volatile,
}

// Helper functions

fn calculate_trend<T: Copy + Into<f64>>(values: &[T]) -> TrendDirection {
    if values.len() < 3 {
        return TrendDirection::Stable;
    }
    
    let vals: Vec<f64> = values.iter().map(|&v| v.into()).collect();
    
    // Calculate linear regression slope
    let n = vals.len() as f64;
    let sum_x: f64 = (0..vals.len()).map(|i| i as f64).sum();
    let sum_y: f64 = vals.iter().sum();
    let sum_xy: f64 = vals.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..vals.len()).map(|i| (i as f64).powi(2)).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
    
    // Calculate variance to detect volatility
    let mean = sum_y / n;
    let variance = vals.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    
    // Determine trend
    if std_dev > mean * 0.3 {
        TrendDirection::Volatile
    } else if slope > 0.01 {
        TrendDirection::Increasing
    } else if slope < -0.01 {
        TrendDirection::Decreasing
    } else {
        TrendDirection::Stable
    }
}

/// Initialize GPU metrics system
pub fn init_gpu_metrics() -> GpuResult<()> {
    tracing::debug!("📊 Initializing GPU metrics system");
    // TODO: Initialize metrics collection infrastructure
    Ok(())
}

/// Create metrics collector with default settings
pub async fn create_default_metrics_collector(
    gpu_manager: Arc<GpuManager>
) -> GpuResult<GpuMetricsCollector> {
    GpuMetricsCollector::new(gpu_manager, Duration::from_secs(1)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_metrics_creation() {
        let metrics = GpuDeviceMetrics::new(0);
        assert_eq!(metrics.device_id, 0);
        assert_eq!(metrics.kernel_launches, 0);
        assert_eq!(metrics.error_count, 0);
    }
    
    #[test]
    fn test_kernel_metrics() {
        let mut kernel_metrics = KernelMetrics::new("test_kernel");
        assert_eq!(kernel_metrics.executions, 0);
        assert_eq!(kernel_metrics.success_rate(), 0.0);
        
        kernel_metrics.record_execution(Duration::from_millis(10), true);
        kernel_metrics.record_execution(Duration::from_millis(20), false);
        
        assert_eq!(kernel_metrics.executions, 2);
        assert_eq!(kernel_metrics.successes, 1);
        assert_eq!(kernel_metrics.failures, 1);
        assert_eq!(kernel_metrics.success_rate(), 0.5);
        assert_eq!(kernel_metrics.average_time(), Duration::from_millis(15));
    }
    
    #[test]
    fn test_trend_calculation() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(calculate_trend(&increasing), TrendDirection::Increasing);
        
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(calculate_trend(&decreasing), TrendDirection::Decreasing);
        
        let stable = vec![3.0, 3.1, 2.9, 3.0, 3.1];
        assert_eq!(calculate_trend(&stable), TrendDirection::Stable);
        
        let volatile = vec![1.0, 5.0, 2.0, 8.0, 1.0];
        assert_eq!(calculate_trend(&volatile), TrendDirection::Volatile);
    }
    
    #[test]
    fn test_global_metrics() {
        let mut global = GlobalGpuMetrics::new();
        assert_eq!(global.total_operations, 0);
        assert_eq!(global.total_errors, 0);
        
        global.total_operations = 100;
        global.total_errors = 5;
        
        let error_rate = global.total_errors as f64 / global.total_operations as f64;
        assert_eq!(error_rate, 0.05);
    }
    
    #[test]
    fn test_memory_transfer_types() {
        let transfer_types = [
            MemoryTransferType::HostToDevice,
            MemoryTransferType::DeviceToHost,
            MemoryTransferType::DeviceToDevice,
            MemoryTransferType::UnifiedMemory,
        ];
        
        for transfer_type in &transfer_types {
            assert_ne!(format!("{:?}", transfer_type), "");
        }
    }
}