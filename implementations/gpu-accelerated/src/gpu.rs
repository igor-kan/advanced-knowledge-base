//! GPU device management and CUDA runtime integration
//!
//! This module provides comprehensive GPU device management, CUDA runtime
//! initialization, and low-level GPU operations for the knowledge graph.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use cudarc::driver::{CudaDevice, CudaFunction, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

use crate::core::{GpuDeviceId, CudaStream, GpuResult};
use crate::error::GpuKnowledgeGraphError;

/// GPU device manager for coordinating multiple CUDA devices
pub struct GpuManager {
    /// Available CUDA devices
    devices: Vec<Arc<GpuDevice>>,
    
    /// Device selection strategy
    selection_strategy: DeviceSelectionStrategy,
    
    /// Load balancer for distributing work across GPUs
    load_balancer: Arc<RwLock<GpuLoadBalancer>>,
    
    /// Device performance metrics
    device_metrics: Arc<RwLock<HashMap<GpuDeviceId, DeviceMetrics>>>,
}

impl GpuManager {
    /// Create new GPU manager and initialize all available devices
    pub async fn new(config: &crate::graph::GpuGraphConfig) -> GpuResult<Self> {
        tracing::info!("🔍 Detecting CUDA devices...");
        
        // Initialize CUDA runtime
        init_cuda_runtime()?;
        
        // Detect available devices
        let device_count = get_device_count()?;
        tracing::info!("Found {} CUDA device(s)", device_count);
        
        let mut devices = Vec::new();
        let target_devices = if config.gpu_devices.is_empty() {
            (0..device_count).collect::<Vec<_>>()
        } else {
            config.gpu_devices.clone()
        };
        
        // Initialize target devices
        for device_id in target_devices {
            if device_id >= device_count {
                tracing::warn!("Device {} not available (only {} devices found)", device_id, device_count);
                continue;
            }
            
            match GpuDevice::new(device_id, config).await {
                Ok(device) => {
                    let device_info = device.get_info();
                    tracing::info!(
                        "✅ Initialized GPU {}: {} (Compute {}.{}, {}GB memory)",
                        device_id,
                        device_info.name,
                        device_info.compute_capability.0,
                        device_info.compute_capability.1,
                        device_info.total_memory / (1024 * 1024 * 1024)
                    );
                    devices.push(Arc::new(device));
                },
                Err(e) => {
                    tracing::error!("Failed to initialize GPU {}: {}", device_id, e);
                }
            }
        }
        
        if devices.is_empty() {
            return Err(GpuKnowledgeGraphError::gpu_initialization("No CUDA devices available"));
        }
        
        let selection_strategy = if devices.len() == 1 {
            DeviceSelectionStrategy::Single
        } else {
            DeviceSelectionStrategy::LoadBalanced
        };
        
        let load_balancer = Arc::new(RwLock::new(GpuLoadBalancer::new(&devices)));
        let device_metrics = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize metrics for all devices
        {
            let mut metrics = device_metrics.write();
            for device in &devices {
                metrics.insert(device.id(), DeviceMetrics::new());
            }
        }
        
        Ok(Self {
            devices,
            selection_strategy,
            load_balancer,
            device_metrics,
        })
    }
    
    /// Get number of available GPU devices
    pub fn get_device_count(&self) -> usize {
        self.devices.len()
    }
    
    /// Get all available devices
    pub fn get_available_devices(&self) -> Vec<GpuDeviceId> {
        self.devices.iter().map(|d| d.id()).collect()
    }
    
    /// Get device by ID
    pub fn get_device(&self, device_id: GpuDeviceId) -> GpuResult<Arc<GpuDevice>> {
        self.devices.iter()
            .find(|d| d.id() == device_id)
            .cloned()
            .ok_or_else(|| GpuKnowledgeGraphError::device_not_found(device_id))
    }
    
    /// Select optimal device for a given workload
    pub async fn select_optimal_device(&self, workload: &WorkloadHint) -> GpuResult<Arc<GpuDevice>> {
        match self.selection_strategy {
            DeviceSelectionStrategy::Single => {
                Ok(Arc::clone(&self.devices[0]))
            },
            DeviceSelectionStrategy::LoadBalanced => {
                let device_id = self.load_balancer.read().select_device(workload);
                self.get_device(device_id)
            },
            DeviceSelectionStrategy::MemoryOptimized => {
                self.select_device_by_available_memory().await
            },
            DeviceSelectionStrategy::ComputeOptimized => {
                self.select_device_by_compute_capability().await
            },
        }
    }
    
    /// Synchronize all GPU devices
    pub async fn synchronize_all_devices(&self) -> GpuResult<()> {
        for device in &self.devices {
            device.synchronize().await?;
        }
        Ok(())
    }
    
    /// Get utilization across all devices
    pub async fn get_utilization(&self) -> GpuResult<Vec<f32>> {
        let mut utilizations = Vec::new();
        for device in &self.devices {
            utilizations.push(device.get_utilization().await?);
        }
        Ok(utilizations)
    }
    
    /// Get memory usage across all devices
    pub async fn get_memory_usage(&self) -> GpuResult<Vec<(usize, usize)>> {
        let mut memory_usage = Vec::new();
        for device in &self.devices {
            memory_usage.push(device.get_memory_usage().await?);
        }
        Ok(memory_usage)
    }
    
    /// Get detailed device information
    pub async fn get_device_info(&self) -> GpuResult<Vec<GpuDeviceInfo>> {
        let mut device_info = Vec::new();
        for device in &self.devices {
            device_info.push(device.get_info());
        }
        Ok(device_info)
    }
    
    /// Update device metrics
    pub fn update_device_metrics(&self, device_id: GpuDeviceId, operation_time: Duration, memory_used: usize) {
        if let Some(metrics) = self.device_metrics.write().get_mut(&device_id) {
            metrics.update(operation_time, memory_used);
        }
    }
    
    // Private helper methods
    
    async fn select_device_by_available_memory(&self) -> GpuResult<Arc<GpuDevice>> {
        let mut best_device = None;
        let mut max_free_memory = 0;
        
        for device in &self.devices {
            let (used, total) = device.get_memory_usage().await?;
            let free = total - used;
            if free > max_free_memory {
                max_free_memory = free;
                best_device = Some(Arc::clone(device));
            }
        }
        
        best_device.ok_or_else(|| GpuKnowledgeGraphError::internal_error("No device with available memory"))
    }
    
    async fn select_device_by_compute_capability(&self) -> GpuResult<Arc<GpuDevice>> {
        let mut best_device = None;
        let mut best_compute_capability = (0, 0);
        
        for device in &self.devices {
            let info = device.get_info();
            if info.compute_capability > best_compute_capability {
                best_compute_capability = info.compute_capability;
                best_device = Some(Arc::clone(device));
            }
        }
        
        best_device.ok_or_else(|| GpuKnowledgeGraphError::internal_error("No device with compute capability"))
    }
}

/// Individual GPU device wrapper
pub struct GpuDevice {
    /// Device ID
    id: GpuDeviceId,
    
    /// CUDA device handle
    cuda_device: Arc<CudaDevice>,
    
    /// Device information
    info: GpuDeviceInfo,
    
    /// CUDA streams for async operations
    streams: Vec<CudaStream>,
    
    /// Current stream index for round-robin allocation
    current_stream: std::sync::atomic::AtomicUsize,
    
    /// Device-specific memory pool
    memory_pool: Arc<RwLock<DeviceMemoryPool>>,
    
    /// Performance monitoring
    performance_monitor: Arc<RwLock<DevicePerformanceMonitor>>,
}

impl GpuDevice {
    /// Create new GPU device
    pub async fn new(device_id: GpuDeviceId, config: &crate::graph::GpuGraphConfig) -> GpuResult<Self> {
        // Initialize CUDA device
        let cuda_device = Arc::new(CudaDevice::new(device_id as usize)
            .map_err(|e| GpuKnowledgeGraphError::cuda_error(format!("Failed to create CUDA device {}: {}", device_id, e)))?);
        
        // Get device information
        let info = query_device_info(&cuda_device, device_id)?;
        
        // Validate minimum requirements
        validate_device_requirements(&info)?;
        
        // Create CUDA streams
        let mut streams = Vec::new();
        for i in 0..config.cuda_streams_per_gpu {
            let stream = create_cuda_stream(&cuda_device)?;
            streams.push(stream);
            tracing::debug!("Created CUDA stream {} for device {}", i, device_id);
        }
        
        // Initialize memory pool
        let memory_pool = Arc::new(RwLock::new(
            DeviceMemoryPool::new(device_id, config.gpu_memory_pool_size)
        ));
        
        // Initialize performance monitor
        let performance_monitor = Arc::new(RwLock::new(
            DevicePerformanceMonitor::new()
        ));
        
        Ok(Self {
            id: device_id,
            cuda_device,
            info,
            streams,
            current_stream: std::sync::atomic::AtomicUsize::new(0),
            memory_pool,
            performance_monitor,
        })
    }
    
    /// Get device ID
    pub fn id(&self) -> GpuDeviceId {
        self.id
    }
    
    /// Get device information
    pub fn get_info(&self) -> GpuDeviceInfo {
        self.info.clone()
    }
    
    /// Get CUDA device handle
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.cuda_device
    }
    
    /// Get next available CUDA stream (round-robin)
    pub fn get_stream(&self) -> CudaStream {
        let index = self.current_stream.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.streams.len();
        self.streams[index]
    }
    
    /// Synchronize device (wait for all operations to complete)
    pub async fn synchronize(&self) -> GpuResult<()> {
        self.cuda_device.synchronize()
            .map_err(|e| GpuKnowledgeGraphError::synchronization_error(format!("Device {} sync failed: {}", self.id, e)))?;
        Ok(())
    }
    
    /// Get current GPU utilization percentage
    pub async fn get_utilization(&self) -> GpuResult<f32> {
        // TODO: Implement actual GPU utilization monitoring
        // For now, return a placeholder based on performance monitor
        let monitor = self.performance_monitor.read();
        Ok(monitor.estimated_utilization())
    }
    
    /// Get memory usage (used, total) in bytes
    pub async fn get_memory_usage(&self) -> GpuResult<(usize, usize)> {
        let pool = self.memory_pool.read();
        Ok((pool.used_memory(), pool.total_memory()))
    }
    
    /// Allocate GPU memory
    pub async fn allocate_memory(&self, size: usize) -> GpuResult<*mut u8> {
        let mut pool = self.memory_pool.write();
        pool.allocate(size)
    }
    
    /// Free GPU memory
    pub async fn free_memory(&self, ptr: *mut u8) -> GpuResult<()> {
        let mut pool = self.memory_pool.write();
        pool.free(ptr)
    }
    
    /// Launch CUDA kernel
    pub async fn launch_kernel(
        &self,
        kernel: &CudaFunction,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        params: &[*const std::ffi::c_void],
    ) -> GpuResult<()> {
        let stream = self.get_stream();
        let start = Instant::now();
        
        let launch_config = LaunchConfig {
            grid_dim: grid_size,
            block_dim: block_size,
            shared_mem_bytes: 0,
        };
        
        unsafe {
            kernel.launch_async(launch_config, params, stream as *mut std::ffi::c_void)
                .map_err(|e| GpuKnowledgeGraphError::kernel_launch(format!("Kernel launch failed on device {}: {}", self.id, e)))?;
        }
        
        // Update performance metrics
        let duration = start.elapsed();
        self.performance_monitor.write().record_kernel_launch(duration);
        
        Ok(())
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device ID
    pub device_id: GpuDeviceId,
    
    /// Device name
    pub name: String,
    
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    
    /// Total global memory (bytes)
    pub total_memory: usize,
    
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    
    /// Maximum blocks per grid
    pub max_blocks_per_grid: (u32, u32, u32),
    
    /// Warp size
    pub warp_size: u32,
    
    /// Shared memory per block (bytes)
    pub shared_memory_per_block: usize,
    
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    
    /// Clock rate (kHz)
    pub clock_rate: u32,
    
    /// Memory clock rate (kHz)
    pub memory_clock_rate: u32,
    
    /// Memory bus width (bits)
    pub memory_bus_width: u32,
    
    /// L2 cache size (bytes)
    pub l2_cache_size: usize,
}

/// Device selection strategies
#[derive(Debug, Clone, Copy)]
pub enum DeviceSelectionStrategy {
    /// Use single GPU (if only one available)
    Single,
    /// Load-balanced selection across multiple GPUs
    LoadBalanced,
    /// Select GPU with most available memory
    MemoryOptimized,
    /// Select GPU with highest compute capability
    ComputeOptimized,
}

/// Workload hint for device selection
#[derive(Debug, Clone)]
pub enum WorkloadHint {
    /// Graph traversal algorithm
    GraphTraversal { estimated_memory: usize },
    /// Linear algebra operation
    LinearAlgebra { matrix_size: (usize, usize) },
    /// Pattern matching
    PatternMatching { pattern_complexity: u32 },
    /// Batch processing
    BatchProcessing { batch_size: usize },
    /// Memory-intensive operation
    MemoryIntensive { memory_requirement: usize },
    /// Compute-intensive operation
    ComputeIntensive { expected_duration: Duration },
}

/// GPU load balancer
pub struct GpuLoadBalancer {
    /// Device loads (normalized 0.0-1.0)
    device_loads: HashMap<GpuDeviceId, f32>,
    
    /// Last selection times
    last_selections: HashMap<GpuDeviceId, Instant>,
    
    /// Selection counts for round-robin fallback
    selection_counts: HashMap<GpuDeviceId, u64>,
}

impl GpuLoadBalancer {
    pub fn new(devices: &[Arc<GpuDevice>]) -> Self {
        let mut device_loads = HashMap::new();
        let mut last_selections = HashMap::new();
        let mut selection_counts = HashMap::new();
        
        for device in devices {
            device_loads.insert(device.id(), 0.0);
            last_selections.insert(device.id(), Instant::now());
            selection_counts.insert(device.id(), 0);
        }
        
        Self {
            device_loads,
            last_selections,
            selection_counts,
        }
    }
    
    pub fn select_device(&mut self, workload: &WorkloadHint) -> GpuDeviceId {
        // Find device with lowest load
        let mut best_device = None;
        let mut lowest_load = f32::INFINITY;
        
        for (&device_id, &load) in &self.device_loads {
            // Adjust load based on workload type and recency
            let adjusted_load = self.calculate_adjusted_load(device_id, load, workload);
            
            if adjusted_load < lowest_load {
                lowest_load = adjusted_load;
                best_device = Some(device_id);
            }
        }
        
        let selected_device = best_device.unwrap_or(0);
        
        // Update selection tracking
        self.last_selections.insert(selected_device, Instant::now());
        *self.selection_counts.get_mut(&selected_device).unwrap() += 1;
        
        // Estimate new load based on workload
        let estimated_load_increase = self.estimate_load_increase(workload);
        *self.device_loads.get_mut(&selected_device).unwrap() += estimated_load_increase;
        
        selected_device
    }
    
    pub fn update_device_load(&mut self, device_id: GpuDeviceId, new_load: f32) {
        self.device_loads.insert(device_id, new_load.clamp(0.0, 1.0));
    }
    
    fn calculate_adjusted_load(&self, device_id: GpuDeviceId, base_load: f32, workload: &WorkloadHint) -> f32 {
        let mut adjusted_load = base_load;
        
        // Penalize recently used devices slightly
        if let Some(&last_used) = self.last_selections.get(&device_id) {
            let time_since_use = last_used.elapsed().as_millis() as f32;
            if time_since_use < 100.0 { // Less than 100ms ago
                adjusted_load += 0.1;
            }
        }
        
        // Adjust based on workload type
        match workload {
            WorkloadHint::MemoryIntensive { memory_requirement } => {
                // Prefer devices with more available memory
                // This would require actual memory info - simplified for now
                adjusted_load += (*memory_requirement as f32 / 1e9) * 0.1;
            },
            WorkloadHint::ComputeIntensive { expected_duration } => {
                // Prefer less loaded devices for long-running tasks
                adjusted_load += expected_duration.as_secs_f32() * 0.05;
            },
            _ => {}
        }
        
        adjusted_load
    }
    
    fn estimate_load_increase(&self, workload: &WorkloadHint) -> f32 {
        match workload {
            WorkloadHint::GraphTraversal { estimated_memory } => {
                (*estimated_memory as f32 / 1e9).min(0.3) // Max 30% load increase
            },
            WorkloadHint::LinearAlgebra { matrix_size } => {
                let elements = (matrix_size.0 * matrix_size.1) as f32;
                (elements / 1e9).min(0.5) // Max 50% load increase
            },
            WorkloadHint::BatchProcessing { batch_size } => {
                (*batch_size as f32 / 1e6).min(0.4) // Max 40% load increase
            },
            WorkloadHint::MemoryIntensive { memory_requirement } => {
                (*memory_requirement as f32 / 1e9).min(0.6) // Max 60% load increase
            },
            WorkloadHint::ComputeIntensive { expected_duration } => {
                expected_duration.as_secs_f32().min(0.8) // Max 80% load increase
            },
            _ => 0.1, // Default small increase
        }
    }
}

/// Device-specific metrics
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    /// Total operations performed
    pub operations_count: u64,
    
    /// Total execution time
    pub total_execution_time: Duration,
    
    /// Average execution time
    pub average_execution_time: Duration,
    
    /// Peak memory usage
    pub peak_memory_usage: usize,
    
    /// Current memory usage
    pub current_memory_usage: usize,
    
    /// Last update time
    pub last_update: Instant,
}

impl DeviceMetrics {
    pub fn new() -> Self {
        Self {
            operations_count: 0,
            total_execution_time: Duration::ZERO,
            average_execution_time: Duration::ZERO,
            peak_memory_usage: 0,
            current_memory_usage: 0,
            last_update: Instant::now(),
        }
    }
    
    pub fn update(&mut self, execution_time: Duration, memory_used: usize) {
        self.operations_count += 1;
        self.total_execution_time += execution_time;
        self.average_execution_time = self.total_execution_time / self.operations_count as u32;
        self.current_memory_usage = memory_used;
        self.peak_memory_usage = self.peak_memory_usage.max(memory_used);
        self.last_update = Instant::now();
    }
}

// Placeholder implementations for lower-level components

struct DeviceMemoryPool {
    device_id: GpuDeviceId,
    total_size: usize,
    used_size: usize,
    allocations: HashMap<*mut u8, usize>,
}

impl DeviceMemoryPool {
    fn new(device_id: GpuDeviceId, total_size: usize) -> Self {
        Self {
            device_id,
            total_size,
            used_size: 0,
            allocations: HashMap::new(),
        }
    }
    
    fn used_memory(&self) -> usize {
        self.used_size
    }
    
    fn total_memory(&self) -> usize {
        self.total_size
    }
    
    fn allocate(&mut self, size: usize) -> GpuResult<*mut u8> {
        if self.used_size + size > self.total_size {
            return Err(GpuKnowledgeGraphError::out_of_memory(
                format!("GPU {} out of memory: requested {}, available {}", 
                       self.device_id, size, self.total_size - self.used_size)
            ));
        }
        
        // TODO: Implement actual GPU memory allocation
        let ptr = Box::into_raw(vec![0u8; size].into_boxed_slice()) as *mut u8;
        self.allocations.insert(ptr, size);
        self.used_size += size;
        
        Ok(ptr)
    }
    
    fn free(&mut self, ptr: *mut u8) -> GpuResult<()> {
        if let Some(size) = self.allocations.remove(&ptr) {
            self.used_size -= size;
            // TODO: Implement actual GPU memory deallocation
            unsafe {
                let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, size));
            }
            Ok(())
        } else {
            Err(GpuKnowledgeGraphError::invalid_memory_address("Attempted to free invalid GPU memory"))
        }
    }
}

struct DevicePerformanceMonitor {
    kernel_launches: u64,
    total_kernel_time: Duration,
    last_activity: Instant,
}

impl DevicePerformanceMonitor {
    fn new() -> Self {
        Self {
            kernel_launches: 0,
            total_kernel_time: Duration::ZERO,
            last_activity: Instant::now(),
        }
    }
    
    fn record_kernel_launch(&mut self, duration: Duration) {
        self.kernel_launches += 1;
        self.total_kernel_time += duration;
        self.last_activity = Instant::now();
    }
    
    fn estimated_utilization(&self) -> f32 {
        // Simple estimation based on recent activity
        let time_since_activity = self.last_activity.elapsed().as_secs_f32();
        if time_since_activity < 1.0 {
            0.8 // High utilization if active recently
        } else if time_since_activity < 5.0 {
            0.3 // Medium utilization
        } else {
            0.1 // Low utilization
        }
    }
}

// CUDA runtime functions

/// Initialize CUDA runtime
pub fn init_cuda_runtime() -> GpuResult<()> {
    // TODO: Initialize CUDA runtime with proper error handling
    tracing::debug!("Initializing CUDA runtime");
    Ok(())
}

/// Get number of CUDA devices
pub fn get_device_count() -> GpuResult<i32> {
    // TODO: Implement actual device count query
    Ok(1) // Placeholder - assume 1 device
}

/// Get total GPU count across all devices
pub fn get_gpu_count() -> GpuResult<usize> {
    get_device_count().map(|count| count as usize)
}

/// Warm up basic GPU operations
pub fn warm_up_basic_ops() -> GpuResult<()> {
    tracing::debug!("Warming up basic GPU operations");
    // TODO: Implement GPU warmup operations
    Ok(())
}

/// Synchronize specific GPU device
pub fn synchronize_device() -> GpuResult<()> {
    // TODO: Implement device synchronization
    Ok(())
}

/// Synchronize specific CUDA stream
pub fn synchronize_stream(_stream: CudaStream) -> GpuResult<()> {
    // TODO: Implement stream synchronization
    Ok(())
}

// Helper functions

fn query_device_info(cuda_device: &CudaDevice, device_id: GpuDeviceId) -> GpuResult<GpuDeviceInfo> {
    // TODO: Query actual device properties
    Ok(GpuDeviceInfo {
        device_id,
        name: format!("CUDA Device {}", device_id),
        compute_capability: (8, 0), // Placeholder
        total_memory: 24 * 1024 * 1024 * 1024, // 24GB placeholder
        max_threads_per_block: 1024,
        max_blocks_per_grid: (65535, 65535, 65535),
        warp_size: 32,
        shared_memory_per_block: 48 * 1024, // 48KB
        multiprocessor_count: 108, // RTX 4090 placeholder
        clock_rate: 2520000, // 2.52 GHz
        memory_clock_rate: 10501000, // ~21 Gbps effective
        memory_bus_width: 384, // bits
        l2_cache_size: 6 * 1024 * 1024, // 6MB
    })
}

fn validate_device_requirements(info: &GpuDeviceInfo) -> GpuResult<()> {
    // Check minimum compute capability
    if info.compute_capability < crate::device_info::MIN_COMPUTE_CAPABILITY {
        return Err(GpuKnowledgeGraphError::unsupported_gpu(
            format!("GPU {} has compute capability {}.{}, minimum required is {}.{}",
                   info.device_id,
                   info.compute_capability.0, info.compute_capability.1,
                   crate::device_info::MIN_COMPUTE_CAPABILITY.0,
                   crate::device_info::MIN_COMPUTE_CAPABILITY.1)
        ));
    }
    
    // Check minimum memory
    if info.total_memory < crate::device_info::MIN_GPU_MEMORY {
        return Err(GpuKnowledgeGraphError::insufficient_gpu_memory(
            format!("GPU {} has {}GB memory, minimum required is {}GB",
                   info.device_id,
                   info.total_memory / (1024 * 1024 * 1024),
                   crate::device_info::MIN_GPU_MEMORY / (1024 * 1024 * 1024))
        ));
    }
    
    Ok(())
}

fn create_cuda_stream(cuda_device: &CudaDevice) -> GpuResult<CudaStream> {
    // TODO: Create actual CUDA stream
    static STREAM_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    Ok(STREAM_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_manager_creation() {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_err() {
            return; // Skip if no GPU
        }
        
        let config = crate::graph::GpuGraphConfig::default();
        match GpuManager::new(&config).await {
            Ok(manager) => {
                assert!(manager.get_device_count() > 0);
                let devices = manager.get_available_devices();
                assert!(!devices.is_empty());
            },
            Err(e) => {
                println!("GPU manager creation failed (expected if no GPU): {}", e);
            }
        }
    }
    
    #[test]
    fn test_load_balancer() {
        let devices = vec![];
        let mut balancer = GpuLoadBalancer::new(&devices);
        
        // Test with mock device IDs
        balancer.device_loads.insert(0, 0.3);
        balancer.device_loads.insert(1, 0.7);
        balancer.last_selections.insert(0, Instant::now());
        balancer.last_selections.insert(1, Instant::now());
        balancer.selection_counts.insert(0, 0);
        balancer.selection_counts.insert(1, 0);
        
        let workload = WorkloadHint::GraphTraversal { estimated_memory: 1024 * 1024 };
        let selected = balancer.select_device(&workload);
        
        // Should select device with lower load (0)
        assert_eq!(selected, 0);
    }
    
    #[test]
    fn test_device_metrics() {
        let mut metrics = DeviceMetrics::new();
        assert_eq!(metrics.operations_count, 0);
        
        metrics.update(Duration::from_millis(10), 1024);
        assert_eq!(metrics.operations_count, 1);
        assert_eq!(metrics.current_memory_usage, 1024);
        assert_eq!(metrics.peak_memory_usage, 1024);
    }
    
    #[test]
    fn test_workload_hints() {
        let workload = WorkloadHint::MemoryIntensive { memory_requirement: 2 * 1024 * 1024 * 1024 };
        
        match workload {
            WorkloadHint::MemoryIntensive { memory_requirement } => {
                assert_eq!(memory_requirement, 2 * 1024 * 1024 * 1024);
            },
            _ => panic!("Wrong workload type"),
        }
    }
}