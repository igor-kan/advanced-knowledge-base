//! Error handling for GPU-accelerated knowledge graph
//!
//! This module provides comprehensive error handling for GPU operations,
//! CUDA runtime errors, and GPU memory management failures.

use std::fmt;

use thiserror::Error;

/// Main error type for GPU-accelerated knowledge graph operations
#[derive(Error, Debug)]
pub enum GpuKnowledgeGraphError {
    /// CUDA runtime errors
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    /// GPU device initialization errors
    #[error("GPU initialization failed: {0}")]
    GpuInitialization(String),
    
    /// GPU device not found
    #[error("GPU device {0} not found")]
    DeviceNotFound(i32),
    
    /// Unsupported GPU hardware
    #[error("Unsupported GPU: {0}")]
    UnsupportedGpu(String),
    
    /// Insufficient GPU memory
    #[error("Insufficient GPU memory: {0}")]
    InsufficientGpuMemory(String),
    
    /// GPU memory allocation failed
    #[error("GPU memory allocation failed: {0}")]
    GpuMemoryAllocation(String),
    
    /// Invalid GPU memory address
    #[error("Invalid GPU memory address: {0}")]
    InvalidMemoryAddress(String),
    
    /// GPU out of memory
    #[error("GPU out of memory: {0}")]
    OutOfMemory(String),
    
    /// CUDA kernel launch failed
    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunch(String),
    
    /// CUDA kernel execution failed
    #[error("CUDA kernel execution failed: {0}")]
    KernelExecution(String),
    
    /// GPU memory transfer failed
    #[error("GPU memory transfer failed: {0}")]
    MemoryTransfer(String),
    
    /// GPU synchronization failed
    #[error("GPU synchronization failed: {0}")]
    Synchronization(String),
    
    /// cuBLAS library error
    #[error("cuBLAS error: {0}")]
    CuBlas(String),
    
    /// cuSPARSE library error
    #[error("cuSPARSE error: {0}")]
    CuSparse(String),
    
    /// cuRAND library error
    #[error("cuRAND error: {0}")]
    CuRand(String),
    
    /// cuGraph library error
    #[error("cuGraph error: {0}")]
    CuGraph(String),
    
    /// Multi-GPU communication error
    #[error("Multi-GPU communication error: {0}")]
    MultiGpuCommunication(String),
    
    /// NCCL (NVIDIA Collective Communication Library) error
    #[error("NCCL error: {0}")]
    Nccl(String),
    
    /// GPU algorithm error
    #[error("GPU algorithm error: {0}")]
    GpuAlgorithm(String),
    
    /// GPU-CPU data synchronization error
    #[error("GPU-CPU synchronization error: {0}")]
    GpuCpuSync(String),
    
    /// Unified memory error
    #[error("Unified memory error: {0}")]
    UnifiedMemory(String),
    
    /// GPU driver error
    #[error("GPU driver error: {0}")]
    GpuDriver(String),
    
    /// GPU context error
    #[error("GPU context error: {0}")]
    GpuContext(String),
    
    /// GPU stream error
    #[error("GPU stream error: {0}")]
    GpuStream(String),
    
    /// GPU event error
    #[error("GPU event error: {0}")]
    GpuEvent(String),
    
    /// GPU profiling error
    #[error("GPU profiling error: {0}")]
    GpuProfiling(String),
    
    /// GPU performance degradation
    #[error("GPU performance degradation detected: {0}")]
    PerformanceDegradation(String),
    
    /// GPU thermal throttling
    #[error("GPU thermal throttling: {0}")]
    ThermalThrottling(String),
    
    /// GPU power limit exceeded
    #[error("GPU power limit exceeded: {0}")]
    PowerLimitExceeded(String),
    
    /// I/O errors from file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Thread synchronization errors
    #[error("Thread synchronization error: {0}")]
    ThreadSync(String),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Internal implementation errors
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Feature not implemented for GPU
    #[error("Feature not implemented for GPU: {0}")]
    NotImplementedGpu(String),
    
    /// Feature not available on this GPU
    #[error("Feature not available on this GPU: {0}")]
    NotAvailableGpu(String),
    
    /// Invalid GPU operation
    #[error("Invalid GPU operation: {0}")]
    InvalidGpuOperation(String),
    
    /// GPU resource limit exceeded
    #[error("GPU resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    /// GPU timeout error
    #[error("GPU operation timed out: {0}")]
    GpuTimeout(String),
    
    /// Data corruption detected on GPU
    #[error("GPU data corruption detected: {0}")]
    GpuDataCorruption(String),
    
    /// GPU hardware failure
    #[error("GPU hardware failure: {0}")]
    HardwareFailure(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Network error (for distributed GPU operations)
    #[error("Network error: {0}")]
    Network(String),
    
    /// Generic error with GPU context
    #[error("GPU Error: {message} (context: {context})")]
    GenericGpu {
        message: String,
        context: String,
    },
}

/// Result type for GPU knowledge graph operations
pub type GpuResult<T> = std::result::Result<T, GpuKnowledgeGraphError>;

impl GpuKnowledgeGraphError {
    /// Create a CUDA error
    pub fn cuda_error(message: impl Into<String>) -> Self {
        Self::Cuda(message.into())
    }
    
    /// Create a GPU initialization error
    pub fn gpu_initialization(message: impl Into<String>) -> Self {
        Self::GpuInitialization(message.into())
    }
    
    /// Create a device not found error
    pub fn device_not_found(device_id: i32) -> Self {
        Self::DeviceNotFound(device_id)
    }
    
    /// Create an unsupported GPU error
    pub fn unsupported_gpu(message: impl Into<String>) -> Self {
        Self::UnsupportedGpu(message.into())
    }
    
    /// Create an insufficient GPU memory error
    pub fn insufficient_gpu_memory(message: impl Into<String>) -> Self {
        Self::InsufficientGpuMemory(message.into())
    }
    
    /// Create a GPU memory allocation error
    pub fn gpu_memory_allocation(message: impl Into<String>) -> Self {
        Self::GpuMemoryAllocation(message.into())
    }
    
    /// Create an invalid memory address error
    pub fn invalid_memory_address(message: impl Into<String>) -> Self {
        Self::InvalidMemoryAddress(message.into())
    }
    
    /// Create an out of memory error
    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Self::OutOfMemory(message.into())
    }
    
    /// Create a kernel launch error
    pub fn kernel_launch(message: impl Into<String>) -> Self {
        Self::KernelLaunch(message.into())
    }
    
    /// Create a kernel execution error
    pub fn kernel_execution(message: impl Into<String>) -> Self {
        Self::KernelExecution(message.into())
    }
    
    /// Create a memory transfer error
    pub fn memory_transfer(message: impl Into<String>) -> Self {
        Self::MemoryTransfer(message.into())
    }
    
    /// Create a synchronization error
    pub fn synchronization_error(message: impl Into<String>) -> Self {
        Self::Synchronization(message.into())
    }
    
    /// Create a cuBLAS error
    pub fn cublas_error(message: impl Into<String>) -> Self {
        Self::CuBlas(message.into())
    }
    
    /// Create a cuSPARSE error
    pub fn cusparse_error(message: impl Into<String>) -> Self {
        Self::CuSparse(message.into())
    }
    
    /// Create a cuRAND error
    pub fn curand_error(message: impl Into<String>) -> Self {
        Self::CuRand(message.into())
    }
    
    /// Create a cuGraph error
    pub fn cugraph_error(message: impl Into<String>) -> Self {
        Self::CuGraph(message.into())
    }
    
    /// Create a multi-GPU communication error
    pub fn multi_gpu_communication(message: impl Into<String>) -> Self {
        Self::MultiGpuCommunication(message.into())
    }
    
    /// Create an NCCL error
    pub fn nccl_error(message: impl Into<String>) -> Self {
        Self::Nccl(message.into())
    }
    
    /// Create a GPU algorithm error
    pub fn gpu_algorithm_error(message: impl Into<String>) -> Self {
        Self::GpuAlgorithm(message.into())
    }
    
    /// Create a GPU-CPU synchronization error
    pub fn gpu_cpu_sync_error(message: impl Into<String>) -> Self {
        Self::GpuCpuSync(message.into())
    }
    
    /// Create a unified memory error
    pub fn unified_memory_error(message: impl Into<String>) -> Self {
        Self::UnifiedMemory(message.into())
    }
    
    /// Create a GPU driver error
    pub fn gpu_driver_error(message: impl Into<String>) -> Self {
        Self::GpuDriver(message.into())
    }
    
    /// Create a GPU context error
    pub fn gpu_context_error(message: impl Into<String>) -> Self {
        Self::GpuContext(message.into())
    }
    
    /// Create a GPU stream error
    pub fn gpu_stream_error(message: impl Into<String>) -> Self {
        Self::GpuStream(message.into())
    }
    
    /// Create a GPU event error
    pub fn gpu_event_error(message: impl Into<String>) -> Self {
        Self::GpuEvent(message.into())
    }
    
    /// Create a GPU profiling error
    pub fn gpu_profiling_error(message: impl Into<String>) -> Self {
        Self::GpuProfiling(message.into())
    }
    
    /// Create a performance degradation error
    pub fn performance_degradation(message: impl Into<String>) -> Self {
        Self::PerformanceDegradation(message.into())
    }
    
    /// Create a thermal throttling error
    pub fn thermal_throttling(message: impl Into<String>) -> Self {
        Self::ThermalThrottling(message.into())
    }
    
    /// Create a power limit exceeded error
    pub fn power_limit_exceeded(message: impl Into<String>) -> Self {
        Self::PowerLimitExceeded(message.into())
    }
    
    /// Create a thread synchronization error
    pub fn thread_sync_error(message: impl Into<String>) -> Self {
        Self::ThreadSync(message.into())
    }
    
    /// Create a configuration error
    pub fn configuration_error(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }
    
    /// Create an internal error
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
    
    /// Create a not implemented error
    pub fn not_implemented_gpu(message: impl Into<String>) -> Self {
        Self::NotImplementedGpu(message.into())
    }
    
    /// Create a not available error
    pub fn not_available_gpu(message: impl Into<String>) -> Self {
        Self::NotAvailableGpu(message.into())
    }
    
    /// Create an invalid operation error
    pub fn invalid_gpu_operation(message: impl Into<String>) -> Self {
        Self::InvalidGpuOperation(message.into())
    }
    
    /// Create a resource limit exceeded error
    pub fn resource_limit_exceeded(message: impl Into<String>) -> Self {
        Self::ResourceLimitExceeded(message.into())
    }
    
    /// Create a GPU timeout error
    pub fn gpu_timeout(message: impl Into<String>) -> Self {
        Self::GpuTimeout(message.into())
    }
    
    /// Create a GPU data corruption error
    pub fn gpu_data_corruption(message: impl Into<String>) -> Self {
        Self::GpuDataCorruption(message.into())
    }
    
    /// Create a hardware failure error
    pub fn hardware_failure(message: impl Into<String>) -> Self {
        Self::HardwareFailure(message.into())
    }
    
    /// Create a network error
    pub fn network_error(message: impl Into<String>) -> Self {
        Self::Network(message.into())
    }
    
    /// Create a generic GPU error with context
    pub fn generic_gpu(message: impl Into<String>, context: impl Into<String>) -> Self {
        Self::GenericGpu {
            message: message.into(),
            context: context.into(),
        }
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Non-recoverable hardware/driver errors
            Self::HardwareFailure(_) |
            Self::GpuDriver(_) |
            Self::UnsupportedGpu(_) |
            Self::GpuDataCorruption(_) => false,
            
            // Memory errors that might be recoverable with cleanup
            Self::OutOfMemory(_) |
            Self::InsufficientGpuMemory(_) |
            Self::GpuMemoryAllocation(_) => true,
            
            // Timeout and synchronization errors are usually recoverable
            Self::GpuTimeout(_) |
            Self::Synchronization(_) |
            Self::ThreadSync(_) => true,
            
            // Performance issues might be recoverable
            Self::PerformanceDegradation(_) |
            Self::ThermalThrottling(_) |
            Self::PowerLimitExceeded(_) => true,
            
            // Network errors for distributed operations
            Self::Network(_) |
            Self::MultiGpuCommunication(_) => true,
            
            // Context-dependent
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> GpuErrorSeverity {
        match self {
            Self::HardwareFailure(_) |
            Self::GpuDataCorruption(_) |
            Self::GpuDriver(_) => GpuErrorSeverity::Critical,
            
            Self::OutOfMemory(_) |
            Self::InsufficientGpuMemory(_) |
            Self::UnsupportedGpu(_) => GpuErrorSeverity::Error,
            
            Self::PerformanceDegradation(_) |
            Self::ThermalThrottling(_) |
            Self::PowerLimitExceeded(_) => GpuErrorSeverity::Warning,
            
            Self::NotImplementedGpu(_) |
            Self::NotAvailableGpu(_) => GpuErrorSeverity::Info,
            
            _ => GpuErrorSeverity::Error,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> GpuErrorCategory {
        match self {
            Self::Cuda(_) | Self::GpuDriver(_) | Self::GpuContext(_) => GpuErrorCategory::Driver,
            
            Self::GpuMemoryAllocation(_) | Self::OutOfMemory(_) | 
            Self::InsufficientGpuMemory(_) | Self::InvalidMemoryAddress(_) |
            Self::UnifiedMemory(_) => GpuErrorCategory::Memory,
            
            Self::KernelLaunch(_) | Self::KernelExecution(_) => GpuErrorCategory::Kernel,
            
            Self::CuBlas(_) | Self::CuSparse(_) | Self::CuRand(_) | 
            Self::CuGraph(_) => GpuErrorCategory::Library,
            
            Self::MultiGpuCommunication(_) | Self::Nccl(_) => GpuErrorCategory::Communication,
            
            Self::PerformanceDegradation(_) | Self::ThermalThrottling(_) |
            Self::PowerLimitExceeded(_) => GpuErrorCategory::Performance,
            
            Self::HardwareFailure(_) => GpuErrorCategory::Hardware,
            
            Self::GpuAlgorithm(_) => GpuErrorCategory::Algorithm,
            
            Self::Configuration(_) => GpuErrorCategory::Configuration,
            
            Self::Network(_) => GpuErrorCategory::Network,
            
            _ => GpuErrorCategory::Other,
        }
    }
    
    /// Get suggested recovery action
    pub fn suggested_recovery(&self) -> Option<String> {
        match self {
            Self::OutOfMemory(_) | Self::InsufficientGpuMemory(_) => {
                Some("Try reducing batch size or enabling memory optimization".to_string())
            },
            Self::ThermalThrottling(_) => {
                Some("Reduce GPU load or improve cooling".to_string())
            },
            Self::PowerLimitExceeded(_) => {
                Some("Reduce GPU power target or improve power supply".to_string())
            },
            Self::PerformanceDegradation(_) => {
                Some("Check GPU utilization and consider load balancing".to_string())
            },
            Self::MultiGpuCommunication(_) => {
                Some("Check network connectivity between GPUs".to_string())
            },
            Self::GpuTimeout(_) => {
                Some("Increase timeout or split operation into smaller chunks".to_string())
            },
            _ => None,
        }
    }
}

/// GPU error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuErrorSeverity {
    /// Informational messages
    Info,
    /// Warning conditions
    Warning,
    /// Error conditions
    Error,
    /// Critical conditions requiring immediate attention
    Critical,
}

impl fmt::Display for GpuErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// GPU error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuErrorCategory {
    /// GPU driver and runtime errors
    Driver,
    /// GPU memory management errors
    Memory,
    /// CUDA kernel errors
    Kernel,
    /// GPU library errors (cuBLAS, cuSPARSE, etc.)
    Library,
    /// Multi-GPU communication errors
    Communication,
    /// Performance-related errors
    Performance,
    /// Hardware failure errors
    Hardware,
    /// Algorithm implementation errors
    Algorithm,
    /// Configuration errors
    Configuration,
    /// Network errors
    Network,
    /// Other errors
    Other,
}

impl fmt::Display for GpuErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver => write!(f, "DRIVER"),
            Self::Memory => write!(f, "MEMORY"),
            Self::Kernel => write!(f, "KERNEL"),
            Self::Library => write!(f, "LIBRARY"),
            Self::Communication => write!(f, "COMMUNICATION"),
            Self::Performance => write!(f, "PERFORMANCE"),
            Self::Hardware => write!(f, "HARDWARE"),
            Self::Algorithm => write!(f, "ALGORITHM"),
            Self::Configuration => write!(f, "CONFIG"),
            Self::Network => write!(f, "NETWORK"),
            Self::Other => write!(f, "OTHER"),
        }
    }
}

/// GPU error context for debugging
#[derive(Debug, Clone)]
pub struct GpuErrorContext {
    /// GPU device ID where error occurred
    pub device_id: Option<i32>,
    
    /// CUDA stream ID if applicable
    pub stream_id: Option<usize>,
    
    /// Operation being performed
    pub operation: String,
    
    /// Component that generated the error
    pub component: String,
    
    /// Thread ID where error occurred
    pub thread_id: Option<std::thread::ThreadId>,
    
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    
    /// Additional GPU-specific metadata
    pub gpu_metadata: std::collections::HashMap<String, String>,
}

impl GpuErrorContext {
    /// Create a new GPU error context
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            device_id: None,
            stream_id: None,
            operation: operation.into(),
            component: component.into(),
            thread_id: Some(std::thread::current().id()),
            timestamp: std::time::SystemTime::now(),
            gpu_metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add GPU device context
    pub fn with_device(mut self, device_id: i32) -> Self {
        self.device_id = Some(device_id);
        self
    }
    
    /// Add CUDA stream context
    pub fn with_stream(mut self, stream_id: usize) -> Self {
        self.stream_id = Some(stream_id);
        self
    }
    
    /// Add GPU metadata
    pub fn with_gpu_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.gpu_metadata.insert(key.into(), value.into());
        self
    }
}

/// Extension trait for adding GPU context to results
pub trait GpuResultExt<T> {
    /// Add GPU context to an error
    fn with_gpu_context(
        self, 
        operation: impl Into<String>, 
        component: impl Into<String>
    ) -> GpuResult<T>;
    
    /// Add GPU device context
    fn with_gpu_device(self, device_id: i32) -> GpuResult<T>;
    
    /// Add CUDA stream context
    fn with_cuda_stream(self, stream_id: usize) -> GpuResult<T>;
    
    /// Add simple GPU message
    fn with_gpu_message(self, message: impl Into<String>) -> GpuResult<T>;
}

impl<T> GpuResultExt<T> for GpuResult<T> {
    fn with_gpu_context(
        self, 
        operation: impl Into<String>, 
        component: impl Into<String>
    ) -> GpuResult<T> {
        self.map_err(|err| {
            let context = GpuErrorContext::new(operation, component);
            GpuKnowledgeGraphError::generic_gpu(
                err.to_string(),
                format!("{}::{} (thread: {:?})", 
                       context.component, context.operation, context.thread_id)
            )
        })
    }
    
    fn with_gpu_device(self, device_id: i32) -> GpuResult<T> {
        self.map_err(|err| {
            GpuKnowledgeGraphError::generic_gpu(
                err.to_string(),
                format!("GPU device {}", device_id)
            )
        })
    }
    
    fn with_cuda_stream(self, stream_id: usize) -> GpuResult<T> {
        self.map_err(|err| {
            GpuKnowledgeGraphError::generic_gpu(
                err.to_string(),
                format!("CUDA stream {}", stream_id)
            )
        })
    }
    
    fn with_gpu_message(self, message: impl Into<String>) -> GpuResult<T> {
        self.map_err(|err| {
            GpuKnowledgeGraphError::generic_gpu(err.to_string(), message.into())
        })
    }
}

/// Macro for creating GPU context-aware errors
#[macro_export]
macro_rules! gpu_error {
    ($kind:ident, $msg:expr) => {
        $crate::error::GpuKnowledgeGraphError::$kind($msg.into())
    };
    ($kind:ident, $fmt:expr, $($arg:tt)*) => {
        $crate::error::GpuKnowledgeGraphError::$kind(format!($fmt, $($arg)*))
    };
}

/// Macro for ensuring GPU conditions with custom errors
#[macro_export]
macro_rules! gpu_ensure {
    ($cond:expr, $err:expr) => {
        if !($cond) {
            return Err($err);
        }
    };
}

/// Macro for early return with GPU error context
#[macro_export]
macro_rules! gpu_bail {
    ($err:expr) => {
        return Err($err)
    };
    ($kind:ident, $msg:expr) => {
        return Err($crate::error::GpuKnowledgeGraphError::$kind($msg.into()))
    };
    ($kind:ident, $fmt:expr, $($arg:tt)*) => {
        return Err($crate::error::GpuKnowledgeGraphError::$kind(format!($fmt, $($arg)*)))
    };
}

// Convert from common error types
impl From<rayon::ThreadPoolBuildError> for GpuKnowledgeGraphError {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        Self::thread_sync_error(format!("Thread pool build error: {}", err))
    }
}

impl<T> From<std::sync::PoisonError<T>> for GpuKnowledgeGraphError {
    fn from(err: std::sync::PoisonError<T>) -> Self {
        Self::thread_sync_error(format!("Mutex poison error: {}", err))
    }
}

impl From<std::num::TryFromIntError> for GpuKnowledgeGraphError {
    fn from(err: std::num::TryFromIntError) -> Self {
        Self::internal_error(format!("Integer conversion error: {}", err))
    }
}

impl From<bincode::Error> for GpuKnowledgeGraphError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization(serde_json::Error::custom(format!("Bincode error: {}", err)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_error_creation() {
        let err = GpuKnowledgeGraphError::cuda_error("CUDA initialization failed");
        assert_eq!(err.to_string(), "CUDA error: CUDA initialization failed");
        assert_eq!(err.severity(), GpuErrorSeverity::Error);
        assert_eq!(err.category(), GpuErrorCategory::Driver);
    }
    
    #[test]
    fn test_gpu_error_recovery() {
        let memory_err = GpuKnowledgeGraphError::out_of_memory("GPU out of memory");
        assert!(memory_err.is_recoverable());
        assert!(memory_err.suggested_recovery().is_some());
        
        let hardware_err = GpuKnowledgeGraphError::hardware_failure("GPU fan failure");
        assert!(!hardware_err.is_recoverable());
    }
    
    #[test]
    fn test_gpu_error_context() {
        let context = GpuErrorContext::new("kernel_launch", "gpu_algorithms")
            .with_device(0)
            .with_stream(1)
            .with_gpu_metadata("block_size", "256")
            .with_gpu_metadata("grid_size", "1024");
        
        assert_eq!(context.operation, "kernel_launch");
        assert_eq!(context.component, "gpu_algorithms");
        assert_eq!(context.device_id, Some(0));
        assert_eq!(context.stream_id, Some(1));
        assert_eq!(context.gpu_metadata.get("block_size"), Some(&"256".to_string()));
    }
    
    #[test]
    fn test_gpu_error_macros() {
        let err = gpu_error!(cuda_error, "Test CUDA error");
        assert!(matches!(err, GpuKnowledgeGraphError::Cuda(_)));
        
        let err = gpu_error!(out_of_memory, "Out of memory: {} bytes", 1024);
        assert!(matches!(err, GpuKnowledgeGraphError::OutOfMemory(_)));
        assert!(err.to_string().contains("1024"));
    }
    
    #[test]
    fn test_gpu_result_extension() {
        let result: GpuResult<i32> = Err(GpuKnowledgeGraphError::cuda_error("Test error"));
        let result_with_context = result.with_gpu_context("test_op", "test_component");
        
        assert!(result_with_context.is_err());
        let err = result_with_context.unwrap_err();
        assert!(matches!(err, GpuKnowledgeGraphError::GenericGpu { .. }));
    }
    
    #[test]
    fn test_error_severity_ordering() {
        assert!(GpuErrorSeverity::Critical > GpuErrorSeverity::Error);
        assert!(GpuErrorSeverity::Error > GpuErrorSeverity::Warning);
        assert!(GpuErrorSeverity::Warning > GpuErrorSeverity::Info);
    }
    
    #[test]
    fn test_error_categories() {
        let categories = [
            GpuErrorCategory::Driver,
            GpuErrorCategory::Memory,
            GpuErrorCategory::Kernel,
            GpuErrorCategory::Library,
            GpuErrorCategory::Communication,
            GpuErrorCategory::Performance,
            GpuErrorCategory::Hardware,
            GpuErrorCategory::Algorithm,
            GpuErrorCategory::Configuration,
            GpuErrorCategory::Network,
            GpuErrorCategory::Other,
        ];
        
        for category in &categories {
            assert!(!category.to_string().is_empty());
        }
    }
}