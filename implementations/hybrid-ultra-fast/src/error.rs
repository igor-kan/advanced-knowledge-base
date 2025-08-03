//! Error handling for hybrid knowledge graph
//!
//! This module provides comprehensive error handling across all hybrid components.

use std::fmt;
use std::sync::PoisonError;

use thiserror::Error;

/// Main error type for hybrid knowledge graph operations
#[derive(Error, Debug)]
pub enum HybridError {
    /// I/O errors from file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Memory allocation errors
    #[error("Memory allocation failed: {0}")]
    Memory(String),
    
    /// SIMD operation errors
    #[error("SIMD operation failed: {0}")]
    Simd(String),
    
    /// FFI bridge errors
    #[error("FFI bridge error: {0}")]
    Ffi(String),
    
    /// C++ component errors
    #[error("C++ component error: {0}")]
    Cpp(String),
    
    /// Assembly kernel errors
    #[error("Assembly kernel error: {0}")]
    Assembly(String),
    
    /// Node not found
    #[error("Node not found: {0}")]
    NodeNotFound(u64),
    
    /// Edge not found
    #[error("Edge not found: {0}")]
    EdgeNotFound(u64),
    
    /// Invalid node ID
    #[error("Invalid node ID: {0}")]
    InvalidNodeId(u64),
    
    /// Invalid edge ID
    #[error("Invalid edge ID: {0}")]
    InvalidEdgeId(u64),
    
    /// Graph is empty
    #[error("Graph is empty")]
    EmptyGraph,
    
    /// Concurrent access error
    #[error("Concurrent access error: {0}")]
    ConcurrentAccess(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),
    
    /// Algorithm error
    #[error("Algorithm error: {0}")]
    Algorithm(String),
    
    /// Query error
    #[error("Query error: {0}")]
    Query(String),
    
    /// Pattern matching error
    #[error("Pattern matching error: {0}")]
    PatternMatching(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),
    
    /// Network error (for distributed operations)
    #[error("Network error: {0}")]
    Network(String),
    
    /// GPU error (for CUDA operations)
    #[cfg(feature = "gpu")]
    #[error("GPU error: {0}")]
    Gpu(String),
    
    /// Distributed system error
    #[cfg(feature = "distributed")]
    #[error("Distributed system error: {0}")]
    Distributed(String),
    
    /// Thread synchronization error
    #[error("Thread synchronization error: {0}")]
    Synchronization(String),
    
    /// Resource exhaustion
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),
    
    /// Capacity exceeded
    #[error("Capacity exceeded: {0}")]
    CapacityExceeded(String),
    
    /// Data corruption detected
    #[error("Data corruption detected: {0}")]
    DataCorruption(String),
    
    /// Feature not implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
    
    /// Feature not available
    #[error("Feature not available: {0}")]
    NotAvailable(String),
    
    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// Property type mismatch
    #[error("Property type mismatch: expected {expected}, got {actual}")]
    PropertyTypeMismatch {
        expected: String,
        actual: String,
    },
    
    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {size}")]
    IndexOutOfBounds {
        index: usize,
        size: usize,
    },
    
    /// Generic error with context
    #[error("Error: {message} (context: {context})")]
    Generic {
        message: String,
        context: String,
    },
}

/// Result type for hybrid knowledge graph operations
pub type HybridResult<T> = std::result::Result<T, HybridError>;

impl HybridError {
    /// Create a memory allocation error
    pub fn memory_allocation(message: impl Into<String>) -> Self {
        Self::Memory(message.into())
    }
    
    /// Create a SIMD operation error
    pub fn simd_operation(message: impl Into<String>) -> Self {
        Self::Simd(message.into())
    }
    
    /// Create an FFI bridge error
    pub fn ffi_bridge(message: impl Into<String>) -> Self {
        Self::Ffi(message.into())
    }
    
    /// Create a C++ component error
    pub fn cpp_component(message: impl Into<String>) -> Self {
        Self::Cpp(message.into())
    }
    
    /// Create an assembly kernel error
    pub fn assembly_kernel(message: impl Into<String>) -> Self {
        Self::Assembly(message.into())
    }
    
    /// Create a concurrent access error
    pub fn concurrent_access(message: impl Into<String>) -> Self {
        Self::ConcurrentAccess(message.into())
    }
    
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }
    
    /// Create a storage error
    pub fn storage_error(message: impl Into<String>) -> Self {
        Self::Storage(message.into())
    }
    
    /// Create an algorithm error
    pub fn algorithm_error(message: impl Into<String>) -> Self {
        Self::Algorithm(message.into())
    }
    
    /// Create a query error
    pub fn query_error(message: impl Into<String>) -> Self {
        Self::Query(message.into())
    }
    
    /// Create a pattern matching error
    pub fn pattern_matching(message: impl Into<String>) -> Self {
        Self::PatternMatching(message.into())
    }
    
    /// Create a network error
    pub fn network_error(message: impl Into<String>) -> Self {
        Self::Network(message.into())
    }
    
    /// Create a GPU error
    #[cfg(feature = "gpu")]
    pub fn gpu_error(message: impl Into<String>) -> Self {
        Self::Gpu(message.into())
    }
    
    /// Create a distributed system error
    #[cfg(feature = "distributed")]
    pub fn distributed_error(message: impl Into<String>) -> Self {
        Self::Distributed(message.into())
    }
    
    /// Create a synchronization error
    pub fn synchronization_error(message: impl Into<String>) -> Self {
        Self::Synchronization(message.into())
    }
    
    /// Create a resource exhausted error
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::ResourceExhausted(message.into())
    }
    
    /// Create a timeout error
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout(message.into())
    }
    
    /// Create a capacity exceeded error
    pub fn capacity_exceeded(message: impl Into<String>) -> Self {
        Self::CapacityExceeded(message.into())
    }
    
    /// Create a data corruption error
    pub fn data_corruption(message: impl Into<String>) -> Self {
        Self::DataCorruption(message.into())
    }
    
    /// Create a not implemented error
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::NotImplemented(message.into())
    }
    
    /// Create a not available error
    pub fn not_available(message: impl Into<String>) -> Self {
        Self::NotAvailable(message.into())
    }
    
    /// Create an invalid operation error
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation(message.into())
    }
    
    /// Create a property type mismatch error
    pub fn property_type_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::PropertyTypeMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
    
    /// Create an index out of bounds error
    pub fn index_out_of_bounds(index: usize, size: usize) -> Self {
        Self::IndexOutOfBounds { index, size }
    }
    
    /// Create a generic error with context
    pub fn generic(message: impl Into<String>, context: impl Into<String>) -> Self {
        Self::Generic {
            message: message.into(),
            context: context.into(),
        }
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Non-recoverable errors
            Self::Memory(_) |
            Self::DataCorruption(_) |
            Self::ResourceExhausted(_) |
            Self::CapacityExceeded(_) => false,
            
            // Potentially recoverable errors
            Self::Timeout(_) |
            Self::Network(_) |
            Self::ConcurrentAccess(_) |
            Self::Synchronization(_) => true,
            
            // Context-dependent
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Memory(_) |
            Self::DataCorruption(_) |
            Self::ResourceExhausted(_) => ErrorSeverity::Critical,
            
            Self::NodeNotFound(_) |
            Self::EdgeNotFound(_) |
            Self::InvalidNodeId(_) |
            Self::InvalidEdgeId(_) |
            Self::PropertyTypeMismatch { .. } |
            Self::IndexOutOfBounds { .. } => ErrorSeverity::Warning,
            
            Self::NotImplemented(_) |
            Self::NotAvailable(_) => ErrorSeverity::Info,
            
            _ => ErrorSeverity::Error,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Io(_) => ErrorCategory::System,
            Self::Memory(_) => ErrorCategory::Memory,
            Self::Simd(_) | Self::Assembly(_) => ErrorCategory::Performance,
            Self::Ffi(_) | Self::Cpp(_) => ErrorCategory::Bridge,
            Self::NodeNotFound(_) | Self::EdgeNotFound(_) | 
            Self::InvalidNodeId(_) | Self::InvalidEdgeId(_) => ErrorCategory::Data,
            Self::ConcurrentAccess(_) | Self::Synchronization(_) => ErrorCategory::Concurrency,
            Self::Configuration(_) => ErrorCategory::Configuration,
            Self::Storage(_) => ErrorCategory::Storage,
            Self::Algorithm(_) => ErrorCategory::Algorithm,
            Self::Query(_) | Self::PatternMatching(_) => ErrorCategory::Query,
            Self::Network(_) => ErrorCategory::Network,
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => ErrorCategory::Gpu,
            #[cfg(feature = "distributed")]
            Self::Distributed(_) => ErrorCategory::Distributed,
            _ => ErrorCategory::Other,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational messages
    Info,
    /// Warning conditions
    Warning,
    /// Error conditions
    Error,
    /// Critical conditions
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Error categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// System-level errors
    System,
    /// Memory-related errors
    Memory,
    /// Performance-related errors
    Performance,
    /// Bridge/FFI errors
    Bridge,
    /// Data-related errors
    Data,
    /// Concurrency errors
    Concurrency,
    /// Configuration errors
    Configuration,
    /// Storage errors
    Storage,
    /// Algorithm errors
    Algorithm,
    /// Query errors
    Query,
    /// Network errors
    Network,
    /// GPU errors
    Gpu,
    /// Distributed system errors
    Distributed,
    /// Other errors
    Other,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System => write!(f, "SYSTEM"),
            Self::Memory => write!(f, "MEMORY"),
            Self::Performance => write!(f, "PERFORMANCE"),
            Self::Bridge => write!(f, "BRIDGE"),
            Self::Data => write!(f, "DATA"),
            Self::Concurrency => write!(f, "CONCURRENCY"),
            Self::Configuration => write!(f, "CONFIG"),
            Self::Storage => write!(f, "STORAGE"),
            Self::Algorithm => write!(f, "ALGORITHM"),
            Self::Query => write!(f, "QUERY"),
            Self::Network => write!(f, "NETWORK"),
            Self::Gpu => write!(f, "GPU"),
            Self::Distributed => write!(f, "DISTRIBUTED"),
            Self::Other => write!(f, "OTHER"),
        }
    }
}

/// Error context for additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed
    pub operation: String,
    /// Component that generated the error
    pub component: String,
    /// Thread ID where error occurred
    pub thread_id: Option<std::thread::ThreadId>,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            component: component.into(),
            thread_id: Some(std::thread::current().id()),
            timestamp: std::time::SystemTime::now(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// Add metadata to the context
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// Convert from common error types
impl From<rayon::ThreadPoolBuildError> for HybridError {
    fn from(err: rayon::ThreadPoolBuildError) -> Self {
        Self::synchronization_error(format!("Thread pool build error: {}", err))
    }
}

impl<T> From<PoisonError<T>> for HybridError {
    fn from(err: PoisonError<T>) -> Self {
        Self::synchronization_error(format!("Mutex poison error: {}", err))
    }
}

impl From<std::num::TryFromIntError> for HybridError {
    fn from(err: std::num::TryFromIntError) -> Self {
        Self::invalid_operation(format!("Integer conversion error: {}", err))
    }
}

impl From<bincode::Error> for HybridError {
    fn from(err: bincode::Error) -> Self {
        Self::Serialization(serde_json::Error::custom(format!("Bincode error: {}", err)))
    }
}

/// Extension trait for adding context to results
pub trait HybridResultExt<T> {
    /// Add context to an error
    fn with_context(self, operation: impl Into<String>, component: impl Into<String>) -> HybridResult<T>;
    
    /// Add simple context message
    fn with_message(self, message: impl Into<String>) -> HybridResult<T>;
}

impl<T> HybridResultExt<T> for HybridResult<T> {
    fn with_context(self, operation: impl Into<String>, component: impl Into<String>) -> HybridResult<T> {
        self.map_err(|err| {
            let context = ErrorContext::new(operation, component);
            HybridError::generic(
                err.to_string(),
                format!("{}::{} (thread: {:?})", context.component, context.operation, context.thread_id)
            )
        })
    }
    
    fn with_message(self, message: impl Into<String>) -> HybridResult<T> {
        self.map_err(|err| {
            HybridError::generic(err.to_string(), message.into())
        })
    }
}

/// Macro for creating context-aware errors
#[macro_export]
macro_rules! hybrid_error {
    ($kind:ident, $msg:expr) => {
        $crate::error::HybridError::$kind($msg.into())
    };
    ($kind:ident, $fmt:expr, $($arg:tt)*) => {
        $crate::error::HybridError::$kind(format!($fmt, $($arg)*))
    };
}

/// Macro for ensuring conditions with custom errors
#[macro_export]
macro_rules! hybrid_ensure {
    ($cond:expr, $err:expr) => {
        if !($cond) {
            return Err($err);
        }
    };
}

/// Macro for early return with error context
#[macro_export]
macro_rules! hybrid_bail {
    ($err:expr) => {
        return Err($err)
    };
    ($kind:ident, $msg:expr) => {
        return Err($crate::error::HybridError::$kind($msg.into()))
    };
    ($kind:ident, $fmt:expr, $($arg:tt)*) => {
        return Err($crate::error::HybridError::$kind(format!($fmt, $($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let err = HybridError::memory_allocation("Out of memory");
        assert_eq!(err.to_string(), "Memory allocation failed: Out of memory");
        assert_eq!(err.severity(), ErrorSeverity::Critical);
        assert_eq!(err.category(), ErrorCategory::Memory);
        assert!(!err.is_recoverable());
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("create_node", "graph")
            .with_metadata("node_id", "123")
            .with_metadata("thread", "worker-1");
        
        assert_eq!(context.operation, "create_node");
        assert_eq!(context.component, "graph");
        assert_eq!(context.metadata.get("node_id"), Some(&"123".to_string()));
    }
    
    #[test]
    fn test_error_macros() {
        let err = hybrid_error!(memory_allocation, "Test error");
        assert!(matches!(err, HybridError::Memory(_)));
        
        let err = hybrid_error!(algorithm_error, "Error code: {}", 404);
        assert!(matches!(err, HybridError::Algorithm(_)));
        assert!(err.to_string().contains("404"));
    }
    
    #[test]
    fn test_result_extension() {
        let result: HybridResult<i32> = Err(HybridError::NodeNotFound(123));
        let result_with_context = result.with_context("test_op", "test_component");
        
        assert!(result_with_context.is_err());
        let err = result_with_context.unwrap_err();
        assert!(matches!(err, HybridError::Generic { .. }));
    }
}