//! Error handling for the Ultra-Fast Knowledge Graph - 2025 Edition
//!
//! Comprehensive error types and handling for all aspects of the knowledge graph
//! system, including performance monitoring and recovery mechanisms.

use std::fmt;
use thiserror::Error;

/// Main error type for the ultra-fast knowledge graph system
#[derive(Error, Debug)]
pub enum UltraFastKnowledgeGraphError {
    /// Graph operation errors
    #[error("Graph operation failed: {0}")]
    GraphOperationError(String),
    
    /// Node-related errors
    #[error("Node error: {0}")]
    NodeError(String),
    
    /// Edge-related errors  
    #[error("Edge error: {0}")]
    EdgeError(String),
    
    /// Storage backend errors
    #[error("Storage error: {0}")]
    StorageError(String),
    
    /// Distributed system errors
    #[error("Distributed system error: {0}")]
    DistributedError(String),
    
    /// Network communication errors
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Memory allocation errors
    #[error("Memory allocation error: {0}")]
    MemoryAllocationError(String),
    
    /// SIMD operation errors
    #[error("SIMD operation error: {0}")]
    SIMDError(String),
    
    /// Assembly optimization errors
    #[error("Assembly optimization error: {0}")]
    AssemblyError(String),
    
    /// Fortran bridge errors
    #[error("Fortran bridge error: {0}")]
    FortranError(String),
    
    /// C++ backend errors
    #[error("C++ backend error: {0}")]
    CppBackendError(String),
    
    /// Mathematical computation errors
    #[error("Mathematical computation error: {0}")]
    ComputationError(String),
    
    /// Index operation errors
    #[error("Index operation error: {0}")]
    IndexError(String),
    
    /// Query processing errors
    #[error("Query processing error: {0}")]
    QueryError(String),
    
    /// Transaction errors
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    /// Concurrency errors
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Permission/security errors
    #[error("Permission error: {0}")]
    PermissionError(String),
    
    /// Resource limit errors
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitError(String),
    
    /// Timeout errors
    #[error("Operation timed out: {0}")]
    TimeoutError(String),
    
    /// System-level errors
    #[error("System error: {0}")]
    SystemError(String),
    
    /// Invalid operation errors
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    /// Data corruption errors
    #[error("Data corruption detected: {0}")]
    DataCorruption(String),
    
    /// Performance degradation errors
    #[error("Performance degradation: {0}")]
    PerformanceDegradation(String),
    
    /// Resource exhaustion errors
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    
    /// Hardware-specific errors
    #[error("Hardware error: {0}")]
    HardwareError(String),
    
    /// GPU/CUDA errors
    #[error("GPU error: {0}")]
    GpuError(String),
    
    /// Machine learning errors
    #[error("ML operation error: {0}")]
    MLError(String),
    
    /// Monitoring/metrics errors
    #[error("Monitoring error: {0}")]
    MonitoringError(String),
    
    /// Backup/recovery errors
    #[error("Backup/recovery error: {0}")]
    BackupRecoveryError(String),
    
    /// Migration errors
    #[error("Migration error: {0}")]
    MigrationError(String),
    
    /// External service errors
    #[error("External service error: {0}")]
    ExternalServiceError(String),
    
    /// Protocol errors
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    /// Encoding/decoding errors
    #[error("Encoding error: {0}")]
    EncodingError(String),
    
    /// Compression errors
    #[error("Compression error: {0}")]
    CompressionError(String),
    
    /// Authentication errors
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    
    /// Authorization errors
    #[error("Authorization error: {0}")]
    AuthorizationError(String),
    
    /// Audit errors
    #[error("Audit error: {0}")]
    AuditError(String),
    
    /// Generic I/O errors
    #[error("I/O error")]
    IoError(#[from] std::io::Error),
    
    /// JSON serialization errors
    #[error("JSON error")]
    JsonError(#[from] serde_json::Error),
    
    /// Bincode serialization errors
    #[error("Bincode error")]
    BincodeError(#[from] bincode::Error),
    
    /// Channel communication errors
    #[error("Channel error")]
    ChannelError(String),
    
    /// Thread join errors
    #[error("Thread join error")]
    ThreadJoinError(String),
    
    /// Lock acquisition errors
    #[error("Lock acquisition error: {0}")]
    LockError(String),
    
    /// Generic anyhow errors for external integrations
    #[error("External error")]
    ExternalError(#[from] anyhow::Error),
}

impl UltraFastKnowledgeGraphError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors
            Self::NetworkError(_) |
            Self::TimeoutError(_) |
            Self::ResourceExhausted(_) |
            Self::ConcurrencyError(_) |
            Self::LockError(_) |
            Self::ChannelError(_) |
            Self::ExternalServiceError(_) => true,
            
            // Non-recoverable errors
            Self::DataCorruption(_) |
            Self::MemoryAllocationError(_) |
            Self::HardwareError(_) |
            Self::SystemError(_) |
            Self::ValidationError(_) |
            Self::ConfigurationError(_) => false,
            
            // Context-dependent errors
            _ => false,
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical errors
            Self::DataCorruption(_) |
            Self::MemoryAllocationError(_) |
            Self::SystemError(_) |
            Self::HardwareError(_) => ErrorSeverity::Critical,
            
            // High severity errors
            Self::StorageError(_) |
            Self::DistributedError(_) |
            Self::BackupRecoveryError(_) |
            Self::AuthenticationError(_) |
            Self::AuthorizationError(_) => ErrorSeverity::High,
            
            // Medium severity errors
            Self::GraphOperationError(_) |
            Self::NetworkError(_) |
            Self::QueryError(_) |
            Self::PerformanceDegradation(_) |
            Self::ResourceExhausted(_) => ErrorSeverity::Medium,
            
            // Low severity errors
            Self::ValidationError(_) |
            Self::ConfigurationError(_) |
            Self::TimeoutError(_) |
            Self::MonitoringError(_) => ErrorSeverity::Low,
            
            // Default to medium
            _ => ErrorSeverity::Medium,
        }
    }
    
    /// Get suggested recovery action
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            // Immediate retry
            Self::NetworkError(_) |
            Self::TimeoutError(_) |
            Self::ChannelError(_) => RecoveryAction::Retry,
            
            // Retry with backoff
            Self::ResourceExhausted(_) |
            Self::ConcurrencyError(_) |
            Self::ExternalServiceError(_) => RecoveryAction::RetryWithBackoff,
            
            // Failover to backup
            Self::StorageError(_) |
            Self::DistributedError(_) => RecoveryAction::Failover,
            
            // Restart component
            Self::MemoryAllocationError(_) |
            Self::PerformanceDegradation(_) => RecoveryAction::RestartComponent,
            
            // Manual intervention required
            Self::DataCorruption(_) |
            Self::HardwareError(_) |
            Self::SystemError(_) => RecoveryAction::ManualIntervention,
            
            // Default to logging
            _ => RecoveryAction::LogAndContinue,
        }
    }
    
    /// Get error context for debugging
    pub fn context(&self) -> ErrorContext {
        match self {
            Self::GraphOperationError(_) => ErrorContext::Graph,
            Self::NodeError(_) | Self::EdgeError(_) => ErrorContext::Data,
            Self::StorageError(_) => ErrorContext::Storage,
            Self::DistributedError(_) | Self::NetworkError(_) => ErrorContext::Network,
            Self::SIMDError(_) | Self::AssemblyError(_) => ErrorContext::Performance,
            Self::FortranError(_) | Self::CppBackendError(_) => ErrorContext::Backend,
            Self::QueryError(_) => ErrorContext::Query,
            Self::TransactionError(_) => ErrorContext::Transaction,
            Self::AuthenticationError(_) | Self::AuthorizationError(_) => ErrorContext::Security,
            Self::MonitoringError(_) => ErrorContext::Monitoring,
            _ => ErrorContext::General,
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - warnings and minor issues
    Low,
    /// Medium severity - operational issues
    Medium,
    /// High severity - service degradation
    High,
    /// Critical severity - system failure
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Suggested recovery actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Log error and continue operation
    LogAndContinue,
    /// Retry operation immediately
    Retry,
    /// Retry with exponential backoff
    RetryWithBackoff,
    /// Failover to backup system
    Failover,
    /// Restart affected component
    RestartComponent,
    /// Graceful shutdown
    GracefulShutdown,
    /// Emergency shutdown
    EmergencyShutdown,
    /// Manual intervention required
    ManualIntervention,
}

/// Error context categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorContext {
    /// General system errors
    General,
    /// Graph structure errors
    Graph,
    /// Data-related errors
    Data,
    /// Storage system errors
    Storage,
    /// Network/distributed errors
    Network,
    /// Performance optimization errors
    Performance,
    /// Backend integration errors
    Backend,
    /// Query processing errors
    Query,
    /// Transaction errors
    Transaction,
    /// Security errors
    Security,
    /// Monitoring errors
    Monitoring,
}

/// Error recovery mechanism
pub struct ErrorRecovery {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Base retry delay in milliseconds
    pub base_retry_delay_ms: u64,
    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Enable circuit breaker
    pub circuit_breaker_enabled: bool,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: usize,
    /// Circuit breaker timeout in seconds
    pub circuit_breaker_timeout_secs: u64,
}

impl Default for ErrorRecovery {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_retry_delay_ms: 100,
            max_retry_delay_ms: 5000,
            backoff_multiplier: 2.0,
            circuit_breaker_enabled: true,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_secs: 60,
        }
    }
}

impl ErrorRecovery {
    /// Calculate retry delay with exponential backoff
    pub fn calculate_retry_delay(&self, attempt: usize) -> std::time::Duration {
        let delay_ms = (self.base_retry_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32))
            .min(self.max_retry_delay_ms as f64) as u64;
        
        std::time::Duration::from_millis(delay_ms)
    }
    
    /// Check if should retry based on error and attempt count
    pub fn should_retry(&self, error: &UltraFastKnowledgeGraphError, attempts: usize) -> bool {
        attempts < self.max_retries && error.is_recoverable()
    }
}

/// Circuit breaker for error handling
pub struct CircuitBreaker {
    /// Failure count
    failures: std::sync::atomic::AtomicUsize,
    /// Last failure time
    last_failure: std::sync::Mutex<Option<std::time::Instant>>,
    /// Circuit breaker state
    state: std::sync::atomic::AtomicU8, // 0=Closed, 1=Open, 2=HalfOpen
    /// Configuration
    config: ErrorRecovery,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(config: ErrorRecovery) -> Self {
        Self {
            failures: std::sync::atomic::AtomicUsize::new(0),
            last_failure: std::sync::Mutex::new(None),
            state: std::sync::atomic::AtomicU8::new(0), // Closed
            config,
        }
    }
    
    /// Check if circuit breaker allows operation
    pub fn allows_request(&self) -> bool {
        if !self.config.circuit_breaker_enabled {
            return true;
        }
        
        let state = self.state.load(std::sync::atomic::Ordering::Relaxed);
        match state {
            0 => true, // Closed - allow requests
            1 => {     // Open - check timeout
                if let Ok(last_failure) = self.last_failure.lock() {
                    if let Some(last) = *last_failure {
                        let timeout = std::time::Duration::from_secs(self.config.circuit_breaker_timeout_secs);
                        if last.elapsed() > timeout {
                            // Move to half-open
                            self.state.store(2, std::sync::atomic::Ordering::Relaxed);
                            return true;
                        }
                    }
                }
                false
            },
            2 => true, // Half-open - allow limited requests
            _ => false,
        }
    }
    
    /// Record successful operation
    pub fn record_success(&self) {
        self.failures.store(0, std::sync::atomic::Ordering::Relaxed);
        self.state.store(0, std::sync::atomic::Ordering::Relaxed); // Closed
    }
    
    /// Record failed operation
    pub fn record_failure(&self) {
        let failure_count = self.failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        
        if failure_count >= self.config.circuit_breaker_threshold {
            self.state.store(1, std::sync::atomic::Ordering::Relaxed); // Open
            if let Ok(mut last_failure) = self.last_failure.lock() {
                *last_failure = Some(std::time::Instant::now());
            }
        }
    }
    
    /// Get current state
    pub fn state(&self) -> CircuitBreakerState {
        match self.state.load(std::sync::atomic::Ordering::Relaxed) {
            0 => CircuitBreakerState::Closed,
            1 => CircuitBreakerState::Open,
            2 => CircuitBreakerState::HalfOpen,
            _ => CircuitBreakerState::Closed,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed, allowing all requests
    Closed,
    /// Circuit is open, blocking all requests
    Open,
    /// Circuit is half-open, allowing limited requests for testing
    HalfOpen,
}

/// Error metrics for monitoring
#[derive(Debug, Default)]
pub struct ErrorMetrics {
    /// Total error count by type
    pub error_counts: std::collections::HashMap<String, u64>,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Mean time between failures
    pub mtbf_seconds: f64,
    /// Mean time to recovery
    pub mttr_seconds: f64,
    /// Last error timestamp
    pub last_error: Option<std::time::Instant>,
    /// Recovery success rate
    pub recovery_success_rate: f64,
}

impl ErrorMetrics {
    /// Record error occurrence
    pub fn record_error(&mut self, error: &UltraFastKnowledgeGraphError) {
        let error_type = std::any::type_name_of_val(error);
        *self.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
        self.last_error = Some(std::time::Instant::now());
    }
    
    /// Calculate current error rate
    pub fn calculate_error_rate(&mut self, window_seconds: u64) -> f64 {
        if let Some(last_error) = self.last_error {
            let elapsed = last_error.elapsed().as_secs();
            if elapsed < window_seconds {
                let total_errors: u64 = self.error_counts.values().sum();
                return total_errors as f64 / window_seconds as f64;
            }
        }
        0.0
    }
    
    /// Get most common error types
    pub fn top_error_types(&self, limit: usize) -> Vec<(String, u64)> {
        let mut errors: Vec<_> = self.error_counts.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        errors.sort_by(|a, b| b.1.cmp(&a.1));
        errors.truncate(limit);
        errors
    }
}

/// Result type alias for convenience
pub type UltraResult<T> = Result<T, UltraFastKnowledgeGraphError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        let critical_error = UltraFastKnowledgeGraphError::DataCorruption("test".to_string());
        assert_eq!(critical_error.severity(), ErrorSeverity::Critical);
        
        let medium_error = UltraFastKnowledgeGraphError::NetworkError("test".to_string());
        assert_eq!(medium_error.severity(), ErrorSeverity::Medium);
    }
    
    #[test]
    fn test_error_recovery() {
        let recoverable_error = UltraFastKnowledgeGraphError::NetworkError("test".to_string());
        assert!(recoverable_error.is_recoverable());
        
        let non_recoverable_error = UltraFastKnowledgeGraphError::DataCorruption("test".to_string());
        assert!(!non_recoverable_error.is_recoverable());
    }
    
    #[test]
    fn test_recovery_delay_calculation() {
        let recovery = ErrorRecovery::default();
        
        let delay1 = recovery.calculate_retry_delay(0);
        let delay2 = recovery.calculate_retry_delay(1);
        let delay3 = recovery.calculate_retry_delay(2);
        
        assert!(delay2 > delay1);
        assert!(delay3 > delay2);
        assert!(delay3.as_millis() <= recovery.max_retry_delay_ms as u128);
    }
    
    #[test]
    fn test_circuit_breaker() {
        let config = ErrorRecovery::default();
        let breaker = CircuitBreaker::new(config);
        
        // Initially closed
        assert_eq!(breaker.state(), CircuitBreakerState::Closed);
        assert!(breaker.allows_request());
        
        // Record failures
        for _ in 0..5 {
            breaker.record_failure();
        }
        
        // Should now be open
        assert_eq!(breaker.state(), CircuitBreakerState::Open);
        assert!(!breaker.allows_request());
        
        // Record success should close it
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitBreakerState::Closed);
        assert!(breaker.allows_request());
    }
    
    #[test]
    fn test_error_metrics() {
        let mut metrics = ErrorMetrics::default();
        
        let error1 = UltraFastKnowledgeGraphError::NetworkError("test1".to_string());
        let error2 = UltraFastKnowledgeGraphError::NetworkError("test2".to_string());
        let error3 = UltraFastKnowledgeGraphError::StorageError("test3".to_string());
        
        metrics.record_error(&error1);
        metrics.record_error(&error2);
        metrics.record_error(&error3);
        
        let top_errors = metrics.top_error_types(2);
        assert_eq!(top_errors.len(), 2);
        
        // NetworkError should be first (2 occurrences)
        assert!(top_errors[0].0.contains("NetworkError"));
        assert_eq!(top_errors[0].1, 2);
    }
}