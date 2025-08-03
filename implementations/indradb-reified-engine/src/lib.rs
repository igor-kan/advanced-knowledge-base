//! # IndraDB-Reified Engine
//!
//! High-performance graph database with edge reification capabilities built on top of IndraDB.
//! This library extends IndraDB's property graph model with advanced relationship reification,
//! allowing edges to be treated as first-class nodes with their own properties and connections.

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod engine;
pub mod reification;
pub mod transaction;
pub mod query;
pub mod types;
pub mod utils;

// Re-exports for convenience
pub use engine::IndraReifiedEngine;
pub use reification::{ReificationManager, ReifiedRelationship};
pub use transaction::{TransactionManager, ReifiedTransaction};
pub use query::{QueryExecutor, PropertyGraphQuery, QueryResult};
pub use types::*;

// Re-export IndraDB types for convenience
pub use indradb::{
    Vertex, Edge, Type, Identifier, VertexProperties, EdgeProperties,
    Transaction as IndraTransaction, Datastore, MemoryDatastore,
};

/// Result type for all operations
pub type Result<T> = std::result::Result<T, IndraReifiedError>;

/// Comprehensive error types for the IndraDB reified engine
#[derive(Debug, thiserror::Error)]
pub enum IndraReifiedError {
    /// IndraDB database errors
    #[error("IndraDB error: {message}")]
    IndraError { message: String },
    
    /// Transaction errors
    #[error("Transaction error: {operation} - {details}")]
    TransactionError { operation: String, details: String },
    
    /// Reification operation errors
    #[error("Reification error: {operation} - {details}")]
    ReificationError { operation: String, details: String },
    
    /// Query execution errors
    #[error("Query error: {query} - {message}")]
    QueryError { query: String, message: String },
    
    /// Property graph errors
    #[error("Property graph error: {entity} - {issue}")]
    PropertyGraphError { entity: String, issue: String },
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Validation errors
    #[error("Validation error: {entity} - {constraint}")]
    ValidationError { entity: String, constraint: String },
    
    /// Concurrency errors
    #[error("Concurrency error: {operation}")]
    ConcurrencyError { operation: String },
    
    /// Configuration errors
    #[error("Configuration error: {parameter} - {issue}")]
    ConfigError { parameter: String, issue: String },
    
    /// Generic internal errors
    #[error("Internal error: {details}")]
    Internal { details: String },
}

/// Engine configuration for IndraDB integration
#[derive(Debug, Clone)]
pub struct IndraReifiedConfig {
    /// Database backend type
    pub backend_type: BackendType,
    /// Database path for persistent backends
    pub database_path: Option<String>,
    /// Number of threads for parallel operations
    pub num_threads: usize,
    /// Enable transaction logging
    pub enable_transaction_log: bool,
    /// Enable reification caching
    pub enable_reification_cache: bool,
    /// Maximum cache size for reified relationships
    pub max_cache_size: usize,
    /// Enable property indexing
    pub enable_property_indexing: bool,
    /// Transaction timeout in milliseconds
    pub transaction_timeout_ms: u64,
    /// Enable concurrent reification
    pub enable_concurrent_reification: bool,
    /// Bulk operation batch size
    pub bulk_batch_size: usize,
}

/// Backend type for IndraDB storage
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// In-memory backend for testing and development
    Memory,
    /// RocksDB backend for production use
    RocksDB,
    /// Custom backend implementation
    Custom(String),
}

impl Default for IndraReifiedConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Memory,
            database_path: None,
            num_threads: num_cpus::get(),
            enable_transaction_log: true,
            enable_reification_cache: true,
            max_cache_size: 100_000,
            enable_property_indexing: true,
            transaction_timeout_ms: 30_000,
            enable_concurrent_reification: true,
            bulk_batch_size: 1_000,
        }
    }
}

impl IndraReifiedConfig {
    /// Create a high-performance configuration for large graphs
    pub fn high_performance() -> Self {
        Self {
            backend_type: BackendType::RocksDB,
            num_threads: num_cpus::get(),
            enable_transaction_log: true,
            enable_reification_cache: true,
            max_cache_size: 1_000_000,
            enable_concurrent_reification: true,
            bulk_batch_size: 10_000,
            enable_property_indexing: true,
            ..Default::default()
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            backend_type: BackendType::Memory,
            num_threads: 4,
            max_cache_size: 10_000,
            bulk_batch_size: 500,
            enable_property_indexing: false,
            ..Default::default()
        }
    }
    
    /// Create a development configuration for testing
    pub fn development() -> Self {
        Self {
            backend_type: BackendType::Memory,
            num_threads: 2,
            max_cache_size: 1_000,
            transaction_timeout_ms: 5_000,
            bulk_batch_size: 100,
            enable_transaction_log: false,
            ..Default::default()
        }
    }
    
    /// Create configuration for RocksDB backend
    pub fn rocksdb(database_path: impl Into<String>) -> Self {
        Self {
            backend_type: BackendType::RocksDB,
            database_path: Some(database_path.into()),
            ..Self::high_performance()
        }
    }
}

/// Initialize the IndraDB reified engine with logging
pub fn init() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .map_err(|e| IndraReifiedError::Internal {
            details: format!("Failed to initialize logging: {}", e),
        })?;
    
    tracing::info!("IndraDB Reified Engine initialized");
    tracing::info!("Features: edge reification, property graphs, ACID transactions");
    
    Ok(())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
    "IndraDB Reified Engine v",
    env!("CARGO_PKG_VERSION"),
    " built with Rust ",
    env!("RUSTC_VERSION"),
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = IndraReifiedConfig::default();
        assert_eq!(config.backend_type, BackendType::Memory);
        assert!(config.num_threads > 0);
        assert!(config.transaction_timeout_ms > 0);
    }
    
    #[test]
    fn test_high_performance_config() {
        let config = IndraReifiedConfig::high_performance();
        assert_eq!(config.backend_type, BackendType::RocksDB);
        assert_eq!(config.max_cache_size, 1_000_000);
        assert!(config.enable_concurrent_reification);
        assert_eq!(config.bulk_batch_size, 10_000);
    }
    
    #[test]
    fn test_memory_optimized_config() {
        let config = IndraReifiedConfig::memory_optimized();
        assert_eq!(config.backend_type, BackendType::Memory);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.max_cache_size, 10_000);
        assert!(!config.enable_property_indexing);
    }
    
    #[test]
    fn test_development_config() {
        let config = IndraReifiedConfig::development();
        assert_eq!(config.backend_type, BackendType::Memory);
        assert_eq!(config.transaction_timeout_ms, 5_000);
        assert!(!config.enable_transaction_log);
    }
    
    #[test]
    fn test_rocksdb_config() {
        let config = IndraReifiedConfig::rocksdb("/tmp/test.db");
        assert_eq!(config.backend_type, BackendType::RocksDB);
        assert_eq!(config.database_path, Some("/tmp/test.db".to_string()));
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(BUILD_INFO.contains("IndraDB Reified Engine"));
    }
    
    #[test]
    fn test_error_types() {
        let indra_error = IndraReifiedError::IndraError {
            message: "Connection failed".to_string(),
        };
        assert!(indra_error.to_string().contains("IndraDB error"));
        
        let transaction_error = IndraReifiedError::TransactionError {
            operation: "commit".to_string(),
            details: "Deadlock detected".to_string(),
        };
        assert!(transaction_error.to_string().contains("Transaction error"));
        
        let reification_error = IndraReifiedError::ReificationError {
            operation: "reify".to_string(),
            details: "Invalid relationship".to_string(),
        };
        assert!(reification_error.to_string().contains("Reification error"));
    }
    
    #[test]
    fn test_init() {
        // Should not panic
        let _ = init();
    }
    
    #[test]
    fn test_backend_types() {
        assert_eq!(BackendType::Memory, BackendType::Memory);
        assert_eq!(BackendType::RocksDB, BackendType::RocksDB);
        assert_ne!(BackendType::Memory, BackendType::RocksDB);
        
        let custom1 = BackendType::Custom("test".to_string());
        let custom2 = BackendType::Custom("test".to_string());
        let custom3 = BackendType::Custom("other".to_string());
        
        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }
}