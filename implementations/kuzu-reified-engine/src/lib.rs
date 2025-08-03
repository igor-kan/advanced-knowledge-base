//! # Kuzu-Reified Engine
//!
//! High-performance graph database with edge reification capabilities built on top of Kuzu.
//! This library extends Kuzu's columnar graph database with advanced relationship modeling,
//! allowing edges to be treated as first-class nodes with their own properties and connections.

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod engine;
pub mod reification;
pub mod schema;
pub mod query;
pub mod types;
pub mod utils;

// Re-exports for convenience
pub use engine::KuzuReifiedEngine;
pub use reification::{ReificationManager, ReifiedRelationship};
pub use schema::{SchemaManager, NodeSchema, RelationshipSchema};
pub use query::{QueryExecutor, CypherExtensions, QueryResult};
pub use types::*;

/// Result type for all operations
pub type Result<T> = std::result::Result<T, KuzuReifiedError>;

/// Comprehensive error types for the Kuzu reified engine
#[derive(Debug, thiserror::Error)]
pub enum KuzuReifiedError {
    /// Kuzu database errors
    #[error("Kuzu error: {message}")]
    KuzuError { message: String },
    
    /// Schema validation errors
    #[error("Schema error: {field} - {issue}")]
    SchemaError { field: String, issue: String },
    
    /// Reification operation errors
    #[error("Reification error: {operation} - {details}")]
    ReificationError { operation: String, details: String },
    
    /// Query execution errors
    #[error("Query error: {query} - {message}")]
    QueryError { query: String, message: String },
    
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

/// Engine configuration
#[derive(Debug, Clone)]
pub struct KuzuReifiedConfig {
    /// Database path
    pub database_path: String,
    /// Number of threads for parallel operations
    pub num_threads: usize,
    /// Buffer pool size in MB
    pub buffer_pool_size_mb: u64,
    /// Enable query optimization
    pub enable_optimization: bool,
    /// Enable reification caching
    pub enable_reification_cache: bool,
    /// Maximum cache size for reified relationships
    pub max_cache_size: usize,
    /// Enable schema validation
    pub enable_schema_validation: bool,
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
    /// Enable concurrent reification
    pub enable_concurrent_reification: bool,
}

impl Default for KuzuReifiedConfig {
    fn default() -> Self {
        Self {
            database_path: "./kuzu_reified.db".to_string(),
            num_threads: num_cpus::get(),
            buffer_pool_size_mb: 4096, // 4GB
            enable_optimization: true,
            enable_reification_cache: true,
            max_cache_size: 100_000,
            enable_schema_validation: true,
            query_timeout_ms: 30_000,
            enable_concurrent_reification: true,
        }
    }
}

impl KuzuReifiedConfig {
    /// Create a high-performance configuration for large graphs
    pub fn high_performance() -> Self {
        Self {
            buffer_pool_size_mb: 16384, // 16GB
            num_threads: num_cpus::get(),
            enable_optimization: true,
            enable_reification_cache: true,
            max_cache_size: 1_000_000,
            enable_concurrent_reification: true,
            ..Default::default()
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            buffer_pool_size_mb: 1024, // 1GB
            num_threads: 4,
            max_cache_size: 10_000,
            ..Default::default()
        }
    }
    
    /// Create a development configuration for testing
    pub fn development() -> Self {
        Self {
            database_path: ":memory:".to_string(),
            buffer_pool_size_mb: 512,
            num_threads: 2,
            max_cache_size: 1_000,
            query_timeout_ms: 5_000,
            ..Default::default()
        }
    }
}

/// Initialize the Kuzu reified engine with logging
pub fn init() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .map_err(|e| KuzuReifiedError::Internal {
            details: format!("Failed to initialize logging: {}", e),
        })?;
    
    tracing::info!("Kuzu Reified Engine initialized");
    tracing::info!("Features: edge reification, columnar storage, high-performance queries");
    
    Ok(())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
    "Kuzu Reified Engine v",
    env!("CARGO_PKG_VERSION"),
    " built with Rust ",
    env!("RUSTC_VERSION"),
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = KuzuReifiedConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.buffer_pool_size_mb > 0);
        assert!(config.query_timeout_ms > 0);
    }
    
    #[test]
    fn test_high_performance_config() {
        let config = KuzuReifiedConfig::high_performance();
        assert_eq!(config.buffer_pool_size_mb, 16384);
        assert!(config.enable_optimization);
        assert!(config.enable_concurrent_reification);
    }
    
    #[test]
    fn test_memory_optimized_config() {
        let config = KuzuReifiedConfig::memory_optimized();
        assert_eq!(config.buffer_pool_size_mb, 1024);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.max_cache_size, 10_000);
    }
    
    #[test]
    fn test_development_config() {
        let config = KuzuReifiedConfig::development();
        assert_eq!(config.database_path, ":memory:");
        assert_eq!(config.query_timeout_ms, 5_000);
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(BUILD_INFO.contains("Kuzu Reified Engine"));
    }
    
    #[test]
    fn test_error_types() {
        let kuzu_error = KuzuReifiedError::KuzuError {
            message: "Connection failed".to_string(),
        };
        assert!(kuzu_error.to_string().contains("Kuzu error"));
        
        let schema_error = KuzuReifiedError::SchemaError {
            field: "name".to_string(),
            issue: "Required field missing".to_string(),
        };
        assert!(schema_error.to_string().contains("Schema error"));
    }
    
    #[test]
    fn test_init() {
        // Should not panic
        let _ = init();
    }
}