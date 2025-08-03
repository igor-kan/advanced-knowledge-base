//! # Neo4j-Reified Engine
//!
//! Advanced graph database with multi-level edge reification capabilities built on top of Neo4j.
//! This library extends Neo4j's native graph model with sophisticated relationship reification,
//! enabling complex relationship hierarchies and meta-relationships for enterprise applications.

#![deny(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod engine;
pub mod reification;
pub mod cypher;
pub mod connection;
pub mod batch;
pub mod temporal;
pub mod types;
pub mod utils;

// Re-exports for convenience
pub use engine::Neo4jReifiedEngine;
pub use reification::{ReificationManager, ReificationPattern, ReificationHierarchy};
pub use cypher::{CypherExtensions, QueryBuilder, ReificationQuery};
pub use connection::{ConnectionManager, Neo4jPool};
pub use batch::{BatchReifier, ReificationRequest, BatchResult};
pub use temporal::{TemporalReifier, VersionStrategy, TemporalConfig};
pub use types::*;

// Re-export Neo4j types for convenience
pub use neo4rs::{Graph, Node, Relation, BoltType, Query, Row};

/// Result type for all operations
pub type Result<T> = std::result::Result<T, Neo4jReifiedError>;

/// Comprehensive error types for the Neo4j reified engine
#[derive(Debug, thiserror::Error)]
pub enum Neo4jReifiedError {
    /// Neo4j database errors
    #[error("Neo4j error: {message}")]
    Neo4jError { message: String },
    
    /// Connection pool errors
    #[error("Connection error: {operation} - {details}")]
    ConnectionError { operation: String, details: String },
    
    /// Reification operation errors  
    #[error("Reification error: {operation} - {details}")]
    ReificationError { operation: String, details: String },
    
    /// Cypher query errors
    #[error("Cypher error: {query} - {message}")]
    CypherError { query: String, message: String },
    
    /// Batch operation errors
    #[error("Batch error: {operation} - {failed_count} of {total_count} operations failed")]
    BatchError { operation: String, failed_count: usize, total_count: usize },
    
    /// Temporal versioning errors
    #[error("Temporal error: {operation} - {details}")]
    TemporalError { operation: String, details: String },
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Validation errors
    #[error("Validation error: {entity} - {constraint}")]
    ValidationError { entity: String, constraint: String },
    
    /// Transaction errors
    #[error("Transaction error: {operation} - {details}")]
    TransactionError { operation: String, details: String },
    
    /// Configuration errors
    #[error("Configuration error: {parameter} - {issue}")]
    ConfigError { parameter: String, issue: String },
    
    /// Multi-database errors
    #[error("Multi-database error: {database} - {operation}")]
    MultiDatabaseError { database: String, operation: String },
    
    /// Generic internal errors
    #[error("Internal error: {details}")]
    Internal { details: String },
}

/// Engine configuration for Neo4j integration
#[derive(Debug, Clone)]
pub struct Neo4jReifiedConfig {
    /// Neo4j connection URI
    pub uri: String,
    /// Database username
    pub username: String,
    /// Database password
    pub password: String,
    /// Default database name
    pub database: Option<String>,
    /// Connection pool configuration
    pub pool_config: PoolConfig,
    /// Enable advanced reification patterns
    pub enable_advanced_reification: bool,
    /// Enable multi-level reification
    pub enable_multi_level_reification: bool,
    /// Enable temporal versioning
    pub enable_temporal_versioning: bool,
    /// Enable batch operations
    pub enable_batch_operations: bool,
    /// Enable query caching
    pub enable_query_caching: bool,
    /// Maximum cache size for reified relationships
    pub max_cache_size: usize,
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
    /// Transaction timeout in milliseconds
    pub transaction_timeout_ms: u64,
    /// Enable concurrent reification
    pub enable_concurrent_reification: bool,
    /// Bulk operation batch size
    pub bulk_batch_size: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Enable audit logging
    pub enable_audit_logging: bool,
}

/// Connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool
    pub max_connections: usize,
    /// Minimum number of connections to maintain
    pub min_connections: usize,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Maximum connection lifetime in milliseconds
    pub max_connection_lifetime_ms: u64,
    /// Connection idle timeout in milliseconds
    pub idle_timeout_ms: u64,
    /// Enable connection health checks
    pub enable_health_checks: bool,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 50,
            min_connections: 5,
            connection_timeout_ms: 30_000,
            max_connection_lifetime_ms: 3_600_000, // 1 hour
            idle_timeout_ms: 600_000, // 10 minutes
            enable_health_checks: true,
            health_check_interval_ms: 30_000,
        }
    }
}

impl Default for Neo4jReifiedConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "password".to_string(),
            database: None,
            pool_config: PoolConfig::default(),
            enable_advanced_reification: true,
            enable_multi_level_reification: true,
            enable_temporal_versioning: false,
            enable_batch_operations: true,
            enable_query_caching: true,
            max_cache_size: 100_000,
            query_timeout_ms: 30_000,
            transaction_timeout_ms: 60_000,
            enable_concurrent_reification: true,
            bulk_batch_size: 1_000,
            enable_monitoring: false,
            enable_audit_logging: false,
        }
    }
}

impl Neo4jReifiedConfig {
    /// Create a high-performance configuration for enterprise use
    pub fn enterprise() -> Self {
        Self {
            pool_config: PoolConfig {
                max_connections: 200,
                min_connections: 20,
                connection_timeout_ms: 10_000,
                max_connection_lifetime_ms: 7_200_000, // 2 hours
                idle_timeout_ms: 300_000, // 5 minutes
                enable_health_checks: true,
                health_check_interval_ms: 15_000,
            },
            enable_advanced_reification: true,
            enable_multi_level_reification: true,
            enable_temporal_versioning: true,
            enable_batch_operations: true,
            enable_query_caching: true,
            max_cache_size: 1_000_000,
            bulk_batch_size: 10_000,
            enable_monitoring: true,
            enable_audit_logging: true,
            ..Default::default()
        }
    }
    
    /// Create a development configuration for testing
    pub fn development() -> Self {
        Self {
            pool_config: PoolConfig {
                max_connections: 10,
                min_connections: 2,
                connection_timeout_ms: 5_000,
                max_connection_lifetime_ms: 1_800_000, // 30 minutes
                idle_timeout_ms: 300_000, // 5 minutes
                enable_health_checks: false,
                health_check_interval_ms: 60_000,
            },
            enable_advanced_reification: true,
            enable_multi_level_reification: false,
            enable_temporal_versioning: false,
            enable_batch_operations: false,
            enable_query_caching: false,
            max_cache_size: 1_000,
            bulk_batch_size: 100,
            enable_monitoring: false,
            enable_audit_logging: false,
            ..Default::default()
        }
    }
    
    /// Create a memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            pool_config: PoolConfig {
                max_connections: 25,
                min_connections: 3,
                ..Default::default()
            },
            enable_query_caching: false, // Reduce memory usage
            max_cache_size: 10_000,
            bulk_batch_size: 500,
            ..Default::default()
        }
    }
    
    /// Create configuration for specific database
    pub fn with_database(mut self, database: impl Into<String>) -> Self {
        self.database = Some(database.into());
        self
    }
    
    /// Create configuration with custom connection details
    pub fn with_connection(
        uri: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            uri: uri.into(),
            username: username.into(),
            password: password.into(),
            ..Default::default()
        }
    }
}

/// Initialize the Neo4j reified engine with logging
pub fn init() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init()
        .map_err(|e| Neo4jReifiedError::Internal {
            details: format!("Failed to initialize logging: {}", e),
        })?;
    
    tracing::info!("Neo4j Reified Engine initialized");
    tracing::info!("Features: advanced reification, multi-level hierarchies, temporal versioning");
    
    Ok(())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: &str = concat!(
    "Neo4j Reified Engine v",
    env!("CARGO_PKG_VERSION"),
    " built with Rust ",
    env!("RUSTC_VERSION"),
);

/// Supported Neo4j versions
pub const SUPPORTED_NEO4J_VERSIONS: &[&str] = &["5.0", "5.1", "5.2", "5.3", "5.4", "5.5"];

/// Maximum reification depth to prevent infinite recursion
pub const MAX_REIFICATION_DEPTH: usize = 10;

/// Default reified edge label
pub const REIFIED_EDGE_LABEL: &str = "ReifiedEdge";

/// Connection relationship types for reification
pub const FROM_RELATIONSHIP: &str = "FROM";
pub const TO_RELATIONSHIP: &str = "TO";

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = Neo4jReifiedConfig::default();
        assert_eq!(config.uri, "bolt://localhost:7687");
        assert!(config.pool_config.max_connections > 0);
        assert!(config.query_timeout_ms > 0);
    }
    
    #[test]
    fn test_enterprise_config() {
        let config = Neo4jReifiedConfig::enterprise();
        assert_eq!(config.pool_config.max_connections, 200);
        assert!(config.enable_advanced_reification);
        assert!(config.enable_temporal_versioning);
        assert!(config.enable_monitoring);
        assert!(config.enable_audit_logging);
    }
    
    #[test]
    fn test_development_config() {
        let config = Neo4jReifiedConfig::development();
        assert_eq!(config.pool_config.max_connections, 10);
        assert!(!config.enable_temporal_versioning);
        assert!(!config.enable_monitoring);
    }
    
    #[test]
    fn test_memory_optimized_config() {
        let config = Neo4jReifiedConfig::memory_optimized();
        assert_eq!(config.pool_config.max_connections, 25);
        assert!(!config.enable_query_caching);
        assert_eq!(config.max_cache_size, 10_000);
    }
    
    #[test]
    fn test_config_with_database() {
        let config = Neo4jReifiedConfig::default()
            .with_database("my_graph");
        assert_eq!(config.database, Some("my_graph".to_string()));
    }
    
    #[test]
    fn test_config_with_connection() {
        let config = Neo4jReifiedConfig::with_connection(
            "bolt://remote:7687",
            "admin",
            "secret"
        );
        assert_eq!(config.uri, "bolt://remote:7687");
        assert_eq!(config.username, "admin");
        assert_eq!(config.password, "secret");
    }
    
    #[test]
    fn test_pool_config_defaults() {
        let pool_config = PoolConfig::default();
        assert_eq!(pool_config.max_connections, 50);
        assert_eq!(pool_config.min_connections, 5);
        assert!(pool_config.enable_health_checks);
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(BUILD_INFO.contains("Neo4j Reified Engine"));
    }
    
    #[test]
    fn test_supported_versions() {
        assert!(!SUPPORTED_NEO4J_VERSIONS.is_empty());
        assert!(SUPPORTED_NEO4J_VERSIONS.contains(&"5.0"));
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(MAX_REIFICATION_DEPTH, 10);
        assert_eq!(REIFIED_EDGE_LABEL, "ReifiedEdge");
        assert_eq!(FROM_RELATIONSHIP, "FROM");
        assert_eq!(TO_RELATIONSHIP, "TO");
    }
    
    #[test]
    fn test_error_types() {
        let neo4j_error = Neo4jReifiedError::Neo4jError {
            message: "Connection failed".to_string(),
        };
        assert!(neo4j_error.to_string().contains("Neo4j error"));
        
        let reification_error = Neo4jReifiedError::ReificationError {
            operation: "reify".to_string(),
            details: "Invalid relationship".to_string(),
        };
        assert!(reification_error.to_string().contains("Reification error"));
        
        let batch_error = Neo4jReifiedError::BatchError {
            operation: "batch_reify".to_string(),
            failed_count: 5,
            total_count: 100,
        };
        assert!(batch_error.to_string().contains("5 of 100"));
    }
    
    #[test]
    fn test_init() {
        // Should not panic
        let _ = init();
    }
}