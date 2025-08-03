//! Connection management and pooling for Neo4j
//!
//! This module provides connection pooling, health monitoring, and failover capabilities
//! for Neo4j database connections in the reified engine.

use crate::{Neo4jReifiedError, Result, PoolConfig};

use neo4rs::Graph;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn, error, debug};

/// Connection manager for Neo4j database
pub struct ConnectionManager {
    /// Primary Neo4j connection
    primary_graph: Arc<Graph>,
    /// Connection pool configuration
    config: PoolConfig,
    /// Connection health status
    is_healthy: std::sync::atomic::AtomicBool,
    /// Last successful health check
    last_health_check: std::sync::Arc<std::sync::Mutex<Instant>>,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new(graph: Arc<Graph>, config: PoolConfig) -> Self {
        Self {
            primary_graph: graph,
            config,
            is_healthy: std::sync::atomic::AtomicBool::new(true),
            last_health_check: std::sync::Arc::new(std::sync::Mutex::new(Instant::now())),
        }
    }

    /// Get a connection from the pool
    pub async fn get_connection(&self) -> Result<Arc<Graph>> {
        if self.config.enable_health_checks {
            self.check_connection_health().await?;
        }
        
        Ok(self.primary_graph.clone())
    }

    /// Check the health of the connection
    pub async fn check_connection_health(&self) -> Result<bool> {
        let now = Instant::now();
        let should_check = {
            let last_check = self.last_health_check.lock().unwrap();
            now.duration_since(*last_check) > Duration::from_millis(self.config.health_check_interval_ms)
        };

        if !should_check {
            return Ok(self.is_healthy.load(std::sync::atomic::Ordering::Relaxed));
        }

        debug!("Performing connection health check");
        
        match self.perform_health_check().await {
            Ok(()) => {
                self.is_healthy.store(true, std::sync::atomic::Ordering::Relaxed);
                *self.last_health_check.lock().unwrap() = now;
                Ok(true)
            }
            Err(e) => {
                warn!("Connection health check failed: {}", e);
                self.is_healthy.store(false, std::sync::atomic::Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Force a connection health check
    pub async fn force_health_check(&self) -> Result<bool> {
        *self.last_health_check.lock().unwrap() = Instant::now() - Duration::from_secs(3600); // Force check
        self.check_connection_health().await
    }

    /// Get current health status
    pub fn is_healthy(&self) -> bool {
        self.is_healthy.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Start background health monitoring
    pub fn start_health_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let manager = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(manager.config.health_check_interval_ms));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = manager.check_connection_health().await {
                    error!("Background health check failed: {}", e);
                }
            }
        })
    }

    // Private methods

    async fn perform_health_check(&self) -> Result<()> {
        let query = neo4rs::Query::new("RETURN 1 as health_check".to_string());
        
        match tokio::time::timeout(
            Duration::from_millis(self.config.connection_timeout_ms),
            self.primary_graph.execute(query)
        ).await {
            Ok(Ok(mut result)) => {
                if let Ok(Some(_)) = result.next().await {
                    debug!("Health check passed");
                    Ok(())
                } else {
                    Err(Neo4jReifiedError::ConnectionError {
                        operation: "health_check".to_string(),
                        details: "No result returned from health check query".to_string(),
                    })
                }
            }
            Ok(Err(e)) => Err(Neo4jReifiedError::ConnectionError {
                operation: "health_check".to_string(),
                details: format!("Health check query failed: {}", e),
            }),
            Err(_) => Err(Neo4jReifiedError::ConnectionError {
                operation: "health_check".to_string(),
                details: "Health check timed out".to_string(),
            }),
        }
    }
}

impl Clone for ConnectionManager {
    fn clone(&self) -> Self {
        Self {
            primary_graph: self.primary_graph.clone(),
            config: self.config.clone(),
            is_healthy: std::sync::atomic::AtomicBool::new(self.is_healthy.load(std::sync::atomic::Ordering::Relaxed)),
            last_health_check: self.last_health_check.clone(),
        }
    }
}

/// Neo4j connection pool with advanced features
pub struct Neo4jPool {
    /// Connection managers
    connections: Vec<Arc<ConnectionManager>>,
    /// Current connection index for round-robin
    current_index: std::sync::atomic::AtomicUsize,
    /// Pool configuration
    config: PoolConfig,
}

impl Neo4jPool {
    /// Create a new connection pool
    pub async fn new(
        uris: Vec<String>,
        username: String,
        password: String,
        config: PoolConfig,
    ) -> Result<Self> {
        if uris.is_empty() {
            return Err(Neo4jReifiedError::ConfigError {
                parameter: "uris".to_string(),
                issue: "At least one URI must be provided".to_string(),
            });
        }

        let mut connections = Vec::new();
        
        for uri in uris {
            info!("Connecting to Neo4j at {}", uri);
            
            let graph = Graph::new(&uri, &username, &password)
                .await
                .map_err(|e| Neo4jReifiedError::ConnectionError {
                    operation: "connect".to_string(),
                    details: format!("Failed to connect to {}: {}", uri, e),
                })?;

            let connection_manager = Arc::new(ConnectionManager::new(Arc::new(graph), config.clone()));
            
            // Verify connection
            connection_manager.force_health_check().await?;
            
            connections.push(connection_manager);
        }

        info!("Created connection pool with {} connections", connections.len());

        Ok(Self {
            connections,
            current_index: std::sync::atomic::AtomicUsize::new(0),
            config,
        })
    }

    /// Get a healthy connection from the pool
    pub async fn get_connection(&self) -> Result<Arc<Graph>> {
        let start_index = self.current_index.load(std::sync::atomic::Ordering::Relaxed);
        let mut attempts = 0;
        
        while attempts < self.connections.len() {
            let index = (start_index + attempts) % self.connections.len();
            let connection = &self.connections[index];
            
            if connection.is_healthy() {
                // Update current index for next request
                self.current_index.store((index + 1) % self.connections.len(), std::sync::atomic::Ordering::Relaxed);
                
                return connection.get_connection().await;
            }
            
            attempts += 1;
        }
        
        // All connections are unhealthy, try to use the first one anyway
        warn!("All connections appear unhealthy, using primary connection");
        self.connections[0].get_connection().await
    }

    /// Get connection statistics
    pub fn get_stats(&self) -> PoolStats {
        let mut healthy_connections = 0;
        let mut unhealthy_connections = 0;
        
        for connection in &self.connections {
            if connection.is_healthy() {
                healthy_connections += 1;
            } else {
                unhealthy_connections += 1;
            }
        }
        
        PoolStats {
            total_connections: self.connections.len(),
            healthy_connections,
            unhealthy_connections,
            current_index: self.current_index.load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Force health check on all connections
    pub async fn check_all_connections(&self) -> Result<Vec<bool>> {
        let mut results = Vec::new();
        
        for connection in &self.connections {
            match connection.force_health_check().await {
                Ok(healthy) => results.push(healthy),
                Err(_) => results.push(false),
            }
        }
        
        Ok(results)
    }

    /// Start background health monitoring for all connections
    pub fn start_monitoring(&self) -> Vec<tokio::task::JoinHandle<()>> {
        self.connections.iter()
            .map(|conn| conn.start_health_monitoring())
            .collect()
    }
}

/// Connection pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub unhealthy_connections: usize,
    pub current_index: usize,
}

/// Connection retry logic with exponential backoff
pub struct ConnectionRetry {
    max_retries: usize,
    base_delay_ms: u64,
    max_delay_ms: u64,
}

impl ConnectionRetry {
    pub fn new(max_retries: usize, base_delay_ms: u64, max_delay_ms: u64) -> Self {
        Self {
            max_retries,
            base_delay_ms,
            max_delay_ms,
        }
    }

    pub async fn retry_with_backoff<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut attempts = 0;
        let mut delay = self.base_delay_ms;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    
                    if attempts >= self.max_retries {
                        return Err(e);
                    }
                    
                    debug!("Operation failed, retrying in {}ms (attempt {}/{})", delay, attempts, self.max_retries);
                    sleep(Duration::from_millis(delay)).await;
                    
                    // Exponential backoff with jitter
                    delay = std::cmp::min(delay * 2, self.max_delay_ms);
                    
                    // Add random jitter (±20%)
                    let jitter = (delay as f64 * 0.2) as u64;
                    let jitter_range = if jitter > 0 { rand::random::<u64>() % (jitter * 2) } else { 0 };
                    delay = delay.saturating_sub(jitter).saturating_add(jitter_range);
                }
            }
        }
    }
}

impl Default for ConnectionRetry {
    fn default() -> Self {
        Self::new(3, 1000, 30000) // 3 retries, 1s base delay, 30s max delay
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_defaults() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections, 50);
        assert_eq!(config.min_connections, 5);
        assert!(config.enable_health_checks);
    }

    #[test]
    fn test_connection_retry_creation() {
        let retry = ConnectionRetry::new(5, 2000, 60000);
        assert_eq!(retry.max_retries, 5);
        assert_eq!(retry.base_delay_ms, 2000);
        assert_eq!(retry.max_delay_ms, 60000);
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats {
            total_connections: 3,
            healthy_connections: 2,
            unhealthy_connections: 1,
            current_index: 0,
        };
        
        assert_eq!(stats.total_connections, 3);
        assert_eq!(stats.healthy_connections, 2);
        assert_eq!(stats.unhealthy_connections, 1);
    }

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_connection_manager() {
        // Integration tests would go here
    }
}