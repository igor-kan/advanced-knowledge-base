//! Core Neo4j reified engine implementation
//!
//! This module provides the main Neo4jReifiedEngine struct and its core functionality
//! for managing reified relationships in Neo4j databases.

use crate::types::*;
use crate::{Neo4jReifiedError, Neo4jReifiedConfig, Result};
use crate::connection::ConnectionManager;
use crate::reification::ReificationManager;
use crate::cypher::CypherExtensions;
use crate::batch::BatchReifier;
use crate::temporal::TemporalReifier;

use neo4rs::{Graph, Query, BoltType};
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;
use tracing::{info, warn, error};

/// Main Neo4j reified engine for advanced graph operations
pub struct Neo4jReifiedEngine {
    /// Neo4j graph connection
    graph: Arc<Graph>,
    /// Connection manager for pooling and health checks
    connection_manager: Arc<ConnectionManager>,
    /// Reification manager for relationship operations
    reification_manager: Arc<ReificationManager>,
    /// Cypher query extensions
    cypher_extensions: Arc<CypherExtensions>,
    /// Batch operations manager
    batch_reifier: Arc<BatchReifier>,
    /// Temporal versioning manager
    temporal_reifier: Option<Arc<TemporalReifier>>,
    /// Engine configuration
    config: Neo4jReifiedConfig,
    /// Performance statistics
    stats: Arc<EngineStats>,
    /// Query cache for performance optimization
    query_cache: Arc<DashMap<String, Arc<CypherResult>>>,
    /// Node cache for frequently accessed nodes
    node_cache: Arc<DashMap<EntityId, Arc<ReifiedNode>>>,
    /// Relationship cache
    relationship_cache: Arc<DashMap<EntityId, Arc<ReifiedRelationship>>>,
}

impl Neo4jReifiedEngine {
    /// Create a new Neo4j reified engine with default configuration
    pub async fn connect(
        uri: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Result<Self> {
        let config = Neo4jReifiedConfig::with_connection(uri, username, password);
        Self::connect_with_config(config).await
    }

    /// Create a new engine with custom configuration
    pub async fn connect_with_config(config: Neo4jReifiedConfig) -> Result<Self> {
        info!("Connecting to Neo4j at {}", config.uri);

        // Create Neo4j connection
        let graph = Graph::new(&config.uri, &config.username, &config.password)
            .await
            .map_err(|e| Neo4jReifiedError::ConnectionError {
                operation: "connect".to_string(),
                details: format!("Failed to connect to Neo4j: {}", e),
            })?;

        // Verify connection
        let query = Query::new("RETURN 1 as test".to_string());
        graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ConnectionError {
                operation: "verify".to_string(),
                details: format!("Connection verification failed: {}", e),
            })?;

        let graph = Arc::new(graph);
        let stats = Arc::new(EngineStats::new());

        // Initialize components
        let connection_manager = Arc::new(ConnectionManager::new(graph.clone(), config.pool_config.clone()));
        let reification_manager = Arc::new(ReificationManager::new(graph.clone(), stats.clone())?);
        let cypher_extensions = Arc::new(CypherExtensions::new(graph.clone()));
        let batch_reifier = Arc::new(BatchReifier::new(graph.clone(), stats.clone(), config.bulk_batch_size));

        // Initialize temporal reifier if enabled
        let temporal_reifier = if config.enable_temporal_versioning {
            Some(Arc::new(TemporalReifier::new(graph.clone())?))
        } else {
            None
        };

        // Initialize caches
        let query_cache = Arc::new(DashMap::with_capacity(config.max_cache_size));
        let node_cache = Arc::new(DashMap::with_capacity(config.max_cache_size / 2));
        let relationship_cache = Arc::new(DashMap::with_capacity(config.max_cache_size / 2));

        info!("Neo4j reified engine initialized successfully");
        info!("Features enabled: reification={}, temporal={}, batch={}, caching={}",
            config.enable_advanced_reification,
            config.enable_temporal_versioning,
            config.enable_batch_operations,
            config.enable_query_caching
        );

        Ok(Self {
            graph,
            connection_manager,
            reification_manager,
            cypher_extensions,
            batch_reifier,
            temporal_reifier,
            config,
            stats,
            query_cache,
            node_cache,
            relationship_cache,
        })
    }

    /// Create a new node with labels and properties
    pub async fn create_node(&self, labels: Vec<String>, properties: PropertyMap) -> Result<ReifiedNode> {
        let mut node = ReifiedNode::new(labels, properties);
        
        // Build Cypher query for node creation
        let labels_str = node.labels.iter()
            .map(|l| format!(":{}", l))
            .collect::<Vec<_>>()
            .join("");

        let mut cypher = format!("CREATE (n{}) ", labels_str);
        let mut params = std::collections::HashMap::new();

        // Add properties if present
        if !node.properties.is_empty() {
            cypher.push_str("SET ");
            let prop_assignments: Vec<String> = node.properties.keys()
                .enumerate()
                .map(|(i, key)| {
                    let param_name = format!("prop_{}", i);
                    params.insert(param_name.clone(), node.properties[key].to_bolt_type());
                    format!("n.{} = ${}", key, param_name)
                })
                .collect();
            cypher.push_str(&prop_assignments.join(", "));
        }

        cypher.push_str(" RETURN id(n) as node_id");

        let mut query = Query::new(cypher);
        for (key, value) in params {
            query = query.param(key, value);
        }

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::Neo4jError {
                message: format!("Failed to create node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::Neo4jError {
                message: format!("Failed to read node creation result: {}", e),
            })? {
            let node_id: i64 = row.get("node_id")
                .map_err(|e| Neo4jReifiedError::Neo4jError {
                    message: format!("Failed to get node ID: {}", e),
                })?;
            
            node.id = EntityId::new(node_id);
            self.stats.inc_nodes_created();

            // Cache the node if caching is enabled
            if self.config.enable_query_caching {
                self.node_cache.insert(node.id, Arc::new(node.clone()));
            }

            info!("Created node {} with labels {:?}", node.id, node.labels);
            Ok(node)
        } else {
            Err(Neo4jReifiedError::Neo4jError {
                message: "No result returned from node creation".to_string(),
            })
        }
    }

    /// Get a node by ID
    pub async fn get_node(&self, id: EntityId) -> Result<Option<ReifiedNode>> {
        // Check cache first
        if self.config.enable_query_caching {
            if let Some(cached_node) = self.node_cache.get(&id) {
                return Ok(Some((**cached_node).clone()));
            }
        }

        let query = Query::new("MATCH (n) WHERE id(n) = $id RETURN n, labels(n) as node_labels".to_string())
            .param("id", id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::Neo4jError {
                message: format!("Failed to get node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::Neo4jError {
                message: format!("Failed to read node result: {}", e),
            })? {
            let neo4j_node: neo4rs::Node = row.get("n")
                .map_err(|e| Neo4jReifiedError::Neo4jError {
                    message: format!("Failed to get node data: {}", e),
                })?;

            let node = ReifiedNode::from_neo4j_node(&neo4j_node);

            // Cache the node
            if self.config.enable_query_caching {
                self.node_cache.insert(id, Arc::new(node.clone()));
            }

            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    /// Create a reified relationship between two nodes
    pub async fn reify_relationship(
        &self,
        from_node_id: EntityId,
        to_node_id: EntityId,
        relationship_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Result<ReificationStructure> {
        let relationship_type = relationship_type.into();
        
        if !self.config.enable_advanced_reification {
            return Err(Neo4jReifiedError::ReificationError {
                operation: "reify_relationship".to_string(),
                details: "Advanced reification is not enabled".to_string(),
            });
        }

        self.reification_manager.reify_relationship(
            from_node_id,
            to_node_id,
            relationship_type,
            properties,
        ).await
    }

    /// Execute a Cypher query with reification support
    pub async fn execute_cypher(&self, cypher: &str) -> Result<CypherResult> {
        let start_time = std::time::Instant::now();

        // Check cache first
        let cache_key = cypher.to_string();
        if self.config.enable_query_caching {
            if let Some(cached_result) = self.query_cache.get(&cache_key) {
                return Ok((**cached_result).clone());
            }
        }

        let query = Query::new(cypher.to_string());
        let mut neo4j_result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: cypher.to_string(),
                message: format!("Query execution failed: {}", e),
            })?;

        let mut columns = Vec::new();
        let mut rows = Vec::new();

        // Get column names from first row
        if let Some(first_row) = neo4j_result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: cypher.to_string(),
                message: format!("Failed to read query result: {}", e),
            })? {
            columns = first_row.keys().map(|k| k.to_string()).collect();
            
            // Process first row
            let mut row_values = Vec::new();
            for column in &columns {
                let value = first_row.get::<BoltType>(column)
                    .map_err(|e| Neo4jReifiedError::CypherError {
                        query: cypher.to_string(),
                        message: format!("Failed to get column '{}': {}", column, e),
                    })?;
                row_values.push(PropertyValue::from_bolt_type(&value));
            }
            rows.push(row_values);

            // Process remaining rows
            while let Some(row) = neo4j_result.next().await
                .map_err(|e| Neo4jReifiedError::CypherError {
                    query: cypher.to_string(),
                    message: format!("Failed to read query result: {}", e),
                })? {
                let mut row_values = Vec::new();
                for column in &columns {
                    let value = row.get::<BoltType>(column)
                        .map_err(|e| Neo4jReifiedError::CypherError {
                            query: cypher.to_string(),
                            message: format!("Failed to get column '{}': {}", column, e),
                        })?;
                    row_values.push(PropertyValue::from_bolt_type(&value));
                }
                rows.push(row_values);
            }
        }

        let execution_time = start_time.elapsed().as_micros() as u64;
        
        let result = CypherResult {
            columns,
            rows,
            stats: QueryStats {
                execution_time_us: execution_time,
                ..Default::default()
            },
            reification_info: ReificationInfo::default(),
        };

        // Update statistics
        self.stats.inc_queries_executed();
        self.stats.add_query_time(execution_time);

        // Cache result if caching is enabled
        if self.config.enable_query_caching && result.len() < 1000 { // Don't cache large results
            self.query_cache.insert(cache_key, Arc::new(result.clone()));
        }

        Ok(result)
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Clear all caches
    pub fn clear_caches(&self) {
        self.query_cache.clear();
        self.node_cache.clear();
        self.relationship_cache.clear();
        info!("All caches cleared");
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize, usize) {
        (
            self.query_cache.len(),
            self.node_cache.len(),
            self.relationship_cache.len(),
        )
    }

    /// Connect to a specific database
    pub async fn with_database(&self, database: impl Into<String>) -> Result<Self> {
        let mut new_config = self.config.clone();
        new_config.database = Some(database.into());
        Self::connect_with_config(new_config).await
    }

    /// Execute a transaction with multiple operations
    pub async fn execute_transaction<F, T>(&self, operations: F) -> Result<T>
    where
        F: FnOnce(&Self) -> Box<dyn std::future::Future<Output = Result<T>> + Send + '_> + Send,
        T: Send,
    {
        // For now, execute operations directly
        // In a full implementation, this would use Neo4j transactions
        let future = operations(self);
        future.await
    }

    /// Health check for the engine
    pub async fn health_check(&self) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        let query = Query::new("RETURN 'health_check' as status, timestamp() as ts".to_string());
        match self.graph.execute(query).await {
            Ok(mut result) => {
                if let Some(_row) = result.next().await.map_err(|e| Neo4jReifiedError::Neo4jError {
                    message: format!("Health check query failed: {}", e),
                })? {
                    let duration = start_time.elapsed();
                    info!("Health check passed in {:?}", duration);
                    Ok(true)
                } else {
                    warn!("Health check returned no results");
                    Ok(false)
                }
            }
            Err(e) => {
                error!("Health check failed: {}", e);
                Err(Neo4jReifiedError::Neo4jError {
                    message: format!("Health check failed: {}", e),
                })
            }
        }
    }

    /// Get the Neo4j graph instance for direct access
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    /// Get the reification manager
    pub fn reification_manager(&self) -> &Arc<ReificationManager> {
        &self.reification_manager
    }

    /// Get the batch reifier for bulk operations
    pub fn batch_reifier(&self) -> &Arc<BatchReifier> {
        &self.batch_reifier
    }

    /// Get the temporal reifier if enabled
    pub fn temporal_reifier(&self) -> Option<&Arc<TemporalReifier>> {
        self.temporal_reifier.as_ref()
    }
}

impl Drop for Neo4jReifiedEngine {
    fn drop(&mut self) {
        info!("Neo4j reified engine shutting down");
        
        // Log final statistics
        let stats = self.get_stats();
        info!("Final statistics: {} nodes created, {} relationships created, {} queries executed",
            stats.nodes_created.load(std::sync::atomic::Ordering::Relaxed),
            stats.relationships_created.load(std::sync::atomic::Ordering::Relaxed),
            stats.queries_executed.load(std::sync::atomic::Ordering::Relaxed)
        );
        
        info!("Average query time: {:.2}µs", stats.average_query_time_us());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties;

    // Note: These tests require a running Neo4j instance
    // They are integration tests and should be run with `cargo test --ignored`

    #[tokio::test]
    #[ignore]
    async fn test_engine_creation() {
        let engine = Neo4jReifiedEngine::connect(
            "bolt://localhost:7687",
            "neo4j",
            "password"
        ).await.unwrap();

        assert!(engine.health_check().await.unwrap());
    }

    #[tokio::test]
    #[ignore]
    async fn test_node_creation() {
        let engine = Neo4jReifiedEngine::connect(
            "bolt://localhost:7687",
            "neo4j",
            "password"
        ).await.unwrap();

        let properties = properties!(
            "name" => "Alice",
            "age" => 30,
            "active" => true
        );

        let node = engine.create_node(
            vec!["Person".to_string(), "Employee".to_string()],
            properties
        ).await.unwrap();

        assert!(!node.id.is_temporary());
        assert!(node.has_label("Person"));
        assert!(node.has_label("Employee"));
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("Alice".to_string())));
    }

    #[tokio::test]
    #[ignore]
    async fn test_node_retrieval() {
        let engine = Neo4jReifiedEngine::connect(
            "bolt://localhost:7687",
            "neo4j",
            "password"
        ).await.unwrap();

        // Create a node first
        let properties = properties!("name" => "Bob", "title" => "Engineer");
        let created_node = engine.create_node(
            vec!["Person".to_string()],
            properties
        ).await.unwrap();

        // Retrieve the node
        let retrieved_node = engine.get_node(created_node.id).await.unwrap();
        assert!(retrieved_node.is_some());
        
        let node = retrieved_node.unwrap();
        assert_eq!(node.id, created_node.id);
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("Bob".to_string())));
    }

    #[tokio::test]
    #[ignore]
    async fn test_cypher_execution() {
        let engine = Neo4jReifiedEngine::connect(
            "bolt://localhost:7687",
            "neo4j",
            "password"
        ).await.unwrap();

        let result = engine.execute_cypher("RETURN 42 as answer, 'hello' as greeting").await.unwrap();
        
        assert_eq!(result.columns.len(), 2);
        assert_eq!(result.rows.len(), 1);
        assert!(result.columns.contains(&"answer".to_string()));
        assert!(result.columns.contains(&"greeting".to_string()));
    }

    #[tokio::test]
    #[ignore]
    async fn test_cache_functionality() {
        let mut config = Neo4jReifiedConfig::default();
        config.enable_query_caching = true;
        
        let engine = Neo4jReifiedEngine::connect_with_config(config).await.unwrap();

        // Execute the same query twice
        let query = "RETURN timestamp() as ts";
        let _result1 = engine.execute_cypher(query).await.unwrap();
        let _result2 = engine.execute_cypher(query).await.unwrap();

        let (query_cache_size, _, _) = engine.get_cache_stats();
        assert!(query_cache_size > 0);

        engine.clear_caches();
        let (query_cache_size, _, _) = engine.get_cache_stats();
        assert_eq!(query_cache_size, 0);
    }
}