//! Main KuzuReifiedEngine implementation
//!
//! This module provides the core engine that integrates Kuzu's columnar graph database
//! with edge reification capabilities, allowing relationships to be treated as first-class nodes.

use crate::reification::ReificationManager;
use crate::schema::SchemaManager;
use crate::query::QueryExecutor;
use crate::types::*;
use crate::{KuzuReifiedConfig, KuzuReifiedError, Result};

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::time::Instant;

/// Main Kuzu reified engine providing high-performance graph operations with edge reification
pub struct KuzuReifiedEngine {
    /// Engine configuration
    config: KuzuReifiedConfig,
    /// Schema manager for type validation and schema evolution
    schema_manager: Arc<SchemaManager>,
    /// Reification manager for converting edges to nodes
    reification_manager: Arc<ReificationManager>,
    /// Query executor with Cypher extensions
    query_executor: Arc<QueryExecutor>,
    /// Engine statistics
    stats: Arc<EngineStats>,
    /// Connection pool for concurrent operations
    connection_pool: Arc<RwLock<Vec<Arc<Mutex<KuzuConnection>>>>>,
    /// Current database state
    is_initialized: Arc<RwLock<bool>>,
}

/// Kuzu connection wrapper for thread-safe operations
pub struct KuzuConnection {
    /// The actual Kuzu database connection
    db: kuzu::Database,
    /// Connection statistics
    stats: ConnectionStats,
    /// Last used timestamp for connection pooling
    last_used: Instant,
}

/// Connection-specific statistics
#[derive(Debug, Default)]
pub struct ConnectionStats {
    /// Total queries executed on this connection
    pub queries_executed: u64,
    /// Total execution time in microseconds
    pub total_execution_time_us: u64,
    /// Number of active transactions
    pub active_transactions: u32,
}

impl KuzuReifiedEngine {
    /// Create a new Kuzu reified engine with default configuration
    pub async fn new(database_path: impl Into<String>) -> Result<Self> {
        let config = KuzuReifiedConfig {
            database_path: database_path.into(),
            ..Default::default()
        };
        
        Self::with_config(config).await
    }
    
    /// Create a new engine with custom configuration
    pub async fn with_config(config: KuzuReifiedConfig) -> Result<Self> {
        info!("Initializing Kuzu Reified Engine with config: {:?}", config);
        
        // Initialize schema manager
        let schema_manager = Arc::new(SchemaManager::new(&config).await?);
        
        // Initialize reification manager
        let reification_manager = Arc::new(ReificationManager::new(&config, schema_manager.clone()).await?);
        
        // Initialize query executor
        let query_executor = Arc::new(QueryExecutor::new(&config, schema_manager.clone(), reification_manager.clone()).await?);
        
        // Create connection pool
        let connection_pool = Arc::new(RwLock::new(Vec::new()));
        
        let engine = Self {
            config,
            schema_manager,
            reification_manager,
            query_executor,
            stats: Arc::new(EngineStats::new()),
            connection_pool,
            is_initialized: Arc::new(RwLock::new(false)),
        };
        
        // Initialize the engine
        engine.initialize().await?;
        
        info!("Kuzu Reified Engine initialized successfully");
        Ok(engine)
    }
    
    /// Initialize the engine and create initial connections
    async fn initialize(&self) -> Result<()> {
        debug!("Initializing engine components...");
        
        // Initialize the Kuzu database
        let db = kuzu::Database::new(&self.config.database_path, self.config.buffer_pool_size_mb)
            .map_err(|e| KuzuReifiedError::KuzuError { 
                message: format!("Failed to create Kuzu database: {}", e) 
            })?;
        
        // Create initial connection pool
        let mut pool = self.connection_pool.write().await;
        for i in 0..self.config.num_threads {
            let connection = KuzuConnection {
                db: db.clone(),
                stats: ConnectionStats::default(),
                last_used: Instant::now(),
            };
            pool.push(Arc::new(Mutex::new(connection)));
            debug!("Created connection {} in pool", i);
        }
        drop(pool);
        
        // Initialize schema
        self.schema_manager.initialize().await?;
        
        // Initialize reification system
        self.reification_manager.initialize().await?;
        
        // Initialize query executor
        self.query_executor.initialize().await?;
        
        // Mark as initialized
        *self.is_initialized.write().await = true;
        
        info!("Engine initialization complete");
        Ok(())
    }
    
    /// Create a new node in the graph
    pub async fn create_node(&self, label: impl Into<String>, properties: PropertyMap) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let label = label.into();
        let node = Node::new(label.clone(), properties);
        
        // Validate against schema
        self.schema_manager.validate_node(&node).await?;
        
        // Execute node creation
        let node_id = self.execute_node_creation(node).await?;
        
        // Update statistics
        self.stats.inc_nodes_created();
        
        debug!("Created node {} with label '{}'", node_id, label);
        Ok(node_id)
    }
    
    /// Create a new relationship between nodes
    pub async fn create_relationship(
        &self,
        from: EntityId,
        to: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let rel_type = rel_type.into();
        let relationship = Relationship::new(from, to, rel_type.clone(), properties);
        
        // Validate against schema
        self.schema_manager.validate_relationship(&relationship).await?;
        
        // Execute relationship creation
        let rel_id = self.execute_relationship_creation(relationship).await?;
        
        // Update statistics
        self.stats.inc_relationships_created();
        
        debug!("Created relationship {} of type '{}'", rel_id, rel_type);
        Ok(rel_id)
    }
    
    /// Reify a relationship, converting it to a node
    pub async fn reify_relationship(
        &self,
        from: EntityId,
        to: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let rel_type = rel_type.into();
        
        // Use the reification manager to convert the relationship
        let reified_node_id = self.reification_manager
            .reify_relationship(from, to, rel_type, properties)
            .await?;
        
        // Update statistics
        self.stats.inc_reifications_performed();
        
        debug!("Reified relationship into node {}", reified_node_id);
        Ok(reified_node_id)
    }
    
    /// Execute a Cypher query with reification extensions
    pub async fn execute_cypher(&self, query: impl Into<String>) -> Result<CypherResult> {
        self.ensure_initialized().await?;
        
        let query = query.into();
        let start_time = Instant::now();
        
        // Execute query through the query executor
        let result = self.query_executor.execute(&query).await?;
        
        // Update statistics
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.inc_queries_executed();
        self.stats.add_query_time(execution_time);
        
        debug!("Executed query in {}μs: {}", execution_time, query);
        Ok(result)
    }
    
    /// Get a node by its ID
    pub async fn get_node(&self, node_id: EntityId) -> Result<Option<Node>> {
        self.ensure_initialized().await?;
        
        let query = format!(
            "MATCH (n) WHERE id(n) = '{}' RETURN n",
            node_id.to_string()
        );
        
        let result = self.execute_cypher(query).await?;
        
        if result.is_empty() {
            return Ok(None);
        }
        
        // Parse the result back to a Node
        self.parse_node_from_result(&result).await
    }
    
    /// Get a relationship by its ID
    pub async fn get_relationship(&self, rel_id: EntityId) -> Result<Option<Relationship>> {
        self.ensure_initialized().await?;
        
        let query = format!(
            "MATCH ()-[r]->() WHERE id(r) = '{}' RETURN r",
            rel_id.to_string()
        );
        
        let result = self.execute_cypher(query).await?;
        
        if result.is_empty() {
            return Ok(None);
        }
        
        // Parse the result back to a Relationship
        self.parse_relationship_from_result(&result).await
    }
    
    /// Update node properties
    pub async fn update_node(&self, node_id: EntityId, properties: PropertyMap) -> Result<()> {
        self.ensure_initialized().await?;
        
        // Build SET clause for properties
        let set_clauses: Vec<String> = properties
            .iter()
            .map(|(key, value)| {
                format!("n.{} = {}", key, self.property_value_to_cypher(value))
            })
            .collect();
        
        if set_clauses.is_empty() {
            return Ok(());
        }
        
        let query = format!(
            "MATCH (n) WHERE id(n) = '{}' SET {}",
            node_id.to_string(),
            set_clauses.join(", ")
        );
        
        self.execute_cypher(query).await?;
        debug!("Updated node {} properties", node_id);
        Ok(())
    }
    
    /// Delete a node and its relationships
    pub async fn delete_node(&self, node_id: EntityId) -> Result<()> {
        self.ensure_initialized().await?;
        
        let query = format!(
            "MATCH (n) WHERE id(n) = '{}' DETACH DELETE n",
            node_id.to_string()
        );
        
        self.execute_cypher(query).await?;
        debug!("Deleted node {}", node_id);
        Ok(())
    }
    
    /// Find shortest path between two nodes
    pub async fn find_shortest_path(&self, from: EntityId, to: EntityId) -> Result<Option<GraphPath>> {
        self.ensure_initialized().await?;
        
        let query = format!(
            "MATCH path = shortestPath((a)-[*]-(b)) WHERE id(a) = '{}' AND id(b) = '{}' RETURN path",
            from.to_string(),
            to.to_string()
        );
        
        let result = self.execute_cypher(query).await?;
        
        if result.is_empty() {
            return Ok(None);
        }
        
        // Parse the path from the result
        self.parse_path_from_result(&result).await
    }
    
    /// Get all reified relationships in the graph
    pub async fn get_reified_relationships(&self) -> Result<Vec<ReifiedRelationship>> {
        self.ensure_initialized().await?;
        
        self.reification_manager.get_all_reified_relationships().await
    }
    
    /// Get engine statistics
    pub fn get_stats(&self) -> &EngineStats {
        &self.stats
    }
    
    /// Get engine configuration
    pub fn get_config(&self) -> &KuzuReifiedConfig {
        &self.config
    }
    
    /// Check if the engine is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.is_initialized.read().await
    }
    
    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Kuzu Reified Engine...");
        
        // Shutdown query executor
        self.query_executor.shutdown().await?;
        
        // Shutdown reification manager
        self.reification_manager.shutdown().await?;
        
        // Shutdown schema manager
        self.schema_manager.shutdown().await?;
        
        // Clear connection pool
        let mut pool = self.connection_pool.write().await;
        pool.clear();
        drop(pool);
        
        // Mark as not initialized
        *self.is_initialized.write().await = false;
        
        info!("Engine shutdown complete");
        Ok(())
    }
    
    // Private helper methods
    
    /// Ensure the engine is initialized
    async fn ensure_initialized(&self) -> Result<()> {
        if !self.is_initialized().await {
            return Err(KuzuReifiedError::Internal {
                details: "Engine not initialized".to_string(),
            });
        }
        Ok(())
    }
    
    /// Execute node creation in Kuzu
    async fn execute_node_creation(&self, node: Node) -> Result<EntityId> {
        let properties_cypher = self.properties_to_cypher(&node.properties);
        let query = format!(
            "CREATE (n:{} {}) RETURN id(n)",
            node.label,
            properties_cypher
        );
        
        let result = self.execute_cypher(query).await?;
        
        if result.is_empty() {
            return Err(KuzuReifiedError::Internal {
                details: "Failed to create node - no ID returned".to_string(),
            });
        }
        
        // Extract the node ID from the result
        self.extract_entity_id_from_result(&result, 0, 0)
    }
    
    /// Execute relationship creation in Kuzu
    async fn execute_relationship_creation(&self, relationship: Relationship) -> Result<EntityId> {
        let properties_cypher = self.properties_to_cypher(&relationship.properties);
        let query = format!(
            "MATCH (a), (b) WHERE id(a) = '{}' AND id(b) = '{}' CREATE (a)-[r:{} {}]->(b) RETURN id(r)",
            relationship.from.to_string(),
            relationship.to.to_string(),
            relationship.rel_type,
            properties_cypher
        );
        
        let result = self.execute_cypher(query).await?;
        
        if result.is_empty() {
            return Err(KuzuReifiedError::Internal {
                details: "Failed to create relationship - no ID returned".to_string(),
            });
        }
        
        // Extract the relationship ID from the result
        self.extract_entity_id_from_result(&result, 0, 0)
    }
    
    /// Convert properties to Cypher syntax
    fn properties_to_cypher(&self, properties: &PropertyMap) -> String {
        if properties.is_empty() {
            return "{}".to_string();
        }
        
        let prop_strings: Vec<String> = properties
            .iter()
            .map(|(key, value)| {
                format!("{}: {}", key, self.property_value_to_cypher(value))
            })
            .collect();
        
        format!("{{{}}}", prop_strings.join(", "))
    }
    
    /// Convert a property value to Cypher syntax
    fn property_value_to_cypher(&self, value: &PropertyValue) -> String {
        match value {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => format!("'{}'", s.replace('\'', "\\'")),
            PropertyValue::Array(arr) => {
                let elements: Vec<String> = arr
                    .iter()
                    .map(|v| self.property_value_to_cypher(v))
                    .collect();
                format!("[{}]", elements.join(", "))
            }
            PropertyValue::Object(_) => "{}".to_string(), // Simplified for now
            PropertyValue::Binary(_) => "null".to_string(), // Not directly supported
            PropertyValue::Timestamp(ts) => {
                format!("datetime('{:?}')", ts)
            }
            PropertyValue::Uuid(uuid) => format!("'{}'", uuid.to_string()),
        }
    }
    
    /// Parse a node from query result
    async fn parse_node_from_result(&self, result: &CypherResult) -> Result<Option<Node>> {
        // This is a simplified implementation
        // In a real implementation, you would parse the Kuzu result format
        if let Some(row) = result.get_row(0) {
            if let Some(PropertyValue::Object(props)) = row.get(0) {
                let mut node_props = PropertyMap::new();
                for (key, value) in props {
                    node_props.insert(key.clone(), value.clone());
                }
                
                let node = Node::new("Unknown", node_props); // Label would be parsed from result
                return Ok(Some(node));
            }
        }
        Ok(None)
    }
    
    /// Parse a relationship from query result
    async fn parse_relationship_from_result(&self, result: &CypherResult) -> Result<Option<Relationship>> {
        // Simplified implementation
        if let Some(row) = result.get_row(0) {
            if let Some(PropertyValue::Object(props)) = row.get(0) {
                let mut rel_props = PropertyMap::new();
                for (key, value) in props {
                    rel_props.insert(key.clone(), value.clone());
                }
                
                // These would be parsed from the actual result
                let from = EntityId::new();
                let to = EntityId::new();
                let rel = Relationship::new(from, to, "UNKNOWN", rel_props);
                return Ok(Some(rel));
            }
        }
        Ok(None)
    }
    
    /// Parse a graph path from query result
    async fn parse_path_from_result(&self, result: &CypherResult) -> Result<Option<GraphPath>> {
        // Simplified implementation
        if !result.is_empty() {
            let mut path = GraphPath::new();
            // Parse nodes and relationships from the path result
            // This would involve parsing Kuzu's path format
            return Ok(Some(path));
        }
        Ok(None)
    }
    
    /// Extract entity ID from query result
    fn extract_entity_id_from_result(&self, result: &CypherResult, row: usize, col: usize) -> Result<EntityId> {
        if let Some(row_data) = result.get_row(row) {
            if let Some(value) = row_data.get(col) {
                match value {
                    PropertyValue::String(s) => {
                        EntityId::from_string(s).map_err(|e| KuzuReifiedError::Internal {
                            details: format!("Failed to parse entity ID: {}", e),
                        })
                    }
                    PropertyValue::Uuid(uuid) => Ok(EntityId::from_uuid(*uuid)),
                    _ => Err(KuzuReifiedError::Internal {
                        details: "Invalid entity ID format in result".to_string(),
                    }),
                }
            } else {
                Err(KuzuReifiedError::Internal {
                    details: "No entity ID found in result".to_string(),
                })
            }
        } else {
            Err(KuzuReifiedError::Internal {
                details: "No row found in result".to_string(),
            })
        }
    }
}

impl Drop for KuzuReifiedEngine {
    fn drop(&mut self) {
        // Async shutdown would need to be handled differently in practice
        warn!("KuzuReifiedEngine dropped - ensure shutdown() was called");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_engine() -> (KuzuReifiedEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        
        let config = KuzuReifiedConfig {
            database_path: db_path.to_string_lossy().to_string(),
            ..KuzuReifiedConfig::development()
        };
        
        let engine = KuzuReifiedEngine::with_config(config).await.unwrap();
        (engine, temp_dir)
    }
    
    #[tokio::test]
    async fn test_engine_creation() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        assert!(engine.is_initialized().await);
        assert_eq!(engine.get_stats().nodes_created.load(std::sync::atomic::Ordering::Relaxed), 0);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_node_creation() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let node_id = engine.create_node("Person", props).await.unwrap();
        
        assert_ne!(node_id.to_string(), "");
        assert_eq!(engine.get_stats().nodes_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_relationship_creation() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        // Create nodes first
        let person1 = engine.create_node("Person", crate::properties!("name" => "Alice")).await.unwrap();
        let person2 = engine.create_node("Person", crate::properties!("name" => "Bob")).await.unwrap();
        
        // Create relationship
        let rel_props = crate::properties!("since" => "2020-01-01");
        let rel_id = engine.create_relationship(person1, person2, "KNOWS", rel_props).await.unwrap();
        
        assert_ne!(rel_id.to_string(), "");
        assert_eq!(engine.get_stats().relationships_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reification() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        // Create nodes
        let person = engine.create_node("Person", crate::properties!("name" => "Alice")).await.unwrap();
        let company = engine.create_node("Company", crate::properties!("name" => "TechCorp")).await.unwrap();
        
        // Reify relationship
        let employment_props = crate::properties!("salary" => 75000, "since" => "2020-01-01");
        let reified_id = engine.reify_relationship(person, company, "WORKS_FOR", employment_props).await.unwrap();
        
        assert_ne!(reified_id.to_string(), "");
        assert_eq!(engine.get_stats().reifications_performed.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_cypher_execution() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        // Create a node first
        let _node_id = engine.create_node("Person", crate::properties!("name" => "Alice")).await.unwrap();
        
        // Execute a query
        let result = engine.execute_cypher("MATCH (p:Person) RETURN p.name").await.unwrap();
        
        assert!(!result.is_empty());
        assert_eq!(engine.get_stats().queries_executed.load(std::sync::atomic::Ordering::Relaxed), 2); // Including the CREATE
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_engine_shutdown() {
        let (engine, _temp_dir) = create_test_engine().await;
        
        assert!(engine.is_initialized().await);
        
        engine.shutdown().await.unwrap();
        
        assert!(!engine.is_initialized().await);
    }
}