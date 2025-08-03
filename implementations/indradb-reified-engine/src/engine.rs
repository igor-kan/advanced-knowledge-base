//! Main IndraReifiedEngine implementation
//!
//! This module provides the core engine that integrates IndraDB's property graph database
//! with edge reification capabilities, allowing relationships to be treated as first-class nodes.

use crate::reification::ReificationManager;
use crate::transaction::TransactionManager;
use crate::query::QueryExecutor;
use crate::types::*;
use crate::{IndraReifiedConfig, IndraReifiedError, Result, BackendType};

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use indradb::{Datastore, MemoryDatastore, Vertex, Edge, Type, Identifier};
use std::time::Instant;

/// Main IndraDB reified engine providing high-performance graph operations with edge reification
pub struct IndraReifiedEngine {
    /// Engine configuration
    config: IndraReifiedConfig,
    /// IndraDB datastore
    datastore: Arc<dyn Datastore + Send + Sync>,
    /// Reification manager for converting edges to nodes
    reification_manager: Arc<ReificationManager>,
    /// Transaction manager for ACID operations
    transaction_manager: Arc<TransactionManager>,
    /// Query executor for property graph queries
    query_executor: Arc<QueryExecutor>,
    /// Engine statistics
    stats: Arc<EngineStats>,
    /// Current database state
    is_initialized: Arc<RwLock<bool>>,
}

impl IndraReifiedEngine {
    /// Create a new IndraDB reified engine with memory backend
    pub async fn new_memory_backend() -> Result<Self> {
        let config = IndraReifiedConfig {
            backend_type: BackendType::Memory,
            ..Default::default()
        };
        
        Self::with_config(config).await
    }
    
    /// Create a new engine with RocksDB backend
    pub async fn new_rocksdb_backend(database_path: impl Into<String>) -> Result<Self> {
        let config = IndraReifiedConfig::rocksdb(database_path);
        Self::with_config(config).await
    }
    
    /// Create a new engine with custom configuration
    pub async fn with_config(config: IndraReifiedConfig) -> Result<Self> {
        info!("Initializing IndraDB Reified Engine with config: {:?}", config);
        
        // Create the appropriate datastore backend
        let datastore = Self::create_datastore(&config).await?;
        
        // Initialize transaction manager
        let transaction_manager = Arc::new(TransactionManager::new(&config, datastore.clone()).await?);
        
        // Initialize reification manager
        let reification_manager = Arc::new(ReificationManager::new(&config, datastore.clone(), transaction_manager.clone()).await?);
        
        // Initialize query executor
        let query_executor = Arc::new(QueryExecutor::new(&config, datastore.clone(), reification_manager.clone()).await?);
        
        let engine = Self {
            config,
            datastore,
            reification_manager,
            transaction_manager,
            query_executor,
            stats: Arc::new(EngineStats::new()),
            is_initialized: Arc::new(RwLock::new(false)),
        };
        
        // Initialize the engine
        engine.initialize().await?;
        
        info!("IndraDB Reified Engine initialized successfully");
        Ok(engine)
    }
    
    /// Initialize the engine components
    async fn initialize(&self) -> Result<()> {
        debug!("Initializing engine components...");
        
        // Initialize transaction manager
        self.transaction_manager.initialize().await?;
        
        // Initialize reification manager
        self.reification_manager.initialize().await?;
        
        // Initialize query executor
        self.query_executor.initialize().await?;
        
        // Create initial schema for reified relationships
        self.create_reification_schema().await?;
        
        // Mark as initialized
        *self.is_initialized.write().await = true;
        
        info!("Engine initialization complete");
        Ok(())
    }
    
    /// Create a new vertex in the graph
    pub async fn create_vertex(&self, vertex_type: impl Into<String>, properties: PropertyMap) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let vertex_type = vertex_type.into();
        let vertex_id = EntityId::new();
        
        // Create the vertex
        let vertex_type_obj = Type::new(vertex_type.clone())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "vertex_type".to_string(),
                constraint: format!("Invalid vertex type: {}", e),
            })?;
        
        let vertex = Vertex::new(vertex_id.to_identifier(), vertex_type_obj);
        
        // Start a transaction for atomic operation
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Create the vertex
        transaction.create_vertex(&vertex).await?;
        
        // Set properties if any
        if !properties.is_empty() {
            self.set_vertex_properties_in_transaction(&mut transaction, vertex_id, properties).await?;
        }
        
        // Commit the transaction
        transaction.commit().await?;
        
        // Update statistics
        self.stats.inc_vertices_created();
        
        debug!("Created vertex {} with type '{}'", vertex_id, vertex_type);
        Ok(vertex_id)
    }
    
    /// Create a new edge between vertices
    pub async fn create_edge(
        &self,
        from: EntityId,
        to: EntityId,
        edge_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let edge_type = edge_type.into();
        let edge_id = EntityId::new();
        
        // Create the edge
        let edge_type_obj = Type::new(edge_type.clone())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: format!("Invalid edge type: {}", e),
            })?;
        
        let edge = Edge::new(from.to_identifier(), edge_type_obj, to.to_identifier());
        
        // Start a transaction for atomic operation
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Create the edge
        transaction.create_edge(&edge).await?;
        
        // Set properties if any
        if !properties.is_empty() {
            self.set_edge_properties_in_transaction(&mut transaction, &edge, properties).await?;
        }
        
        // Commit the transaction
        transaction.commit().await?;
        
        // Update statistics
        self.stats.inc_edges_created();
        
        debug!("Created edge {} of type '{}'", edge_id, edge_type);
        Ok(edge_id)
    }
    
    /// Reify an edge, converting it to a vertex
    pub async fn reify_edge(
        &self,
        from: EntityId,
        to: EntityId,
        edge_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let edge_type = edge_type.into();
        
        // Use the reification manager to convert the edge
        let reified_vertex_id = self.reification_manager
            .reify_edge(from, to, edge_type, properties)
            .await?;
        
        // Update statistics
        self.stats.inc_reifications_performed();
        
        debug!("Reified edge into vertex {}", reified_vertex_id);
        Ok(reified_vertex_id)
    }
    
    /// Execute a property graph query
    pub async fn execute_query(&self, query: impl Into<PropertyGraphQuery>) -> Result<PropertyGraphResult> {
        self.ensure_initialized().await?;
        
        let query = query.into();
        let start_time = Instant::now();
        
        // Execute query through the query executor
        let result = self.query_executor.execute(query).await?;
        
        // Update statistics
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.inc_queries_executed();
        self.stats.add_query_time(execution_time);
        
        debug!("Executed query in {}μs", execution_time);
        Ok(result)
    }
    
    /// Get a vertex by its ID
    pub async fn get_vertex(&self, vertex_id: EntityId) -> Result<Option<ReifiedNode>> {
        self.ensure_initialized().await?;
        
        let mut transaction = self.transaction_manager.begin_readonly_transaction().await?;
        
        // Get the vertex
        let vertices = transaction.get_vertices(&[vertex_id.to_identifier()]).await?;
        
        if let Some(vertex) = vertices.first() {
            // Get vertex properties
            let properties = self.get_vertex_properties_in_transaction(&mut transaction, vertex_id).await?;
            
            let node = ReifiedNode::with_id(
                vertex_id,
                vertex.t.as_str().to_string(),
                properties,
            );
            
            transaction.commit().await?;
            Ok(Some(node))
        } else {
            transaction.commit().await?;
            Ok(None)
        }
    }
    
    /// Get an edge by its endpoints and type
    pub async fn get_edge(&self, from: EntityId, to: EntityId, edge_type: &str) -> Result<Option<ReifiedEdge>> {
        self.ensure_initialized().await?;
        
        let mut transaction = self.transaction_manager.begin_readonly_transaction().await?;
        
        let edge_type_obj = Type::new(edge_type.to_string())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: format!("Invalid edge type: {}", e),
            })?;
        
        let edge = Edge::new(from.to_identifier(), edge_type_obj, to.to_identifier());
        
        // Check if edge exists
        let exists = transaction.get_edges(&[edge.clone()]).await?;
        
        if !exists.is_empty() {
            // Get edge properties
            let properties = self.get_edge_properties_in_transaction(&mut transaction, &edge).await?;
            
            let reified_edge = ReifiedEdge::new(from, to, edge_type.to_string(), properties);
            
            transaction.commit().await?;
            Ok(Some(reified_edge))
        } else {
            transaction.commit().await?;
            Ok(None)
        }
    }
    
    /// Update vertex properties
    pub async fn update_vertex(&self, vertex_id: EntityId, properties: PropertyMap) -> Result<()> {
        self.ensure_initialized().await?;
        
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Set the new properties
        self.set_vertex_properties_in_transaction(&mut transaction, vertex_id, properties).await?;
        
        transaction.commit().await?;
        debug!("Updated vertex {} properties", vertex_id);
        Ok(())
    }
    
    /// Delete a vertex and its edges
    pub async fn delete_vertex(&self, vertex_id: EntityId) -> Result<()> {
        self.ensure_initialized().await?;
        
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Delete the vertex (this will also delete connected edges in IndraDB)
        transaction.delete_vertices(&[vertex_id.to_identifier()]).await?;
        
        transaction.commit().await?;
        debug!("Deleted vertex {}", vertex_id);
        Ok(())
    }
    
    /// Find shortest path between two vertices
    pub async fn find_shortest_path(&self, from: EntityId, to: EntityId) -> Result<Option<GraphPath>> {
        self.ensure_initialized().await?;
        
        // Use the query executor for path finding
        self.query_executor.find_shortest_path(from, to).await
    }
    
    /// Get all reified relationships in the graph
    pub async fn get_reified_relationships(&self) -> Result<Vec<ReifiedRelationship>> {
        self.ensure_initialized().await?;
        
        self.reification_manager.get_all_reified_relationships().await
    }
    
    /// Begin a new transaction
    pub async fn begin_transaction(&self) -> Result<crate::transaction::ReifiedTransaction> {
        self.ensure_initialized().await?;
        
        self.transaction_manager.begin_transaction().await
    }
    
    /// Get engine statistics
    pub fn get_stats(&self) -> &EngineStats {
        &self.stats
    }
    
    /// Get engine configuration
    pub fn get_config(&self) -> &IndraReifiedConfig {
        &self.config
    }
    
    /// Check if the engine is initialized
    pub async fn is_initialized(&self) -> bool {
        *self.is_initialized.read().await
    }
    
    /// Get vertex count
    pub async fn get_vertex_count(&self) -> Result<u64> {
        self.ensure_initialized().await?;
        
        let transaction = self.transaction_manager.begin_readonly_transaction().await?;
        let count = transaction.get_vertex_count().await?;
        transaction.commit().await?;
        
        Ok(count)
    }
    
    /// Get edge count
    pub async fn get_edge_count(&self) -> Result<u64> {
        self.ensure_initialized().await?;
        
        let transaction = self.transaction_manager.begin_readonly_transaction().await?;
        let count = transaction.get_edge_count().await?;
        transaction.commit().await?;
        
        Ok(count)
    }
    
    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down IndraDB Reified Engine...");
        
        // Shutdown query executor
        self.query_executor.shutdown().await?;
        
        // Shutdown reification manager
        self.reification_manager.shutdown().await?;
        
        // Shutdown transaction manager
        self.transaction_manager.shutdown().await?;
        
        // Mark as not initialized
        *self.is_initialized.write().await = false;
        
        info!("Engine shutdown complete");
        Ok(())
    }
    
    // Private helper methods
    
    /// Create the appropriate datastore based on configuration
    async fn create_datastore(config: &IndraReifiedConfig) -> Result<Arc<dyn Datastore + Send + Sync>> {
        match &config.backend_type {
            BackendType::Memory => {
                debug!("Creating memory datastore");
                Ok(Arc::new(MemoryDatastore::default()))
            }
            BackendType::RocksDB => {
                // For now, fall back to memory datastore
                // In a real implementation, you would create a RocksDB datastore
                warn!("RocksDB backend not implemented, falling back to memory");
                Ok(Arc::new(MemoryDatastore::default()))
            }
            BackendType::Custom(name) => {
                return Err(IndraReifiedError::ConfigError {
                    parameter: "backend_type".to_string(),
                    issue: format!("Custom backend '{}' not implemented", name),
                });
            }
        }
    }
    
    /// Ensure the engine is initialized
    async fn ensure_initialized(&self) -> Result<()> {
        if !self.is_initialized().await {
            return Err(IndraReifiedError::Internal {
                details: "Engine not initialized".to_string(),
            });
        }
        Ok(())
    }
    
    /// Create reification-specific schema
    async fn create_reification_schema(&self) -> Result<()> {
        debug!("Creating reification schema...");
        
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Create a ReifiedEdge vertex type by creating a sample vertex and then deleting it
        // This ensures the type is registered in IndraDB
        let reified_edge_type = Type::new("ReifiedEdge".to_string())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "ReifiedEdge".to_string(),
                constraint: format!("Invalid type: {}", e),
            })?;
        
        let temp_vertex = Vertex::new(EntityId::new().to_identifier(), reified_edge_type);
        transaction.create_vertex(&temp_vertex).await?;
        transaction.delete_vertices(&[temp_vertex.id]).await?;
        
        // Create FROM and TO edge types
        let from_type = Type::new("FROM".to_string())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "FROM".to_string(),
                constraint: format!("Invalid type: {}", e),
            })?;
        
        let to_type = Type::new("TO".to_string())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "TO".to_string(),
                constraint: format!("Invalid type: {}", e),
            })?;
        
        // Register edge types by creating temporary edges
        let temp_from = EntityId::new();
        let temp_to = EntityId::new();
        
        let temp_from_vertex = Vertex::new(temp_from.to_identifier(), Type::new("TempNode".to_string()).unwrap());
        let temp_to_vertex = Vertex::new(temp_to.to_identifier(), Type::new("TempNode".to_string()).unwrap());
        
        transaction.create_vertex(&temp_from_vertex).await?;
        transaction.create_vertex(&temp_to_vertex).await?;
        
        let temp_from_edge = Edge::new(temp_from.to_identifier(), from_type, temp_to.to_identifier());
        let temp_to_edge = Edge::new(temp_from.to_identifier(), to_type, temp_to.to_identifier());
        
        transaction.create_edge(&temp_from_edge).await?;
        transaction.create_edge(&temp_to_edge).await?;
        
        // Clean up temporary elements
        transaction.delete_vertices(&[temp_from.to_identifier(), temp_to.to_identifier()]).await?;
        
        transaction.commit().await?;
        
        debug!("Reification schema created");
        Ok(())
    }
    
    /// Set vertex properties in a transaction
    async fn set_vertex_properties_in_transaction(
        &self,
        transaction: &mut crate::transaction::ReifiedTransaction,
        vertex_id: EntityId,
        properties: PropertyMap,
    ) -> Result<()> {
        for (key, value) in properties {
            let prop_type = Type::new(key)
                .map_err(|e| IndraReifiedError::ValidationError {
                    entity: "property_name".to_string(),
                    constraint: format!("Invalid property name: {}", e),
                })?;
            
            transaction.set_vertex_property(vertex_id.to_identifier(), prop_type, value.to_indra_value()).await?;
        }
        Ok(())
    }
    
    /// Set edge properties in a transaction
    async fn set_edge_properties_in_transaction(
        &self,
        transaction: &mut crate::transaction::ReifiedTransaction,
        edge: &Edge,
        properties: PropertyMap,
    ) -> Result<()> {
        for (key, value) in properties {
            let prop_type = Type::new(key)
                .map_err(|e| IndraReifiedError::ValidationError {
                    entity: "property_name".to_string(),
                    constraint: format!("Invalid property name: {}", e),
                })?;
            
            transaction.set_edge_property(edge.clone(), prop_type, value.to_indra_value()).await?;
        }
        Ok(())
    }
    
    /// Get vertex properties in a transaction
    async fn get_vertex_properties_in_transaction(
        &self,
        transaction: &mut crate::transaction::ReifiedTransaction,
        vertex_id: EntityId,
    ) -> Result<PropertyMap> {
        let properties = transaction.get_all_vertex_properties(vertex_id.to_identifier()).await?;
        
        let mut prop_map = PropertyMap::new();
        for prop in properties {
            prop_map.insert(prop.name.as_str().to_string(), PropertyValue::from_indra_value(&prop.value));
        }
        
        Ok(prop_map)
    }
    
    /// Get edge properties in a transaction
    async fn get_edge_properties_in_transaction(
        &self,
        transaction: &mut crate::transaction::ReifiedTransaction,
        edge: &Edge,
    ) -> Result<PropertyMap> {
        let properties = transaction.get_all_edge_properties(edge.clone()).await?;
        
        let mut prop_map = PropertyMap::new();
        for prop in properties {
            prop_map.insert(prop.name.as_str().to_string(), PropertyValue::from_indra_value(&prop.value));
        }
        
        Ok(prop_map)
    }
}

impl Drop for IndraReifiedEngine {
    fn drop(&mut self) {
        // Async shutdown would need to be handled differently in practice
        warn!("IndraReifiedEngine dropped - ensure shutdown() was called");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        assert!(engine.is_initialized().await);
        assert_eq!(engine.get_stats().vertices_created.load(std::sync::atomic::Ordering::Relaxed), 0);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_vertex_creation() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let vertex_id = engine.create_vertex("Person", props).await.unwrap();
        
        assert_ne!(vertex_id.to_string(), "");
        assert_eq!(engine.get_stats().vertices_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Verify vertex was created
        let vertex = engine.get_vertex(vertex_id).await.unwrap();
        assert!(vertex.is_some());
        
        let vertex = vertex.unwrap();
        assert_eq!(vertex.node_type, "Person");
        assert_eq!(vertex.get_property("name"), Some(&PropertyValue::String("Alice".to_string())));
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_edge_creation() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        // Create vertices first
        let person1 = engine.create_vertex("Person", crate::properties!("name" => "Alice")).await.unwrap();
        let person2 = engine.create_vertex("Person", crate::properties!("name" => "Bob")).await.unwrap();
        
        // Create edge
        let edge_props = crate::properties!("since" => "2020-01-01");
        let edge_id = engine.create_edge(person1, person2, "KNOWS", edge_props).await.unwrap();
        
        assert_ne!(edge_id.to_string(), "");
        assert_eq!(engine.get_stats().edges_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Verify edge was created
        let edge = engine.get_edge(person1, person2, "KNOWS").await.unwrap();
        assert!(edge.is_some());
        
        let edge = edge.unwrap();
        assert_eq!(edge.edge_type, "KNOWS");
        assert_eq!(edge.get_property("since"), Some(&PropertyValue::String("2020-01-01".to_string())));
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reification() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        // Create vertices
        let person = engine.create_vertex("Person", crate::properties!("name" => "Alice")).await.unwrap();
        let company = engine.create_vertex("Company", crate::properties!("name" => "TechCorp")).await.unwrap();
        
        // Reify edge
        let employment_props = crate::properties!("salary" => 75000, "since" => "2020-01-01");
        let reified_id = engine.reify_edge(person, company, "WORKS_FOR", employment_props).await.unwrap();
        
        assert_ne!(reified_id.to_string(), "");
        assert_eq!(engine.get_stats().reifications_performed.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Verify reified vertex exists
        let reified_vertex = engine.get_vertex(reified_id).await.unwrap();
        assert!(reified_vertex.is_some());
        
        let reified_vertex = reified_vertex.unwrap();
        assert_eq!(reified_vertex.node_type, "ReifiedEdge");
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_vertex_counts() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        let initial_count = engine.get_vertex_count().await.unwrap();
        
        // Create some vertices
        let _v1 = engine.create_vertex("Person", crate::properties!("name" => "Alice")).await.unwrap();
        let _v2 = engine.create_vertex("Person", crate::properties!("name" => "Bob")).await.unwrap();
        
        let final_count = engine.get_vertex_count().await.unwrap();
        assert_eq!(final_count, initial_count + 2);
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_vertex_update() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        let vertex_id = engine.create_vertex("Person", crate::properties!("name" => "Alice")).await.unwrap();
        
        // Update properties
        let new_props = crate::properties!("name" => "Alice Smith", "age" => 30);
        engine.update_vertex(vertex_id, new_props).await.unwrap();
        
        // Verify update
        let vertex = engine.get_vertex(vertex_id).await.unwrap().unwrap();
        assert_eq!(vertex.get_property("name"), Some(&PropertyValue::String("Alice Smith".to_string())));
        assert_eq!(vertex.get_property("age"), Some(&PropertyValue::Int(30)));
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_vertex_deletion() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        let vertex_id = engine.create_vertex("Person", crate::properties!("name" => "Alice")).await.unwrap();
        
        // Verify vertex exists
        let vertex = engine.get_vertex(vertex_id).await.unwrap();
        assert!(vertex.is_some());
        
        // Delete vertex
        engine.delete_vertex(vertex_id).await.unwrap();
        
        // Verify vertex is gone
        let vertex = engine.get_vertex(vertex_id).await.unwrap();
        assert!(vertex.is_none());
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_transaction() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        // Begin transaction
        let mut transaction = engine.begin_transaction().await.unwrap();
        
        // Create vertex in transaction
        let vertex_type = Type::new("Person".to_string()).unwrap();
        let vertex = Vertex::new(EntityId::new().to_identifier(), vertex_type);
        transaction.create_vertex(&vertex).await.unwrap();
        
        // Commit transaction
        transaction.commit().await.unwrap();
        
        // Verify vertex exists
        let vertex_opt = engine.get_vertex(EntityId::from(vertex.id)).await.unwrap();
        assert!(vertex_opt.is_some());
        
        engine.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_config_variations() {
        // Test memory optimized config
        let memory_config = IndraReifiedConfig::memory_optimized();
        let engine1 = IndraReifiedEngine::with_config(memory_config).await.unwrap();
        assert_eq!(engine1.get_config().backend_type, BackendType::Memory);
        engine1.shutdown().await.unwrap();
        
        // Test development config
        let dev_config = IndraReifiedConfig::development();
        let engine2 = IndraReifiedEngine::with_config(dev_config).await.unwrap();
        assert_eq!(engine2.get_config().bulk_batch_size, 100);
        engine2.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_engine_shutdown() {
        let engine = IndraReifiedEngine::new_memory_backend().await.unwrap();
        
        assert!(engine.is_initialized().await);
        
        engine.shutdown().await.unwrap();
        
        assert!(!engine.is_initialized().await);
    }
}