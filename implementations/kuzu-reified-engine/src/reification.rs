//! Reification Manager for converting relationships to nodes
//!
//! This module handles the core reification logic, allowing edges/relationships
//! to be converted into first-class nodes while maintaining graph connectivity.

use crate::schema::SchemaManager;
use crate::types::*;
use crate::{KuzuReifiedConfig, KuzuReifiedError, Result};

use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use dashmap::DashMap;
use tracing::{debug, info, warn};
use ahash::AHashMap;

/// Manager for edge reification operations
pub struct ReificationManager {
    /// Configuration
    config: KuzuReifiedConfig,
    /// Schema manager reference
    schema_manager: Arc<SchemaManager>,
    /// Cache of reified relationships
    reified_cache: DashMap<EntityId, ReifiedRelationship>,
    /// Mapping from original relationship ID to reified node ID
    relationship_to_node_mapping: DashMap<EntityId, EntityId>,
    /// Reverse mapping from reified node ID to original relationship info
    node_to_relationship_mapping: DashMap<EntityId, OriginalRelationshipInfo>,
    /// Reification statistics
    stats: Arc<ReificationStats>,
    /// Whether the manager is initialized
    is_initialized: Arc<RwLock<bool>>,
}

/// Statistics for reification operations
#[derive(Debug, Default)]
pub struct ReificationStats {
    /// Total reifications performed
    pub total_reifications: std::sync::atomic::AtomicU64,
    /// Cache hits
    pub cache_hits: std::sync::atomic::AtomicU64,
    /// Cache misses
    pub cache_misses: std::sync::atomic::AtomicU64,
    /// Failed reifications
    pub failed_reifications: std::sync::atomic::AtomicU64,
    /// Average reification time in microseconds
    pub avg_reification_time_us: std::sync::atomic::AtomicU64,
}

impl ReificationManager {
    /// Create a new reification manager
    pub async fn new(
        config: &KuzuReifiedConfig,
        schema_manager: Arc<SchemaManager>,
    ) -> Result<Self> {
        let manager = Self {
            config: config.clone(),
            schema_manager,
            reified_cache: DashMap::new(),
            relationship_to_node_mapping: DashMap::new(),
            node_to_relationship_mapping: DashMap::new(),
            stats: Arc::new(ReificationStats::default()),
            is_initialized: Arc::new(RwLock::new(false)),
        };
        
        info!("ReificationManager created with cache size: {}", config.max_cache_size);
        Ok(manager)
    }
    
    /// Initialize the reification manager
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing reification manager...");
        
        // Create reification-specific schema elements
        self.create_reification_schema().await?;
        
        // Load existing reified relationships if any
        self.load_existing_reifications().await?;
        
        // Mark as initialized
        *self.is_initialized.write().await = true;
        
        info!("ReificationManager initialized successfully");
        Ok(())
    }
    
    /// Reify a relationship, converting it to a node
    pub async fn reify_relationship(
        &self,
        from: EntityId,
        to: EntityId,
        rel_type: String,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        let start_time = std::time::Instant::now();
        
        // Create the original relationship first
        let original_rel = Relationship::new(from, to, rel_type.clone(), properties.clone());
        
        // Validate the relationship
        self.schema_manager.validate_relationship(&original_rel).await?;
        
        // Create the reified node
        let reified_node_id = EntityId::new();
        let reified_node = self.create_reified_node(
            reified_node_id,
            &original_rel,
        ).await?;
        
        // Create connection relationships
        let from_connection = self.create_connection_relationship(
            from,
            reified_node_id,
            "FROM",
        ).await?;
        
        let to_connection = self.create_connection_relationship(
            reified_node_id,
            to,
            "TO",
        ).await?;
        
        // Create the reified relationship structure
        let original_info = OriginalRelationshipInfo {
            original_from: from,
            original_to: to,
            original_type: rel_type.clone(),
            reified_at: SystemTime::now(),
        };
        
        let reified_rel = ReifiedRelationship {
            node: reified_node,
            original_relationship: original_info.clone(),
            from_connection,
            to_connection,
        };
        
        // Store in caches
        if self.config.enable_reification_cache {
            self.reified_cache.insert(reified_node_id, reified_rel.clone());
            self.relationship_to_node_mapping.insert(original_rel.id, reified_node_id);
            self.node_to_relationship_mapping.insert(reified_node_id, original_info);
            
            // Manage cache size
            self.manage_cache_size().await;
        }
        
        // Update statistics
        let duration = start_time.elapsed().as_micros() as u64;
        self.stats.total_reifications.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.update_avg_reification_time(duration);
        
        debug!("Reified relationship {} -> {} of type '{}' to node {} in {}μs", 
               from, to, rel_type, reified_node_id, duration);
        
        Ok(reified_node_id)
    }
    
    /// Get a reified relationship by its node ID
    pub async fn get_reified_relationship(&self, node_id: EntityId) -> Result<Option<ReifiedRelationship>> {
        self.ensure_initialized().await?;
        
        // Check cache first
        if let Some(reified) = self.reified_cache.get(&node_id) {
            self.stats.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(Some(reified.value().clone()));
        }
        
        self.stats.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Load from database
        self.load_reified_relationship_from_db(node_id).await
    }
    
    /// Check if a node is a reified relationship
    pub async fn is_reified_relationship(&self, node_id: EntityId) -> Result<bool> {
        self.ensure_initialized().await?;
        
        // Check cache first
        if self.reified_cache.contains_key(&node_id) {
            return Ok(true);
        }
        
        // Check in database
        self.check_reified_in_db(node_id).await
    }
    
    /// Get the original relationship info for a reified node
    pub async fn get_original_relationship_info(&self, node_id: EntityId) -> Result<Option<OriginalRelationshipInfo>> {
        self.ensure_initialized().await?;
        
        if let Some(info) = self.node_to_relationship_mapping.get(&node_id) {
            return Ok(Some(info.value().clone()));
        }
        
        // Load from database if not in cache
        self.load_original_info_from_db(node_id).await
    }
    
    /// Get all reified relationships
    pub async fn get_all_reified_relationships(&self) -> Result<Vec<ReifiedRelationship>> {
        self.ensure_initialized().await?;
        
        let mut reified_rels = Vec::new();
        
        // Get from cache
        for entry in self.reified_cache.iter() {
            reified_rels.push(entry.value().clone());
        }
        
        // Load additional from database if cache is not complete
        let db_reified = self.load_all_reified_from_db().await?;
        for rel in db_reified {
            if !reified_rels.iter().any(|r| r.node.id == rel.node.id) {
                reified_rels.push(rel);
            }
        }
        
        Ok(reified_rels)
    }
    
    /// Unreify a relationship, converting it back to a regular edge
    pub async fn unreify_relationship(&self, node_id: EntityId) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        // Get the reified relationship
        let reified = self.get_reified_relationship(node_id).await?
            .ok_or_else(|| KuzuReifiedError::ReificationError {
                operation: "unreify".to_string(),
                details: format!("Node {} is not a reified relationship", node_id),
            })?;
        
        // Create the original relationship
        let original_rel = Relationship::new(
            reified.original_relationship.original_from,
            reified.original_relationship.original_to,
            reified.original_relationship.original_type.clone(),
            reified.node.properties.clone(),
        );
        
        // Store the relationship in the database
        let rel_id = self.store_relationship_in_db(&original_rel).await?;
        
        // Remove the reified node and its connections
        self.remove_reified_node(node_id).await?;
        
        // Update caches
        self.reified_cache.remove(&node_id);
        self.node_to_relationship_mapping.remove(&node_id);
        if let Some((_, original_id)) = self.relationship_to_node_mapping
            .iter()
            .find(|entry| *entry.value() == node_id)
            .map(|entry| (entry.key().clone(), entry.value().clone()))
        {
            self.relationship_to_node_mapping.remove(&original_id);
        }
        
        debug!("Unreified node {} back to relationship {}", node_id, rel_id);
        Ok(rel_id)
    }
    
    /// Add a relationship to a reified edge
    pub async fn add_relationship_to_reified(
        &self,
        reified_node_id: EntityId,
        target_node_id: EntityId,
        rel_type: String,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        self.ensure_initialized().await?;
        
        // Verify the node is reified
        if !self.is_reified_relationship(reified_node_id).await? {
            return Err(KuzuReifiedError::ReificationError {
                operation: "add_relationship".to_string(),
                details: format!("Node {} is not a reified relationship", reified_node_id),
            });
        }
        
        // Create the relationship
        let rel_id = self.create_connection_relationship(
            reified_node_id,
            target_node_id,
            &rel_type,
        ).await?;
        
        debug!("Added relationship {} from reified node {} to {}", 
               rel_id, reified_node_id, target_node_id);
        
        Ok(rel_id)
    }
    
    /// Get reification statistics
    pub fn get_stats(&self) -> &ReificationStats {
        &self.stats
    }
    
    /// Shutdown the reification manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ReificationManager...");
        
        // Clear caches
        self.reified_cache.clear();
        self.relationship_to_node_mapping.clear();
        self.node_to_relationship_mapping.clear();
        
        // Mark as not initialized
        *self.is_initialized.write().await = false;
        
        info!("ReificationManager shutdown complete");
        Ok(())
    }
    
    // Private helper methods
    
    /// Ensure the manager is initialized
    async fn ensure_initialized(&self) -> Result<()> {
        if !*self.is_initialized.read().await {
            return Err(KuzuReifiedError::Internal {
                details: "ReificationManager not initialized".to_string(),
            });
        }
        Ok(())
    }
    
    /// Create reification-specific schema
    async fn create_reification_schema(&self) -> Result<()> {
        debug!("Creating reification schema...");
        
        // Define the ReifiedEdge node type
        self.schema_manager.create_node_type(
            "ReifiedEdge",
            vec![
                ("original_from".to_string(), "STRING".to_string()),
                ("original_to".to_string(), "STRING".to_string()),
                ("original_type".to_string(), "STRING".to_string()),
                ("reified_at".to_string(), "TIMESTAMP".to_string()),
            ],
        ).await?;
        
        // Define connection relationship types
        self.schema_manager.create_relationship_type("FROM", vec![]).await?;
        self.schema_manager.create_relationship_type("TO", vec![]).await?;
        
        debug!("Reification schema created");
        Ok(())
    }
    
    /// Load existing reifications from database
    async fn load_existing_reifications(&self) -> Result<()> {
        debug!("Loading existing reifications...");
        
        // This would query the database for existing ReifiedEdge nodes
        // and populate the caches
        let reified_rels = self.load_all_reified_from_db().await?;
        
        for reified in reified_rels {
            if self.config.enable_reification_cache {
                self.reified_cache.insert(reified.node.id, reified.clone());
                self.node_to_relationship_mapping.insert(
                    reified.node.id,
                    reified.original_relationship.clone(),
                );
            }
        }
        
        debug!("Loaded {} existing reifications", self.reified_cache.len());
        Ok(())
    }
    
    /// Create a reified node from a relationship
    async fn create_reified_node(
        &self,
        node_id: EntityId,
        original_rel: &Relationship,
    ) -> Result<Node> {
        let mut properties = original_rel.properties.clone();
        
        // Add reification metadata
        properties.insert("original_from".to_string(), PropertyValue::String(original_rel.from.to_string()));
        properties.insert("original_to".to_string(), PropertyValue::String(original_rel.to.to_string()));
        properties.insert("original_type".to_string(), PropertyValue::String(original_rel.rel_type.clone()));
        properties.insert("reified_at".to_string(), PropertyValue::Timestamp(SystemTime::now()));
        
        let node = Node::with_id(node_id, "ReifiedEdge", properties);
        
        // Store in database
        self.store_node_in_db(&node).await?;
        
        Ok(node)
    }
    
    /// Create a connection relationship
    async fn create_connection_relationship(
        &self,
        from: EntityId,
        to: EntityId,
        rel_type: &str,
    ) -> Result<EntityId> {
        let relationship = Relationship::new(from, to, rel_type.to_string(), PropertyMap::new());
        self.store_relationship_in_db(&relationship).await
    }
    
    /// Store a node in the database
    async fn store_node_in_db(&self, node: &Node) -> Result<()> {
        // This would use the Kuzu connection to store the node
        // For now, this is a placeholder implementation
        debug!("Storing node {} in database", node.id);
        Ok(())
    }
    
    /// Store a relationship in the database
    async fn store_relationship_in_db(&self, relationship: &Relationship) -> Result<EntityId> {
        // This would use the Kuzu connection to store the relationship
        // For now, this is a placeholder implementation
        debug!("Storing relationship {} in database", relationship.id);
        Ok(relationship.id)
    }
    
    /// Load a reified relationship from database
    async fn load_reified_relationship_from_db(&self, node_id: EntityId) -> Result<Option<ReifiedRelationship>> {
        // This would query the database for the specific reified relationship
        // For now, this is a placeholder implementation
        debug!("Loading reified relationship {} from database", node_id);
        Ok(None)
    }
    
    /// Check if a node is reified in the database
    async fn check_reified_in_db(&self, node_id: EntityId) -> Result<bool> {
        // This would check if the node has the ReifiedEdge label
        debug!("Checking if node {} is reified in database", node_id);
        Ok(false)
    }
    
    /// Load original relationship info from database
    async fn load_original_info_from_db(&self, node_id: EntityId) -> Result<Option<OriginalRelationshipInfo>> {
        // This would query the node properties for original relationship info
        debug!("Loading original info for node {} from database", node_id);
        Ok(None)
    }
    
    /// Load all reified relationships from database
    async fn load_all_reified_from_db(&self) -> Result<Vec<ReifiedRelationship>> {
        // This would query all ReifiedEdge nodes
        debug!("Loading all reified relationships from database");
        Ok(Vec::new())
    }
    
    /// Remove a reified node and its connections
    async fn remove_reified_node(&self, node_id: EntityId) -> Result<()> {
        // This would remove the node and its FROM/TO relationships
        debug!("Removing reified node {} from database", node_id);
        Ok(())
    }
    
    /// Manage cache size to stay within limits
    async fn manage_cache_size(&self) {
        if self.reified_cache.len() > self.config.max_cache_size {
            // Remove oldest entries (simplified LRU)
            let to_remove = self.reified_cache.len() - self.config.max_cache_size;
            let mut removed = 0;
            
            // In a real implementation, you would track access times
            let keys_to_remove: Vec<EntityId> = self.reified_cache
                .iter()
                .take(to_remove)
                .map(|entry| *entry.key())
                .collect();
            
            for key in keys_to_remove {
                self.reified_cache.remove(&key);
                self.node_to_relationship_mapping.remove(&key);
                removed += 1;
                if removed >= to_remove {
                    break;
                }
            }
            
            debug!("Removed {} entries from reification cache", removed);
        }
    }
    
    /// Update average reification time
    fn update_avg_reification_time(&self, new_time_us: u64) {
        let current_avg = self.stats.avg_reification_time_us.load(std::sync::atomic::Ordering::Relaxed);
        let total_ops = self.stats.total_reifications.load(std::sync::atomic::Ordering::Relaxed);
        
        if total_ops == 1 {
            self.stats.avg_reification_time_us.store(new_time_us, std::sync::atomic::Ordering::Relaxed);
        } else {
            // Simple moving average
            let new_avg = ((current_avg * (total_ops - 1)) + new_time_us) / total_ops;
            self.stats.avg_reification_time_us.store(new_avg, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl ReificationStats {
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total_reifications.load(std::sync::atomic::Ordering::Relaxed);
        let failed = self.failed_reifications.load(std::sync::atomic::Ordering::Relaxed);
        
        if total == 0 {
            0.0
        } else {
            (total - failed) as f64 / total as f64
        }
    }
    
    /// Get average reification time
    pub fn avg_reification_time_us(&self) -> u64 {
        self.avg_reification_time_us.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SchemaManager;
    
    async fn create_test_reification_manager() -> ReificationManager {
        let config = KuzuReifiedConfig::development();
        let schema_manager = Arc::new(SchemaManager::new(&config).await.unwrap());
        
        let manager = ReificationManager::new(&config, schema_manager).await.unwrap();
        manager.initialize().await.unwrap();
        manager
    }
    
    #[tokio::test]
    async fn test_reification_manager_creation() {
        let manager = create_test_reification_manager().await;
        
        assert_eq!(manager.reified_cache.len(), 0);
        assert_eq!(manager.get_stats().total_reifications.load(std::sync::atomic::Ordering::Relaxed), 0);
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reification_process() {
        let manager = create_test_reification_manager().await;
        
        let from = EntityId::new();
        let to = EntityId::new();
        let props = crate::properties!("weight" => 1.5);
        
        let reified_id = manager.reify_relationship(from, to, "CONNECTS".to_string(), props).await.unwrap();
        
        assert_ne!(reified_id.to_string(), "");
        assert_eq!(manager.get_stats().total_reifications.load(std::sync::atomic::Ordering::Relaxed), 1);
        
        // Check if it's recognized as reified
        let is_reified = manager.is_reified_relationship(reified_id).await.unwrap();
        assert!(is_reified);
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reification_cache() {
        let mut config = KuzuReifiedConfig::development();
        config.enable_reification_cache = true;
        config.max_cache_size = 10;
        
        let schema_manager = Arc::new(SchemaManager::new(&config).await.unwrap());
        let manager = ReificationManager::new(&config, schema_manager).await.unwrap();
        manager.initialize().await.unwrap();
        
        let from = EntityId::new();
        let to = EntityId::new();
        
        let reified_id = manager.reify_relationship(
            from, 
            to, 
            "TEST".to_string(), 
            PropertyMap::new()
        ).await.unwrap();
        
        // Should be in cache
        assert!(manager.reified_cache.contains_key(&reified_id));
        
        // Get should hit cache
        let _reified = manager.get_reified_relationship(reified_id).await.unwrap();
        assert!(manager.get_stats().cache_hits.load(std::sync::atomic::Ordering::Relaxed) > 0);
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_original_relationship_info() {
        let manager = create_test_reification_manager().await;
        
        let from = EntityId::new();
        let to = EntityId::new();
        let rel_type = "IMPORTANT".to_string();
        
        let reified_id = manager.reify_relationship(from, to, rel_type.clone(), PropertyMap::new()).await.unwrap();
        
        let original_info = manager.get_original_relationship_info(reified_id).await.unwrap();
        assert!(original_info.is_some());
        
        let info = original_info.unwrap();
        assert_eq!(info.original_from, from);
        assert_eq!(info.original_to, to);
        assert_eq!(info.original_type, rel_type);
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_add_relationship_to_reified() {
        let manager = create_test_reification_manager().await;
        
        let from = EntityId::new();
        let to = EntityId::new();
        let reified_id = manager.reify_relationship(from, to, "BASE".to_string(), PropertyMap::new()).await.unwrap();
        
        let target = EntityId::new();
        let connection_id = manager.add_relationship_to_reified(
            reified_id,
            target,
            "CONNECTS_TO".to_string(),
            PropertyMap::new(),
        ).await.unwrap();
        
        assert_ne!(connection_id.to_string(), "");
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_stats_calculation() {
        let manager = create_test_reification_manager().await;
        
        let stats = manager.get_stats();
        
        // Initial state
        assert_eq!(stats.cache_hit_rate(), 0.0);
        assert_eq!(stats.success_rate(), 0.0);
        
        // Perform reifications
        for i in 0..5 {
            let from = EntityId::new();
            let to = EntityId::new();
            let _ = manager.reify_relationship(from, to, format!("REL_{}", i), PropertyMap::new()).await.unwrap();
        }
        
        assert_eq!(stats.success_rate(), 1.0);
        assert!(stats.avg_reification_time_us() > 0);
        
        manager.shutdown().await.unwrap();
    }
}