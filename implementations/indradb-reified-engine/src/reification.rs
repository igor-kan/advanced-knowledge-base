//! Reification Manager for converting edges to nodes
//!
//! This module handles the core reification logic for IndraDB, allowing edges/relationships
//! to be converted into first-class vertices while maintaining graph connectivity through
//! FROM and TO connections.

use crate::transaction::TransactionManager;
use crate::types::*;
use crate::{IndraReifiedConfig, IndraReifiedError, Result};

use std::sync::Arc;
use std::time::SystemTime;
use indradb::{Datastore, Vertex, Edge, Type, Identifier};
use tracing::{debug, info};

/// Manager for edge reification operations in IndraDB
pub struct ReificationManager {
    /// Configuration
    config: IndraReifiedConfig,
    /// IndraDB datastore
    datastore: Arc<dyn Datastore + Send + Sync>,
    /// Transaction manager
    transaction_manager: Arc<TransactionManager>,
    /// Reification statistics
    stats: Arc<ReificationStats>,
}

/// Statistics for reification operations
#[derive(Debug, Default)]
pub struct ReificationStats {
    /// Total reifications performed
    pub total_reifications: std::sync::atomic::AtomicU64,
    /// Failed reifications
    pub failed_reifications: std::sync::atomic::AtomicU64,
    /// Average reification time in microseconds
    pub avg_reification_time_us: std::sync::atomic::AtomicU64,
}

impl ReificationManager {
    /// Create a new reification manager
    pub async fn new(
        config: &IndraReifiedConfig,
        datastore: Arc<dyn Datastore + Send + Sync>,
        transaction_manager: Arc<TransactionManager>,
    ) -> Result<Self> {
        let manager = Self {
            config: config.clone(),
            datastore,
            transaction_manager,
            stats: Arc::new(ReificationStats::default()),
        };
        
        info!("ReificationManager created");
        Ok(manager)
    }
    
    /// Initialize the reification manager
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing reification manager...");
        info!("ReificationManager initialized successfully");
        Ok(())
    }
    
    /// Reify an edge, converting it to a vertex
    pub async fn reify_edge(
        &self,
        from: EntityId,
        to: EntityId,
        edge_type: String,
        properties: PropertyMap,
    ) -> Result<EntityId> {
        let start_time = std::time::Instant::now();
        
        // Start a transaction for atomic reification
        let mut transaction = self.transaction_manager.begin_transaction().await?;
        
        // Create the reified vertex
        let reified_vertex_id = EntityId::new();
        let reified_vertex_type = Type::new("ReifiedEdge".to_string())
            .map_err(|e| IndraReifiedError::ReificationError {
                operation: "create_reified_vertex".to_string(),
                details: format!("Invalid vertex type: {}", e),
            })?;
        
        let reified_vertex = Vertex::new(reified_vertex_id.to_identifier(), reified_vertex_type);
        transaction.create_vertex(&reified_vertex).await?;
        
        // Set properties on the reified vertex
        let mut reified_properties = properties;
        reified_properties.insert("original_from".to_string(), PropertyValue::Uuid(from.as_uuid()));
        reified_properties.insert("original_to".to_string(), PropertyValue::Uuid(to.as_uuid()));
        reified_properties.insert("original_type".to_string(), PropertyValue::String(edge_type.clone()));
        reified_properties.insert("reified_at".to_string(), PropertyValue::Timestamp(SystemTime::now()));
        
        for (key, value) in reified_properties {
            let prop_type = Type::new(key)
                .map_err(|e| IndraReifiedError::ReificationError {
                    operation: "set_property".to_string(),
                    details: format!("Invalid property name: {}", e),
                })?;
            
            transaction.set_vertex_property(
                reified_vertex_id.to_identifier(),
                prop_type,
                value.to_indra_value(),
            ).await?;
        }
        
        // Create FROM connection (original source -> reified vertex)
        let from_type = Type::new("FROM".to_string())
            .map_err(|e| IndraReifiedError::ReificationError {
                operation: "create_from_connection".to_string(),
                details: format!("Invalid edge type: {}", e),
            })?;
        
        let from_edge = Edge::new(from.to_identifier(), from_type, reified_vertex_id.to_identifier());
        transaction.create_edge(&from_edge).await?;
        
        // Create TO connection (reified vertex -> original target)
        let to_type = Type::new("TO".to_string())
            .map_err(|e| IndraReifiedError::ReificationError {
                operation: "create_to_connection".to_string(),
                details: format!("Invalid edge type: {}", e),
            })?;
        
        let to_edge = Edge::new(reified_vertex_id.to_identifier(), to_type, to.to_identifier());
        transaction.create_edge(&to_edge).await?;
        
        // Commit the transaction
        transaction.commit().await?;
        
        // Update statistics
        let duration = start_time.elapsed().as_micros() as u64;
        self.stats.total_reifications.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.update_avg_reification_time(duration);
        
        debug!("Reified edge {} -> {} of type '{}' to vertex {} in {}μs", 
               from, to, edge_type, reified_vertex_id, duration);
        
        Ok(reified_vertex_id)
    }
    
    /// Get all reified relationships
    pub async fn get_all_reified_relationships(&self) -> Result<Vec<ReifiedRelationship>> {
        // This is a simplified implementation
        // In a real implementation, you would query all ReifiedEdge vertices
        debug!("Getting all reified relationships");
        Ok(Vec::new())
    }
    
    /// Get reification statistics
    pub fn get_stats(&self) -> &ReificationStats {
        &self.stats
    }
    
    /// Shutdown the reification manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down ReificationManager...");
        info!("ReificationManager shutdown complete");
        Ok(())
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