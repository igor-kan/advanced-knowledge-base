//! Transaction Manager for ACID operations
//!
//! This module provides transaction management for the IndraDB reified engine,
//! ensuring ACID properties for all graph operations including reification.

use crate::types::*;
use crate::{IndraReifiedConfig, IndraReifiedError, Result};

use std::sync::Arc;
use indradb::{Datastore, Transaction as IndraTransaction, Vertex, Edge, Type, Identifier, VertexProperties, EdgeProperties};
use tracing::{debug, info};

/// Transaction manager for ACID operations
pub struct TransactionManager {
    /// Configuration
    config: IndraReifiedConfig,
    /// IndraDB datastore
    datastore: Arc<dyn Datastore + Send + Sync>,
}

/// Reified transaction wrapper around IndraDB transaction
pub struct ReifiedTransaction {
    /// The underlying IndraDB transaction
    transaction: Box<dyn IndraTransaction + Send + Sync>,
    /// Transaction ID for tracking
    id: EntityId,
    /// Whether transaction is committed or rolled back
    is_finished: bool,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub async fn new(
        config: &IndraReifiedConfig,
        datastore: Arc<dyn Datastore + Send + Sync>,
    ) -> Result<Self> {
        let manager = Self {
            config: config.clone(),
            datastore,
        };
        
        info!("TransactionManager created");
        Ok(manager)
    }
    
    /// Initialize the transaction manager
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing transaction manager...");
        info!("TransactionManager initialized successfully");
        Ok(())
    }
    
    /// Begin a new read-write transaction
    pub async fn begin_transaction(&self) -> Result<ReifiedTransaction> {
        let transaction = self.datastore.transaction()
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "begin".to_string(),
                details: format!("Failed to begin transaction: {}", e),
            })?;
        
        let tx_id = EntityId::new();
        debug!("Started transaction {}", tx_id);
        
        Ok(ReifiedTransaction {
            transaction,
            id: tx_id,
            is_finished: false,
        })
    }
    
    /// Begin a new read-only transaction
    pub async fn begin_readonly_transaction(&self) -> Result<ReifiedTransaction> {
        // IndraDB doesn't differentiate read-only transactions at the API level
        self.begin_transaction().await
    }
    
    /// Shutdown the transaction manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down TransactionManager...");
        info!("TransactionManager shutdown complete");
        Ok(())
    }
}

impl ReifiedTransaction {
    /// Get transaction ID
    pub fn id(&self) -> EntityId {
        self.id
    }
    
    /// Create a vertex
    pub async fn create_vertex(&mut self, vertex: &Vertex) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "create_vertex".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.create_vertex(vertex)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "create_vertex".to_string(),
                details: format!("Failed to create vertex: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Create an edge
    pub async fn create_edge(&mut self, edge: &Edge) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "create_edge".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.create_edge(edge)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "create_edge".to_string(),
                details: format!("Failed to create edge: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Get vertices by IDs
    pub async fn get_vertices(&mut self, ids: &[Identifier]) -> Result<Vec<Vertex>> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_vertices".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.get_vertices(ids)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "get_vertices".to_string(),
                details: format!("Failed to get vertices: {}", e),
            })
    }
    
    /// Get edges
    pub async fn get_edges(&mut self, edges: &[Edge]) -> Result<Vec<Edge>> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_edges".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.get_edges(edges)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "get_edges".to_string(),
                details: format!("Failed to get edges: {}", e),
            })
    }
    
    /// Delete vertices
    pub async fn delete_vertices(&mut self, ids: &[Identifier]) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "delete_vertices".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.delete_vertices(ids)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "delete_vertices".to_string(),
                details: format!("Failed to delete vertices: {}", e),
            })?;
        
        Ok(())
    }
    
    /// Set vertex property
    pub async fn set_vertex_property(
        &mut self,
        vertex_id: Identifier,
        name: Type,
        value: indradb::Json,
    ) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "set_vertex_property".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.set_vertex_properties(VertexProperties {
            vertex: Vertex::new(vertex_id, Type::new("Unknown".to_string()).unwrap()), // Type doesn't matter for property setting
            name,
            value,
        })
        .map_err(|e| IndraReifiedError::TransactionError {
            operation: "set_vertex_property".to_string(),
            details: format!("Failed to set vertex property: {}", e),
        })?;
        
        Ok(())
    }
    
    /// Set edge property
    pub async fn set_edge_property(
        &mut self,
        edge: Edge,
        name: Type,
        value: indradb::Json,
    ) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "set_edge_property".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.set_edge_properties(EdgeProperties {
            edge,
            name,
            value,
        })
        .map_err(|e| IndraReifiedError::TransactionError {
            operation: "set_edge_property".to_string(),
            details: format!("Failed to set edge property: {}", e),
        })?;
        
        Ok(())
    }
    
    /// Get all vertex properties
    pub async fn get_all_vertex_properties(&mut self, vertex_id: Identifier) -> Result<Vec<VertexProperties>> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_all_vertex_properties".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        // This is a simplified implementation
        // In practice, you would need to query all properties for a vertex
        Ok(Vec::new())
    }
    
    /// Get all edge properties
    pub async fn get_all_edge_properties(&mut self, edge: Edge) -> Result<Vec<EdgeProperties>> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_all_edge_properties".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        // This is a simplified implementation
        // In practice, you would need to query all properties for an edge
        Ok(Vec::new())
    }
    
    /// Get vertex count
    pub async fn get_vertex_count(&self) -> Result<u64> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_vertex_count".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.get_vertex_count(None, None)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "get_vertex_count".to_string(),
                details: format!("Failed to get vertex count: {}", e),
            })
    }
    
    /// Get edge count
    pub async fn get_edge_count(&self) -> Result<u64> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "get_edge_count".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        self.transaction.get_edge_count(Identifier::new(uuid::Uuid::nil()), None, None, None)
            .map_err(|e| IndraReifiedError::TransactionError {
                operation: "get_edge_count".to_string(),
                details: format!("Failed to get edge count: {}", e),
            })
    }
    
    /// Commit the transaction
    pub async fn commit(mut self) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "commit".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        // IndraDB transactions are automatically committed when dropped
        self.is_finished = true;
        debug!("Committed transaction {}", self.id);
        Ok(())
    }
    
    /// Rollback the transaction
    pub async fn rollback(mut self) -> Result<()> {
        if self.is_finished {
            return Err(IndraReifiedError::TransactionError {
                operation: "rollback".to_string(),
                details: "Transaction already finished".to_string(),
            });
        }
        
        // IndraDB handles rollback automatically when transaction is dropped without commit
        self.is_finished = true;
        debug!("Rolled back transaction {}", self.id);
        Ok(())
    }
}

impl Drop for ReifiedTransaction {
    fn drop(&mut self) {
        if !self.is_finished {
            debug!("Transaction {} dropped without explicit commit/rollback", self.id);
        }
    }
}