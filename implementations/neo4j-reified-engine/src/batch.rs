//! Batch operations for efficient reification processing
//!
//! This module provides high-performance batch operations for creating, updating,
//! and managing large numbers of reified relationships in Neo4j.

use crate::types::*;
use crate::{Neo4jReifiedError, Result, FROM_RELATIONSHIP, TO_RELATIONSHIP, REIFIED_EDGE_LABEL};

use neo4rs::{Graph, Query, BoltType};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::Semaphore;
use futures::stream::{self, StreamExt};
use tracing::{info, warn, error, debug};

/// Batch reification manager
pub struct BatchReifier {
    /// Neo4j graph connection
    graph: Arc<Graph>,
    /// Performance statistics
    stats: Arc<EngineStats>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Concurrency limit for parallel operations
    concurrency_limit: Arc<Semaphore>,
}

impl BatchReifier {
    /// Create a new batch reifier
    pub fn new(graph: Arc<Graph>, stats: Arc<EngineStats>, max_batch_size: usize) -> Self {
        let concurrency_limit = Arc::new(Semaphore::new(10)); // Default to 10 concurrent operations
        
        Self {
            graph,
            stats,
            max_batch_size,
            concurrency_limit,
        }
    }

    /// Batch reify multiple relationships
    pub async fn batch_reify_relationships(
        &self,
        requests: Vec<ReificationRequest>,
    ) -> Result<BatchResult> {
        if requests.is_empty() {
            return Ok(BatchResult::empty());
        }

        info!("Starting batch reification of {} relationships", requests.len());
        let start_time = std::time::Instant::now();

        // Split into batches
        let batches: Vec<_> = requests.chunks(self.max_batch_size).collect();
        let mut all_results = Vec::new();
        let mut total_successful = 0;
        let mut errors = Vec::new();

        // Process batches concurrently
        let batch_futures = batches.into_iter().enumerate().map(|(batch_idx, batch)| {
            let graph = self.graph.clone();
            let stats = self.stats.clone();
            let permit = self.concurrency_limit.clone();
            
            async move {
                let _permit = permit.acquire().await.unwrap();
                debug!("Processing batch {} with {} requests", batch_idx, batch.len());
                
                self.process_reification_batch(batch.to_vec(), graph, stats).await
            }
        });

        let batch_results = stream::iter(batch_futures)
            .buffer_unordered(5) // Process up to 5 batches concurrently
            .collect::<Vec<_>>()
            .await;

        // Aggregate results
        for result in batch_results {
            match result {
                Ok(batch_result) => {
                    total_successful += batch_result.successful_count;
                    all_results.extend(batch_result.reified_structures);
                    errors.extend(batch_result.errors);
                }
                Err(e) => {
                    errors.push(BatchError {
                        operation: "batch_process".to_string(),
                        error: e.to_string(),
                        affected_items: Vec::new(),
                    });
                }
            }
        }

        let execution_time = start_time.elapsed();
        
        info!(
            "Batch reification completed: {}/{} successful in {:?}",
            total_successful,
            requests.len(),
            execution_time
        );

        // Update statistics
        self.stats.inc_batch_operations();
        
        Ok(BatchResult {
            total_requested: requests.len(),
            successful_count: total_successful,
            failed_count: requests.len() - total_successful,
            reified_structures: all_results,
            errors,
            execution_time_ms: execution_time.as_millis() as u64,
        })
    }

    /// Batch unreify relationships
    pub async fn batch_unreify_relationships(
        &self,
        reified_node_ids: Vec<EntityId>,
    ) -> Result<BatchResult> {
        if reified_node_ids.is_empty() {
            return Ok(BatchResult::empty());
        }

        info!("Starting batch unreification of {} relationships", reified_node_ids.len());
        let start_time = std::time::Instant::now();

        let batches: Vec<_> = reified_node_ids.chunks(self.max_batch_size).collect();
        let mut total_successful = 0;
        let mut errors = Vec::new();

        for (batch_idx, batch) in batches.iter().enumerate() {
            debug!("Processing unreification batch {} with {} items", batch_idx, batch.len());
            
            match self.process_unreification_batch(batch.to_vec()).await {
                Ok(count) => total_successful += count,
                Err(e) => {
                    errors.push(BatchError {
                        operation: "batch_unreify".to_string(),
                        error: e.to_string(),
                        affected_items: batch.iter().map(|id| id.to_string()).collect(),
                    });
                }
            }
        }

        let execution_time = start_time.elapsed();
        
        info!(
            "Batch unreification completed: {}/{} successful in {:?}",
            total_successful,
            reified_node_ids.len(),
            execution_time
        );

        Ok(BatchResult {
            total_requested: reified_node_ids.len(),
            successful_count: total_successful,
            failed_count: reified_node_ids.len() - total_successful,
            reified_structures: Vec::new(), // Unreification doesn't return structures
            errors,
            execution_time_ms: execution_time.as_millis() as u64,
        })
    }

    /// Batch update properties on reified relationships
    pub async fn batch_update_properties(
        &self,
        updates: Vec<PropertyUpdate>,
    ) -> Result<BatchResult> {
        if updates.is_empty() {
            return Ok(BatchResult::empty());
        }

        info!("Starting batch property update of {} items", updates.len());
        let start_time = std::time::Instant::now();

        let batches: Vec<_> = updates.chunks(self.max_batch_size).collect();
        let mut total_successful = 0;
        let mut errors = Vec::new();

        for (batch_idx, batch) in batches.iter().enumerate() {
            debug!("Processing property update batch {} with {} items", batch_idx, batch.len());
            
            match self.process_property_update_batch(batch.to_vec()).await {
                Ok(count) => total_successful += count,
                Err(e) => {
                    errors.push(BatchError {
                        operation: "batch_update_properties".to_string(),
                        error: e.to_string(),
                        affected_items: batch.iter().map(|u| u.entity_id.to_string()).collect(),
                    });
                }
            }
        }

        let execution_time = start_time.elapsed();
        
        info!(
            "Batch property update completed: {}/{} successful in {:?}",
            total_successful,
            updates.len(),
            execution_time
        );

        Ok(BatchResult {
            total_requested: updates.len(),
            successful_count: total_successful,
            failed_count: updates.len() - total_successful,
            reified_structures: Vec::new(),
            errors,
            execution_time_ms: execution_time.as_millis() as u64,
        })
    }

    /// Batch delete reified relationships
    pub async fn batch_delete_reified_relationships(
        &self,
        reified_node_ids: Vec<EntityId>,
    ) -> Result<BatchResult> {
        if reified_node_ids.is_empty() {
            return Ok(BatchResult::empty());
        }

        info!("Starting batch deletion of {} reified relationships", reified_node_ids.len());
        let start_time = std::time::Instant::now();

        let cypher = format!(r#"
            UNWIND $node_ids as node_id
            MATCH (reified:{})
            WHERE id(reified) = node_id
            OPTIONAL MATCH (reified)-[r:{}|{}]-()
            DELETE r, reified
            RETURN count(reified) as deleted_count
        "#, REIFIED_EDGE_LABEL, FROM_RELATIONSHIP, TO_RELATIONSHIP);

        let node_ids: Vec<BoltType> = reified_node_ids.iter()
            .map(|id| BoltType::Integer(id.value()))
            .collect();

        let query = Query::new(cypher).param("node_ids", BoltType::List(node_ids));
        
        match self.graph.execute(query).await {
            Ok(mut result) => {
                if let Ok(Some(row)) = result.next().await {
                    let deleted_count: i64 = row.get("deleted_count").unwrap_or(0);
                    let execution_time = start_time.elapsed();
                    
                    info!("Batch deletion completed: {} items deleted in {:?}", deleted_count, execution_time);
                    
                    Ok(BatchResult {
                        total_requested: reified_node_ids.len(),
                        successful_count: deleted_count as usize,
                        failed_count: reified_node_ids.len() - (deleted_count as usize),
                        reified_structures: Vec::new(),
                        errors: Vec::new(),
                        execution_time_ms: execution_time.as_millis() as u64,
                    })
                } else {
                    Err(Neo4jReifiedError::BatchError {
                        operation: "batch_delete".to_string(),
                        failed_count: reified_node_ids.len(),
                        total_count: reified_node_ids.len(),
                    })
                }
            }
            Err(e) => Err(Neo4jReifiedError::BatchError {
                operation: "batch_delete".to_string(),
                failed_count: reified_node_ids.len(),
                total_count: reified_node_ids.len(),
            }),
        }
    }

    // Private methods

    async fn process_reification_batch(
        &self,
        requests: Vec<ReificationRequest>,
        graph: Arc<Graph>,
        stats: Arc<EngineStats>,
    ) -> Result<BatchResult> {
        let cypher = format!(r#"
            UNWIND $requests as req
            MATCH (from), (to) 
            WHERE id(from) = req.from_id AND id(to) = req.to_id
            CREATE (from)-[:{}]->(reified:{})-[:{}]->(to)
            SET reified.original_type = req.rel_type,
                reified += req.properties,
                reified.created_at = datetime(),
                reified.reified_at = datetime()
            RETURN id(reified) as reified_id, req.from_id as from_id, req.to_id as to_id, req.rel_type as rel_type
        "#, FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP);

        // Convert requests to Neo4j parameters
        let request_maps: Vec<BoltType> = requests.iter().map(|req| {
            let mut map = HashMap::new();
            map.insert("from_id".to_string(), BoltType::Integer(req.from_node_id.value()));
            map.insert("to_id".to_string(), BoltType::Integer(req.to_node_id.value()));
            map.insert("rel_type".to_string(), BoltType::String(req.relationship_type.clone()));
            
            let props_map: HashMap<String, BoltType> = req.properties.iter()
                .map(|(k, v)| (k.clone(), v.to_bolt_type()))
                .collect();
            map.insert("properties".to_string(), BoltType::Map(props_map));
            
            BoltType::Map(map)
        }).collect();

        let query = Query::new(cypher).param("requests", BoltType::List(request_maps));
        
        match graph.execute(query).await {
            Ok(mut result) => {
                let mut reified_structures = Vec::new();
                let mut successful_count = 0;

                while let Ok(Some(row)) = result.next().await {
                    let reified_id: i64 = row.get("reified_id").unwrap_or(-1);
                    let from_id: i64 = row.get("from_id").unwrap_or(-1);
                    let to_id: i64 = row.get("to_id").unwrap_or(-1);
                    let rel_type: String = row.get("rel_type").unwrap_or_default();

                    if reified_id > 0 && from_id > 0 && to_id > 0 {
                        // Find the original request to get properties
                        if let Some(original_request) = requests.iter().find(|r| 
                            r.from_node_id.value() == from_id && 
                            r.to_node_id.value() == to_id &&
                            r.relationship_type == rel_type) {
                            
                            let reified_structure = ReificationStructure {
                                reified_node: ReifiedNode::with_id(
                                    EntityId::new(reified_id),
                                    vec![REIFIED_EDGE_LABEL.to_string()],
                                    original_request.properties.clone(),
                                ),
                                original_relationship: OriginalRelationshipInfo {
                                    original_start_node_id: EntityId::new(from_id),
                                    original_end_node_id: EntityId::new(to_id),
                                    original_type: rel_type.clone(),
                                    original_properties: original_request.properties.clone(),
                                    reified_at: chrono::Utc::now(),
                                    reified_by: None,
                                    reification_context: Some("batch_operation".to_string()),
                                },
                                from_connection: ReifiedRelationship::new(
                                    EntityId::new(from_id),
                                    EntityId::new(reified_id),
                                    FROM_RELATIONSHIP,
                                    HashMap::new(),
                                ),
                                to_connection: ReifiedRelationship::new(
                                    EntityId::new(reified_id),
                                    EntityId::new(to_id),
                                    TO_RELATIONSHIP,
                                    HashMap::new(),
                                ),
                                reification_level: 0,
                            };

                            reified_structures.push(reified_structure);
                            successful_count += 1;
                            stats.inc_reifications_performed();
                        }
                    }
                }

                Ok(BatchResult {
                    total_requested: requests.len(),
                    successful_count,
                    failed_count: requests.len() - successful_count,
                    reified_structures,
                    errors: Vec::new(),
                    execution_time_ms: 0, // Set by caller
                })
            }
            Err(e) => Err(Neo4jReifiedError::BatchError {
                operation: "process_reification_batch".to_string(),
                failed_count: requests.len(),
                total_count: requests.len(),
            }),
        }
    }

    async fn process_unreification_batch(&self, reified_node_ids: Vec<EntityId>) -> Result<usize> {
        let cypher = format!(r#"
            UNWIND $node_ids as node_id
            MATCH (from)-[:{}]->(reified:{})-[:{}]->(to)
            WHERE id(reified) = node_id
            WITH from, to, reified.original_type as rel_type, properties(reified) as props, reified
            CREATE (from)-[r]->(to)
            SET r = props, type(r) = rel_type
            DELETE reified
            RETURN count(r) as unreified_count
        "#, FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP);

        let node_ids: Vec<BoltType> = reified_node_ids.iter()
            .map(|id| BoltType::Integer(id.value()))
            .collect();

        let query = Query::new(cypher).param("node_ids", BoltType::List(node_ids));
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::BatchError {
                operation: "unreify_batch".to_string(),
                failed_count: reified_node_ids.len(),
                total_count: reified_node_ids.len(),
            })?;

        if let Ok(Some(row)) = result.next().await {
            let unreified_count: i64 = row.get("unreified_count").unwrap_or(0);
            Ok(unreified_count as usize)
        } else {
            Ok(0)
        }
    }

    async fn process_property_update_batch(&self, updates: Vec<PropertyUpdate>) -> Result<usize> {
        let cypher = r#"
            UNWIND $updates as update
            MATCH (n) WHERE id(n) = update.entity_id
            SET n += update.properties
            RETURN count(n) as updated_count
        "#;

        let update_maps: Vec<BoltType> = updates.iter().map(|update| {
            let mut map = HashMap::new();
            map.insert("entity_id".to_string(), BoltType::Integer(update.entity_id.value()));
            
            let props_map: HashMap<String, BoltType> = update.properties.iter()
                .map(|(k, v)| (k.clone(), v.to_bolt_type()))
                .collect();
            map.insert("properties".to_string(), BoltType::Map(props_map));
            
            BoltType::Map(map)
        }).collect();

        let query = Query::new(cypher.to_string()).param("updates", BoltType::List(update_maps));
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::BatchError {
                operation: "property_update_batch".to_string(),
                failed_count: updates.len(),
                total_count: updates.len(),
            })?;

        if let Ok(Some(row)) = result.next().await {
            let updated_count: i64 = row.get("updated_count").unwrap_or(0);
            Ok(updated_count as usize)
        } else {
            Ok(0)
        }
    }
}

/// Request for reifying a relationship
#[derive(Debug, Clone)]
pub struct ReificationRequest {
    pub from_node_id: EntityId,
    pub to_node_id: EntityId,
    pub relationship_type: String,
    pub properties: PropertyMap,
}

impl ReificationRequest {
    pub fn new(
        from_node_id: EntityId,
        to_node_id: EntityId,
        relationship_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            from_node_id,
            to_node_id,
            relationship_type: relationship_type.into(),
            properties,
        }
    }
}

/// Property update request
#[derive(Debug, Clone)]
pub struct PropertyUpdate {
    pub entity_id: EntityId,
    pub properties: PropertyMap,
}

impl PropertyUpdate {
    pub fn new(entity_id: EntityId, properties: PropertyMap) -> Self {
        Self { entity_id, properties }
    }
}

/// Result of batch operations
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub total_requested: usize,
    pub successful_count: usize,
    pub failed_count: usize,
    pub reified_structures: Vec<ReificationStructure>,
    pub errors: Vec<BatchError>,
    pub execution_time_ms: u64,
}

impl BatchResult {
    pub fn empty() -> Self {
        Self {
            total_requested: 0,
            successful_count: 0,
            failed_count: 0,
            reified_structures: Vec::new(),
            errors: Vec::new(),
            execution_time_ms: 0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_requested == 0 {
            1.0
        } else {
            self.successful_count as f64 / self.total_requested as f64
        }
    }
}

/// Batch operation error
#[derive(Debug, Clone)]
pub struct BatchError {
    pub operation: String,
    pub error: String,
    pub affected_items: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties;

    #[test]
    fn test_reification_request_creation() {
        let from_id = EntityId::new(1);
        let to_id = EntityId::new(2);
        let properties = properties!("weight" => 1.5, "active" => true);

        let request = ReificationRequest::new(from_id, to_id, "CONNECTS_TO", properties.clone());
        
        assert_eq!(request.from_node_id, from_id);
        assert_eq!(request.to_node_id, to_id);
        assert_eq!(request.relationship_type, "CONNECTS_TO");
        assert_eq!(request.properties.len(), 2);
    }

    #[test]
    fn test_batch_result_success_rate() {
        let result = BatchResult {
            total_requested: 100,
            successful_count: 85,
            failed_count: 15,
            reified_structures: Vec::new(),
            errors: Vec::new(),
            execution_time_ms: 1500,
        };
        
        assert_eq!(result.success_rate(), 0.85);
        
        let empty_result = BatchResult::empty();
        assert_eq!(empty_result.success_rate(), 1.0);
    }

    #[test]
    fn test_property_update_creation() {
        let entity_id = EntityId::new(123);
        let properties = properties!("status" => "updated", "modified_at" => "2024-01-01");

        let update = PropertyUpdate::new(entity_id, properties.clone());
        
        assert_eq!(update.entity_id, entity_id);
        assert_eq!(update.properties.len(), 2);
        assert!(update.properties.contains_key("status"));
        assert!(update.properties.contains_key("modified_at"));
    }

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_batch_reification() {
        // Integration tests would go here
    }
}