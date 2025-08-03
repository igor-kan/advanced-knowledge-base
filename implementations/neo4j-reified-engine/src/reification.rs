//! Reification management for Neo4j relationships
//!
//! This module handles the conversion of relationships into nodes and the management
//! of reification hierarchies in Neo4j databases.

use crate::types::*;
use crate::{Neo4jReifiedError, Result, FROM_RELATIONSHIP, TO_RELATIONSHIP, REIFIED_EDGE_LABEL, MAX_REIFICATION_DEPTH};

use neo4rs::{Graph, Query, BoltType};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;
use tracing::{info, warn, error, debug};

/// Manager for reification operations
pub struct ReificationManager {
    /// Neo4j graph connection
    graph: Arc<Graph>,
    /// Performance statistics
    stats: Arc<EngineStats>,
}

impl ReificationManager {
    /// Create a new reification manager
    pub fn new(graph: Arc<Graph>, stats: Arc<EngineStats>) -> Result<Self> {
        Ok(Self { graph, stats })
    }

    /// Reify a relationship between two nodes
    pub async fn reify_relationship(
        &self,
        from_node_id: EntityId,
        to_node_id: EntityId,
        relationship_type: String,
        properties: PropertyMap,
    ) -> Result<ReificationStructure> {
        info!("Reifying relationship {} -> {} ({})", from_node_id, to_node_id, relationship_type);

        // Check if nodes exist
        self.verify_node_exists(from_node_id).await?;
        self.verify_node_exists(to_node_id).await?;

        // Create the reified node
        let reified_node = self.create_reified_node(&relationship_type, &properties).await?;

        // Create FROM connection
        let from_connection = self.create_connection(
            from_node_id,
            reified_node.id,
            FROM_RELATIONSHIP,
            HashMap::new(),
        ).await?;

        // Create TO connection
        let to_connection = self.create_connection(
            reified_node.id,
            to_node_id,
            TO_RELATIONSHIP,
            HashMap::new(),
        ).await?;

        // Create original relationship info
        let original_relationship = OriginalRelationshipInfo {
            original_start_node_id: from_node_id,
            original_end_node_id: to_node_id,
            original_type: relationship_type,
            original_properties: properties,
            reified_at: Utc::now(),
            reified_by: None, // Could be extended to track user/system
            reification_context: None,
        };

        let reification_structure = ReificationStructure {
            reified_node,
            original_relationship,
            from_connection,
            to_connection,
            reification_level: 0, // Base level reification
        };

        self.stats.inc_reifications_performed();
        info!("Successfully reified relationship, created node {}", reification_structure.reified_node.id);

        Ok(reification_structure)
    }

    /// Reify an existing reified relationship (meta-reification)
    pub async fn reify_reified_relationship(
        &self,
        reified_node_id: EntityId,
        target_node_id: EntityId,
        relationship_type: String,
        properties: PropertyMap,
    ) -> Result<ReificationStructure> {
        info!("Creating meta-reification from {} to {} ({})", reified_node_id, target_node_id, relationship_type);

        // Verify the reified node exists and is indeed a reified relationship
        let reified_node = self.get_reified_node(reified_node_id).await?
            .ok_or_else(|| Neo4jReifiedError::ReificationError {
                operation: "meta_reify".to_string(),
                details: format!("Reified node {} not found", reified_node_id),
            })?;

        if !reified_node.metadata.is_reified_relationship {
            return Err(Neo4jReifiedError::ReificationError {
                operation: "meta_reify".to_string(),
                details: "Source node is not a reified relationship".to_string(),
            });
        }

        // Get current reification depth
        let current_depth = self.get_reification_depth(reified_node_id).await?;
        if current_depth >= MAX_REIFICATION_DEPTH {
            return Err(Neo4jReifiedError::ReificationError {
                operation: "meta_reify".to_string(),
                details: format!("Maximum reification depth ({}) exceeded", MAX_REIFICATION_DEPTH),
            });
        }

        // Create meta-reified node
        let meta_reified_node = self.create_reified_node(&relationship_type, &properties).await?;

        // Create connections
        let from_connection = self.create_connection(
            reified_node_id,
            meta_reified_node.id,
            FROM_RELATIONSHIP,
            HashMap::new(),
        ).await?;

        let to_connection = self.create_connection(
            meta_reified_node.id,
            target_node_id,
            TO_RELATIONSHIP,
            HashMap::new(),
        ).await?;

        let original_relationship = OriginalRelationshipInfo {
            original_start_node_id: reified_node_id,
            original_end_node_id: target_node_id,
            original_type: relationship_type,
            original_properties: properties,
            reified_at: Utc::now(),
            reified_by: None,
            reification_context: Some("meta_reification".to_string()),
        };

        let reification_structure = ReificationStructure {
            reified_node: meta_reified_node,
            original_relationship,
            from_connection,
            to_connection,
            reification_level: current_depth + 1,
        };

        self.stats.inc_reifications_performed();
        info!("Successfully created meta-reification at level {}", reification_structure.reification_level);

        Ok(reification_structure)
    }

    /// Unreify a relationship back to a simple edge
    pub async fn unreify_relationship(&self, reified_node_id: EntityId) -> Result<ReifiedRelationship> {
        info!("Unreifying relationship node {}", reified_node_id);

        // Get the reified node and its connections
        let reification_info = self.get_reification_info(reified_node_id).await?;

        // Create the simple relationship
        let simple_relationship = ReifiedRelationship::new(
            reification_info.from_node_id,
            reification_info.to_node_id,
            reification_info.original_type,
            reification_info.properties,
        );

        // Execute unreification in Neo4j
        let cypher = r#"
            MATCH (from)-[:FROM]->(reified:ReifiedEdge)-[:TO]->(to)
            WHERE id(reified) = $reified_id
            CREATE (from)-[r:$rel_type]->(to)
            SET r = $properties
            DELETE reified
            WITH from, to, r
            MATCH (from)-[:FROM]->(reified)-[:TO]->(to)
            DELETE reified
            RETURN id(r) as rel_id
        "#;

        let mut params = HashMap::new();
        params.insert("reified_id".to_string(), BoltType::Integer(reified_node_id.value()));
        params.insert("rel_type".to_string(), BoltType::String(reification_info.original_type));
        
        // Convert properties to BoltType map
        let props_map: HashMap<String, BoltType> = reification_info.properties
            .into_iter()
            .map(|(k, v)| (k, v.to_bolt_type()))
            .collect();
        params.insert("properties".to_string(), BoltType::Map(props_map));

        let mut query = Query::new(cypher.to_string());
        for (key, value) in params {
            query = query.param(key, value);
        }

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "unreify".to_string(),
                details: format!("Failed to unreify relationship: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "unreify".to_string(),
                details: format!("Failed to read unreify result: {}", e),
            })? {
            let rel_id: i64 = row.get("rel_id")
                .map_err(|e| Neo4jReifiedError::ReificationError {
                    operation: "unreify".to_string(),
                    details: format!("Failed to get relationship ID: {}", e),
                })?;

            let mut unreified_rel = simple_relationship;
            unreified_rel.id = EntityId::new(rel_id);

            info!("Successfully unreified relationship, created edge {}", unreified_rel.id);
            Ok(unreified_rel)
        } else {
            Err(Neo4jReifiedError::ReificationError {
                operation: "unreify".to_string(),
                details: "No result returned from unreification".to_string(),
            })
        }
    }

    /// Get reification hierarchy for a node
    pub async fn get_reification_hierarchy(&self, node_id: EntityId) -> Result<ReificationHierarchy> {
        debug!("Getting reification hierarchy for node {}", node_id);

        let mut hierarchy = ReificationHierarchy {
            root_node_id: node_id,
            levels: Vec::new(),
            max_depth: 0,
            total_reified_relationships: 0,
        };

        // Build hierarchy recursively
        self.build_hierarchy_recursive(node_id, 0, &mut hierarchy).await?;

        Ok(hierarchy)
    }

    /// Find all reified relationships of a specific type
    pub async fn find_reified_relationships_by_type(&self, relationship_type: &str) -> Result<Vec<ReificationStructure>> {
        let cypher = r#"
            MATCH (from)-[:FROM]->(reified:ReifiedEdge)-[:TO]->(to)
            WHERE reified.original_type = $rel_type
            RETURN id(from) as from_id, id(reified) as reified_id, id(to) as to_id,
                   reified, from, to
        "#;

        let query = Query::new(cypher.to_string())
            .param("rel_type", relationship_type);

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "find_by_type".to_string(),
                details: format!("Failed to find reified relationships: {}", e),
            })?;

        let mut reified_relationships = Vec::new();

        while let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "find_by_type".to_string(),
                details: format!("Failed to read search results: {}", e),
            })? {
            
            let from_id: i64 = row.get("from_id")?;
            let reified_id: i64 = row.get("reified_id")?;
            let to_id: i64 = row.get("to_id")?;

            // Build reification structure
            let reified_node = self.get_reified_node(EntityId::new(reified_id)).await?
                .ok_or_else(|| Neo4jReifiedError::ReificationError {
                    operation: "find_by_type".to_string(),
                    details: format!("Reified node {} not found", reified_id),
                })?;

            let from_connection = ReifiedRelationship::new(
                EntityId::new(from_id),
                EntityId::new(reified_id),
                FROM_RELATIONSHIP,
                HashMap::new(),
            );

            let to_connection = ReifiedRelationship::new(
                EntityId::new(reified_id),
                EntityId::new(to_id),
                TO_RELATIONSHIP,
                HashMap::new(),
            );

            let original_relationship = OriginalRelationshipInfo {
                original_start_node_id: EntityId::new(from_id),
                original_end_node_id: EntityId::new(to_id),
                original_type: relationship_type.to_string(),
                original_properties: reified_node.properties.clone(),
                reified_at: reified_node.metadata.created_at,
                reified_by: None,
                reification_context: None,
            };

            let reification_structure = ReificationStructure {
                reified_node,
                original_relationship,
                from_connection,
                to_connection,
                reification_level: 0, // This would need to be calculated
            };

            reified_relationships.push(reification_structure);
        }

        Ok(reified_relationships)
    }

    /// Get statistics about reified relationships
    pub async fn get_reification_stats(&self) -> Result<ReificationStats> {
        let cypher = r#"
            MATCH (n:ReifiedEdge)
            WITH count(n) as total_reified
            MATCH (n:ReifiedEdge)
            RETURN total_reified,
                   collect(distinct n.original_type) as types,
                   avg(size(keys(n))) as avg_properties
        "#;

        let query = Query::new(cypher.to_string());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_stats".to_string(),
                details: format!("Failed to get reification statistics: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_stats".to_string(),
                details: format!("Failed to read statistics: {}", e),
            })? {
            
            let total_reified: i64 = row.get("total_reified").unwrap_or(0);
            let types: Vec<String> = row.get("types").unwrap_or_default();
            let avg_properties: f64 = row.get("avg_properties").unwrap_or(0.0);

            Ok(ReificationStats {
                total_reified_relationships: total_reified as u64,
                relationship_types: types,
                average_properties_per_reification: avg_properties,
                max_reification_depth: MAX_REIFICATION_DEPTH,
            })
        } else {
            Ok(ReificationStats {
                total_reified_relationships: 0,
                relationship_types: Vec::new(),
                average_properties_per_reification: 0.0,
                max_reification_depth: MAX_REIFICATION_DEPTH,
            })
        }
    }

    // Private helper methods

    async fn verify_node_exists(&self, node_id: EntityId) -> Result<()> {
        let query = Query::new("MATCH (n) WHERE id(n) = $id RETURN count(n) as node_count".to_string())
            .param("id", node_id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ValidationError {
                entity: "node".to_string(),
                constraint: format!("Failed to verify node existence: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ValidationError {
                entity: "node".to_string(),
                constraint: format!("Failed to read node count: {}", e),
            })? {
            let count: i64 = row.get("node_count")
                .map_err(|e| Neo4jReifiedError::ValidationError {
                    entity: "node".to_string(),
                    constraint: format!("Failed to get node count: {}", e),
                })?;

            if count == 0 {
                return Err(Neo4jReifiedError::ValidationError {
                    entity: "node".to_string(),
                    constraint: format!("Node {} does not exist", node_id),
                });
            }
        }

        Ok(())
    }

    async fn create_reified_node(&self, relationship_type: &str, properties: &PropertyMap) -> Result<ReifiedNode> {
        let mut cypher = format!("CREATE (n:{}) ", REIFIED_EDGE_LABEL);
        let mut params = HashMap::new();

        // Set original relationship type
        cypher.push_str("SET n.original_type = $original_type ");
        params.insert("original_type".to_string(), BoltType::String(relationship_type.to_string()));

        // Add properties
        if !properties.is_empty() {
            cypher.push_str(", ");
            let prop_assignments: Vec<String> = properties.keys()
                .enumerate()
                .map(|(i, key)| {
                    let param_name = format!("prop_{}", i);
                    params.insert(param_name.clone(), properties[key].to_bolt_type());
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
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "create_reified_node".to_string(),
                details: format!("Failed to create reified node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "create_reified_node".to_string(),
                details: format!("Failed to read reified node result: {}", e),
            })? {
            let node_id: i64 = row.get("node_id")
                .map_err(|e| Neo4jReifiedError::ReificationError {
                    operation: "create_reified_node".to_string(),
                    details: format!("Failed to get reified node ID: {}", e),
                })?;

            let mut node = ReifiedNode::with_id(
                EntityId::new(node_id),
                vec![REIFIED_EDGE_LABEL.to_string()],
                properties.clone(),
            );

            // Mark as reified relationship
            node.metadata.mark_as_reified_relationship(relationship_type.to_string());

            Ok(node)
        } else {
            Err(Neo4jReifiedError::ReificationError {
                operation: "create_reified_node".to_string(),
                details: "No result returned from reified node creation".to_string(),
            })
        }
    }

    async fn create_connection(
        &self,
        from_id: EntityId,
        to_id: EntityId,
        connection_type: &str,
        properties: PropertyMap,
    ) -> Result<ReifiedRelationship> {
        let cypher = format!(
            "MATCH (from), (to) WHERE id(from) = $from_id AND id(to) = $to_id CREATE (from)-[r:{}]->(to) RETURN id(r) as rel_id",
            connection_type
        );

        let query = Query::new(cypher)
            .param("from_id", from_id.value())
            .param("to_id", to_id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "create_connection".to_string(),
                details: format!("Failed to create {} connection: {}", connection_type, e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "create_connection".to_string(),
                details: format!("Failed to read connection result: {}", e),
            })? {
            let rel_id: i64 = row.get("rel_id")
                .map_err(|e| Neo4jReifiedError::ReificationError {
                    operation: "create_connection".to_string(),
                    details: format!("Failed to get connection ID: {}", e),
                })?;

            let mut connection = ReifiedRelationship::with_id(
                EntityId::new(rel_id),
                from_id,
                to_id,
                connection_type,
                properties,
            );

            // Mark as reification connection
            connection.metadata.mark_as_reification_connection();

            Ok(connection)
        } else {
            Err(Neo4jReifiedError::ReificationError {
                operation: "create_connection".to_string(),
                details: format!("No result returned from {} connection creation", connection_type),
            })
        }
    }

    async fn get_reified_node(&self, node_id: EntityId) -> Result<Option<ReifiedNode>> {
        let query = Query::new(format!("MATCH (n:{}) WHERE id(n) = $id RETURN n", REIFIED_EDGE_LABEL))
            .param("id", node_id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_reified_node".to_string(),
                details: format!("Failed to get reified node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_reified_node".to_string(),
                details: format!("Failed to read reified node: {}", e),
            })? {
            let neo4j_node: neo4rs::Node = row.get("n")
                .map_err(|e| Neo4jReifiedError::ReificationError {
                    operation: "get_reified_node".to_string(),
                    details: format!("Failed to get node data: {}", e),
                })?;

            let mut node = ReifiedNode::from_neo4j_node(&neo4j_node);
            node.metadata.is_reified_relationship = true;

            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    async fn get_reification_depth(&self, node_id: EntityId) -> Result<usize> {
        // For now, return 0 - this would need recursive depth calculation
        Ok(0)
    }

    async fn get_reification_info(&self, reified_node_id: EntityId) -> Result<UnreifyInfo> {
        let cypher = r#"
            MATCH (from)-[:FROM]->(reified:ReifiedEdge)-[:TO]->(to)
            WHERE id(reified) = $reified_id
            RETURN id(from) as from_id, id(to) as to_id,
                   reified.original_type as original_type,
                   properties(reified) as props
        "#;

        let query = Query::new(cypher.to_string())
            .param("reified_id", reified_node_id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_reification_info".to_string(),
                details: format!("Failed to get reification info: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::ReificationError {
                operation: "get_reification_info".to_string(),
                details: format!("Failed to read reification info: {}", e),
            })? {
            
            let from_id: i64 = row.get("from_id")?;
            let to_id: i64 = row.get("to_id")?;
            let original_type: String = row.get("original_type")?;
            
            // Convert properties - this is simplified
            let properties = PropertyMap::new(); // Would need proper conversion

            Ok(UnreifyInfo {
                from_node_id: EntityId::new(from_id),
                to_node_id: EntityId::new(to_id),
                original_type,
                properties,
            })
        } else {
            Err(Neo4jReifiedError::ReificationError {
                operation: "get_reification_info".to_string(),
                details: format!("Reified node {} not found or not properly structured", reified_node_id),
            })
        }
    }

    async fn build_hierarchy_recursive(
        &self,
        node_id: EntityId,
        depth: usize,
        hierarchy: &mut ReificationHierarchy,
    ) -> Result<()> {
        // Implementation would recursively build the hierarchy
        // For now, this is a placeholder
        hierarchy.max_depth = hierarchy.max_depth.max(depth);
        Ok(())
    }
}

/// Pattern for reification operations
pub struct ReificationPattern {
    pub pattern_name: String,
    pub description: String,
    pub cypher_template: String,
}

impl ReificationPattern {
    pub fn simple_reification() -> Self {
        Self {
            pattern_name: "simple_reification".to_string(),
            description: "Convert a relationship into a reified node with FROM/TO connections".to_string(),
            cypher_template: r#"
                MATCH (a)-[r:$REL_TYPE]->(b)
                CREATE (a)-[:FROM]->(reified:ReifiedEdge {original_type: '$REL_TYPE'})-[:TO]->(b)
                SET reified += $PROPERTIES
                DELETE r
                RETURN reified
            "#.to_string(),
        }
    }

    pub fn meta_reification() -> Self {
        Self {
            pattern_name: "meta_reification".to_string(),
            description: "Reify a relationship that connects to an already reified relationship".to_string(),
            cypher_template: r#"
                MATCH (reified1:ReifiedEdge), (target)
                WHERE id(reified1) = $REIFIED_ID AND id(target) = $TARGET_ID
                CREATE (reified1)-[:FROM]->(meta:ReifiedEdge {original_type: '$REL_TYPE'})-[:TO]->(target)
                SET meta += $PROPERTIES
                RETURN meta
            "#.to_string(),
        }
    }
}

/// Hierarchy information for reified relationships
pub struct ReificationHierarchy {
    pub root_node_id: EntityId,
    pub levels: Vec<Vec<EntityId>>,
    pub max_depth: usize,
    pub total_reified_relationships: u64,
}

/// Statistics about reification in the database
pub struct ReificationStats {
    pub total_reified_relationships: u64,
    pub relationship_types: Vec<String>,
    pub average_properties_per_reification: f64,
    pub max_reification_depth: usize,
}

/// Helper struct for unreification
struct UnreifyInfo {
    pub from_node_id: EntityId,
    pub to_node_id: EntityId,
    pub original_type: String,
    pub properties: PropertyMap,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties;

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_reification_pattern_creation() {
        let simple = ReificationPattern::simple_reification();
        assert_eq!(simple.pattern_name, "simple_reification");
        assert!(!simple.cypher_template.is_empty());

        let meta = ReificationPattern::meta_reification();
        assert_eq!(meta.pattern_name, "meta_reification");
        assert!(!meta.cypher_template.is_empty());
    }
}