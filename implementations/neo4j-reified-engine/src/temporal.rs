//! Temporal versioning for reified relationships
//!
//! This module provides time-aware reification with versioning capabilities,
//! allowing tracking of changes over time in reified relationship structures.

use crate::types::*;
use crate::{Neo4jReifiedError, Result, REIFIED_EDGE_LABEL, FROM_RELATIONSHIP, TO_RELATIONSHIP};

use neo4rs::{Graph, Query, BoltType};
use std::sync::Arc;
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug};

/// Temporal reification manager
pub struct TemporalReifier {
    /// Neo4j graph connection
    graph: Arc<Graph>,
}

impl TemporalReifier {
    /// Create a new temporal reifier
    pub fn new(graph: Arc<Graph>) -> Result<Self> {
        Ok(Self { graph })
    }

    /// Create a temporal reified relationship with versioning
    pub async fn reify_with_temporal_versioning(
        &self,
        from_node_id: EntityId,
        to_node_id: EntityId,
        relationship_type: String,
        properties: PropertyMap,
        temporal_config: TemporalConfig,
    ) -> Result<TemporalReificationStructure> {
        info!("Creating temporal reified relationship {} -> {} ({})", from_node_id, to_node_id, relationship_type);

        // Create base reified node with temporal properties
        let mut temporal_properties = properties.clone();
        
        // Add temporal metadata
        temporal_properties.insert("valid_from".to_string(), 
            PropertyValue::Date(temporal_config.valid_from));
        
        if let Some(valid_to) = temporal_config.valid_to {
            temporal_properties.insert("valid_to".to_string(), 
                PropertyValue::Date(valid_to));
        }
        
        temporal_properties.insert("version".to_string(), 
            PropertyValue::String(temporal_config.version.clone()));
        
        temporal_properties.insert("version_strategy".to_string(), 
            PropertyValue::String(temporal_config.version_strategy.to_string()));

        // Create the temporal reified node
        let reified_node = self.create_temporal_reified_node(
            &relationship_type,
            &temporal_properties,
            &temporal_config,
        ).await?;

        // Create temporal connections
        let from_connection = self.create_temporal_connection(
            from_node_id,
            reified_node.id,
            FROM_RELATIONSHIP,
            temporal_config.valid_from,
            temporal_config.valid_to,
        ).await?;

        let to_connection = self.create_temporal_connection(
            reified_node.id,
            to_node_id,
            TO_RELATIONSHIP,
            temporal_config.valid_from,
            temporal_config.valid_to,
        ).await?;

        let temporal_structure = TemporalReificationStructure {
            reified_node,
            from_connection,
            to_connection,
            temporal_config: temporal_config.clone(),
            version_history: Vec::new(),
            is_current_version: true,
        };

        info!("Successfully created temporal reification version {}", temporal_config.version);
        Ok(temporal_structure)
    }

    /// Create a new version of an existing temporal reified relationship
    pub async fn create_new_version(
        &self,
        current_reified_id: EntityId,
        new_properties: PropertyMap,
        version_config: VersionConfig,
    ) -> Result<TemporalReificationStructure> {
        info!("Creating new version of temporal reified relationship {}", current_reified_id);

        // Get current version
        let current_version = self.get_temporal_reified_node(current_reified_id).await?
            .ok_or_else(|| Neo4jReifiedError::TemporalError {
                operation: "create_new_version".to_string(),
                details: format!("Current reified node {} not found", current_reified_id),
            })?;

        // Determine new version number
        let new_version = self.generate_next_version(
            &current_version.version,
            &version_config.strategy,
        )?;

        // End current version
        self.end_current_version(current_reified_id, version_config.effective_date).await?;

        // Create new temporal config
        let new_temporal_config = TemporalConfig {
            valid_from: version_config.effective_date,
            valid_to: None, // Open-ended
            version: new_version.clone(),
            version_strategy: version_config.strategy.clone(),
        };

        // Get connection information from current version
        let connections = self.get_temporal_connections(current_reified_id).await?;

        // Create new version with updated properties
        let mut version_properties = new_properties;
        version_properties.insert("original_type".to_string(), 
            current_version.original_type.clone().into());

        let new_reified_structure = self.reify_with_temporal_versioning(
            connections.from_node_id,
            connections.to_node_id,
            current_version.original_type,
            version_properties,
            new_temporal_config,
        ).await?;

        // Link to previous version
        self.link_versions(current_reified_id, new_reified_structure.reified_node.id).await?;

        info!("Successfully created new version {} from {}", new_version, current_version.version);
        Ok(new_reified_structure)
    }

    /// Get the version history of a temporal reified relationship
    pub async fn get_version_history(&self, reified_id: EntityId) -> Result<Vec<TemporalVersion>> {
        let cypher = format!(r#"
            MATCH (current:{}) WHERE id(current) = $reified_id
            OPTIONAL MATCH path = (current)-[:PREVIOUS_VERSION*0..]->(version:{})
            WITH version
            RETURN id(version) as version_id,
                   version.version as version_number,
                   version.valid_from as valid_from,
                   version.valid_to as valid_to,
                   version.created_at as created_at,
                   properties(version) as properties
            ORDER BY version.created_at DESC
        "#, REIFIED_EDGE_LABEL, REIFIED_EDGE_LABEL);

        let query = Query::new(cypher).param("reified_id", reified_id.value());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_version_history".to_string(),
                details: format!("Failed to get version history: {}", e),
            })?;

        let mut versions = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            let version_id: i64 = row.get("version_id").unwrap_or(-1);
            let version_number: String = row.get("version_number").unwrap_or_default();
            
            if version_id > 0 {
                let version = TemporalVersion {
                    id: EntityId::new(version_id),
                    version_number,
                    valid_from: row.get::<DateTime<Utc>>("valid_from").unwrap_or_else(|_| Utc::now()),
                    valid_to: row.get::<DateTime<Utc>>("valid_to").ok(),
                    created_at: row.get::<DateTime<Utc>>("created_at").unwrap_or_else(|_| Utc::now()),
                    properties: HashMap::new(), // Would need proper property conversion
                };
                versions.push(version);
            }
        }

        Ok(versions)
    }

    /// Get the current active version of a temporal reified relationship
    pub async fn get_current_version(&self, base_id: EntityId) -> Result<Option<TemporalReificationStructure>> {
        let cypher = format!(r#"
            MATCH (reified:{}) 
            WHERE id(reified) = $base_id OR reified.base_id = $base_id
            AND (reified.valid_to IS NULL OR reified.valid_to > datetime())
            AND reified.valid_from <= datetime()
            RETURN reified
            ORDER BY reified.created_at DESC
            LIMIT 1
        "#, REIFIED_EDGE_LABEL);

        let query = Query::new(cypher).param("base_id", base_id.value());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_current_version".to_string(),
                details: format!("Failed to get current version: {}", e),
            })?;

        if let Ok(Some(row)) = result.next().await {
            let neo4j_node: neo4rs::Node = row.get("reified")
                .map_err(|e| Neo4jReifiedError::TemporalError {
                    operation: "get_current_version".to_string(),
                    details: format!("Failed to get reified node: {}", e),
                })?;

            // Convert to temporal structure - simplified implementation
            let reified_node = ReifiedNode::from_neo4j_node(&neo4j_node);
            
            // Get temporal config from properties
            let temporal_config = self.extract_temporal_config(&reified_node)?;
            
            // Get connections
            let connections = self.get_temporal_connections(reified_node.id).await?;
            
            let temporal_structure = TemporalReificationStructure {
                reified_node,
                from_connection: ReifiedRelationship::new(
                    connections.from_node_id,
                    connections.to_node_id,
                    FROM_RELATIONSHIP,
                    HashMap::new(),
                ),
                to_connection: ReifiedRelationship::new(
                    connections.from_node_id,
                    connections.to_node_id,
                    TO_RELATIONSHIP,
                    HashMap::new(),
                ),
                temporal_config,
                version_history: Vec::new(),
                is_current_version: true,
            };

            Ok(Some(temporal_structure))
        } else {
            Ok(None)
        }
    }

    /// Get version at a specific point in time
    pub async fn get_version_at_time(
        &self,
        base_id: EntityId,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<TemporalReificationStructure>> {
        let cypher = format!(r#"
            MATCH (reified:{}) 
            WHERE id(reified) = $base_id OR reified.base_id = $base_id
            AND reified.valid_from <= $timestamp
            AND (reified.valid_to IS NULL OR reified.valid_to > $timestamp)
            RETURN reified
            ORDER BY reified.created_at DESC
            LIMIT 1
        "#, REIFIED_EDGE_LABEL);

        let query = Query::new(cypher)
            .param("base_id", base_id.value())
            .param("timestamp", timestamp.to_rfc3339());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_version_at_time".to_string(),
                details: format!("Failed to get version at time: {}", e),
            })?;

        if let Ok(Some(row)) = result.next().await {
            let neo4j_node: neo4rs::Node = row.get("reified")
                .map_err(|e| Neo4jReifiedError::TemporalError {
                    operation: "get_version_at_time".to_string(),
                    details: format!("Failed to get reified node: {}", e),
                })?;

            let reified_node = ReifiedNode::from_neo4j_node(&neo4j_node);
            let temporal_config = self.extract_temporal_config(&reified_node)?;
            let connections = self.get_temporal_connections(reified_node.id).await?;
            
            let temporal_structure = TemporalReificationStructure {
                reified_node,
                from_connection: ReifiedRelationship::new(
                    connections.from_node_id,
                    connections.to_node_id,
                    FROM_RELATIONSHIP,
                    HashMap::new(),
                ),
                to_connection: ReifiedRelationship::new(
                    connections.from_node_id,
                    connections.to_node_id,
                    TO_RELATIONSHIP,
                    HashMap::new(),
                ),
                temporal_config,
                version_history: Vec::new(),
                is_current_version: false,
            };

            Ok(Some(temporal_structure))
        } else {
            Ok(None)
        }
    }

    // Private helper methods

    async fn create_temporal_reified_node(
        &self,
        relationship_type: &str,
        properties: &PropertyMap,
        temporal_config: &TemporalConfig,
    ) -> Result<ReifiedNode> {
        let mut cypher = format!("CREATE (n:{}) ", REIFIED_EDGE_LABEL);
        let mut params = HashMap::new();

        // Set temporal and reification properties
        cypher.push_str("SET n.original_type = $original_type, ");
        cypher.push_str("n.version = $version, ");
        cypher.push_str("n.version_strategy = $version_strategy, ");
        cypher.push_str("n.valid_from = $valid_from, ");
        cypher.push_str("n.created_at = datetime(), ");
        cypher.push_str("n.is_temporal = true ");

        params.insert("original_type".to_string(), BoltType::String(relationship_type.to_string()));
        params.insert("version".to_string(), BoltType::String(temporal_config.version.clone()));
        params.insert("version_strategy".to_string(), BoltType::String(temporal_config.version_strategy.to_string()));
        params.insert("valid_from".to_string(), BoltType::String(temporal_config.valid_from.to_rfc3339()));

        if let Some(valid_to) = temporal_config.valid_to {
            cypher.push_str(", n.valid_to = $valid_to ");
            params.insert("valid_to".to_string(), BoltType::String(valid_to.to_rfc3339()));
        }

        // Add user properties
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
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "create_temporal_reified_node".to_string(),
                details: format!("Failed to create temporal reified node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "create_temporal_reified_node".to_string(),
                details: format!("Failed to read temporal node result: {}", e),
            })? {
            let node_id: i64 = row.get("node_id")
                .map_err(|e| Neo4jReifiedError::TemporalError {
                    operation: "create_temporal_reified_node".to_string(),
                    details: format!("Failed to get temporal node ID: {}", e),
                })?;

            let mut node = ReifiedNode::with_id(
                EntityId::new(node_id),
                vec![REIFIED_EDGE_LABEL.to_string()],
                properties.clone(),
            );

            // Mark as temporal reified relationship
            node.metadata.mark_as_reified_relationship(relationship_type.to_string());

            Ok(node)
        } else {
            Err(Neo4jReifiedError::TemporalError {
                operation: "create_temporal_reified_node".to_string(),
                details: "No result returned from temporal reified node creation".to_string(),
            })
        }
    }

    async fn create_temporal_connection(
        &self,
        from_id: EntityId,
        to_id: EntityId,
        connection_type: &str,
        valid_from: DateTime<Utc>,
        valid_to: Option<DateTime<Utc>>,
    ) -> Result<ReifiedRelationship> {
        let mut cypher = format!(
            "MATCH (from), (to) WHERE id(from) = $from_id AND id(to) = $to_id CREATE (from)-[r:{}]->(to) ",
            connection_type
        );
        
        cypher.push_str("SET r.valid_from = $valid_from, r.created_at = datetime(), r.is_temporal = true ");
        
        if valid_to.is_some() {
            cypher.push_str(", r.valid_to = $valid_to ");
        }
        
        cypher.push_str("RETURN id(r) as rel_id");

        let mut query = Query::new(cypher)
            .param("from_id", from_id.value())
            .param("to_id", to_id.value())
            .param("valid_from", valid_from.to_rfc3339());

        if let Some(valid_to) = valid_to {
            query = query.param("valid_to", valid_to.to_rfc3339());
        }

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "create_temporal_connection".to_string(),
                details: format!("Failed to create temporal connection: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "create_temporal_connection".to_string(),
                details: format!("Failed to read temporal connection result: {}", e),
            })? {
            let rel_id: i64 = row.get("rel_id")
                .map_err(|e| Neo4jReifiedError::TemporalError {
                    operation: "create_temporal_connection".to_string(),
                    details: format!("Failed to get temporal connection ID: {}", e),
                })?;

            let mut connection = ReifiedRelationship::with_id(
                EntityId::new(rel_id),
                from_id,
                to_id,
                connection_type,
                HashMap::new(),
            );

            connection.metadata.mark_as_reification_connection();

            Ok(connection)
        } else {
            Err(Neo4jReifiedError::TemporalError {
                operation: "create_temporal_connection".to_string(),
                details: format!("No result returned from temporal {} connection creation", connection_type),
            })
        }
    }

    async fn get_temporal_reified_node(&self, node_id: EntityId) -> Result<Option<TemporalReifiedNode>> {
        let query = Query::new(format!("MATCH (n:{}) WHERE id(n) = $id AND n.is_temporal = true RETURN n", REIFIED_EDGE_LABEL))
            .param("id", node_id.value());

        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_temporal_reified_node".to_string(),
                details: format!("Failed to get temporal reified node: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_temporal_reified_node".to_string(),
                details: format!("Failed to read temporal reified node: {}", e),
            })? {
            let neo4j_node: neo4rs::Node = row.get("n")
                .map_err(|e| Neo4jReifiedError::TemporalError {
                    operation: "get_temporal_reified_node".to_string(),
                    details: format!("Failed to get node data: {}", e),
                })?;

            // Extract temporal properties
            let version = neo4j_node.get::<String>("version").unwrap_or_default();
            let original_type = neo4j_node.get::<String>("original_type").unwrap_or_default();

            Ok(Some(TemporalReifiedNode {
                id: EntityId::new(neo4j_node.id()),
                version,
                original_type,
            }))
        } else {
            Ok(None)
        }
    }

    async fn get_temporal_connections(&self, reified_id: EntityId) -> Result<ConnectionInfo> {
        let cypher = format!(r#"
            MATCH (from)-[:{}]->(reified)-[:{}]->(to)
            WHERE id(reified) = $reified_id
            RETURN id(from) as from_id, id(to) as to_id
        "#, FROM_RELATIONSHIP, TO_RELATIONSHIP);

        let query = Query::new(cypher).param("reified_id", reified_id.value());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_temporal_connections".to_string(),
                details: format!("Failed to get temporal connections: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "get_temporal_connections".to_string(),
                details: format!("Failed to read temporal connections: {}", e),
            })? {
            let from_id: i64 = row.get("from_id")?;
            let to_id: i64 = row.get("to_id")?;

            Ok(ConnectionInfo {
                from_node_id: EntityId::new(from_id),
                to_node_id: EntityId::new(to_id),
            })
        } else {
            Err(Neo4jReifiedError::TemporalError {
                operation: "get_temporal_connections".to_string(),
                details: format!("No connections found for reified node {}", reified_id),
            })
        }
    }

    async fn end_current_version(&self, reified_id: EntityId, end_time: DateTime<Utc>) -> Result<()> {
        let cypher = format!(r#"
            MATCH (reified:{})
            WHERE id(reified) = $reified_id
            SET reified.valid_to = $end_time
        "#, REIFIED_EDGE_LABEL);

        let query = Query::new(cypher)
            .param("reified_id", reified_id.value())
            .param("end_time", end_time.to_rfc3339());

        self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "end_current_version".to_string(),
                details: format!("Failed to end current version: {}", e),
            })?;

        Ok(())
    }

    async fn link_versions(&self, previous_id: EntityId, current_id: EntityId) -> Result<()> {
        let cypher = r#"
            MATCH (previous), (current)
            WHERE id(previous) = $previous_id AND id(current) = $current_id
            CREATE (current)-[:PREVIOUS_VERSION]->(previous)
        "#;

        let query = Query::new(cypher.to_string())
            .param("previous_id", previous_id.value())
            .param("current_id", current_id.value());

        self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::TemporalError {
                operation: "link_versions".to_string(),
                details: format!("Failed to link versions: {}", e),
            })?;

        Ok(())
    }

    fn generate_next_version(&self, current_version: &str, strategy: &VersionStrategy) -> Result<String> {
        match strategy {
            VersionStrategy::Semantic => {
                // Parse semantic version (e.g., "1.2.3" -> "1.2.4")
                let parts: Vec<&str> = current_version.split('.').collect();
                if parts.len() != 3 {
                    return Ok(format!("{}.1", current_version));
                }
                
                let patch: u32 = parts[2].parse().unwrap_or(0);
                Ok(format!("{}.{}.{}", parts[0], parts[1], patch + 1))
            }
            VersionStrategy::Sequential => {
                // Parse sequential version (e.g., "v5" -> "v6")
                if let Some(num_str) = current_version.strip_prefix('v') {
                    let num: u32 = num_str.parse().unwrap_or(0);
                    Ok(format!("v{}", num + 1))
                } else {
                    let num: u32 = current_version.parse().unwrap_or(0);
                    Ok((num + 1).to_string())
                }
            }
            VersionStrategy::Timestamp => {
                Ok(Utc::now().timestamp().to_string())
            }
        }
    }

    fn extract_temporal_config(&self, node: &ReifiedNode) -> Result<TemporalConfig> {
        let valid_from = node.get_property("valid_from")
            .and_then(|v| match v {
                PropertyValue::Date(dt) => Some(*dt),
                PropertyValue::String(s) => s.parse().ok(),
                _ => None,
            })
            .unwrap_or_else(Utc::now);

        let valid_to = node.get_property("valid_to")
            .and_then(|v| match v {
                PropertyValue::Date(dt) => Some(*dt),
                PropertyValue::String(s) => s.parse().ok(),
                _ => None,
            });

        let version = node.get_property("version")
            .and_then(|v| match v {
                PropertyValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "1.0.0".to_string());

        let version_strategy = node.get_property("version_strategy")
            .and_then(|v| match v {
                PropertyValue::String(s) => s.parse().ok(),
                _ => None,
            })
            .unwrap_or(VersionStrategy::Semantic);

        Ok(TemporalConfig {
            valid_from,
            valid_to,
            version,
            version_strategy,
        })
    }
}

/// Temporal configuration for reified relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub version: String,
    pub version_strategy: VersionStrategy,
}

/// Version strategy for temporal relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionStrategy {
    /// Semantic versioning (e.g., 1.0.0, 1.0.1, 1.1.0)
    Semantic,
    /// Sequential versioning (e.g., v1, v2, v3)
    Sequential,
    /// Timestamp-based versioning
    Timestamp,
}

impl std::fmt::Display for VersionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionStrategy::Semantic => write!(f, "semantic"),
            VersionStrategy::Sequential => write!(f, "sequential"),
            VersionStrategy::Timestamp => write!(f, "timestamp"),
        }
    }
}

impl std::str::FromStr for VersionStrategy {
    type Err = Neo4jReifiedError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "semantic" => Ok(VersionStrategy::Semantic),
            "sequential" => Ok(VersionStrategy::Sequential),
            "timestamp" => Ok(VersionStrategy::Timestamp),
            _ => Err(Neo4jReifiedError::ValidationError {
                entity: "version_strategy".to_string(),
                constraint: format!("Unknown version strategy: {}", s),
            }),
        }
    }
}

/// Configuration for creating a new version
#[derive(Debug, Clone)]
pub struct VersionConfig {
    pub effective_date: DateTime<Utc>,
    pub strategy: VersionStrategy,
}

/// Temporal reification structure
#[derive(Debug, Clone)]
pub struct TemporalReificationStructure {
    pub reified_node: ReifiedNode,
    pub from_connection: ReifiedRelationship,
    pub to_connection: ReifiedRelationship,
    pub temporal_config: TemporalConfig,
    pub version_history: Vec<TemporalVersion>,
    pub is_current_version: bool,
}

/// Version information
#[derive(Debug, Clone)]
pub struct TemporalVersion {
    pub id: EntityId,
    pub version_number: String,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub properties: PropertyMap,
}

// Helper structs
struct TemporalReifiedNode {
    id: EntityId,
    version: String,
    original_type: String,
}

struct ConnectionInfo {
    from_node_id: EntityId,
    to_node_id: EntityId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_strategy_parsing() {
        assert!(matches!("semantic".parse::<VersionStrategy>().unwrap(), VersionStrategy::Semantic));
        assert!(matches!("sequential".parse::<VersionStrategy>().unwrap(), VersionStrategy::Sequential));
        assert!(matches!("timestamp".parse::<VersionStrategy>().unwrap(), VersionStrategy::Timestamp));
        assert!("invalid".parse::<VersionStrategy>().is_err());
    }

    #[test]
    fn test_version_strategy_display() {
        assert_eq!(VersionStrategy::Semantic.to_string(), "semantic");
        assert_eq!(VersionStrategy::Sequential.to_string(), "sequential");
        assert_eq!(VersionStrategy::Timestamp.to_string(), "timestamp");
    }

    #[test]
    fn test_temporal_config_creation() {
        let config = TemporalConfig {
            valid_from: Utc::now(),
            valid_to: None,
            version: "1.0.0".to_string(),
            version_strategy: VersionStrategy::Semantic,
        };
        
        assert_eq!(config.version, "1.0.0");
        assert!(matches!(config.version_strategy, VersionStrategy::Semantic));
        assert!(config.valid_to.is_none());
    }

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_temporal_reification() {
        // Integration tests would go here
    }
}