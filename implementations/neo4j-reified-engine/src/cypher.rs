//! Cypher query extensions for reification operations
//!
//! This module provides extended Cypher functionality for working with reified relationships,
//! including custom procedures and query builders.

use crate::types::*;
use crate::{Neo4jReifiedError, Result, FROM_RELATIONSHIP, TO_RELATIONSHIP, REIFIED_EDGE_LABEL};

use neo4rs::{Graph, Query, BoltType};
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value as JsonValue;

/// Extended Cypher functionality for reification
pub struct CypherExtensions {
    graph: Arc<Graph>,
}

impl CypherExtensions {
    /// Create new Cypher extensions
    pub fn new(graph: Arc<Graph>) -> Self {
        Self { graph }
    }

    /// Execute a reification query using custom syntax
    pub async fn execute_reification_query(&self, query: &ReificationQuery) -> Result<CypherResult> {
        let cypher = self.build_reification_cypher(query)?;
        let neo4j_query = self.prepare_query(&cypher, &query.parameters)?;
        
        self.execute_prepared_query(neo4j_query).await
    }

    /// Find all reified relationships of a specific type
    pub async fn find_reified_by_type(&self, relationship_type: &str) -> Result<Vec<EntityId>> {
        let cypher = format!(
            "MATCH (n:{}) WHERE n.original_type = $rel_type RETURN id(n) as node_id",
            REIFIED_EDGE_LABEL
        );

        let query = Query::new(cypher).param("rel_type", relationship_type);
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: format!("find_reified_by_type({})", relationship_type),
                message: format!("Query execution failed: {}", e),
            })?;

        let mut entity_ids = Vec::new();
        while let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: format!("find_reified_by_type({})", relationship_type),
                message: format!("Failed to read result: {}", e),
            })? {
            let node_id: i64 = row.get("node_id")
                .map_err(|e| Neo4jReifiedError::CypherError {
                    query: format!("find_reified_by_type({})", relationship_type),
                    message: format!("Failed to get node_id: {}", e),
                })?;
            entity_ids.push(EntityId::new(node_id));
        }

        Ok(entity_ids)
    }

    /// Get the reification hierarchy for a node
    pub async fn get_reification_hierarchy(&self, node_id: EntityId) -> Result<Vec<GraphPath>> {
        let cypher = r#"
            MATCH path = (n)-[:FROM|TO*1..10]-(reified:ReifiedEdge)
            WHERE id(n) = $node_id
            RETURN path, length(path) as depth
            ORDER BY depth
        "#;

        let query = Query::new(cypher.to_string()).param("node_id", node_id.value());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "get_reification_hierarchy".to_string(),
                message: format!("Query execution failed: {}", e),
            })?;

        let mut paths = Vec::new();
        while let Some(_row) = result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "get_reification_hierarchy".to_string(),
                message: format!("Failed to read result: {}", e),
            })? {
            // Build GraphPath from Neo4j path - simplified for now
            let path = GraphPath::new();
            paths.push(path);
        }

        Ok(paths)
    }

    /// Batch reify multiple relationships
    pub async fn batch_reify(&self, requests: &[BatchReificationRequest]) -> Result<Vec<EntityId>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let mut cypher = String::from("UNWIND $requests as req\n");
        cypher.push_str("MATCH (from), (to) WHERE id(from) = req.from_id AND id(to) = req.to_id\n");
        cypher.push_str(&format!("CREATE (from)-[:{}]->(reified:{})-[:{}]->(to)\n", 
            FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP));
        cypher.push_str("SET reified.original_type = req.rel_type, reified += req.properties\n");
        cypher.push_str("RETURN id(reified) as reified_id");

        let request_maps: Vec<HashMap<String, BoltType>> = requests.iter().map(|req| {
            let mut map = HashMap::new();
            map.insert("from_id".to_string(), BoltType::Integer(req.from_node_id.value()));
            map.insert("to_id".to_string(), BoltType::Integer(req.to_node_id.value()));
            map.insert("rel_type".to_string(), BoltType::String(req.relationship_type.clone()));
            
            let props_map: HashMap<String, BoltType> = req.properties.iter()
                .map(|(k, v)| (k.clone(), v.to_bolt_type()))
                .collect();
            map.insert("properties".to_string(), BoltType::Map(props_map));
            
            map
        }).collect();

        let bolt_requests: Vec<BoltType> = request_maps.into_iter()
            .map(BoltType::Map)
            .collect();

        let query = Query::new(cypher).param("requests", BoltType::List(bolt_requests));
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "batch_reify".to_string(),
                message: format!("Batch reification failed: {}", e),
            })?;

        let mut reified_ids = Vec::new();
        while let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "batch_reify".to_string(),
                message: format!("Failed to read batch result: {}", e),
            })? {
            let reified_id: i64 = row.get("reified_id")
                .map_err(|e| Neo4jReifiedError::CypherError {
                    query: "batch_reify".to_string(),
                    message: format!("Failed to get reified_id: {}", e),
                })?;
            reified_ids.push(EntityId::new(reified_id));
        }

        Ok(reified_ids)
    }

    /// Unreify a relationship back to a simple edge
    pub async fn unreify(&self, reified_node_id: EntityId) -> Result<EntityId> {
        let cypher = r#"
            MATCH (from)-[:FROM]->(reified:ReifiedEdge)-[:TO]->(to)
            WHERE id(reified) = $reified_id
            WITH from, to, reified.original_type as rel_type, properties(reified) as props
            CREATE (from)-[r]->(to)
            SET r = props, type(r) = rel_type
            WITH r, reified
            MATCH (reified)-[:FROM|TO]-()
            DELETE reified
            RETURN id(r) as rel_id
        "#;

        let query = Query::new(cypher.to_string()).param("reified_id", reified_node_id.value());
        let mut result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "unreify".to_string(),
                message: format!("Unreification failed: {}", e),
            })?;

        if let Some(row) = result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "unreify".to_string(),
                message: format!("Failed to read unreify result: {}", e),
            })? {
            let rel_id: i64 = row.get("rel_id")
                .map_err(|e| Neo4jReifiedError::CypherError {
                    query: "unreify".to_string(),
                    message: format!("Failed to get rel_id: {}", e),
                })?;
            Ok(EntityId::new(rel_id))
        } else {
            Err(Neo4jReifiedError::CypherError {
                query: "unreify".to_string(),
                message: "No relationship created during unreification".to_string(),
            })
        }
    }

    /// Execute advanced pattern matching for reified relationships
    pub async fn match_reified_pattern(&self, pattern: &ReifiedPattern) -> Result<CypherResult> {
        let cypher = self.build_pattern_cypher(pattern)?;
        let query = Query::new(cypher);
        
        self.execute_prepared_query(query).await
    }

    // Private helper methods

    fn build_reification_cypher(&self, query: &ReificationQuery) -> Result<String> {
        match &query.operation {
            ReificationOperation::Reify { from_var, to_var, rel_type } => {
                Ok(format!(
                    "MATCH ({}), ({}) CREATE ({})-[:{}]->(reified:{})-[:{}]->({}) SET reified.original_type = '{}' RETURN reified",
                    from_var, to_var, from_var, FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP, to_var, rel_type
                ))
            }
            ReificationOperation::Unreify { reified_var } => {
                Ok(format!(
                    "MATCH (from)-[:{}]->({0}:{})-[:{}]->(to) WITH from, to, {0}.original_type as rel_type CREATE (from)-[r]->(to) SET type(r) = rel_type DELETE {0} RETURN r",
                    FROM_RELATIONSHIP, reified_var, REIFIED_EDGE_LABEL, TO_RELATIONSHIP
                ))
            }
            ReificationOperation::FindReified { rel_type } => {
                Ok(format!(
                    "MATCH (n:{}) WHERE n.original_type = '{}' RETURN n",
                    REIFIED_EDGE_LABEL, rel_type
                ))
            }
        }
    }

    fn build_pattern_cypher(&self, pattern: &ReifiedPattern) -> Result<String> {
        match pattern {
            ReifiedPattern::SimpleReified { rel_type } => {
                Ok(format!(
                    "MATCH (from)-[:{}]->(reified:{})-[:{}]->(to) WHERE reified.original_type = '{}' RETURN from, reified, to",
                    FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP, rel_type
                ))
            }
            ReifiedPattern::MetaReified { original_type, meta_type } => {
                Ok(format!(
                    "MATCH (from)-[:{}]->(reified1:{})-[:{}]->(to1)-[:{}]->(reified2:{})-[:{}]->(to2) WHERE reified1.original_type = '{}' AND reified2.original_type = '{}' RETURN from, reified1, to1, reified2, to2",
                    FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP,
                    FROM_RELATIONSHIP, REIFIED_EDGE_LABEL, TO_RELATIONSHIP,
                    original_type, meta_type
                ))
            }
            ReifiedPattern::Custom { cypher_template } => {
                Ok(cypher_template.clone())
            }
        }
    }

    fn prepare_query(&self, cypher: &str, parameters: &HashMap<String, JsonValue>) -> Result<Query> {
        let mut query = Query::new(cypher.to_string());
        
        for (key, value) in parameters {
            let bolt_value = self.json_to_bolt_type(value)?;
            query = query.param(key.clone(), bolt_value);
        }
        
        Ok(query)
    }

    fn json_to_bolt_type(&self, value: &JsonValue) -> Result<BoltType> {
        match value {
            JsonValue::Null => Ok(BoltType::Null),
            JsonValue::Bool(b) => Ok(BoltType::Boolean(*b)),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(BoltType::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(BoltType::Float(f))
                } else {
                    Err(Neo4jReifiedError::ValidationError {
                        entity: "parameter".to_string(),
                        constraint: "Invalid number format".to_string(),
                    })
                }
            }
            JsonValue::String(s) => Ok(BoltType::String(s.clone())),
            JsonValue::Array(arr) => {
                let bolt_array: Result<Vec<BoltType>> = arr.iter()
                    .map(|v| self.json_to_bolt_type(v))
                    .collect();
                Ok(BoltType::List(bolt_array?))
            }
            JsonValue::Object(obj) => {
                let mut bolt_map = HashMap::new();
                for (key, value) in obj {
                    bolt_map.insert(key.clone(), self.json_to_bolt_type(value)?);
                }
                Ok(BoltType::Map(bolt_map))
            }
        }
    }

    async fn execute_prepared_query(&self, query: Query) -> Result<CypherResult> {
        let start_time = std::time::Instant::now();
        let mut neo4j_result = self.graph.execute(query).await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "prepared_query".to_string(),
                message: format!("Query execution failed: {}", e),
            })?;

        let mut columns = Vec::new();
        let mut rows = Vec::new();

        if let Some(first_row) = neo4j_result.next().await
            .map_err(|e| Neo4jReifiedError::CypherError {
                query: "prepared_query".to_string(),
                message: format!("Failed to read result: {}", e),
            })? {
            columns = first_row.keys().map(|k| k.to_string()).collect();
            
            let mut row_values = Vec::new();
            for column in &columns {
                let value = first_row.get::<BoltType>(column)
                    .map_err(|e| Neo4jReifiedError::CypherError {
                        query: "prepared_query".to_string(),
                        message: format!("Failed to get column '{}': {}", column, e),
                    })?;
                row_values.push(PropertyValue::from_bolt_type(&value));
            }
            rows.push(row_values);

            while let Some(row) = neo4j_result.next().await
                .map_err(|e| Neo4jReifiedError::CypherError {
                    query: "prepared_query".to_string(),
                    message: format!("Failed to read result: {}", e),
                })? {
                let mut row_values = Vec::new();
                for column in &columns {
                    let value = row.get::<BoltType>(column)
                        .map_err(|e| Neo4jReifiedError::CypherError {
                            query: "prepared_query".to_string(),
                            message: format!("Failed to get column '{}': {}", column, e),
                        })?;
                    row_values.push(PropertyValue::from_bolt_type(&value));
                }
                rows.push(row_values);
            }
        }

        let execution_time = start_time.elapsed().as_micros() as u64;
        
        Ok(CypherResult {
            columns,
            rows,
            stats: QueryStats {
                execution_time_us: execution_time,
                ..Default::default()
            },
            reification_info: ReificationInfo::default(),
        })
    }
}

/// Query builder for reification operations
pub struct QueryBuilder {
    cypher: String,
    parameters: HashMap<String, JsonValue>,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            cypher: String::new(),
            parameters: HashMap::new(),
        }
    }

    /// Start a MATCH clause
    pub fn match_clause(mut self, pattern: &str) -> Self {
        if !self.cypher.is_empty() {
            self.cypher.push(' ');
        }
        self.cypher.push_str("MATCH ");
        self.cypher.push_str(pattern);
        self
    }

    /// Add a WHERE clause
    pub fn where_clause(mut self, condition: &str) -> Self {
        self.cypher.push_str(" WHERE ");
        self.cypher.push_str(condition);
        self
    }

    /// Add a CREATE clause for reification
    pub fn create_reified(mut self, from_var: &str, to_var: &str, rel_type: &str) -> Self {
        self.cypher.push_str(" CREATE (");
        self.cypher.push_str(from_var);
        self.cypher.push_str(")-[:");
        self.cypher.push_str(FROM_RELATIONSHIP);
        self.cypher.push_str("]->(reified:");
        self.cypher.push_str(REIFIED_EDGE_LABEL);
        self.cypher.push_str(")-[:");
        self.cypher.push_str(TO_RELATIONSHIP);
        self.cypher.push_str("]->(");
        self.cypher.push_str(to_var);
        self.cypher.push_str(") SET reified.original_type = '");
        self.cypher.push_str(rel_type);
        self.cypher.push('\'');
        self
    }

    /// Add a RETURN clause
    pub fn return_clause(mut self, items: &str) -> Self {
        self.cypher.push_str(" RETURN ");
        self.cypher.push_str(items);
        self
    }

    /// Add a parameter
    pub fn parameter(mut self, name: impl Into<String>, value: JsonValue) -> Self {
        self.parameters.insert(name.into(), value);
        self
    }

    /// Build the final query
    pub fn build(self) -> ReificationQuery {
        ReificationQuery {
            operation: ReificationOperation::Custom {
                cypher: self.cypher.clone(),
            },
            parameters: self.parameters,
            cypher: Some(self.cypher),
        }
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Reification query structure
pub struct ReificationQuery {
    pub operation: ReificationOperation,
    pub parameters: HashMap<String, JsonValue>,
    pub cypher: Option<String>,
}

/// Reification operations
pub enum ReificationOperation {
    Reify {
        from_var: String,
        to_var: String,
        rel_type: String,
    },
    Unreify {
        reified_var: String,
    },
    FindReified {
        rel_type: String,
    },
    Custom {
        cypher: String,
    },
}

/// Reified relationship patterns
pub enum ReifiedPattern {
    SimpleReified {
        rel_type: String,
    },
    MetaReified {
        original_type: String,
        meta_type: String,
    },
    Custom {
        cypher_template: String,
    },
}

/// Batch reification request
pub struct BatchReificationRequest {
    pub from_node_id: EntityId,
    pub to_node_id: EntityId,
    pub relationship_type: String,
    pub properties: PropertyMap,
}

impl BatchReificationRequest {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .match_clause("(a:Person)-[r:KNOWS]->(b:Person)")
            .where_clause("a.name = $name")
            .create_reified("a", "b", "KNOWS")
            .return_clause("reified")
            .parameter("name", json!("Alice"))
            .build();

        assert!(query.cypher.is_some());
        assert!(query.parameters.contains_key("name"));
    }

    #[test]
    fn test_batch_reification_request() {
        let from_id = EntityId::new(1);
        let to_id = EntityId::new(2);
        let properties = HashMap::new();

        let request = BatchReificationRequest::new(from_id, to_id, "WORKS_FOR", properties);
        
        assert_eq!(request.from_node_id, from_id);
        assert_eq!(request.to_node_id, to_id);
        assert_eq!(request.relationship_type, "WORKS_FOR");
    }

    #[tokio::test]
    #[ignore] // Requires Neo4j instance
    async fn test_cypher_extensions() {
        // Integration tests would go here
    }
}