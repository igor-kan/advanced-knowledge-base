//! Core types and data structures for the Neo4j reified engine
//!
//! This module defines the fundamental data types used throughout the Neo4j reified engine,
//! extending Neo4j's native graph model with advanced reification-specific types.

use crate::{Neo4jReifiedError, Result};
use neo4rs::{Node, Relation, BoltType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Unique identifier for entities in the reified graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub i64);

impl EntityId {
    /// Create a new entity ID from Neo4j internal ID
    pub fn new(id: i64) -> Self {
        Self(id)
    }
    
    /// Get the inner ID value
    pub fn value(&self) -> i64 {
        self.0
    }
    
    /// Create a temporary ID for new entities (will be replaced by Neo4j)
    pub fn temporary() -> Self {
        Self(-1)
    }
    
    /// Check if this is a temporary ID
    pub fn is_temporary(&self) -> bool {
        self.0 < 0
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
    
    /// Parse from string
    pub fn from_string(s: &str) -> Result<Self> {
        let id = s.parse::<i64>()
            .map_err(|e| Neo4jReifiedError::ValidationError {
                entity: "entity_id".to_string(),
                constraint: format!("Invalid ID format: {}", e),
            })?;
        Ok(Self(id))
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for EntityId {
    fn from(id: i64) -> Self {
        Self(id)
    }
}

impl From<EntityId> for i64 {
    fn from(id: EntityId) -> Self {
        id.0
    }
}

/// Property value types supported in Neo4j
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// String value
    String(String),
    /// List of values
    List(Vec<PropertyValue>),
    /// Map of key-value pairs
    Map(HashMap<String, PropertyValue>),
    /// Date value
    Date(DateTime<Utc>),
    /// Duration in milliseconds
    Duration(i64),
    /// Point in 2D space
    Point2D { x: f64, y: f64, srid: i32 },
    /// Point in 3D space
    Point3D { x: f64, y: f64, z: f64, srid: i32 },
}

impl PropertyValue {
    /// Check if the value is null
    pub fn is_null(&self) -> bool {
        matches!(self, PropertyValue::Null)
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        match self {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => s.clone(),
            PropertyValue::List(_) => "[list]".to_string(),
            PropertyValue::Map(_) => "{map}".to_string(),
            PropertyValue::Date(d) => d.to_rfc3339(),
            PropertyValue::Duration(d) => format!("{}ms", d),
            PropertyValue::Point2D { x, y, srid } => format!("Point2D({}, {}, srid={})", x, y, srid),
            PropertyValue::Point3D { x, y, z, srid } => format!("Point3D({}, {}, {}, srid={})", x, y, z, srid),
        }
    }
    
    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            PropertyValue::Null => "null",
            PropertyValue::Bool(_) => "boolean",
            PropertyValue::Int(_) => "integer",
            PropertyValue::Float(_) => "float",
            PropertyValue::String(_) => "string",
            PropertyValue::List(_) => "list",
            PropertyValue::Map(_) => "map",
            PropertyValue::Date(_) => "date",
            PropertyValue::Duration(_) => "duration",
            PropertyValue::Point2D { .. } => "point2d",
            PropertyValue::Point3D { .. } => "point3d",
        }
    }
    
    /// Convert to Neo4j BoltType
    pub fn to_bolt_type(&self) -> BoltType {
        match self {
            PropertyValue::Null => BoltType::Null,
            PropertyValue::Bool(b) => BoltType::Boolean(*b),
            PropertyValue::Int(i) => BoltType::Integer(*i),
            PropertyValue::Float(f) => BoltType::Float(*f),
            PropertyValue::String(s) => BoltType::String(s.clone()),
            PropertyValue::List(list) => {
                let bolt_list: Vec<BoltType> = list.iter().map(|v| v.to_bolt_type()).collect();
                BoltType::List(bolt_list)
            }
            PropertyValue::Map(map) => {
                let bolt_map: HashMap<String, BoltType> = map.iter()
                    .map(|(k, v)| (k.clone(), v.to_bolt_type()))
                    .collect();
                BoltType::Map(bolt_map)
            }
            PropertyValue::Date(dt) => {
                // Convert to Neo4j Date type (days since epoch)
                let days_since_epoch = dt.timestamp() / 86400;
                BoltType::Date(neo4rs::types::Date::new(days_since_epoch))
            }
            PropertyValue::Duration(ms) => {
                BoltType::Duration(neo4rs::types::Duration::new(0, 0, 0, *ms * 1_000_000)) // Convert ms to nanoseconds
            }
            PropertyValue::Point2D { x, y, srid } => {
                BoltType::Point2D(neo4rs::types::Point2D::new(*srid, *x, *y))
            }
            PropertyValue::Point3D { x, y, z, srid } => {
                BoltType::Point3D(neo4rs::types::Point3D::new(*srid, *x, *y, *z))
            }
        }
    }
    
    /// Create from Neo4j BoltType
    pub fn from_bolt_type(bolt_type: &BoltType) -> Self {
        match bolt_type {
            BoltType::Null => PropertyValue::Null,
            BoltType::Boolean(b) => PropertyValue::Bool(*b),
            BoltType::Integer(i) => PropertyValue::Int(*i),
            BoltType::Float(f) => PropertyValue::Float(*f),
            BoltType::String(s) => PropertyValue::String(s.clone()),
            BoltType::List(list) => {
                let prop_list: Vec<PropertyValue> = list.iter().map(PropertyValue::from_bolt_type).collect();
                PropertyValue::List(prop_list)
            }
            BoltType::Map(map) => {
                let prop_map: HashMap<String, PropertyValue> = map.iter()
                    .map(|(k, v)| (k.clone(), PropertyValue::from_bolt_type(v)))
                    .collect();
                PropertyValue::Map(prop_map)
            }
            BoltType::Date(date) => {
                let timestamp = date.days() * 86400;
                let dt = DateTime::from_timestamp(timestamp, 0).unwrap_or_else(|| Utc::now());
                PropertyValue::Date(dt)
            }
            BoltType::Duration(duration) => {
                let ms = duration.nanoseconds() / 1_000_000;
                PropertyValue::Duration(ms)
            }
            BoltType::Point2D(point) => {
                PropertyValue::Point2D {
                    x: point.x(),
                    y: point.y(),
                    srid: point.srid(),
                }
            }
            BoltType::Point3D(point) => {
                PropertyValue::Point3D {
                    x: point.x(),
                    y: point.y(),
                    z: point.z(),
                    srid: point.srid(),
                }
            }
            _ => PropertyValue::Null, // Fallback for unsupported types
        }
    }
}

/// Convenient type alias for property maps
pub type PropertyMap = HashMap<String, PropertyValue>;

/// Node in the reified graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedNode {
    /// Neo4j internal ID (may be temporary for new nodes)
    pub id: EntityId,
    /// Node labels
    pub labels: Vec<String>,
    /// Node properties
    pub properties: PropertyMap,
    /// Metadata for reification tracking
    pub metadata: NodeMetadata,
}

impl ReifiedNode {
    /// Create a new node with labels and properties
    pub fn new(labels: Vec<String>, properties: PropertyMap) -> Self {
        Self {
            id: EntityId::temporary(),
            labels,
            properties,
            metadata: NodeMetadata::new(),
        }
    }
    
    /// Create a node with specific ID (from existing Neo4j node)
    pub fn with_id(id: EntityId, labels: Vec<String>, properties: PropertyMap) -> Self {
        Self {
            id,
            labels,
            properties,
            metadata: NodeMetadata::new(),
        }
    }
    
    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.properties.get(key)
    }
    
    /// Set a property value
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
        self.metadata.touch();
    }
    
    /// Check if node has a specific label
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.contains(&label.to_string())
    }
    
    /// Add a label to the node
    pub fn add_label(&mut self, label: impl Into<String>) {
        let label = label.into();
        if !self.labels.contains(&label) {
            self.labels.push(label);
            self.metadata.touch();
        }
    }
    
    /// Convert from Neo4j Node
    pub fn from_neo4j_node(node: &Node) -> Self {
        let mut properties = PropertyMap::new();
        for (key, value) in node.properties() {
            properties.insert(key.clone(), PropertyValue::from_bolt_type(value));
        }
        
        Self::with_id(
            EntityId::new(node.id()),
            node.labels().to_vec(),
            properties,
        )
    }
    
    /// Get primary label (first label)
    pub fn primary_label(&self) -> Option<&String> {
        self.labels.first()
    }
}

/// Relationship in the reified graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedRelationship {
    /// Neo4j internal ID (may be temporary for new relationships)
    pub id: EntityId,
    /// Source node ID
    pub start_node_id: EntityId,
    /// Target node ID
    pub end_node_id: EntityId,
    /// Relationship type
    pub rel_type: String,
    /// Relationship properties
    pub properties: PropertyMap,
    /// Metadata for reification tracking
    pub metadata: RelationshipMetadata,
}

impl ReifiedRelationship {
    /// Create a new relationship
    pub fn new(
        start_node_id: EntityId,
        end_node_id: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id: EntityId::temporary(),
            start_node_id,
            end_node_id,
            rel_type: rel_type.into(),
            properties,
            metadata: RelationshipMetadata::new(),
        }
    }
    
    /// Create a relationship with specific ID (from existing Neo4j relationship)
    pub fn with_id(
        id: EntityId,
        start_node_id: EntityId,
        end_node_id: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id,
            start_node_id,
            end_node_id,
            rel_type: rel_type.into(),
            properties,
            metadata: RelationshipMetadata::new(),
        }
    }
    
    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.properties.get(key)
    }
    
    /// Set a property value
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
        self.metadata.touch();
    }
    
    /// Check if this relationship is reified
    pub fn is_reified(&self) -> bool {
        self.metadata.is_reified
    }
    
    /// Mark as reified
    pub fn mark_as_reified(&mut self, reified_node_id: EntityId) {
        self.metadata.is_reified = true;
        self.metadata.reified_node_id = Some(reified_node_id);
        self.metadata.touch();
    }
    
    /// Convert from Neo4j Relation
    pub fn from_neo4j_relation(relation: &Relation) -> Self {
        let mut properties = PropertyMap::new();
        for (key, value) in relation.properties() {
            properties.insert(key.clone(), PropertyValue::from_bolt_type(value));
        }
        
        Self::with_id(
            EntityId::new(relation.id()),
            EntityId::new(relation.start_node_id()),
            EntityId::new(relation.end_node_id()),
            relation.typ().to_string(),
            properties,
        )
    }
}

/// Complete reified relationship structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReificationStructure {
    /// The node representing the reified relationship
    pub reified_node: ReifiedNode,
    /// Original relationship information
    pub original_relationship: OriginalRelationshipInfo,
    /// FROM connection (original source -> reified node)
    pub from_connection: ReifiedRelationship,
    /// TO connection (reified node -> original target)
    pub to_connection: ReifiedRelationship,
    /// Reification level (0 = base relationship, 1+ = meta-relationship)
    pub reification_level: usize,
}

/// Information about the original relationship before reification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalRelationshipInfo {
    /// Original source node ID
    pub original_start_node_id: EntityId,
    /// Original target node ID
    pub original_end_node_id: EntityId,
    /// Original relationship type
    pub original_type: String,
    /// Original relationship properties (preserved in reified node)
    pub original_properties: PropertyMap,
    /// When the reification occurred
    pub reified_at: DateTime<Utc>,
    /// Who/what initiated the reification
    pub reified_by: Option<String>,
    /// Reification reason/context
    pub reification_context: Option<String>,
}

/// Node metadata for tracking and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub updated_at: DateTime<Utc>,
    /// Version for optimistic locking
    pub version: u64,
    /// Access count for optimization
    pub access_count: u64,
    /// Custom metadata tags
    pub tags: Vec<String>,
    /// Whether this node is a reified relationship
    pub is_reified_relationship: bool,
    /// Original relationship type if this is a reified relationship
    pub original_relationship_type: Option<String>,
}

impl NodeMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            access_count: 0,
            tags: Vec::new(),
            is_reified_relationship: false,
            original_relationship_type: None,
        }
    }
    
    /// Update the modification timestamp and increment version
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
        self.version += 1;
    }
    
    /// Record access
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
    
    /// Mark as reified relationship
    pub fn mark_as_reified_relationship(&mut self, original_type: String) {
        self.is_reified_relationship = true;
        self.original_relationship_type = Some(original_type);
        self.touch();
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Relationship metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last modification timestamp
    pub updated_at: DateTime<Utc>,
    /// Version for optimistic locking
    pub version: u64,
    /// Whether this relationship has been reified
    pub is_reified: bool,
    /// Reified node ID if reified
    pub reified_node_id: Option<EntityId>,
    /// Access count
    pub access_count: u64,
    /// Weight/importance for algorithms
    pub weight: f64,
    /// Whether this is a connection in a reification pattern (FROM/TO)
    pub is_reification_connection: bool,
}

impl RelationshipMetadata {
    /// Create new relationship metadata
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            is_reified: false,
            reified_node_id: None,
            access_count: 0,
            weight: 1.0,
            is_reification_connection: false,
        }
    }
    
    /// Update timestamp and version
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
        self.version += 1;
    }
    
    /// Record access
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
    
    /// Set as reified
    pub fn set_reified(&mut self, reified_node_id: EntityId) {
        self.is_reified = true;
        self.reified_node_id = Some(reified_node_id);
        self.touch();
    }
    
    /// Mark as reification connection
    pub fn mark_as_reification_connection(&mut self) {
        self.is_reification_connection = true;
        self.touch();
    }
}

impl Default for RelationshipMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Query result from Cypher execution with reification support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypherResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows of data
    pub rows: Vec<Vec<PropertyValue>>,
    /// Execution statistics
    pub stats: QueryStats,
    /// Reification-specific metadata
    pub reification_info: ReificationInfo,
}

impl CypherResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: QueryStats::default(),
            reification_info: ReificationInfo::default(),
        }
    }
    
    /// Get number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    
    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    
    /// Get a specific row
    pub fn get_row(&self, index: usize) -> Option<&Vec<PropertyValue>> {
        self.rows.get(index)
    }
    
    /// Convert to JSON representation
    pub fn to_json(&self) -> serde_json::Value {
        let mut objects = Vec::new();
        
        for row in &self.rows {
            let mut obj = serde_json::Map::new();
            for (i, column) in self.columns.iter().enumerate() {
                if let Some(value) = row.get(i) {
                    obj.insert(column.clone(), serde_json::to_value(value).unwrap_or(serde_json::Value::Null));
                }
            }
            objects.push(serde_json::Value::Object(obj));
        }
        
        serde_json::json!({
            "data": objects,
            "stats": self.stats,
            "reification_info": self.reification_info
        })
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Number of nodes accessed
    pub nodes_created: u64,
    /// Number of nodes updated
    pub nodes_updated: u64,
    /// Number of nodes deleted
    pub nodes_deleted: u64,
    /// Number of relationships created
    pub relationships_created: u64,
    /// Number of relationships updated
    pub relationships_updated: u64,
    /// Number of relationships deleted
    pub relationships_deleted: u64,
    /// Number of properties set
    pub properties_set: u64,
    /// Number of labels added
    pub labels_added: u64,
    /// Number of labels removed
    pub labels_removed: u64,
    /// Whether query used an index
    pub index_used: bool,
    /// Query plan information
    pub query_plan: Option<String>,
}

/// Reification-specific information in query results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReificationInfo {
    /// Number of reified relationships involved
    pub reified_relationships_count: u64,
    /// Maximum reification depth encountered
    pub max_reification_depth: usize,
    /// Reification patterns used
    pub patterns_used: Vec<String>,
    /// Performance impact of reification
    pub reification_overhead_us: u64,
}

/// Graph path result from traversal queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Nodes in the path
    pub nodes: Vec<EntityId>,
    /// Relationships in the path
    pub relationships: Vec<EntityId>,
    /// Total path weight/cost
    pub weight: f64,
    /// Path metadata
    pub metadata: PathMetadata,
}

impl GraphPath {
    /// Create new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            relationships: Vec::new(),
            weight: 0.0,
            metadata: PathMetadata::default(),
        }
    }
    
    /// Add a hop to the path
    pub fn add_hop(&mut self, node: EntityId, relationship: Option<EntityId>, weight: f64) {
        self.nodes.push(node);
        if let Some(rel) = relationship {
            self.relationships.push(rel);
        }
        self.weight += weight;
    }
    
    /// Get path length (number of relationships)
    pub fn length(&self) -> usize {
        self.relationships.len()
    }
    
    /// Check if path is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    
    /// Get start node
    pub fn start_node(&self) -> Option<EntityId> {
        self.nodes.first().copied()
    }
    
    /// Get end node
    pub fn end_node(&self) -> Option<EntityId> {
        self.nodes.last().copied()
    }
}

impl Default for GraphPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Path metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathMetadata {
    /// Algorithm used to find the path
    pub algorithm: String,
    /// Computation time in microseconds
    pub computation_time_us: u64,
    /// Number of nodes explored
    pub nodes_explored: usize,
    /// Whether path is optimal
    pub is_optimal: bool,
    /// Number of reified relationships in path
    pub reified_count: usize,
    /// Reification depth levels encountered
    pub reification_depths: Vec<usize>,
}

/// Engine statistics for monitoring
#[derive(Debug, Default)]
pub struct EngineStats {
    /// Total nodes created
    pub nodes_created: std::sync::atomic::AtomicU64,
    /// Total relationships created
    pub relationships_created: std::sync::atomic::AtomicU64,
    /// Total reifications performed
    pub reifications_performed: std::sync::atomic::AtomicU64,
    /// Total queries executed
    pub queries_executed: std::sync::atomic::AtomicU64,
    /// Total query time in microseconds
    pub total_query_time_us: std::sync::atomic::AtomicU64,
    /// Total connection pool hits
    pub pool_hits: std::sync::atomic::AtomicU64,
    /// Total connection pool misses
    pub pool_misses: std::sync::atomic::AtomicU64,
    /// Total batch operations
    pub batch_operations: std::sync::atomic::AtomicU64,
}

impl EngineStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Increment nodes created
    pub fn inc_nodes_created(&self) {
        self.nodes_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment relationships created
    pub fn inc_relationships_created(&self) {
        self.relationships_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment reifications performed
    pub fn inc_reifications_performed(&self) {
        self.reifications_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment queries executed
    pub fn inc_queries_executed(&self) {
        self.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Add query time
    pub fn add_query_time(&self, time_us: u64) {
        self.total_query_time_us.fetch_add(time_us, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment pool hits
    pub fn inc_pool_hits(&self) {
        self.pool_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment pool misses
    pub fn inc_pool_misses(&self) {
        self.pool_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment batch operations
    pub fn inc_batch_operations(&self) {
        self.batch_operations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Get average query time
    pub fn average_query_time_us(&self) -> f64 {
        let total_time = self.total_query_time_us.load(std::sync::atomic::Ordering::Relaxed);
        let total_queries = self.queries_executed.load(std::sync::atomic::Ordering::Relaxed);
        
        if total_queries == 0 {
            0.0
        } else {
            total_time as f64 / total_queries as f64
        }
    }
    
    /// Get pool hit rate
    pub fn pool_hit_rate(&self) -> f64 {
        let hits = self.pool_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.pool_misses.load(std::sync::atomic::Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

// Conversion implementations for PropertyValue
impl From<bool> for PropertyValue {
    fn from(value: bool) -> Self {
        PropertyValue::Bool(value)
    }
}

impl From<i32> for PropertyValue {
    fn from(value: i32) -> Self {
        PropertyValue::Int(value as i64)
    }
}

impl From<i64> for PropertyValue {
    fn from(value: i64) -> Self {
        PropertyValue::Int(value)
    }
}

impl From<f32> for PropertyValue {
    fn from(value: f32) -> Self {
        PropertyValue::Float(value as f64)
    }
}

impl From<f64> for PropertyValue {
    fn from(value: f64) -> Self {
        PropertyValue::Float(value)
    }
}

impl From<String> for PropertyValue {
    fn from(value: String) -> Self {
        PropertyValue::String(value)
    }
}

impl From<&str> for PropertyValue {
    fn from(value: &str) -> Self {
        PropertyValue::String(value.to_string())
    }
}

impl From<DateTime<Utc>> for PropertyValue {
    fn from(value: DateTime<Utc>) -> Self {
        PropertyValue::Date(value)
    }
}

impl From<SystemTime> for PropertyValue {
    fn from(value: SystemTime) -> Self {
        let dt = DateTime::<Utc>::from(value);
        PropertyValue::Date(dt)
    }
}

/// Macro for creating property maps easily
#[macro_export]
macro_rules! properties {
    () => {
        HashMap::new()
    };
    ($($key:expr => $value:expr),+ $(,)?) => {
        {
            let mut map = HashMap::new();
            $(
                map.insert($key.to_string(), $value.into());
            )+
            map
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_id_creation() {
        let id1 = EntityId::new(123);
        let id2 = EntityId::new(456);
        assert_ne!(id1, id2);
        assert_eq!(id1.value(), 123);
        
        let temp_id = EntityId::temporary();
        assert!(temp_id.is_temporary());
        
        let id_from_str = EntityId::from_string("789").unwrap();
        assert_eq!(id_from_str.value(), 789);
    }
    
    #[test]
    fn test_property_value_conversions() {
        let prop: PropertyValue = true.into();
        assert!(matches!(prop, PropertyValue::Bool(true)));
        
        let prop: PropertyValue = 42i64.into();
        assert!(matches!(prop, PropertyValue::Int(42)));
        
        let prop: PropertyValue = "hello".into();
        assert!(matches!(prop, PropertyValue::String(ref s) if s == "hello"));
        
        let now = Utc::now();
        let prop: PropertyValue = now.into();
        assert!(matches!(prop, PropertyValue::Date(dt) if dt == now));
    }
    
    #[test]
    fn test_reified_node_creation() {
        let props = properties!("name" => "Alice", "age" => 30);
        let node = ReifiedNode::new(vec!["Person".to_string()], props);
        
        assert!(node.has_label("Person"));
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("Alice".to_string())));
        assert_eq!(node.get_property("age"), Some(&PropertyValue::Int(30)));
        assert_eq!(node.primary_label(), Some(&"Person".to_string()));
    }
    
    #[test]
    fn test_reified_relationship_creation() {
        let from = EntityId::new(1);
        let to = EntityId::new(2);
        let props = properties!("since" => "2020-01-01");
        
        let rel = ReifiedRelationship::new(from, to, "KNOWS", props);
        assert_eq!(rel.start_node_id, from);
        assert_eq!(rel.end_node_id, to);
        assert_eq!(rel.rel_type, "KNOWS");
        assert!(!rel.is_reified());
        
        let mut rel = rel;
        rel.mark_as_reified(EntityId::new(3));
        assert!(rel.is_reified());
        assert_eq!(rel.metadata.reified_node_id, Some(EntityId::new(3)));
    }
    
    #[test]
    fn test_cypher_result() {
        let mut result = CypherResult::empty();
        result.columns = vec!["name".to_string(), "age".to_string()];
        result.rows = vec![
            vec![PropertyValue::String("Alice".to_string()), PropertyValue::Int(30)],
            vec![PropertyValue::String("Bob".to_string()), PropertyValue::Int(25)],
        ];
        
        assert_eq!(result.len(), 2);
        assert!(!result.is_empty());
        
        let json = result.to_json();
        assert!(json["data"].is_array());
        assert!(json["stats"].is_object());
    }
    
    #[test]
    fn test_graph_path() {
        let mut path = GraphPath::new();
        let node1 = EntityId::new(1);
        let node2 = EntityId::new(2);
        let rel1 = EntityId::new(10);
        
        path.add_hop(node1, None, 0.0);
        path.add_hop(node2, Some(rel1), 1.5);
        
        assert_eq!(path.length(), 1);
        assert_eq!(path.weight, 1.5);
        assert_eq!(path.start_node(), Some(node1));
        assert_eq!(path.end_node(), Some(node2));
        assert!(!path.is_empty());
    }
    
    #[test]
    fn test_engine_stats() {
        let stats = EngineStats::new();
        
        stats.inc_nodes_created();
        stats.inc_relationships_created();
        stats.inc_queries_executed();
        stats.add_query_time(1000);
        stats.inc_pool_hits();
        
        assert_eq!(stats.nodes_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.relationships_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.queries_executed.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.average_query_time_us(), 1000.0);
        assert_eq!(stats.pool_hit_rate(), 1.0);
    }
    
    #[test] 
    fn test_metadata_operations() {
        let mut node_metadata = NodeMetadata::new();
        let initial_version = node_metadata.version;
        
        node_metadata.touch();
        assert!(node_metadata.version > initial_version);
        
        node_metadata.record_access();
        assert_eq!(node_metadata.access_count, 1);
        
        node_metadata.mark_as_reified_relationship("WORKS_FOR".to_string());
        assert!(node_metadata.is_reified_relationship);
        assert_eq!(node_metadata.original_relationship_type, Some("WORKS_FOR".to_string()));
        
        let mut rel_metadata = RelationshipMetadata::new();
        assert!(!rel_metadata.is_reified);
        assert!(!rel_metadata.is_reification_connection);
        
        rel_metadata.set_reified(EntityId::new(123));
        assert!(rel_metadata.is_reified);
        assert_eq!(rel_metadata.reified_node_id, Some(EntityId::new(123)));
        
        rel_metadata.mark_as_reification_connection();
        assert!(rel_metadata.is_reification_connection);
    }
    
    #[test]
    fn test_property_macros() {
        let props = properties!(
            "name" => "Alice",
            "age" => 30,
            "active" => true,
            "created" => Utc::now()
        );
        
        assert_eq!(props.len(), 4);
        assert!(props.contains_key("name"));
        assert!(props.contains_key("age"));
        assert!(props.contains_key("active"));
        assert!(props.contains_key("created"));
    }
    
    #[test]
    fn test_property_value_types() {
        let point2d = PropertyValue::Point2D { x: 1.0, y: 2.0, srid: 4326 };
        assert_eq!(point2d.type_name(), "point2d");
        
        let point3d = PropertyValue::Point3D { x: 1.0, y: 2.0, z: 3.0, srid: 4326 };
        assert_eq!(point3d.type_name(), "point3d");
        
        let duration = PropertyValue::Duration(5000);
        assert_eq!(duration.type_name(), "duration");
        
        let list = PropertyValue::List(vec![PropertyValue::Int(1), PropertyValue::String("test".to_string())]);
        assert_eq!(list.type_name(), "list");
    }
}