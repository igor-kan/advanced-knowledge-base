//! Core types and data structures for the IndraDB reified engine
//!
//! This module defines the fundamental data types used throughout the IndraDB reified engine,
//! extending IndraDB's property graph model with reification-specific types.

use crate::{IndraReifiedError, Result};
use indradb::{Vertex, Edge, Identifier, Type, VertexProperties, EdgeProperties};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
use uuid::Uuid;

/// Unique identifier for entities in the reified graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntityId(pub Uuid);

impl EntityId {
    /// Create a new random entity ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create an entity ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the inner UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
    
    /// Convert to string representation
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
    
    /// Parse from string
    pub fn from_string(s: &str) -> Result<Self> {
        let uuid = Uuid::parse_str(s)
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "entity_id".to_string(),
                constraint: format!("Invalid UUID format: {}", e),
            })?;
        Ok(Self(uuid))
    }
    
    /// Convert to IndraDB Identifier
    pub fn to_identifier(&self) -> Identifier {
        Identifier::new(self.0)
    }
}

impl Default for EntityId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EntityId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for EntityId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<EntityId> for Uuid {
    fn from(id: EntityId) -> Self {
        id.0
    }
}

impl From<Identifier> for EntityId {
    fn from(id: Identifier) -> Self {
        Self(id.into())
    }
}

impl From<EntityId> for Identifier {
    fn from(id: EntityId) -> Self {
        Identifier::new(id.0)
    }
}

/// Property value types supported in the reified graph
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
    /// Array of values
    Array(Vec<PropertyValue>),
    /// Nested object
    Object(HashMap<String, PropertyValue>),
    /// Binary data
    Binary(Vec<u8>),
    /// Timestamp
    Timestamp(SystemTime),
    /// UUID value
    Uuid(Uuid),
    /// JSON value for complex structures
    Json(serde_json::Value),
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
            PropertyValue::Array(_) => "[array]".to_string(),
            PropertyValue::Object(_) => "{object}".to_string(),
            PropertyValue::Binary(data) => format!("[binary:{}]", data.len()),
            PropertyValue::Timestamp(ts) => format!("{:?}", ts),
            PropertyValue::Uuid(uuid) => uuid.to_string(),
            PropertyValue::Json(json) => json.to_string(),
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
            PropertyValue::Array(_) => "array",
            PropertyValue::Object(_) => "object",
            PropertyValue::Binary(_) => "binary",
            PropertyValue::Timestamp(_) => "timestamp",
            PropertyValue::Uuid(_) => "uuid",
            PropertyValue::Json(_) => "json",
        }
    }
    
    /// Convert to IndraDB JSON value
    pub fn to_indra_value(&self) -> indradb::Json {
        match self {
            PropertyValue::Null => indradb::Json::Null,
            PropertyValue::Bool(b) => indradb::Json::Bool(*b),
            PropertyValue::Int(i) => indradb::Json::Number(serde_json::Number::from(*i)),
            PropertyValue::Float(f) => {
                if let Some(num) = serde_json::Number::from_f64(*f) {
                    indradb::Json::Number(num)
                } else {
                    indradb::Json::Null
                }
            }
            PropertyValue::String(s) => indradb::Json::String(s.clone()),
            PropertyValue::Array(arr) => {
                let indra_array: Vec<indradb::Json> = arr.iter().map(|v| v.to_indra_value()).collect();
                indradb::Json::Array(indra_array)
            }
            PropertyValue::Object(obj) => {
                let mut indra_obj = serde_json::Map::new();
                for (k, v) in obj {
                    indra_obj.insert(k.clone(), v.to_indra_value());
                }
                indradb::Json::Object(indra_obj)
            }
            PropertyValue::Binary(data) => {
                // Encode binary data as base64 string
                indradb::Json::String(base64::encode(data))
            }
            PropertyValue::Timestamp(ts) => {
                let millis = ts.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                indradb::Json::Number(serde_json::Number::from(millis))
            }
            PropertyValue::Uuid(uuid) => indradb::Json::String(uuid.to_string()),
            PropertyValue::Json(json) => json.clone(),
        }
    }
    
    /// Create from IndraDB JSON value
    pub fn from_indra_value(value: &indradb::Json) -> Self {
        match value {
            indradb::Json::Null => PropertyValue::Null,
            indradb::Json::Bool(b) => PropertyValue::Bool(*b),
            indradb::Json::Number(n) => {
                if let Some(i) = n.as_i64() {
                    PropertyValue::Int(i)
                } else if let Some(f) = n.as_f64() {
                    PropertyValue::Float(f)
                } else {
                    PropertyValue::Null
                }
            }
            indradb::Json::String(s) => PropertyValue::String(s.clone()),
            indradb::Json::Array(arr) => {
                let prop_array: Vec<PropertyValue> = arr.iter().map(PropertyValue::from_indra_value).collect();
                PropertyValue::Array(prop_array)
            }
            indradb::Json::Object(obj) => {
                let mut prop_obj = HashMap::new();
                for (k, v) in obj {
                    prop_obj.insert(k.clone(), PropertyValue::from_indra_value(v));
                }
                PropertyValue::Object(prop_obj)
            }
        }
    }
}

/// Convenient type alias for property maps
pub type PropertyMap = HashMap<String, PropertyValue>;

/// Node in the reified graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedNode {
    /// Unique identifier
    pub id: EntityId,
    /// Node type/label
    pub node_type: String,
    /// Node properties
    pub properties: PropertyMap,
    /// Metadata
    pub metadata: NodeMetadata,
}

impl ReifiedNode {
    /// Create a new node
    pub fn new(node_type: impl Into<String>, properties: PropertyMap) -> Self {
        Self {
            id: EntityId::new(),
            node_type: node_type.into(),
            properties,
            metadata: NodeMetadata::new(),
        }
    }
    
    /// Create a node with specific ID
    pub fn with_id(id: EntityId, node_type: impl Into<String>, properties: PropertyMap) -> Self {
        Self {
            id,
            node_type: node_type.into(),
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
    
    /// Convert to IndraDB Vertex
    pub fn to_vertex(&self) -> Vertex {
        Vertex::new(self.id.to_identifier(), Type::new(self.node_type.clone()).unwrap())
    }
    
    /// Convert to IndraDB VertexProperties
    pub fn to_vertex_properties(&self) -> Vec<VertexProperties> {
        let mut props = Vec::new();
        
        for (key, value) in &self.properties {
            if let Ok(prop_type) = Type::new(key.clone()) {
                props.push(VertexProperties {
                    vertex: self.to_vertex(),
                    name: prop_type,
                    value: value.to_indra_value(),
                });
            }
        }
        
        props
    }
}

/// Relationship/Edge in the reified graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedEdge {
    /// Unique identifier
    pub id: EntityId,
    /// Source node ID
    pub from: EntityId,
    /// Target node ID
    pub to: EntityId,
    /// Edge type
    pub edge_type: String,
    /// Edge properties
    pub properties: PropertyMap,
    /// Metadata
    pub metadata: EdgeMetadata,
}

impl ReifiedEdge {
    /// Create a new edge
    pub fn new(
        from: EntityId,
        to: EntityId,
        edge_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id: EntityId::new(),
            from,
            to,
            edge_type: edge_type.into(),
            properties,
            metadata: EdgeMetadata::new(),
        }
    }
    
    /// Create an edge with specific ID
    pub fn with_id(
        id: EntityId,
        from: EntityId,
        to: EntityId,
        edge_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id,
            from,
            to,
            edge_type: edge_type.into(),
            properties,
            metadata: EdgeMetadata::new(),
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
    
    /// Check if this edge is reified
    pub fn is_reified(&self) -> bool {
        self.metadata.is_reified
    }
    
    /// Mark as reified
    pub fn mark_as_reified(&mut self, reified_node_id: EntityId) {
        self.metadata.is_reified = true;
        self.metadata.reified_node_id = Some(reified_node_id);
        self.metadata.touch();
    }
    
    /// Convert to IndraDB Edge
    pub fn to_edge(&self) -> Result<Edge> {
        let edge_type = Type::new(self.edge_type.clone())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: format!("Invalid edge type: {}", e),
            })?;
        
        Ok(Edge::new(
            self.from.to_identifier(),
            edge_type,
            self.to.to_identifier(),
        ))
    }
    
    /// Convert to IndraDB EdgeProperties
    pub fn to_edge_properties(&self) -> Result<Vec<EdgeProperties>> {
        let edge = self.to_edge()?;
        let mut props = Vec::new();
        
        for (key, value) in &self.properties {
            if let Ok(prop_type) = Type::new(key.clone()) {
                props.push(EdgeProperties {
                    edge: edge.clone(),
                    name: prop_type,
                    value: value.to_indra_value(),
                });
            }
        }
        
        Ok(props)
    }
}

/// Reified relationship - an edge that has been converted to a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedRelationship {
    /// The node representing the reified relationship
    pub node: ReifiedNode,
    /// Original edge information
    pub original_edge: OriginalEdgeInfo,
    /// Connection from the original source to the reified node
    pub from_connection: EntityId,
    /// Connection from the reified node to the original target
    pub to_connection: EntityId,
}

/// Information about the original edge before reification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalEdgeInfo {
    /// Original source node
    pub original_from: EntityId,
    /// Original target node
    pub original_to: EntityId,
    /// Original edge type
    pub original_type: String,
    /// Original edge properties
    pub original_properties: PropertyMap,
    /// When the reification occurred
    pub reified_at: SystemTime,
}

/// Node metadata for tracking and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Version for optimistic locking
    pub version: u64,
    /// Access count for optimization
    pub access_count: u64,
    /// Custom metadata tags
    pub tags: Vec<String>,
}

impl NodeMetadata {
    /// Create new metadata
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            access_count: 0,
            tags: Vec::new(),
        }
    }
    
    /// Update the modification timestamp and increment version
    pub fn touch(&mut self) {
        self.updated_at = SystemTime::now();
        self.version += 1;
    }
    
    /// Record access
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
    /// Version for optimistic locking
    pub version: u64,
    /// Whether this edge has been reified
    pub is_reified: bool,
    /// Reified node ID if reified
    pub reified_node_id: Option<EntityId>,
    /// Access count
    pub access_count: u64,
    /// Weight/importance for algorithms
    pub weight: f64,
}

impl EdgeMetadata {
    /// Create new edge metadata
    pub fn new() -> Self {
        let now = SystemTime::now();
        Self {
            created_at: now,
            updated_at: now,
            version: 1,
            is_reified: false,
            reified_node_id: None,
            access_count: 0,
            weight: 1.0,
        }
    }
    
    /// Update timestamp and version
    pub fn touch(&mut self) {
        self.updated_at = SystemTime::now();
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
}

impl Default for EdgeMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Query result from property graph execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyGraphResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows of data
    pub rows: Vec<Vec<PropertyValue>>,
    /// Execution statistics
    pub stats: QueryStats,
}

impl PropertyGraphResult {
    /// Create empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            stats: QueryStats::default(),
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
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Number of vertices accessed
    pub vertices_accessed: u64,
    /// Number of edges traversed
    pub edges_traversed: u64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// Whether query was cached
    pub was_cached: bool,
    /// Number of reified relationships involved
    pub reified_relationships: u64,
    /// Number of transactions used
    pub transactions_used: u64,
}

/// Graph path result from traversal queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Vertices in the path
    pub vertices: Vec<EntityId>,
    /// Edges in the path
    pub edges: Vec<EntityId>,
    /// Total path weight/cost
    pub weight: f64,
    /// Path metadata
    pub metadata: PathMetadata,
}

impl GraphPath {
    /// Create new empty path
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            weight: 0.0,
            metadata: PathMetadata::default(),
        }
    }
    
    /// Add a hop to the path
    pub fn add_hop(&mut self, vertex: EntityId, edge: Option<EntityId>, weight: f64) {
        self.vertices.push(vertex);
        if let Some(e) = edge {
            self.edges.push(e);
        }
        self.weight += weight;
    }
    
    /// Get path length (number of edges)
    pub fn length(&self) -> usize {
        self.edges.len()
    }
    
    /// Check if path is empty
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
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
    /// Number of vertices explored
    pub vertices_explored: usize,
    /// Whether path is optimal
    pub is_optimal: bool,
    /// Number of reified relationships in path
    pub reified_count: usize,
}

/// Engine statistics for monitoring
#[derive(Debug, Default)]
pub struct EngineStats {
    /// Total vertices created
    pub vertices_created: std::sync::atomic::AtomicU64,
    /// Total edges created
    pub edges_created: std::sync::atomic::AtomicU64,
    /// Total reifications performed
    pub reifications_performed: std::sync::atomic::AtomicU64,
    /// Total queries executed
    pub queries_executed: std::sync::atomic::AtomicU64,
    /// Total query time in microseconds
    pub total_query_time_us: std::sync::atomic::AtomicU64,
    /// Total transactions committed
    pub transactions_committed: std::sync::atomic::AtomicU64,
    /// Total transactions rolled back
    pub transactions_rolled_back: std::sync::atomic::AtomicU64,
}

impl EngineStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Increment vertices created
    pub fn inc_vertices_created(&self) {
        self.vertices_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment edges created
    pub fn inc_edges_created(&self) {
        self.edges_created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    
    /// Increment transactions committed
    pub fn inc_transactions_committed(&self) {
        self.transactions_committed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Increment transactions rolled back
    pub fn inc_transactions_rolled_back(&self) {
        self.transactions_rolled_back.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    
    /// Get transaction success rate
    pub fn transaction_success_rate(&self) -> f64 {
        let committed = self.transactions_committed.load(std::sync::atomic::Ordering::Relaxed);
        let rolled_back = self.transactions_rolled_back.load(std::sync::atomic::Ordering::Relaxed);
        let total = committed + rolled_back;
        
        if total == 0 {
            0.0
        } else {
            committed as f64 / total as f64
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

impl From<Vec<u8>> for PropertyValue {
    fn from(value: Vec<u8>) -> Self {
        PropertyValue::Binary(value)
    }
}

impl From<SystemTime> for PropertyValue {
    fn from(value: SystemTime) -> Self {
        PropertyValue::Timestamp(value)
    }
}

impl From<Uuid> for PropertyValue {
    fn from(value: Uuid) -> Self {
        PropertyValue::Uuid(value)
    }
}

impl From<serde_json::Value> for PropertyValue {
    fn from(value: serde_json::Value) -> Self {
        PropertyValue::Json(value)
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
        let id1 = EntityId::new();
        let id2 = EntityId::new();
        assert_ne!(id1, id2);
        
        let uuid = Uuid::new_v4();
        let id3 = EntityId::from_uuid(uuid);
        assert_eq!(id3.as_uuid(), uuid);
        
        let id_str = id3.to_string();
        let id4 = EntityId::from_string(&id_str).unwrap();
        assert_eq!(id3, id4);
    }
    
    #[test]
    fn test_property_value_conversions() {
        let prop: PropertyValue = true.into();
        assert!(matches!(prop, PropertyValue::Bool(true)));
        
        let prop: PropertyValue = 42i64.into();
        assert!(matches!(prop, PropertyValue::Int(42)));
        
        let prop: PropertyValue = "hello".into();
        assert!(matches!(prop, PropertyValue::String(ref s) if s == "hello"));
        
        let json_val = serde_json::json!({"key": "value"});
        let prop: PropertyValue = json_val.clone().into();
        assert!(matches!(prop, PropertyValue::Json(ref j) if j == &json_val));
    }
    
    #[test]
    fn test_indra_value_conversion() {
        let prop = PropertyValue::String("test".to_string());
        let indra_val = prop.to_indra_value();
        assert!(matches!(indra_val, indradb::Json::String(ref s) if s == "test"));
        
        let converted_back = PropertyValue::from_indra_value(&indra_val);
        assert_eq!(prop, converted_back);
    }
    
    #[test]
    fn test_reified_node_creation() {
        let props = properties!("name" => "Alice", "age" => 30);
        let node = ReifiedNode::new("Person", props);
        
        assert_eq!(node.node_type, "Person");
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("Alice".to_string())));
        assert_eq!(node.get_property("age"), Some(&PropertyValue::Int(30)));
        
        let vertex = node.to_vertex();
        assert_eq!(vertex.t.as_str(), "Person");
    }
    
    #[test]
    fn test_reified_edge_creation() {
        let from = EntityId::new();
        let to = EntityId::new();
        let props = properties!("since" => "2020-01-01");
        
        let edge = ReifiedEdge::new(from, to, "KNOWS", props);
        assert_eq!(edge.from, from);
        assert_eq!(edge.to, to);
        assert_eq!(edge.edge_type, "KNOWS");
        assert!(!edge.is_reified());
        
        let indra_edge = edge.to_edge().unwrap();
        assert_eq!(indra_edge.outbound_id, from.to_identifier());
        assert_eq!(indra_edge.inbound_id, to.to_identifier());
    }
    
    #[test]
    fn test_property_graph_result() {
        let mut result = PropertyGraphResult::empty();
        result.columns = vec!["name".to_string(), "age".to_string()];
        result.rows = vec![
            vec![PropertyValue::String("Alice".to_string()), PropertyValue::Int(30)],
            vec![PropertyValue::String("Bob".to_string()), PropertyValue::Int(25)],
        ];
        
        assert_eq!(result.len(), 2);
        assert!(!result.is_empty());
        
        let first_row = result.get_row(0).unwrap();
        assert_eq!(first_row.len(), 2);
    }
    
    #[test]
    fn test_graph_path() {
        let mut path = GraphPath::new();
        let v1 = EntityId::new();
        let v2 = EntityId::new();
        let e1 = EntityId::new();
        
        path.add_hop(v1, None, 0.0);
        path.add_hop(v2, Some(e1), 1.5);
        
        assert_eq!(path.length(), 1);
        assert_eq!(path.weight, 1.5);
        assert!(!path.is_empty());
    }
    
    #[test]
    fn test_engine_stats() {
        let stats = EngineStats::new();
        
        stats.inc_vertices_created();
        stats.inc_edges_created();
        stats.inc_queries_executed();
        stats.add_query_time(1000);
        stats.inc_transactions_committed();
        
        assert_eq!(stats.vertices_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.edges_created.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.queries_executed.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(stats.average_query_time_us(), 1000.0);
        assert_eq!(stats.transaction_success_rate(), 1.0);
    }
    
    #[test]
    fn test_metadata_operations() {
        let mut metadata = NodeMetadata::new();
        let initial_version = metadata.version;
        
        metadata.touch();
        assert!(metadata.version > initial_version);
        
        metadata.record_access();
        assert_eq!(metadata.access_count, 1);
        
        let mut edge_metadata = EdgeMetadata::new();
        assert!(!edge_metadata.is_reified);
        
        let reified_id = EntityId::new();
        edge_metadata.set_reified(reified_id);
        assert!(edge_metadata.is_reified);
        assert_eq!(edge_metadata.reified_node_id, Some(reified_id));
    }
    
    #[test]
    fn test_property_macros() {
        let props = properties!(
            "name" => "Alice",
            "age" => 30,
            "active" => true
        );
        
        assert_eq!(props.len(), 3);
        assert!(props.contains_key("name"));
        assert!(props.contains_key("age"));
        assert!(props.contains_key("active"));
    }
}