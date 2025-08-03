//! Core types and data structures for the Kuzu reified engine

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use uuid::Uuid;

/// Unique identifier for nodes and relationships
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
    pub fn from_string(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
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

/// Node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier
    pub id: EntityId,
    /// Node type/label
    pub label: String,
    /// Node properties
    pub properties: PropertyMap,
    /// Metadata
    pub metadata: NodeMetadata,
}

impl Node {
    /// Create a new node
    pub fn new(label: impl Into<String>, properties: PropertyMap) -> Self {
        Self {
            id: EntityId::new(),
            label: label.into(),
            properties,
            metadata: NodeMetadata::new(),
        }
    }
    
    /// Create a node with specific ID
    pub fn with_id(id: EntityId, label: impl Into<String>, properties: PropertyMap) -> Self {
        Self {
            id,
            label: label.into(),
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
    
    /// Remove a property
    pub fn remove_property(&mut self, key: &str) -> Option<PropertyValue> {
        let result = self.properties.remove(key);
        if result.is_some() {
            self.metadata.touch();
        }
        result
    }
    
    /// Check if node has a specific property
    pub fn has_property(&self, key: &str) -> bool {
        self.properties.contains_key(key)
    }
}

/// Relationship/Edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Unique identifier
    pub id: EntityId,
    /// Source node ID
    pub from: EntityId,
    /// Target node ID
    pub to: EntityId,
    /// Relationship type
    pub rel_type: String,
    /// Relationship properties
    pub properties: PropertyMap,
    /// Metadata
    pub metadata: RelationshipMetadata,
}

impl Relationship {
    /// Create a new relationship
    pub fn new(
        from: EntityId,
        to: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id: EntityId::new(),
            from,
            to,
            rel_type: rel_type.into(),
            properties,
            metadata: RelationshipMetadata::new(),
        }
    }
    
    /// Create a relationship with specific ID
    pub fn with_id(
        id: EntityId,
        from: EntityId,
        to: EntityId,
        rel_type: impl Into<String>,
        properties: PropertyMap,
    ) -> Self {
        Self {
            id,
            from,
            to,
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
    
    /// Check if this is a reified relationship
    pub fn is_reified(&self) -> bool {
        self.metadata.is_reified
    }
    
    /// Mark as reified
    pub fn mark_as_reified(&mut self) {
        self.metadata.is_reified = true;
        self.metadata.touch();
    }
}

/// Reified relationship - a relationship that has been converted to a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReifiedRelationship {
    /// The node representing the reified relationship
    pub node: Node,
    /// Original relationship information
    pub original_relationship: OriginalRelationshipInfo,
    /// Connections from the original source
    pub from_connection: EntityId,
    /// Connections to the original target
    pub to_connection: EntityId,
}

/// Information about the original relationship before reification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalRelationshipInfo {
    /// Original source node
    pub original_from: EntityId,
    /// Original target node
    pub original_to: EntityId,
    /// Original relationship type
    pub original_type: String,
    /// When the reification occurred
    pub reified_at: SystemTime,
}

/// Property value types supported in the graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// Null value
    Null,
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<PropertyValue>),
    /// Object (nested properties)
    Object(HashMap<String, PropertyValue>),
    /// Binary data
    Binary(Vec<u8>),
    /// Timestamp
    Timestamp(SystemTime),
    /// UUID value
    Uuid(Uuid),
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
        }
    }
}

/// Convenient type alias for property maps
pub type PropertyMap = HashMap<String, PropertyValue>;

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
    
    /// Increment access count
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
    
    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        let tag = tag.into();
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
            self.touch();
        }
    }
    
    /// Remove a tag
    pub fn remove_tag(&mut self, tag: &str) -> bool {
        if let Some(pos) = self.tags.iter().position(|t| t == tag) {
            self.tags.remove(pos);
            self.touch();
            true
        } else {
            false
        }
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
    pub created_at: SystemTime,
    /// Last modification timestamp
    pub updated_at: SystemTime,
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
}

impl RelationshipMetadata {
    /// Create new relationship metadata
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

impl Default for RelationshipMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Query result from Cypher execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CypherResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows of data
    pub rows: Vec<Vec<PropertyValue>>,
    /// Execution statistics
    pub stats: QueryStats,
}

impl CypherResult {
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
        
        serde_json::Value::Array(objects)
    }
}

/// Query execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Number of nodes accessed
    pub nodes_accessed: u64,
    /// Number of relationships traversed
    pub relationships_traversed: u64,
    /// Memory used in bytes
    pub memory_used: u64,
    /// Whether query was cached
    pub was_cached: bool,
    /// Number of reified relationships involved
    pub reified_relationships: u64,
}

impl QueryStats {
    /// Create new stats with execution time
    pub fn with_execution_time(execution_time_us: u64) -> Self {
        Self {
            execution_time_us,
            ..Default::default()
        }
    }
    
    /// Add nodes accessed
    pub fn add_nodes_accessed(&mut self, count: u64) {
        self.nodes_accessed += count;
    }
    
    /// Add relationships traversed
    pub fn add_relationships_traversed(&mut self, count: u64) {
        self.relationships_traversed += count;
    }
    
    /// Set memory usage
    pub fn set_memory_used(&mut self, bytes: u64) {
        self.memory_used = bytes;
    }
    
    /// Mark as cached
    pub fn mark_cached(&mut self) {
        self.was_cached = true;
    }
    
    /// Add reified relationships count
    pub fn add_reified_relationships(&mut self, count: u64) {
        self.reified_relationships += count;
    }
}

/// Path result from path-finding queries
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
    
    /// Get path length (number of hops)
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
}

/// Engine statistics for monitoring
#[derive(Debug, Default)]
pub struct EngineStats {
    /// Total nodes created
    pub nodes_created: AtomicU64,
    /// Total relationships created
    pub relationships_created: AtomicU64,
    /// Total reifications performed
    pub reifications_performed: AtomicU64,
    /// Total queries executed
    pub queries_executed: AtomicU64,
    /// Total query time in microseconds
    pub total_query_time_us: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
}

impl EngineStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Increment nodes created
    pub fn inc_nodes_created(&self) {
        self.nodes_created.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment relationships created
    pub fn inc_relationships_created(&self) {
        self.relationships_created.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment reifications performed
    pub fn inc_reifications_performed(&self) {
        self.reifications_performed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment queries executed
    pub fn inc_queries_executed(&self) {
        self.queries_executed.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Add query time
    pub fn add_query_time(&self, time_us: u64) {
        self.total_query_time_us.fetch_add(time_us, Ordering::Relaxed);
    }
    
    /// Increment cache hits
    pub fn inc_cache_hits(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Increment cache misses
    pub fn inc_cache_misses(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// Get average query time
    pub fn average_query_time_us(&self) -> f64 {
        let total_time = self.total_query_time_us.load(Ordering::Relaxed);
        let total_queries = self.queries_executed.load(Ordering::Relaxed);
        
        if total_queries == 0 {
            0.0
        } else {
            total_time as f64 / total_queries as f64
        }
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

/// Macro for creating property values
#[macro_export]
macro_rules! prop_value {
    (null) => { PropertyValue::Null };
    ($value:expr) => {
        match $value {
            v if v.is_null() => PropertyValue::Null,
            v => PropertyValue::from(v),
        }
    };
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
    }
    
    #[test]
    fn test_node_creation() {
        let props = properties!("name" => "Alice", "age" => 30);
        let node = Node::new("Person", props);
        
        assert_eq!(node.label, "Person");
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("Alice".to_string())));
        assert_eq!(node.get_property("age"), Some(&PropertyValue::Int(30)));
    }
    
    #[test]
    fn test_relationship_creation() {
        let from = EntityId::new();
        let to = EntityId::new();
        let props = properties!("since" => "2020-01-01");
        
        let rel = Relationship::new(from, to, "KNOWS", props);
        assert_eq!(rel.from, from);
        assert_eq!(rel.to, to);
        assert_eq!(rel.rel_type, "KNOWS");
        assert!(!rel.is_reified());
    }
    
    #[test]
    fn test_property_value_conversions() {
        let prop: PropertyValue = true.into();
        assert!(matches!(prop, PropertyValue::Bool(true)));
        
        let prop: PropertyValue = 42.into();
        assert!(matches!(prop, PropertyValue::Int(42)));
        
        let prop: PropertyValue = "hello".into();
        assert!(matches!(prop, PropertyValue::String(ref s) if s == "hello"));
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
        assert!(json.is_array());
    }
    
    #[test]
    fn test_graph_path() {
        let mut path = GraphPath::new();
        let node1 = EntityId::new();
        let node2 = EntityId::new();
        let rel1 = EntityId::new();
        
        path.add_hop(node1, None, 0.0);
        path.add_hop(node2, Some(rel1), 1.5);
        
        assert_eq!(path.length(), 1);
        assert_eq!(path.weight, 1.5);
        assert_eq!(path.start_node(), Some(node1));
        assert_eq!(path.end_node(), Some(node2));
    }
    
    #[test]
    fn test_engine_stats() {
        let stats = EngineStats::new();
        
        stats.inc_nodes_created();
        stats.inc_relationships_created();
        stats.inc_queries_executed();
        stats.add_query_time(1000);
        
        assert_eq!(stats.nodes_created.load(Ordering::Relaxed), 1);
        assert_eq!(stats.relationships_created.load(Ordering::Relaxed), 1);
        assert_eq!(stats.queries_executed.load(Ordering::Relaxed), 1);
        assert_eq!(stats.average_query_time_us(), 1000.0);
    }
    
    #[test]
    fn test_metadata_operations() {
        let mut metadata = NodeMetadata::new();
        let initial_version = metadata.version;
        
        metadata.touch();
        assert!(metadata.version > initial_version);
        
        metadata.add_tag("important");
        metadata.add_tag("test");
        assert_eq!(metadata.tags.len(), 2);
        
        assert!(metadata.remove_tag("important"));
        assert!(!metadata.remove_tag("nonexistent"));
        assert_eq!(metadata.tags.len(), 1);
    }
}