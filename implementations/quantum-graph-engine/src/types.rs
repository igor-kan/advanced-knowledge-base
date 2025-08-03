//! Core data types for the Quantum Graph Engine
//!
//! This module defines the fundamental data structures used throughout the engine,
//! optimized for performance and memory efficiency.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use ahash::AHasher;

/// Fast hash function using AHash
pub fn fast_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = AHasher::default();
    item.hash(&mut hasher);
    hasher.finish()
}

/// Unique identifier for nodes (128-bit for massive scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u128);

impl NodeId {
    /// Create a new node ID from a string
    pub fn new(s: &str) -> Self {
        Self(fast_hash(&s) as u128)
    }
    
    /// Create a node ID from a number
    pub fn from_u64(id: u64) -> Self {
        Self(id as u128)
    }
    
    /// Get the underlying ID value
    pub fn as_u128(self) -> u128 {
        self.0
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self::new(&s)
    }
}

impl From<u64> for NodeId {
    fn from(id: u64) -> Self {
        Self::from_u64(id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node_{:032x}", self.0)
    }
}

/// Unique identifier for edges (128-bit for massive scale)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeId(pub u128);

impl EdgeId {
    /// Create a new edge ID
    pub fn new(from: NodeId, to: NodeId, edge_type: &str) -> Self {
        let combined = format!("{}->{}:{}", from, to, edge_type);
        Self(fast_hash(&combined) as u128)
    }
    
    /// Get the underlying ID value
    pub fn as_u128(self) -> u128 {
        self.0
    }
}

impl std::fmt::Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "edge_{:032x}", self.0)
    }
}

/// Node data structure optimized for performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier
    pub id: NodeId,
    /// Node type/label
    pub node_type: String,
    /// Node properties/data
    pub data: NodeData,
    /// Metadata for versioning and tracking
    pub metadata: NodeMetadata,
}

impl Node {
    /// Create a new node
    pub fn new<S: Into<String>>(id: S, data: NodeData) -> Self {
        let id_str = id.into();
        Self {
            id: NodeId::new(&id_str),
            node_type: "generic".to_string(),
            data,
            metadata: NodeMetadata::default(),
        }
    }
    
    /// Create a typed node
    pub fn with_type<S: Into<String>>(id: S, node_type: String, data: NodeData) -> Self {
        let id_str = id.into();
        Self {
            id: NodeId::new(&id_str),
            node_type,
            data,
            metadata: NodeMetadata::default(),
        }
    }
    
    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.data.properties.get(key)
    }
    
    /// Set a property value
    pub fn set_property(&mut self, key: String, value: PropertyValue) {
        self.data.properties.insert(key, value);
        self.metadata.update_modified();
    }
}

/// Node data container
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeData {
    /// Key-value properties
    pub properties: HashMap<String, PropertyValue>,
    /// Optional content/payload
    pub content: Option<String>,
    /// Optional embedding vector for AI/ML operations
    pub embedding: Option<Vec<f32>>,
}

impl NodeData {
    /// Create new empty node data
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create node data with properties
    pub fn with_properties(properties: HashMap<String, PropertyValue>) -> Self {
        Self {
            properties,
            content: None,
            embedding: None,
        }
    }
    
    /// Add a property
    pub fn property<K: Into<String>, V: Into<PropertyValue>>(mut self, key: K, value: V) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
    
    /// Set content
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = Some(content.into());
        self
    }
    
    /// Set embedding vector
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

/// Node metadata for tracking and versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creation timestamp
    pub created_at: u64,
    /// Last modification timestamp  
    pub modified_at: u64,
    /// Version number
    pub version: u64,
    /// Optional tags
    pub tags: Vec<String>,
    /// Access count for optimization
    pub access_count: u64,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            created_at: now,
            modified_at: now,
            version: 1,
            tags: Vec::new(),
            access_count: 0,
        }
    }
}

impl NodeMetadata {
    /// Update modified timestamp and increment version
    pub fn update_modified(&mut self) {
        self.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.version += 1;
    }
    
    /// Increment access count
    pub fn record_access(&mut self) {
        self.access_count += 1;
    }
}

/// Edge data structure optimized for performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier
    pub id: EdgeId,
    /// Source node
    pub from: NodeId,
    /// Target node
    pub to: NodeId,
    /// Edge type/label
    pub edge_type: String,
    /// Edge properties/data
    pub data: EdgeData,
    /// Metadata for versioning and tracking
    pub metadata: EdgeMetadata,
}

impl Edge {
    /// Create a new edge
    pub fn new(from: NodeId, to: NodeId, edge_type: String) -> Self {
        let id = EdgeId::new(from, to, &edge_type);
        Self {
            id,
            from,
            to,
            edge_type,
            data: EdgeData::default(),
            metadata: EdgeMetadata::default(),
        }
    }
    
    /// Create an edge with properties
    pub fn with_data(from: NodeId, to: NodeId, edge_type: String, data: EdgeData) -> Self {
        let id = EdgeId::new(from, to, &edge_type);
        Self {
            id,
            from,
            to,
            edge_type,
            data,
            metadata: EdgeMetadata::default(),
        }
    }
    
    /// Get a property value
    pub fn get_property(&self, key: &str) -> Option<&PropertyValue> {
        self.data.properties.get(key)
    }
    
    /// Set a property value
    pub fn set_property(&mut self, key: String, value: PropertyValue) {
        self.data.properties.insert(key, value);
        self.metadata.update_modified();
    }
    
    /// Check if edge is directed
    pub fn is_directed(&self) -> bool {
        self.data.directed
    }
    
    /// Get edge weight
    pub fn weight(&self) -> f64 {
        self.data.weight
    }
}

/// Edge data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    /// Key-value properties
    pub properties: HashMap<String, PropertyValue>,
    /// Edge weight (default: 1.0)
    pub weight: f64,
    /// Whether edge is directed (default: true)
    pub directed: bool,
    /// Optional temporal information
    pub temporal: Option<TemporalInfo>,
}

impl Default for EdgeData {
    fn default() -> Self {
        Self {
            properties: HashMap::new(),
            weight: 1.0,
            directed: true,
            temporal: None,
        }
    }
}

impl EdgeData {
    /// Create new empty edge data
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set edge weight
    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
    
    /// Set directed flag
    pub fn directed(mut self, directed: bool) -> Self {
        self.directed = directed;
        self
    }
    
    /// Add a property
    pub fn property<K: Into<String>, V: Into<PropertyValue>>(mut self, key: K, value: V) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
    
    /// Set temporal information
    pub fn temporal(mut self, temporal: TemporalInfo) -> Self {
        self.temporal = Some(temporal);
        self
    }
}

/// Edge metadata for tracking and versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
    /// Version number
    pub version: u64,
    /// Traversal count for optimization
    pub traversal_count: u64,
}

impl Default for EdgeMetadata {
    fn default() -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            created_at: now,
            modified_at: now,
            version: 1,
            traversal_count: 0,
        }
    }
}

impl EdgeMetadata {
    /// Update modified timestamp and increment version
    pub fn update_modified(&mut self) {
        self.modified_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.version += 1;
    }
    
    /// Record edge traversal
    pub fn record_traversal(&mut self) {
        self.traversal_count += 1;
    }
}

/// Temporal information for time-bounded relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalInfo {
    /// Start timestamp
    pub start: Option<u64>,
    /// End timestamp (None for ongoing)
    pub end: Option<u64>,
    /// Duration in seconds
    pub duration: Option<u64>,
}

impl TemporalInfo {
    /// Create temporal info with start time
    pub fn from_start(start: u64) -> Self {
        Self {
            start: Some(start),
            end: None,
            duration: None,
        }
    }
    
    /// Create temporal info with start and end times
    pub fn from_range(start: u64, end: u64) -> Self {
        Self {
            start: Some(start),
            end: Some(end),
            duration: Some(end - start),
        }
    }
    
    /// Check if temporal info is active at given timestamp
    pub fn is_active_at(&self, timestamp: u64) -> bool {
        if let Some(start) = self.start {
            if timestamp < start {
                return false;
            }
        }
        
        if let Some(end) = self.end {
            if timestamp > end {
                return false;
            }
        }
        
        true
    }
}

/// Property value types for nodes and edges
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Array of values
    Array(Vec<PropertyValue>),
    /// Nested object
    Object(HashMap<String, PropertyValue>),
    /// Binary data
    Binary(Vec<u8>),
    /// Null value
    Null,
}

impl From<String> for PropertyValue {
    fn from(s: String) -> Self {
        PropertyValue::String(s)
    }
}

impl From<&str> for PropertyValue {
    fn from(s: &str) -> Self {
        PropertyValue::String(s.to_string())
    }
}

impl From<i64> for PropertyValue {
    fn from(i: i64) -> Self {
        PropertyValue::Integer(i)
    }
}

impl From<f64> for PropertyValue {
    fn from(f: f64) -> Self {
        PropertyValue::Float(f)
    }
}

impl From<bool> for PropertyValue {
    fn from(b: bool) -> Self {
        PropertyValue::Boolean(b)
    }
}

impl From<Vec<PropertyValue>> for PropertyValue {
    fn from(v: Vec<PropertyValue>) -> Self {
        PropertyValue::Array(v)
    }
}

impl From<HashMap<String, PropertyValue>> for PropertyValue {
    fn from(m: HashMap<String, PropertyValue>) -> Self {
        PropertyValue::Object(m)
    }
}

/// Path between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Path {
    /// Nodes in the path
    pub nodes: Vec<NodeId>,
    /// Edges in the path
    pub edges: Vec<EdgeId>,
    /// Total path weight
    pub weight: f64,
    /// Path length (number of hops)
    pub length: usize,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            weight: 0.0,
            length: 0,
        }
    }
    
    /// Add a hop to the path
    pub fn add_hop(&mut self, node: NodeId, edge: Option<EdgeId>, weight: f64) {
        self.nodes.push(node);
        if let Some(edge_id) = edge {
            self.edges.push(edge_id);
        }
        self.weight += weight;
        self.length = self.edges.len();
    }
    
    /// Check if path is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: u64,
    /// Total number of edges
    pub edge_count: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Average node degree
    pub avg_degree: f64,
    /// Graph density
    pub density: f64,
    /// Connected components count
    pub connected_components: u64,
}

impl GraphStats {
    /// Calculate graph density
    pub fn calculate_density(&mut self) {
        if self.node_count > 1 {
            let max_edges = self.node_count * (self.node_count - 1);
            self.density = self.edge_count as f64 / max_edges as f64;
        }
    }
    
    /// Calculate average degree
    pub fn calculate_avg_degree(&mut self) {
        if self.node_count > 0 {
            self.avg_degree = (2.0 * self.edge_count as f64) / self.node_count as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let data = NodeData::new()
            .property("name", "test_node")
            .property("value", 42i64)
            .content("test content");
            
        let node = Node::new("test_id", data);
        assert_eq!(node.get_property("name"), Some(&PropertyValue::String("test_node".to_string())));
        assert_eq!(node.get_property("value"), Some(&PropertyValue::Integer(42)));
    }
    
    #[test]
    fn test_edge_creation() {
        let from = NodeId::new("node1");
        let to = NodeId::new("node2");
        
        let data = EdgeData::new()
            .weight(0.5)
            .directed(true)
            .property("type", "test_edge");
            
        let edge = Edge::with_data(from, to, "CONNECTS".to_string(), data);
        assert_eq!(edge.weight(), 0.5);
        assert!(edge.is_directed());
        assert_eq!(edge.get_property("type"), Some(&PropertyValue::String("test_edge".to_string())));
    }
    
    #[test]
    fn test_temporal_info() {
        let temporal = TemporalInfo::from_range(1000, 2000);
        assert!(temporal.is_active_at(1500));
        assert!(!temporal.is_active_at(500));
        assert!(!temporal.is_active_at(2500));
    }
    
    #[test]
    fn test_property_values() {
        let string_prop = PropertyValue::from("test");
        let int_prop = PropertyValue::from(42i64);
        let float_prop = PropertyValue::from(3.14f64);
        let bool_prop = PropertyValue::from(true);
        
        assert_eq!(string_prop, PropertyValue::String("test".to_string()));
        assert_eq!(int_prop, PropertyValue::Integer(42));
        assert_eq!(float_prop, PropertyValue::Float(3.14));
        assert_eq!(bool_prop, PropertyValue::Boolean(true));
    }
}