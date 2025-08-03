//! Universal Graph Engine - Rust Bindings
//! 
//! Safe Rust wrapper around the Universal Graph Engine C API
//! Provides memory safety, zero-cost abstractions, and idiomatic Rust patterns
//! 
//! Copyright (c) 2025 Universal Graph Engine Project
//! Licensed under MIT License

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::collections::HashMap;
use std::fmt;

// Re-export C API types
pub use universal_graph_sys::*;

/// Result type for Universal Graph operations
pub type UgResult<T> = Result<T, UgError>;

/// Error types for Universal Graph operations
#[derive(Debug, Clone)]
pub enum UgError {
    InvalidId,
    InvalidOperation(String),
    MemoryError,
    TypeMismatch,
    NotFound,
    Other(String),
}

impl fmt::Display for UgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UgError::InvalidId => write!(f, "Invalid ID"),
            UgError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            UgError::MemoryError => write!(f, "Memory allocation error"),
            UgError::TypeMismatch => write!(f, "Type mismatch"),
            UgError::NotFound => write!(f, "Not found"),
            UgError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for UgError {}

/// Type identifiers
pub type NodeId = ug_node_id_t;
pub type RelationshipId = ug_relationship_id_t;
pub type GraphId = ug_graph_id_t;
pub type Weight = ug_weight_t;
pub type Confidence = ug_confidence_t;

/// Universal value that can hold any type safely in Rust
#[derive(Debug)]
pub enum UniversalValue {
    Bool(bool),
    I8(i8), I16(i16), I32(i32), I64(i64),
    U8(u8), U16(u16), U32(u32), U64(u64),
    F32(f32), F64(f64),
    String(String),
    Bytes(Vec<u8>),
    Custom(Box<dyn CustomValue>),
}

/// Trait for custom value types
pub trait CustomValue: fmt::Debug + Send + Sync {
    fn type_name(&self) -> &'static str;
    fn size(&self) -> usize;
    fn as_bytes(&self) -> &[u8];
    fn clone_boxed(&self) -> Box<dyn CustomValue>;
}

impl Clone for UniversalValue {
    fn clone(&self) -> Self {
        match self {
            UniversalValue::Bool(v) => UniversalValue::Bool(*v),
            UniversalValue::I8(v) => UniversalValue::I8(*v),
            UniversalValue::I16(v) => UniversalValue::I16(*v),
            UniversalValue::I32(v) => UniversalValue::I32(*v),
            UniversalValue::I64(v) => UniversalValue::I64(*v),
            UniversalValue::U8(v) => UniversalValue::U8(*v),
            UniversalValue::U16(v) => UniversalValue::U16(*v),
            UniversalValue::U32(v) => UniversalValue::U32(*v),
            UniversalValue::U64(v) => UniversalValue::U64(*v),
            UniversalValue::F32(v) => UniversalValue::F32(*v),
            UniversalValue::F64(v) => UniversalValue::F64(*v),
            UniversalValue::String(v) => UniversalValue::String(v.clone()),
            UniversalValue::Bytes(v) => UniversalValue::Bytes(v.clone()),
            UniversalValue::Custom(v) => UniversalValue::Custom(v.clone_boxed()),
        }
    }
}

impl UniversalValue {
    /// Get the corresponding C type
    fn c_type(&self) -> ug_type_t {
        match self {
            UniversalValue::Bool(_) => ug_type_t_UG_TYPE_BOOL,
            UniversalValue::I8(_) => ug_type_t_UG_TYPE_CHAR,
            UniversalValue::I16(_) => ug_type_t_UG_TYPE_SHORT,
            UniversalValue::I32(_) => ug_type_t_UG_TYPE_INT,
            UniversalValue::I64(_) => ug_type_t_UG_TYPE_LLONG,
            UniversalValue::U8(_) => ug_type_t_UG_TYPE_UCHAR,
            UniversalValue::U16(_) => ug_type_t_UG_TYPE_USHORT,
            UniversalValue::U32(_) => ug_type_t_UG_TYPE_UINT,
            UniversalValue::U64(_) => ug_type_t_UG_TYPE_ULLONG,
            UniversalValue::F32(_) => ug_type_t_UG_TYPE_FLOAT,
            UniversalValue::F64(_) => ug_type_t_UG_TYPE_DOUBLE,
            UniversalValue::String(_) => ug_type_t_UG_TYPE_STRING,
            UniversalValue::Bytes(_) => ug_type_t_UG_TYPE_ARRAY,
            UniversalValue::Custom(_) => ug_type_t_UG_TYPE_CUSTOM_STRUCT,
        }
    }

    /// Convert to bytes for C API
    fn as_bytes(&self) -> Vec<u8> {
        match self {
            UniversalValue::Bool(v) => vec![*v as u8],
            UniversalValue::I8(v) => v.to_le_bytes().to_vec(),
            UniversalValue::I16(v) => v.to_le_bytes().to_vec(),
            UniversalValue::I32(v) => v.to_le_bytes().to_vec(),
            UniversalValue::I64(v) => v.to_le_bytes().to_vec(),
            UniversalValue::U8(v) => v.to_le_bytes().to_vec(),
            UniversalValue::U16(v) => v.to_le_bytes().to_vec(),
            UniversalValue::U32(v) => v.to_le_bytes().to_vec(),
            UniversalValue::U64(v) => v.to_le_bytes().to_vec(),
            UniversalValue::F32(v) => v.to_le_bytes().to_vec(),
            UniversalValue::F64(v) => v.to_le_bytes().to_vec(),
            UniversalValue::String(v) => v.as_bytes().to_vec(),
            UniversalValue::Bytes(v) => v.clone(),
            UniversalValue::Custom(v) => v.as_bytes().to_vec(),
        }
    }
}

/// Safe wrapper around a C graph node
pub struct Node<'g> {
    id: NodeId,
    graph: PhantomData<&'g UniversalGraph>,
}

impl<'g> Node<'g> {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            graph: PhantomData,
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get node data if it matches the expected type
    pub fn data<T>(&self, graph: &UniversalGraph) -> UgResult<T> 
    where
        T: TryFrom<UniversalValue>,
        T::Error: Into<UgError>,
    {
        let node_ptr = unsafe { ug_get_node(graph.inner.as_ptr(), self.id) };
        if node_ptr.is_null() {
            return Err(UgError::NotFound);
        }

        // In a full implementation, we'd convert the C data to UniversalValue
        // and then try to convert to T
        Err(UgError::Other("Not implemented".to_string()))
    }

    /// Get a property value
    pub fn property(&self, graph: &UniversalGraph, key: &str) -> UgResult<Option<UniversalValue>> {
        let key_c = CString::new(key).map_err(|_| UgError::InvalidOperation("Invalid key".to_string()))?;
        
        let prop_ptr = unsafe { 
            ug_get_node_property(graph.inner.as_ptr(), self.id, key_c.as_ptr())
        };
        
        if prop_ptr.is_null() {
            return Ok(None);
        }

        // In a full implementation, we'd convert the C property to UniversalValue
        Ok(Some(UniversalValue::String("placeholder".to_string())))
    }

    /// Set a property value
    pub fn set_property(&self, graph: &UniversalGraph, key: &str, value: UniversalValue) -> UgResult<()> {
        let key_c = CString::new(key).map_err(|_| UgError::InvalidOperation("Invalid key".to_string()))?;
        
        // In a full implementation, we'd convert UniversalValue to C representation
        let success = unsafe {
            // ug_set_node_property(graph.inner.as_ptr(), self.id, key_c.as_ptr(), c_value)
            false // Placeholder
        };
        
        if success {
            Ok(())
        } else {
            Err(UgError::InvalidOperation("Failed to set property".to_string()))
        }
    }
}

/// Safe wrapper around a C graph relationship
pub struct Relationship<'g> {
    id: RelationshipId,
    graph: PhantomData<&'g UniversalGraph>,
}

impl<'g> Relationship<'g> {
    fn new(id: RelationshipId) -> Self {
        Self {
            id,
            graph: PhantomData,
        }
    }

    pub fn id(&self) -> RelationshipId {
        self.id
    }

    /// Get participants in this relationship
    pub fn participants(&self, graph: &UniversalGraph) -> UgResult<Vec<NodeId>> {
        let rel_ptr = unsafe { ug_get_relationship(graph.inner.as_ptr(), self.id) };
        if rel_ptr.is_null() {
            return Err(UgError::NotFound);
        }

        let rel = unsafe { &*rel_ptr };
        let mut participants = Vec::new();
        
        if !rel.participants.is_null() {
            for i in 0..rel.participant_count {
                let participant = unsafe { &*rel.participants.add(i) };
                participants.push(participant.node_id);
            }
        }

        Ok(participants)
    }

    /// Get relationship weight
    pub fn weight(&self, graph: &UniversalGraph) -> UgResult<Weight> {
        let rel_ptr = unsafe { ug_get_relationship(graph.inner.as_ptr(), self.id) };
        if rel_ptr.is_null() {
            return Err(UgError::NotFound);
        }

        let rel = unsafe { &*rel_ptr };
        Ok(rel.weight)
    }
}

/// Main Universal Graph wrapper providing memory safety
pub struct UniversalGraph {
    inner: NonNull<ug_graph_t>,
}

impl UniversalGraph {
    /// Create a new graph
    pub fn new() -> UgResult<Self> {
        Self::with_type(ug_graph_type_t_UG_GRAPH_TYPE_SIMPLE)
    }

    /// Create a new graph with specific type
    pub fn with_type(graph_type: ug_graph_type_t) -> UgResult<Self> {
        let ptr = unsafe { ug_create_graph_with_type(graph_type) };
        
        NonNull::new(ptr)
            .map(|inner| Self { inner })
            .ok_or(UgError::MemoryError)
    }

    /// Create a hypergraph
    pub fn hypergraph() -> UgResult<Self> {
        Self::with_type(ug_graph_type_t_UG_GRAPH_TYPE_HYPERGRAPH)
    }

    /// Create a temporal graph
    pub fn temporal() -> UgResult<Self> {
        Self::with_type(ug_graph_type_t_UG_GRAPH_TYPE_TEMPORAL)
    }

    /// Create a quantum graph
    pub fn quantum() -> UgResult<Self> {
        Self::with_type(ug_graph_type_t_UG_GRAPH_TYPE_QUANTUM)
    }

    /// Create a node with data
    pub fn create_node(&self, value: UniversalValue) -> UgResult<Node> {
        let data_bytes = value.as_bytes();
        let c_type = value.c_type();
        
        let node_id = unsafe {
            ug_create_node(
                self.inner.as_ptr(),
                c_type,
                data_bytes.as_ptr() as *const std::ffi::c_void
            )
        };
        
        if node_id == ug_node_id_t_UG_INVALID_ID {
            Err(UgError::InvalidOperation("Failed to create node".to_string()))
        } else {
            Ok(Node::new(node_id))
        }
    }

    /// Create a node with string data (convenience method)
    pub fn create_string_node(&self, data: &str) -> UgResult<Node> {
        self.create_node(UniversalValue::String(data.to_string()))
    }

    /// Create a node with numeric data (convenience method)
    pub fn create_numeric_node<T>(&self, data: T) -> UgResult<Node> 
    where 
        T: Into<UniversalValue>
    {
        self.create_node(data.into())
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> UgResult<Node> {
        let node_ptr = unsafe { ug_get_node(self.inner.as_ptr(), id) };
        
        if node_ptr.is_null() {
            Err(UgError::NotFound)
        } else {
            Ok(Node::new(id))
        }
    }

    /// Create a simple edge between two nodes
    pub fn create_edge(&self, from: NodeId, to: NodeId, edge_type: &str, weight: Weight) -> UgResult<Relationship> {
        let type_c = CString::new(edge_type)
            .map_err(|_| UgError::InvalidOperation("Invalid edge type".to_string()))?;
        
        let rel_id = unsafe {
            ug_create_edge(
                self.inner.as_ptr(),
                from,
                to,
                type_c.as_ptr(),
                weight
            )
        };
        
        if rel_id == ug_relationship_id_t_UG_INVALID_ID {
            Err(UgError::InvalidOperation("Failed to create edge".to_string()))
        } else {
            Ok(Relationship::new(rel_id))
        }
    }

    /// Create a hyperedge connecting multiple nodes
    pub fn create_hyperedge(&self, participants: &[NodeId], edge_type: &str) -> UgResult<Relationship> {
        let type_c = CString::new(edge_type)
            .map_err(|_| UgError::InvalidOperation("Invalid edge type".to_string()))?;
        
        let rel_id = unsafe {
            ug_create_hyperedge(
                self.inner.as_ptr(),
                participants.as_ptr() as *mut ug_node_id_t,
                participants.len(),
                type_c.as_ptr()
            )
        };
        
        if rel_id == ug_relationship_id_t_UG_INVALID_ID {
            Err(UgError::InvalidOperation("Failed to create hyperedge".to_string()))
        } else {
            Ok(Relationship::new(rel_id))
        }
    }

    /// Get a relationship by ID
    pub fn get_relationship(&self, id: RelationshipId) -> UgResult<Relationship> {
        let rel_ptr = unsafe { ug_get_relationship(self.inner.as_ptr(), id) };
        
        if rel_ptr.is_null() {
            Err(UgError::NotFound)
        } else {
            Ok(Relationship::new(id))
        }
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        unsafe { ug_get_node_count(self.inner.as_ptr()) }
    }

    /// Get relationship count
    pub fn relationship_count(&self) -> usize {
        unsafe { ug_get_relationship_count(self.inner.as_ptr()) }
    }

    /// Print graph statistics
    pub fn print_stats(&self) {
        unsafe { ug_print_graph_stats(self.inner.as_ptr()) }
    }

    /// Export graph to file
    pub fn export(&self, format: &str, filename: &str) -> UgResult<()> {
        let format_c = CString::new(format)
            .map_err(|_| UgError::InvalidOperation("Invalid format".to_string()))?;
        let filename_c = CString::new(filename)
            .map_err(|_| UgError::InvalidOperation("Invalid filename".to_string()))?;
        
        let success = unsafe {
            ug_export_graph(
                self.inner.as_ptr(),
                format_c.as_ptr(),
                filename_c.as_ptr()
            )
        };
        
        if success {
            Ok(())
        } else {
            Err(UgError::InvalidOperation("Export failed".to_string()))
        }
    }
}

impl Drop for UniversalGraph {
    fn drop(&mut self) {
        unsafe {
            ug_destroy_graph(self.inner.as_ptr());
        }
    }
}

// Safety: UniversalGraph can be safely sent between threads
unsafe impl Send for UniversalGraph {}
// Safety: UniversalGraph can be safely shared between threads (with proper synchronization)
unsafe impl Sync for UniversalGraph {}

/// Convert basic Rust types to UniversalValue
impl From<bool> for UniversalValue {
    fn from(v: bool) -> Self { UniversalValue::Bool(v) }
}

impl From<i32> for UniversalValue {
    fn from(v: i32) -> Self { UniversalValue::I32(v) }
}

impl From<i64> for UniversalValue {
    fn from(v: i64) -> Self { UniversalValue::I64(v) }
}

impl From<f32> for UniversalValue {
    fn from(v: f32) -> Self { UniversalValue::F32(v) }
}

impl From<f64> for UniversalValue {
    fn from(v: f64) -> Self { UniversalValue::F64(v) }
}

impl From<String> for UniversalValue {
    fn from(v: String) -> Self { UniversalValue::String(v) }
}

impl From<&str> for UniversalValue {
    fn from(v: &str) -> Self { UniversalValue::String(v.to_string()) }
}

/// Builder pattern for complex graphs
pub struct GraphBuilder {
    graph: UniversalGraph,
    nodes: HashMap<String, NodeId>,
}

impl GraphBuilder {
    pub fn new() -> UgResult<Self> {
        Ok(Self {
            graph: UniversalGraph::new()?,
            nodes: HashMap::new(),
        })
    }

    pub fn hypergraph() -> UgResult<Self> {
        Ok(Self {
            graph: UniversalGraph::hypergraph()?,
            nodes: HashMap::new(),
        })
    }

    pub fn add_node<T>(&mut self, name: &str, data: T) -> UgResult<&mut Self> 
    where
        T: Into<UniversalValue>
    {
        let node = self.graph.create_node(data.into())?;
        self.nodes.insert(name.to_string(), node.id());
        Ok(self)
    }

    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: &str, weight: Weight) -> UgResult<&mut Self> {
        let from_id = self.nodes.get(from).ok_or(UgError::NotFound)?;
        let to_id = self.nodes.get(to).ok_or(UgError::NotFound)?;
        
        self.graph.create_edge(*from_id, *to_id, edge_type, weight)?;
        Ok(self)
    }

    pub fn add_hyperedge(&mut self, participants: &[&str], edge_type: &str) -> UgResult<&mut Self> {
        let mut participant_ids = Vec::new();
        
        for &name in participants {
            let id = self.nodes.get(name).ok_or(UgError::NotFound)?;
            participant_ids.push(*id);
        }
        
        self.graph.create_hyperedge(&participant_ids, edge_type)?;
        Ok(self)
    }

    pub fn build(self) -> UniversalGraph {
        self.graph
    }
}

/// Fluent query interface
pub struct QueryBuilder {
    query: String,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            query: String::new(),
        }
    }

    pub fn match_pattern(mut self, pattern: &str) -> Self {
        self.query.push_str(&format!("MATCH {} ", pattern));
        self
    }

    pub fn where_clause(mut self, condition: &str) -> Self {
        self.query.push_str(&format!("WHERE {} ", condition));
        self
    }

    pub fn return_items(mut self, items: &str) -> Self {
        self.query.push_str(&format!("RETURN {} ", items));
        self
    }

    pub fn build(self) -> String {
        self.query
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = UniversalGraph::new().unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.relationship_count(), 0);
    }

    #[test]
    fn test_node_creation() {
        let graph = UniversalGraph::new().unwrap();
        let node = graph.create_string_node("test data").unwrap();
        assert_ne!(node.id(), ug_node_id_t_UG_INVALID_ID);
    }

    #[test]
    fn test_builder_pattern() {
        let mut builder = GraphBuilder::new().unwrap();
        let graph = builder
            .add_node("alice", "Alice Smith")
            .unwrap()
            .add_node("bob", "Bob Johnson")
            .unwrap()
            .add_edge("alice", "bob", "KNOWS", 1.0)
            .unwrap()
            .build();
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.relationship_count(), 1);
    }

    #[test]
    fn test_hypergraph() {
        let mut builder = GraphBuilder::hypergraph().unwrap();
        let graph = builder
            .add_node("a", 1)
            .unwrap()
            .add_node("b", 2)
            .unwrap()
            .add_node("c", 3)
            .unwrap()
            .add_hyperedge(&["a", "b", "c"], "COLLABORATION")
            .unwrap()
            .build();
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.relationship_count(), 1);
    }
}

/// Re-export common types for convenience
pub mod prelude {
    pub use super::{
        UniversalGraph, Node, Relationship, UniversalValue,
        GraphBuilder, QueryBuilder, UgResult, UgError,
        NodeId, RelationshipId, Weight, Confidence
    };
}