//! Utility functions and helpers for the IndraDB reified engine
//!
//! This module provides common utilities, performance helpers, and convenience functions
//! for working with reified relationships and the IndraDB database.

use crate::types::*;
use crate::{IndraReifiedError, Result};

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use indradb::{Vertex, Edge, Type, Identifier};

/// Conversion utilities between IndraDB and reified types
pub mod conversion {
    use super::*;
    
    /// Convert IndraDB Vertex to ReifiedNode
    pub fn vertex_to_reified_node(vertex: &Vertex, properties: PropertyMap) -> ReifiedNode {
        ReifiedNode::with_id(
            EntityId::from(vertex.id),
            vertex.t.as_str().to_string(),
            properties,
        )
    }
    
    /// Convert ReifiedNode to IndraDB Vertex
    pub fn reified_node_to_vertex(node: &ReifiedNode) -> Result<Vertex> {
        let vertex_type = Type::new(node.node_type.clone())
            .map_err(|e| IndraReifiedError::ValidationError {
                entity: "node_type".to_string(),
                constraint: format!("Invalid node type: {}", e),
            })?;
        
        Ok(Vertex::new(node.id.to_identifier(), vertex_type))
    }
    
    /// Convert IndraDB Edge to ReifiedEdge
    pub fn edge_to_reified_edge(edge: &Edge, properties: PropertyMap) -> ReifiedEdge {
        ReifiedEdge::new(
            EntityId::from(edge.outbound_id),
            EntityId::from(edge.inbound_id),
            edge.t.as_str().to_string(),
            properties,
        )
    }
    
    /// Convert ReifiedEdge to IndraDB Edge
    pub fn reified_edge_to_edge(edge: &ReifiedEdge) -> Result<Edge> {
        edge.to_edge()
    }
    
    /// Convert system time to microsecond timestamp
    pub fn system_time_to_us(time: SystemTime) -> u64 {
        time.duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
    
    /// Convert microsecond timestamp to system time
    pub fn us_to_system_time(us: u64) -> SystemTime {
        UNIX_EPOCH + std::time::Duration::from_micros(us)
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate entity ID format
    pub fn validate_entity_id(id: &EntityId) -> Result<()> {
        if id.as_uuid().is_nil() {
            return Err(IndraReifiedError::ValidationError {
                entity: "entity_id".to_string(),
                constraint: "Entity ID cannot be nil UUID".to_string(),
            });
        }
        Ok(())
    }
    
    /// Validate vertex type name
    pub fn validate_vertex_type(vertex_type: &str) -> Result<()> {
        if vertex_type.is_empty() {
            return Err(IndraReifiedError::ValidationError {
                entity: "vertex_type".to_string(),
                constraint: "Vertex type cannot be empty".to_string(),
            });
        }
        
        if vertex_type.len() > 64 {
            return Err(IndraReifiedError::ValidationError {
                entity: "vertex_type".to_string(),
                constraint: "Vertex type cannot exceed 64 characters".to_string(),
            });
        }
        
        // Check for valid characters (alphanumeric and underscore)
        if !vertex_type.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(IndraReifiedError::ValidationError {
                entity: "vertex_type".to_string(),
                constraint: "Vertex type can only contain alphanumeric characters and underscores".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate edge type name
    pub fn validate_edge_type(edge_type: &str) -> Result<()> {
        if edge_type.is_empty() {
            return Err(IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: "Edge type cannot be empty".to_string(),
            });
        }
        
        if edge_type.len() > 64 {
            return Err(IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: "Edge type cannot exceed 64 characters".to_string(),
            });
        }
        
        // Check for valid characters (alphanumeric, underscore, and uppercase)
        if !edge_type.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(IndraReifiedError::ValidationError {
                entity: "edge_type".to_string(),
                constraint: "Edge type can only contain alphanumeric characters and underscores".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate property name
    pub fn validate_property_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(IndraReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name cannot be empty".to_string(),
            });
        }
        
        if name.len() > 128 {
            return Err(IndraReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name cannot exceed 128 characters".to_string(),
            });
        }
        
        // Property names should be valid identifiers
        if !name.chars().next().unwrap().is_alphabetic() && name.chars().next().unwrap() != '_' {
            return Err(IndraReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name must start with a letter or underscore".to_string(),
            });
        }
        
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(IndraReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name can only contain alphanumeric characters and underscores".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate property map
    pub fn validate_property_map(properties: &PropertyMap) -> Result<()> {
        for (key, _value) in properties {
            validate_property_name(key)?;
        }
        Ok(())
    }
    
    /// Validate reified node structure
    pub fn validate_reified_node(node: &ReifiedNode) -> Result<()> {
        validate_entity_id(&node.id)?;
        validate_vertex_type(&node.node_type)?;
        validate_property_map(&node.properties)?;
        Ok(())
    }
    
    /// Validate reified edge structure
    pub fn validate_reified_edge(edge: &ReifiedEdge) -> Result<()> {
        validate_entity_id(&edge.id)?;
        validate_entity_id(&edge.from)?;
        validate_entity_id(&edge.to)?;
        validate_edge_type(&edge.edge_type)?;
        validate_property_map(&edge.properties)?;
        
        // Check that from and to are different
        if edge.from == edge.to {
            tracing::warn!("Self-referencing edge detected: {}", edge.id);
        }
        
        Ok(())
    }
}

/// Serialization utilities
pub mod serialization {
    use super::*;
    use serde_json;
    
    /// Serialize a property map to JSON string
    pub fn serialize_properties(properties: &PropertyMap) -> Result<String> {
        serde_json::to_string(properties)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Deserialize properties from JSON string
    pub fn deserialize_properties(json: &str) -> Result<PropertyMap> {
        serde_json::from_str(json)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Serialize a reified node to JSON
    pub fn serialize_node(node: &ReifiedNode) -> Result<String> {
        serde_json::to_string(node)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Deserialize a reified node from JSON
    pub fn deserialize_node(json: &str) -> Result<ReifiedNode> {
        serde_json::from_str(json)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Serialize a reified edge to JSON
    pub fn serialize_edge(edge: &ReifiedEdge) -> Result<String> {
        serde_json::to_string(edge)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Deserialize a reified edge from JSON
    pub fn deserialize_edge(json: &str) -> Result<ReifiedEdge> {
        serde_json::from_str(json)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
    
    /// Serialize query result to JSON
    pub fn serialize_result(result: &PropertyGraphResult) -> Result<String> {
        serde_json::to_string(result)
            .map_err(|e| IndraReifiedError::SerializationError(e))
    }
}

/// Hash utilities for efficient lookups
pub mod hashing {
    use super::*;
    use ahash::AHasher;
    use std::hash::{Hash, Hasher};
    
    /// Fast hash for entity IDs
    pub fn hash_entity_id(id: &EntityId) -> u64 {
        let mut hasher = AHasher::default();
        id.as_uuid().hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash a property map for caching
    pub fn hash_property_map(properties: &PropertyMap) -> u64 {
        let mut hasher = AHasher::default();
        
        // Sort keys for consistent hashing
        let mut sorted_props: Vec<_> = properties.iter().collect();
        sorted_props.sort_by_key(|(k, _)| *k);
        
        for (key, value) in sorted_props {
            key.hash(&mut hasher);
            hash_property_value(value).hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    /// Hash a property value
    pub fn hash_property_value(value: &PropertyValue) -> u64 {
        let mut hasher = AHasher::default();
        
        match value {
            PropertyValue::Null => 0u8.hash(&mut hasher),
            PropertyValue::Bool(b) => {
                1u8.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            PropertyValue::Int(i) => {
                2u8.hash(&mut hasher);
                i.hash(&mut hasher);
            }
            PropertyValue::Float(f) => {
                3u8.hash(&mut hasher);
                f.to_bits().hash(&mut hasher);
            }
            PropertyValue::String(s) => {
                4u8.hash(&mut hasher);
                s.hash(&mut hasher);
            }
            PropertyValue::Array(arr) => {
                5u8.hash(&mut hasher);
                arr.len().hash(&mut hasher);
                for item in arr {
                    hash_property_value(item).hash(&mut hasher);
                }
            }
            PropertyValue::Object(obj) => {
                6u8.hash(&mut hasher);
                hash_property_map(obj).hash(&mut hasher);
            }
            PropertyValue::Binary(data) => {
                7u8.hash(&mut hasher);
                data.hash(&mut hasher);
            }
            PropertyValue::Timestamp(ts) => {
                8u8.hash(&mut hasher);
                conversion::system_time_to_us(*ts).hash(&mut hasher);
            }
            PropertyValue::Uuid(uuid) => {
                9u8.hash(&mut hasher);
                uuid.hash(&mut hasher);
            }
            PropertyValue::Json(json) => {
                10u8.hash(&mut hasher);
                json.to_string().hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }
    
    /// Hash a reified node for caching
    pub fn hash_node(node: &ReifiedNode) -> u64 {
        let mut hasher = AHasher::default();
        hash_entity_id(&node.id).hash(&mut hasher);
        node.node_type.hash(&mut hasher);
        hash_property_map(&node.properties).hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash a reified edge for caching
    pub fn hash_edge(edge: &ReifiedEdge) -> u64 {
        let mut hasher = AHasher::default();
        hash_entity_id(&edge.id).hash(&mut hasher);
        hash_entity_id(&edge.from).hash(&mut hasher);
        hash_entity_id(&edge.to).hash(&mut hasher);
        edge.edge_type.hash(&mut hasher);
        hash_property_map(&edge.properties).hash(&mut hasher);
        hasher.finish()
    }
}

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    use std::time::Instant;
    use std::sync::atomic::{AtomicU64, Ordering};
    
    /// Performance counter for tracking operations
    #[derive(Debug)]
    pub struct PerformanceCounter {
        /// Total operations
        pub total_ops: AtomicU64,
        /// Total time in microseconds
        pub total_time_us: AtomicU64,
    }
    
    impl Default for PerformanceCounter {
        fn default() -> Self {
            Self {
                total_ops: AtomicU64::new(0),
                total_time_us: AtomicU64::new(0),
            }
        }
    }
    
    impl PerformanceCounter {
        /// Record an operation with its execution time
        pub fn record_operation(&self, duration_us: u64) {
            self.total_ops.fetch_add(1, Ordering::Relaxed);
            self.total_time_us.fetch_add(duration_us, Ordering::Relaxed);
        }
        
        /// Get average operation time in microseconds
        pub fn avg_operation_time_us(&self) -> f64 {
            let total_ops = self.total_ops.load(Ordering::Relaxed);
            let total_time = self.total_time_us.load(Ordering::Relaxed);
            
            if total_ops > 0 {
                total_time as f64 / total_ops as f64
            } else {
                0.0
            }
        }
        
        /// Reset all counters
        pub fn reset(&self) {
            self.total_ops.store(0, Ordering::Relaxed);
            self.total_time_us.store(0, Ordering::Relaxed);
        }
    }
    
    /// Scoped timing utility for automatic performance measurement
    pub struct ScopedTimer<'a> {
        counter: &'a PerformanceCounter,
        start_time: Instant,
    }
    
    impl<'a> ScopedTimer<'a> {
        /// Create a new scoped timer
        pub fn new(counter: &'a PerformanceCounter) -> Self {
            Self {
                counter,
                start_time: Instant::now(),
            }
        }
    }
    
    impl<'a> Drop for ScopedTimer<'a> {
        fn drop(&mut self) {
            let duration = self.start_time.elapsed().as_micros() as u64;
            self.counter.record_operation(duration);
        }
    }
    
    /// Create a scoped timer for a performance counter
    pub fn time_operation(counter: &PerformanceCounter) -> ScopedTimer {
        ScopedTimer::new(counter)
    }
}

/// Memory utilities
pub mod memory {
    use super::*;
    
    /// Estimate memory usage of a property value
    pub fn estimate_property_value_size(value: &PropertyValue) -> usize {
        match value {
            PropertyValue::Null => 1,
            PropertyValue::Bool(_) => 1,
            PropertyValue::Int(_) => 8,
            PropertyValue::Float(_) => 8,
            PropertyValue::String(s) => s.len() + 24, // String overhead
            PropertyValue::Array(arr) => {
                24 + arr.iter().map(estimate_property_value_size).sum::<usize>() // Vec overhead
            }
            PropertyValue::Object(obj) => {
                32 + obj.iter().map(|(k, v)| k.len() + estimate_property_value_size(v)).sum::<usize>() // HashMap overhead
            }
            PropertyValue::Binary(data) => data.len() + 24, // Vec overhead
            PropertyValue::Timestamp(_) => 16,
            PropertyValue::Uuid(_) => 16,
            PropertyValue::Json(json) => json.to_string().len() + 32, // JSON overhead
        }
    }
    
    /// Estimate memory usage of a property map
    pub fn estimate_property_map_size(properties: &PropertyMap) -> usize {
        32 + properties.iter().map(|(k, v)| k.len() + estimate_property_value_size(v)).sum::<usize>()
    }
    
    /// Estimate memory usage of a reified node
    pub fn estimate_node_size(node: &ReifiedNode) -> usize {
        16 + // EntityId
        node.node_type.len() + 24 + // String overhead
        estimate_property_map_size(&node.properties) +
        64 // NodeMetadata estimate
    }
    
    /// Estimate memory usage of a reified edge
    pub fn estimate_edge_size(edge: &ReifiedEdge) -> usize {
        48 + // 3 EntityIds
        edge.edge_type.len() + 24 + // String overhead
        estimate_property_map_size(&edge.properties) +
        64 // EdgeMetadata estimate
    }
}

/// Utility functions
/// Get current timestamp in microseconds
pub fn timestamp_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Generate a new entity ID
pub fn generate_entity_id() -> EntityId {
    EntityId::new()
}

/// Generate a deterministic entity ID from data
pub fn generate_deterministic_entity_id(data: &[u8]) -> EntityId {
    use sha2::{Sha256, Digest};
    
    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = hasher.finalize();
    
    // Use first 16 bytes of hash as UUID bytes
    let mut uuid_bytes = [0u8; 16];
    uuid_bytes.copy_from_slice(&hash[..16]);
    
    EntityId::from_uuid(Uuid::from_bytes(uuid_bytes))
}

/// Create a property map from key-value pairs
pub fn create_properties<I, K, V>(iter: I) -> PropertyMap
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<PropertyValue>,
{
    iter.into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect()
}

/// Merge two property maps, with the second taking priority
pub fn merge_properties(base: PropertyMap, overlay: PropertyMap) -> PropertyMap {
    let mut result = base;
    for (key, value) in overlay {
        result.insert(key, value);
    }
    result
}

/// Check if two property values are equivalent (deep equality)
pub fn property_values_equal(a: &PropertyValue, b: &PropertyValue) -> bool {
    match (a, b) {
        (PropertyValue::Float(f1), PropertyValue::Float(f2)) => {
            // Handle floating point comparison with epsilon
            (f1 - f2).abs() < f64::EPSILON
        }
        _ => a == b,
    }
}

/// Format a property map for display
pub fn format_properties(properties: &PropertyMap) -> String {
    if properties.is_empty() {
        return "{}".to_string();
    }
    
    let mut sorted_props: Vec<_> = properties.iter().collect();
    sorted_props.sort_by_key(|(k, _)| *k);
    
    let prop_strings: Vec<String> = sorted_props
        .into_iter()
        .map(|(k, v)| format!("{}: {}", k, v.to_string()))
        .collect();
    
    format!("{{{}}}", prop_strings.join(", "))
}

/// Create a simple graph path from vertices and edges
pub fn create_simple_path(vertices: Vec<EntityId>, edges: Vec<EntityId>) -> GraphPath {
    let mut path = GraphPath::new();
    
    for (i, vertex_id) in vertices.iter().enumerate() {
        let edge_id = if i < edges.len() {
            Some(edges[i])
        } else {
            None
        };
        
        path.add_hop(*vertex_id, edge_id, 1.0); // Default weight of 1.0
    }
    
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::validation::*;
    use super::hashing::*;
    use super::serialization::*;
    use super::conversion::*;
    use super::performance::*;
    use super::memory::*;
    
    #[test]
    fn test_validation_functions() {
        // Test entity ID validation
        let valid_id = EntityId::new();
        assert!(validate_entity_id(&valid_id).is_ok());
        
        // Test vertex type validation
        assert!(validate_vertex_type("ValidType").is_ok());
        assert!(validate_vertex_type("").is_err());
        assert!(validate_vertex_type("a".repeat(65).as_str()).is_err());
        assert!(validate_vertex_type("Invalid-Type").is_err());
        
        // Test edge type validation
        assert!(validate_edge_type("VALID_TYPE").is_ok());
        assert!(validate_edge_type("").is_err());
        
        // Test property name validation
        assert!(validate_property_name("valid_property").is_ok());
        assert!(validate_property_name("_private").is_ok());
        assert!(validate_property_name("123invalid").is_err());
        assert!(validate_property_name("").is_err());
    }
    
    #[test]
    fn test_hashing_functions() {
        let id = EntityId::new();
        let hash1 = hash_entity_id(&id);
        let hash2 = hash_entity_id(&id);
        assert_eq!(hash1, hash2); // Should be deterministic
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let props_hash = hash_property_map(&props);
        assert!(props_hash != 0);
        
        let node = ReifiedNode::new("Person", props.clone());
        let node_hash = hash_node(&node);
        assert!(node_hash != 0);
    }
    
    #[test]
    fn test_serialization_functions() {
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let json = serialize_properties(&props).unwrap();
        let deserialized = deserialize_properties(&json).unwrap();
        
        assert_eq!(props.len(), deserialized.len());
        assert!(deserialized.contains_key("name"));
        assert!(deserialized.contains_key("age"));
        
        let node = ReifiedNode::new("Person", props);
        let node_json = serialize_node(&node).unwrap();
        let deserialized_node = deserialize_node(&node_json).unwrap();
        
        assert_eq!(node.node_type, deserialized_node.node_type);
        assert_eq!(node.properties.len(), deserialized_node.properties.len());
    }
    
    #[test]
    fn test_conversion_functions() {
        let now = SystemTime::now();
        let us = system_time_to_us(now);
        let back_to_time = us_to_system_time(us);
        
        // Should be very close (within a millisecond)
        let diff = now.duration_since(back_to_time).unwrap_or_else(|_| back_to_time.duration_since(now).unwrap());
        assert!(diff.as_millis() < 1);
    }
    
    #[test]
    fn test_performance_counter() {
        let counter = PerformanceCounter::default();
        
        // Record some operations
        counter.record_operation(1000);
        counter.record_operation(2000);
        counter.record_operation(1500);
        
        assert_eq!(counter.total_ops.load(std::sync::atomic::Ordering::Relaxed), 3);
        assert_eq!(counter.total_time_us.load(std::sync::atomic::Ordering::Relaxed), 4500);
        assert_eq!(counter.avg_operation_time_us(), 1500.0);
        
        counter.reset();
        assert_eq!(counter.total_ops.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_memory_estimation() {
        let value = PropertyValue::String("test".to_string());
        let size = estimate_property_value_size(&value);
        assert!(size > 4); // At least the string length + overhead
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let props_size = estimate_property_map_size(&props);
        assert!(props_size > 0);
        
        let node = ReifiedNode::new("Person", props);
        let node_size = estimate_node_size(&node);
        assert!(node_size > 0);
    }
    
    #[test]
    fn test_utility_functions() {
        let timestamp = timestamp_us();
        assert!(timestamp > 0);
        
        let id = generate_entity_id();
        assert_ne!(id.to_string(), "");
        
        let data = b"test data";
        let det_id1 = generate_deterministic_entity_id(data);
        let det_id2 = generate_deterministic_entity_id(data);
        assert_eq!(det_id1, det_id2); // Should be deterministic
        
        let props1 = crate::properties!("a" => 1);
        let props2 = crate::properties!("b" => 2);
        let merged = merge_properties(props1, props2);
        assert_eq!(merged.len(), 2);
    }
    
    #[test]
    fn test_property_equality() {
        let float1 = PropertyValue::Float(1.0);
        let float2 = PropertyValue::Float(1.0 + f64::EPSILON / 2.0);
        assert!(property_values_equal(&float1, &float2));
        
        let string1 = PropertyValue::String("test".to_string());
        let string2 = PropertyValue::String("test".to_string());
        assert!(property_values_equal(&string1, &string2));
        
        let string3 = PropertyValue::String("different".to_string());
        assert!(!property_values_equal(&string1, &string3));
    }
    
    #[test]
    fn test_format_properties() {
        let empty_props = PropertyMap::new();
        assert_eq!(format_properties(&empty_props), "{}");
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let formatted = format_properties(&props);
        assert!(formatted.contains("name"));
        assert!(formatted.contains("age"));
        assert!(formatted.contains("Alice"));
    }
    
    #[test]
    fn test_create_simple_path() {
        let vertices = vec![EntityId::new(), EntityId::new(), EntityId::new()];
        let edges = vec![EntityId::new(), EntityId::new()];
        
        let path = create_simple_path(vertices.clone(), edges);
        
        assert_eq!(path.vertices.len(), 3);
        assert_eq!(path.edges.len(), 2);
        assert_eq!(path.weight, 3.0); // 3 vertices * 1.0 weight each
    }
}