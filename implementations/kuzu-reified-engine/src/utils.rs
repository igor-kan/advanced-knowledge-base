//! Utility functions and helpers for the Kuzu reified engine
//!
//! This module provides common utilities, performance helpers, and convenience functions
//! for working with reified relationships and the Kuzu database.

use crate::types::*;
use crate::{KuzuReifiedError, Result};

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use tracing::{debug, warn};
use ahash::AHasher;
use std::hash::{Hash, Hasher};

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    use std::time::Instant;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    
    /// Performance counter for tracking operations
    #[derive(Debug)]
    pub struct PerformanceCounter {
        /// Total operations
        pub total_ops: AtomicU64,
        /// Total time in microseconds
        pub total_time_us: AtomicU64,
        /// Current operations per second
        pub ops_per_second: AtomicU64,
        /// Peak operations per second
        pub peak_ops_per_second: AtomicU64,
        /// Last update timestamp
        pub last_update: AtomicU64,
    }
    
    impl Default for PerformanceCounter {
        fn default() -> Self {
            Self {
                total_ops: AtomicU64::new(0),
                total_time_us: AtomicU64::new(0),
                ops_per_second: AtomicU64::new(0),
                peak_ops_per_second: AtomicU64::new(0),
                last_update: AtomicU64::new(0),
            }
        }
    }
    
    impl PerformanceCounter {
        /// Record an operation with its execution time
        pub fn record_operation(&self, duration_us: u64) {
            self.total_ops.fetch_add(1, Ordering::Relaxed);
            self.total_time_us.fetch_add(duration_us, Ordering::Relaxed);
            
            // Update ops per second periodically
            let now = timestamp_us();
            let last_update = self.last_update.load(Ordering::Relaxed);
            
            if now - last_update > 1_000_000 { // Update every second
                let current_ops = self.total_ops.load(Ordering::Relaxed);
                let time_diff = now - last_update;
                let ops_per_sec = if time_diff > 0 {
                    (current_ops * 1_000_000) / time_diff
                } else {
                    0
                };
                
                self.ops_per_second.store(ops_per_sec, Ordering::Relaxed);
                
                // Update peak if necessary
                let current_peak = self.peak_ops_per_second.load(Ordering::Relaxed);
                if ops_per_sec > current_peak {
                    self.peak_ops_per_second.store(ops_per_sec, Ordering::Relaxed);
                }
                
                self.last_update.store(now, Ordering::Relaxed);
            }
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
        
        /// Get current operations per second
        pub fn ops_per_second(&self) -> u64 {
            self.ops_per_second.load(Ordering::Relaxed)
        }
        
        /// Get peak operations per second
        pub fn peak_ops_per_second(&self) -> u64 {
            self.peak_ops_per_second.load(Ordering::Relaxed)
        }
        
        /// Reset all counters
        pub fn reset(&self) {
            self.total_ops.store(0, Ordering::Relaxed);
            self.total_time_us.store(0, Ordering::Relaxed);
            self.ops_per_second.store(0, Ordering::Relaxed);
            self.peak_ops_per_second.store(0, Ordering::Relaxed);
            self.last_update.store(timestamp_us(), Ordering::Relaxed);
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

/// Serialization utilities
pub mod serialization {
    use super::*;
    use serde_json;
    
    /// Serialize a property map to JSON string
    pub fn serialize_properties(properties: &PropertyMap) -> Result<String> {
        serde_json::to_string(properties)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Deserialize properties from JSON string
    pub fn deserialize_properties(json: &str) -> Result<PropertyMap> {
        serde_json::from_str(json)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Serialize a node to JSON
    pub fn serialize_node(node: &Node) -> Result<String> {
        serde_json::to_string(node)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Deserialize a node from JSON
    pub fn deserialize_node(json: &str) -> Result<Node> {
        serde_json::from_str(json)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Serialize a relationship to JSON
    pub fn serialize_relationship(relationship: &Relationship) -> Result<String> {
        serde_json::to_string(relationship)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Deserialize a relationship from JSON
    pub fn deserialize_relationship(json: &str) -> Result<Relationship> {
        serde_json::from_str(json)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Serialize query result to JSON
    pub fn serialize_result(result: &CypherResult) -> Result<String> {
        serde_json::to_string(result)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Convert property value to a more compact representation
    pub fn compact_property_value(value: &PropertyValue) -> String {
        match value {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => b.to_string(),
            PropertyValue::Int(i) => i.to_string(),
            PropertyValue::Float(f) => f.to_string(),
            PropertyValue::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            PropertyValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(compact_property_value).collect();
                format!("[{}]", items.join(","))
            }
            PropertyValue::Object(_) => "{...}".to_string(),
            PropertyValue::Binary(data) => format!("binary({})", data.len()),
            PropertyValue::Timestamp(ts) => format!("ts({})", timestamp_from_system_time(*ts)),
            PropertyValue::Uuid(uuid) => format!("uuid({})", uuid.simple()),
        }
    }
}

/// Validation utilities
pub mod validation {
    use super::*;
    
    /// Validate entity ID format
    pub fn validate_entity_id(id: &EntityId) -> Result<()> {
        // Entity IDs are UUIDs, so they're always valid if constructed properly
        if id.as_uuid().is_nil() {
            return Err(KuzuReifiedError::ValidationError {
                entity: "entity_id".to_string(),
                constraint: "Entity ID cannot be nil UUID".to_string(),
            });
        }
        Ok(())
    }
    
    /// Validate node label
    pub fn validate_node_label(label: &str) -> Result<()> {
        if label.is_empty() {
            return Err(KuzuReifiedError::ValidationError {
                entity: "node_label".to_string(),
                constraint: "Node label cannot be empty".to_string(),
            });
        }
        
        if label.len() > 64 {
            return Err(KuzuReifiedError::ValidationError {
                entity: "node_label".to_string(),
                constraint: "Node label cannot exceed 64 characters".to_string(),
            });
        }
        
        // Check for valid characters (alphanumeric and underscore)
        if !label.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(KuzuReifiedError::ValidationError {
                entity: "node_label".to_string(),
                constraint: "Node label can only contain alphanumeric characters and underscores".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate relationship type
    pub fn validate_relationship_type(rel_type: &str) -> Result<()> {
        if rel_type.is_empty() {
            return Err(KuzuReifiedError::ValidationError {
                entity: "relationship_type".to_string(),
                constraint: "Relationship type cannot be empty".to_string(),
            });
        }
        
        if rel_type.len() > 64 {
            return Err(KuzuReifiedError::ValidationError {
                entity: "relationship_type".to_string(),
                constraint: "Relationship type cannot exceed 64 characters".to_string(),
            });
        }
        
        // Check for valid characters (alphanumeric, underscore, and uppercase)
        if !rel_type.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(KuzuReifiedError::ValidationError {
                entity: "relationship_type".to_string(),
                constraint: "Relationship type can only contain alphanumeric characters and underscores".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Validate property name
    pub fn validate_property_name(name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(KuzuReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name cannot be empty".to_string(),
            });
        }
        
        if name.len() > 128 {
            return Err(KuzuReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name cannot exceed 128 characters".to_string(),
            });
        }
        
        // Property names should be valid identifiers
        if !name.chars().next().unwrap().is_alphabetic() && name.chars().next().unwrap() != '_' {
            return Err(KuzuReifiedError::ValidationError {
                entity: "property_name".to_string(),
                constraint: "Property name must start with a letter or underscore".to_string(),
            });
        }
        
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(KuzuReifiedError::ValidationError {
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
    
    /// Validate node structure
    pub fn validate_node_structure(node: &Node) -> Result<()> {
        validate_entity_id(&node.id)?;
        validate_node_label(&node.label)?;
        validate_property_map(&node.properties)?;
        Ok(())
    }
    
    /// Validate relationship structure
    pub fn validate_relationship_structure(relationship: &Relationship) -> Result<()> {
        validate_entity_id(&relationship.id)?;
        validate_entity_id(&relationship.from)?;
        validate_entity_id(&relationship.to)?;
        validate_relationship_type(&relationship.rel_type)?;
        validate_property_map(&relationship.properties)?;
        
        // Check that from and to are different
        if relationship.from == relationship.to {
            warn!("Self-referencing relationship detected: {}", relationship.id);
        }
        
        Ok(())
    }
}

/// Hash utilities for efficient lookups
pub mod hashing {
    use super::*;
    
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
                timestamp_from_system_time(*ts).hash(&mut hasher);
            }
            PropertyValue::Uuid(uuid) => {
                9u8.hash(&mut hasher);
                uuid.hash(&mut hasher);
            }
        }
        
        hasher.finish()
    }
    
    /// Hash a node for caching
    pub fn hash_node(node: &Node) -> u64 {
        let mut hasher = AHasher::default();
        hash_entity_id(&node.id).hash(&mut hasher);
        node.label.hash(&mut hasher);
        hash_property_map(&node.properties).hash(&mut hasher);
        hasher.finish()
    }
    
    /// Hash a relationship for caching
    pub fn hash_relationship(relationship: &Relationship) -> u64 {
        let mut hasher = AHasher::default();
        hash_entity_id(&relationship.id).hash(&mut hasher);
        hash_entity_id(&relationship.from).hash(&mut hasher);
        hash_entity_id(&relationship.to).hash(&mut hasher);
        relationship.rel_type.hash(&mut hasher);
        hash_property_map(&relationship.properties).hash(&mut hasher);
        hasher.finish()
    }
}

/// Conversion utilities
pub mod conversion {
    use super::*;
    
    /// Convert property value to string with type information
    pub fn property_value_to_typed_string(value: &PropertyValue) -> String {
        match value {
            PropertyValue::Null => "null".to_string(),
            PropertyValue::Bool(b) => format!("bool:{}", b),
            PropertyValue::Int(i) => format!("int:{}", i),
            PropertyValue::Float(f) => format!("float:{}", f),
            PropertyValue::String(s) => format!("string:{}", s),
            PropertyValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(property_value_to_typed_string).collect();
                format!("array:[{}]", items.join(","))
            }
            PropertyValue::Object(_) => "object:{...}".to_string(),
            PropertyValue::Binary(data) => format!("binary:{}bytes", data.len()),
            PropertyValue::Timestamp(ts) => format!("timestamp:{}", timestamp_from_system_time(*ts)),
            PropertyValue::Uuid(uuid) => format!("uuid:{}", uuid),
        }
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
    
    /// Convert reified relationship to a summary map
    pub fn reified_to_summary(reified: &ReifiedRelationship) -> PropertyMap {
        let mut summary = PropertyMap::new();
        
        summary.insert("node_id".to_string(), PropertyValue::Uuid(reified.node.id.as_uuid()));
        summary.insert("original_from".to_string(), PropertyValue::Uuid(reified.original_relationship.original_from.as_uuid()));
        summary.insert("original_to".to_string(), PropertyValue::Uuid(reified.original_relationship.original_to.as_uuid()));
        summary.insert("original_type".to_string(), PropertyValue::String(reified.original_relationship.original_type.clone()));
        summary.insert("reified_at".to_string(), PropertyValue::Timestamp(reified.original_relationship.reified_at));
        summary.insert("from_connection".to_string(), PropertyValue::Uuid(reified.from_connection.as_uuid()));
        summary.insert("to_connection".to_string(), PropertyValue::Uuid(reified.to_connection.as_uuid()));
        
        // Add node properties with prefix
        for (key, value) in &reified.node.properties {
            summary.insert(format!("node_{}", key), value.clone());
        }
        
        summary
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
        }
    }
    
    /// Estimate memory usage of a property map
    pub fn estimate_property_map_size(properties: &PropertyMap) -> usize {
        32 + properties.iter().map(|(k, v)| k.len() + estimate_property_value_size(v)).sum::<usize>()
    }
    
    /// Estimate memory usage of a node
    pub fn estimate_node_size(node: &Node) -> usize {
        16 + // EntityId
        node.label.len() + 24 + // String overhead
        estimate_property_map_size(&node.properties) +
        64 // NodeMetadata estimate
    }
    
    /// Estimate memory usage of a relationship
    pub fn estimate_relationship_size(relationship: &Relationship) -> usize {
        48 + // 3 EntityIds
        relationship.rel_type.len() + 24 + // String overhead
        estimate_property_map_size(&relationship.properties) +
        64 // RelationshipMetadata estimate
    }
    
    /// Estimate memory usage of a reified relationship
    pub fn estimate_reified_relationship_size(reified: &ReifiedRelationship) -> usize {
        estimate_node_size(&reified.node) +
        64 + // OriginalRelationshipInfo
        32 // Connection EntityIds
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

/// Convert SystemTime to microsecond timestamp
pub fn timestamp_from_system_time(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
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

/// Deep clone a property value
pub fn deep_clone_property_value(value: &PropertyValue) -> PropertyValue {
    value.clone() // PropertyValue already implements Clone properly
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

/// Sanitize a string for use in Cypher queries
pub fn sanitize_cypher_string(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Extract node IDs from a query result
pub fn extract_node_ids_from_result(result: &CypherResult, column_index: usize) -> Vec<EntityId> {
    let mut node_ids = Vec::new();
    
    for row in &result.rows {
        if let Some(value) = row.get(column_index) {
            match value {
                PropertyValue::String(s) => {
                    if let Ok(id) = EntityId::from_string(s) {
                        node_ids.push(id);
                    }
                }
                PropertyValue::Uuid(uuid) => {
                    node_ids.push(EntityId::from_uuid(*uuid));
                }
                _ => {
                    debug!("Unexpected value type for node ID extraction: {:?}", value);
                }
            }
        }
    }
    
    node_ids
}

/// Create a simple graph path from node and relationship IDs
pub fn create_simple_path(nodes: Vec<EntityId>, relationships: Vec<EntityId>) -> GraphPath {
    let mut path = GraphPath::new();
    
    for (i, node_id) in nodes.iter().enumerate() {
        let rel_id = if i < relationships.len() {
            Some(relationships[i])
        } else {
            None
        };
        
        path.add_hop(*node_id, rel_id, 1.0); // Default weight of 1.0
    }
    
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::performance::*;
    use super::validation::*;
    use super::hashing::*;
    use super::serialization::*;
    use super::conversion::*;
    use super::memory::*;
    
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
    fn test_scoped_timer() {
        let counter = PerformanceCounter::default();
        
        {
            let _timer = time_operation(&counter);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        
        assert_eq!(counter.total_ops.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(counter.total_time_us.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }
    
    #[test]
    fn test_validation_functions() {
        // Test entity ID validation
        let valid_id = EntityId::new();
        assert!(validate_entity_id(&valid_id).is_ok());
        
        // Test node label validation
        assert!(validate_node_label("ValidLabel").is_ok());
        assert!(validate_node_label("").is_err());
        assert!(validate_node_label("a".repeat(65).as_str()).is_err());
        assert!(validate_node_label("Invalid-Label").is_err());
        
        // Test relationship type validation
        assert!(validate_relationship_type("VALID_TYPE").is_ok());
        assert!(validate_relationship_type("").is_err());
        
        // Test property name validation
        assert!(validate_property_name("valid_property").is_ok());
        assert!(validate_property_name("_private").is_ok());
        assert!(validate_property_name("123invalid").is_err());
        assert!(validate_property_name("").is_err());
    }
    
    #[test]
    fn test_node_structure_validation() {
        let valid_node = Node::new("Person", crate::properties!("name" => "Alice"));
        assert!(validate_node_structure(&valid_node).is_ok());
        
        // Test with invalid label
        let mut invalid_node = valid_node.clone();
        invalid_node.label = "".to_string();
        assert!(validate_node_structure(&invalid_node).is_err());
    }
    
    #[test]
    fn test_relationship_structure_validation() {
        let from = EntityId::new();
        let to = EntityId::new();
        let valid_rel = Relationship::new(from, to, "KNOWS", crate::properties!());
        
        assert!(validate_relationship_structure(&valid_rel).is_ok());
        
        // Test self-referencing relationship (should warn but not fail)
        let self_rel = Relationship::new(from, from, "KNOWS", crate::properties!());
        assert!(validate_relationship_structure(&self_rel).is_ok());
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
        
        let node = Node::new("Person", props.clone());
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
        
        let node = Node::new("Person", props);
        let node_json = serialize_node(&node).unwrap();
        let deserialized_node = deserialize_node(&node_json).unwrap();
        
        assert_eq!(node.label, deserialized_node.label);
        assert_eq!(node.properties.len(), deserialized_node.properties.len());
    }
    
    #[test]
    fn test_conversion_functions() {
        let value = PropertyValue::String("test".to_string());
        let typed_string = property_value_to_typed_string(&value);
        assert_eq!(typed_string, "string:test");
        
        let now = SystemTime::now();
        let us = system_time_to_us(now);
        let back_to_time = us_to_system_time(us);
        
        // Should be very close (within a millisecond)
        let diff = now.duration_since(back_to_time).unwrap_or_else(|_| back_to_time.duration_since(now).unwrap());
        assert!(diff.as_millis() < 1);
    }
    
    #[test]
    fn test_memory_estimation() {
        let value = PropertyValue::String("test".to_string());
        let size = estimate_property_value_size(&value);
        assert!(size > 4); // At least the string length + overhead
        
        let props = crate::properties!("name" => "Alice", "age" => 30);
        let props_size = estimate_property_map_size(&props);
        assert!(props_size > 0);
        
        let node = Node::new("Person", props);
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
        
        let cypher_str = sanitize_cypher_string("test'string\"with\nnewlines");
        assert!(!cypher_str.contains('\''));
        assert!(!cypher_str.contains('"'));
        assert!(!cypher_str.contains('\n'));
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
    fn test_extract_node_ids() {
        let mut result = CypherResult::empty();
        result.columns = vec!["id".to_string()];
        result.rows = vec![
            vec![PropertyValue::Uuid(Uuid::new_v4())],
            vec![PropertyValue::String("invalid-uuid".to_string())],
        ];
        
        let ids = extract_node_ids_from_result(&result, 0);
        assert_eq!(ids.len(), 1); // Only the valid UUID should be extracted
    }
    
    #[test]
    fn test_create_simple_path() {
        let nodes = vec![EntityId::new(), EntityId::new(), EntityId::new()];
        let rels = vec![EntityId::new(), EntityId::new()];
        
        let path = create_simple_path(nodes.clone(), rels);
        
        assert_eq!(path.nodes.len(), 3);
        assert_eq!(path.relationships.len(), 2);
        assert_eq!(path.weight, 3.0); // 3 nodes * 1.0 weight each
        assert_eq!(path.start_node(), Some(nodes[0]));
        assert_eq!(path.end_node(), Some(nodes[2]));
    }
}