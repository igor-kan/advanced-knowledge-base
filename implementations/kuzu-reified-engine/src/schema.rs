//! Schema Manager for type validation and schema evolution
//!
//! This module manages graph schema definitions, validates node and relationship types,
//! and handles schema evolution for reified relationships.

use crate::types::*;
use crate::{KuzuReifiedConfig, KuzuReifiedError, Result};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};

/// Schema manager for graph type validation and evolution
pub struct SchemaManager {
    /// Configuration
    config: KuzuReifiedConfig,
    /// Node type definitions
    node_schemas: Arc<RwLock<HashMap<String, NodeSchema>>>,
    /// Relationship type definitions
    relationship_schemas: Arc<RwLock<HashMap<String, RelationshipSchema>>>,
    /// Schema validation statistics
    stats: Arc<SchemaStats>,
    /// Whether validation is enabled
    validation_enabled: bool,
    /// Schema version for migration tracking
    schema_version: Arc<RwLock<u32>>,
}

/// Node schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSchema {
    /// Node type name
    pub name: String,
    /// Required properties
    pub required_properties: HashMap<String, PropertyType>,
    /// Optional properties
    pub optional_properties: HashMap<String, PropertyType>,
    /// Property constraints
    pub constraints: Vec<PropertyConstraint>,
    /// Whether this is a reified edge node
    pub is_reified_edge: bool,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    /// Schema version
    pub version: u32,
}

/// Relationship schema definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipSchema {
    /// Relationship type name
    pub name: String,
    /// Required properties
    pub required_properties: HashMap<String, PropertyType>,
    /// Optional properties
    pub optional_properties: HashMap<String, PropertyType>,
    /// Property constraints
    pub constraints: Vec<PropertyConstraint>,
    /// Valid source node types
    pub valid_from_types: Vec<String>,
    /// Valid target node types
    pub valid_to_types: Vec<String>,
    /// Whether this is a reification connection
    pub is_reification_connection: bool,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    /// Schema version
    pub version: u32,
}

/// Property type definitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<PropertyType>),
    Object,
    Binary,
    Timestamp,
    Uuid,
    Null,
}

/// Property constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyConstraint {
    /// Minimum value for numeric types
    MinValue(f64),
    /// Maximum value for numeric types
    MaxValue(f64),
    /// Minimum length for strings/arrays
    MinLength(usize),
    /// Maximum length for strings/arrays
    MaxLength(usize),
    /// Pattern matching for strings
    Pattern(String),
    /// Enum values for restricted choices
    EnumValues(Vec<String>),
    /// Unique constraint
    Unique,
    /// Custom validation function name
    CustomValidator(String),
}

/// Schema validation statistics
#[derive(Debug, Default)]
pub struct SchemaStats {
    /// Total validations performed
    pub validations_performed: std::sync::atomic::AtomicU64,
    /// Validation failures
    pub validation_failures: std::sync::atomic::AtomicU64,
    /// Schema evolutions
    pub schema_evolutions: std::sync::atomic::AtomicU64,
    /// Node type registrations
    pub node_types_registered: std::sync::atomic::AtomicU64,
    /// Relationship type registrations
    pub relationship_types_registered: std::sync::atomic::AtomicU64,
}

impl SchemaManager {
    /// Create a new schema manager
    pub async fn new(config: &KuzuReifiedConfig) -> Result<Self> {
        let manager = Self {
            config: config.clone(),
            node_schemas: Arc::new(RwLock::new(HashMap::new())),
            relationship_schemas: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(SchemaStats::default()),
            validation_enabled: config.enable_schema_validation,
            schema_version: Arc::new(RwLock::new(1)),
        };
        
        info!("SchemaManager created with validation: {}", manager.validation_enabled);
        Ok(manager)
    }
    
    /// Initialize the schema manager with default schemas
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing schema manager...");
        
        // Create default node schemas
        self.create_default_node_schemas().await?;
        
        // Create default relationship schemas
        self.create_default_relationship_schemas().await?;
        
        info!("SchemaManager initialized with {} node types and {} relationship types",
              self.node_schemas.read().await.len(),
              self.relationship_schemas.read().await.len());
        
        Ok(())
    }
    
    /// Create a new node type schema
    pub async fn create_node_type(
        &self,
        name: impl Into<String>,
        properties: Vec<(String, String)>, // (name, type)
    ) -> Result<()> {
        let name = name.into();
        let mut required_properties = HashMap::new();
        
        for (prop_name, prop_type_str) in properties {
            let prop_type = self.parse_property_type(&prop_type_str)?;
            required_properties.insert(prop_name, prop_type);
        }
        
        let schema = NodeSchema {
            name: name.clone(),
            required_properties,
            optional_properties: HashMap::new(),
            constraints: Vec::new(),
            is_reified_edge: name == "ReifiedEdge",
            created_at: std::time::SystemTime::now(),
            version: *self.schema_version.read().await,
        };
        
        self.node_schemas.write().await.insert(name.clone(), schema);
        self.stats.node_types_registered.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        debug!("Created node type schema: {}", name);
        Ok(())
    }
    
    /// Create a new relationship type schema
    pub async fn create_relationship_type(
        &self,
        name: impl Into<String>,
        properties: Vec<(String, String)>, // (name, type)
    ) -> Result<()> {
        let name = name.into();
        let mut required_properties = HashMap::new();
        
        for (prop_name, prop_type_str) in properties {
            let prop_type = self.parse_property_type(&prop_type_str)?;
            required_properties.insert(prop_name, prop_type);
        }
        
        let schema = RelationshipSchema {
            name: name.clone(),
            required_properties,
            optional_properties: HashMap::new(),
            constraints: Vec::new(),
            valid_from_types: Vec::new(), // Allow any by default
            valid_to_types: Vec::new(),   // Allow any by default
            is_reification_connection: name == "FROM" || name == "TO",
            created_at: std::time::SystemTime::now(),
            version: *self.schema_version.read().await,
        };
        
        self.relationship_schemas.write().await.insert(name.clone(), schema);
        self.stats.relationship_types_registered.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        debug!("Created relationship type schema: {}", name);
        Ok(())
    }
    
    /// Validate a node against its schema
    pub async fn validate_node(&self, node: &Node) -> Result<()> {
        if !self.validation_enabled {
            return Ok(());
        }
        
        self.stats.validations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let schemas = self.node_schemas.read().await;
        let schema = schemas.get(&node.label)
            .ok_or_else(|| {
                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                KuzuReifiedError::SchemaError {
                    field: node.label.clone(),
                    issue: "Unknown node type".to_string(),
                }
            })?;
        
        // Validate required properties
        for (prop_name, expected_type) in &schema.required_properties {
            let actual_value = node.properties.get(prop_name)
                .ok_or_else(|| {
                    self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KuzuReifiedError::SchemaError {
                        field: prop_name.clone(),
                        issue: "Required property missing".to_string(),
                    }
                })?;
            
            self.validate_property_type(actual_value, expected_type, prop_name).await?;
        }
        
        // Validate property constraints
        for constraint in &schema.constraints {
            self.validate_constraint(&node.properties, constraint).await?;
        }
        
        // Validate optional properties if present
        for (prop_name, prop_value) in &node.properties {
            if !schema.required_properties.contains_key(prop_name) {
                if let Some(expected_type) = schema.optional_properties.get(prop_name) {
                    self.validate_property_type(prop_value, expected_type, prop_name).await?;
                }
            }
        }
        
        debug!("Validated node {} against schema {}", node.id, node.label);
        Ok(())
    }
    
    /// Validate a relationship against its schema
    pub async fn validate_relationship(&self, relationship: &Relationship) -> Result<()> {
        if !self.validation_enabled {
            return Ok(());
        }
        
        self.stats.validations_performed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        let schemas = self.relationship_schemas.read().await;
        let schema = schemas.get(&relationship.rel_type)
            .ok_or_else(|| {
                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                KuzuReifiedError::SchemaError {
                    field: relationship.rel_type.clone(),
                    issue: "Unknown relationship type".to_string(),
                }
            })?;
        
        // Validate required properties
        for (prop_name, expected_type) in &schema.required_properties {
            let actual_value = relationship.properties.get(prop_name)
                .ok_or_else(|| {
                    self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KuzuReifiedError::SchemaError {
                        field: prop_name.clone(),
                        issue: "Required property missing".to_string(),
                    }
                })?;
            
            self.validate_property_type(actual_value, expected_type, prop_name).await?;
        }
        
        // Validate property constraints
        for constraint in &schema.constraints {
            self.validate_constraint(&relationship.properties, constraint).await?;
        }
        
        debug!("Validated relationship {} against schema {}", relationship.id, relationship.rel_type);
        Ok(())
    }
    
    /// Get node schema by name
    pub async fn get_node_schema(&self, name: &str) -> Option<NodeSchema> {
        self.node_schemas.read().await.get(name).cloned()
    }
    
    /// Get relationship schema by name
    pub async fn get_relationship_schema(&self, name: &str) -> Option<RelationshipSchema> {
        self.relationship_schemas.read().await.get(name).cloned()
    }
    
    /// List all node type names
    pub async fn list_node_types(&self) -> Vec<String> {
        self.node_schemas.read().await.keys().cloned().collect()
    }
    
    /// List all relationship type names
    pub async fn list_relationship_types(&self) -> Vec<String> {
        self.relationship_schemas.read().await.keys().cloned().collect()
    }
    
    /// Evolve schema (add optional property to existing type)
    pub async fn add_optional_property(
        &self,
        type_name: &str,
        property_name: impl Into<String>,
        property_type: impl Into<String>,
        is_node_type: bool,
    ) -> Result<()> {
        let property_name = property_name.into();
        let prop_type = self.parse_property_type(&property_type.into())?;
        
        if is_node_type {
            let mut schemas = self.node_schemas.write().await;
            if let Some(schema) = schemas.get_mut(type_name) {
                schema.optional_properties.insert(property_name.clone(), prop_type);
                schema.version += 1;
                debug!("Added optional property {} to node type {}", property_name, type_name);
            } else {
                return Err(KuzuReifiedError::SchemaError {
                    field: type_name.to_string(),
                    issue: "Node type not found".to_string(),
                });
            }
        } else {
            let mut schemas = self.relationship_schemas.write().await;
            if let Some(schema) = schemas.get_mut(type_name) {
                schema.optional_properties.insert(property_name.clone(), prop_type);
                schema.version += 1;
                debug!("Added optional property {} to relationship type {}", property_name, type_name);
            } else {
                return Err(KuzuReifiedError::SchemaError {
                    field: type_name.to_string(),
                    issue: "Relationship type not found".to_string(),
                });
            }
        }
        
        self.stats.schema_evolutions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.schema_version.write().await += 1;
        
        Ok(())
    }
    
    /// Add constraint to a type
    pub async fn add_constraint(
        &self,
        type_name: &str,
        constraint: PropertyConstraint,
        is_node_type: bool,
    ) -> Result<()> {
        if is_node_type {
            let mut schemas = self.node_schemas.write().await;
            if let Some(schema) = schemas.get_mut(type_name) {
                schema.constraints.push(constraint);
                schema.version += 1;
                debug!("Added constraint to node type {}", type_name);
            } else {
                return Err(KuzuReifiedError::SchemaError {
                    field: type_name.to_string(),
                    issue: "Node type not found".to_string(),
                });
            }
        } else {
            let mut schemas = self.relationship_schemas.write().await;
            if let Some(schema) = schemas.get_mut(type_name) {
                schema.constraints.push(constraint);
                schema.version += 1;
                debug!("Added constraint to relationship type {}", type_name);
            } else {
                return Err(KuzuReifiedError::SchemaError {
                    field: type_name.to_string(),
                    issue: "Relationship type not found".to_string(),
                });
            }
        }
        
        self.stats.schema_evolutions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self.schema_version.write().await += 1;
        
        Ok(())
    }
    
    /// Get schema statistics
    pub fn get_stats(&self) -> &SchemaStats {
        &self.stats
    }
    
    /// Get current schema version
    pub async fn get_schema_version(&self) -> u32 {
        *self.schema_version.read().await
    }
    
    /// Export schema to JSON
    pub async fn export_schema(&self) -> Result<String> {
        #[derive(Serialize)]
        struct SchemaExport {
            version: u32,
            node_schemas: HashMap<String, NodeSchema>,
            relationship_schemas: HashMap<String, RelationshipSchema>,
        }
        
        let export = SchemaExport {
            version: *self.schema_version.read().await,
            node_schemas: self.node_schemas.read().await.clone(),
            relationship_schemas: self.relationship_schemas.read().await.clone(),
        };
        
        serde_json::to_string_pretty(&export)
            .map_err(|e| KuzuReifiedError::SerializationError(e))
    }
    
    /// Import schema from JSON
    pub async fn import_schema(&self, schema_json: &str) -> Result<()> {
        #[derive(Deserialize)]
        struct SchemaImport {
            version: u32,
            node_schemas: HashMap<String, NodeSchema>,
            relationship_schemas: HashMap<String, RelationshipSchema>,
        }
        
        let import: SchemaImport = serde_json::from_str(schema_json)
            .map_err(|e| KuzuReifiedError::SerializationError(e))?;
        
        *self.node_schemas.write().await = import.node_schemas;
        *self.relationship_schemas.write().await = import.relationship_schemas;
        *self.schema_version.write().await = import.version;
        
        info!("Imported schema version {}", import.version);
        Ok(())
    }
    
    /// Shutdown the schema manager
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down SchemaManager...");
        
        self.node_schemas.write().await.clear();
        self.relationship_schemas.write().await.clear();
        
        info!("SchemaManager shutdown complete");
        Ok(())
    }
    
    // Private helper methods
    
    /// Create default node schemas
    async fn create_default_node_schemas(&self) -> Result<()> {
        // Create generic Node schema
        self.create_node_type("Node", vec![]).await?;
        
        // Create Person schema
        self.create_node_type("Person", vec![
            ("name".to_string(), "STRING".to_string()),
        ]).await?;
        
        // Create Company schema
        self.create_node_type("Company", vec![
            ("name".to_string(), "STRING".to_string()),
        ]).await?;
        
        debug!("Created default node schemas");
        Ok(())
    }
    
    /// Create default relationship schemas
    async fn create_default_relationship_schemas(&self) -> Result<()> {
        // Create generic relationship
        self.create_relationship_type("RELATES_TO", vec![]).await?;
        
        // Create common relationships
        self.create_relationship_type("KNOWS", vec![]).await?;
        self.create_relationship_type("WORKS_FOR", vec![]).await?;
        self.create_relationship_type("CONNECTS", vec![]).await?;
        
        debug!("Created default relationship schemas");
        Ok(())
    }
    
    /// Parse property type from string
    fn parse_property_type(&self, type_str: &str) -> Result<PropertyType> {
        match type_str.to_uppercase().as_str() {
            "STRING" => Ok(PropertyType::String),
            "INTEGER" | "INT" => Ok(PropertyType::Integer),
            "FLOAT" | "DOUBLE" => Ok(PropertyType::Float),
            "BOOLEAN" | "BOOL" => Ok(PropertyType::Boolean),
            "BINARY" => Ok(PropertyType::Binary),
            "TIMESTAMP" => Ok(PropertyType::Timestamp),
            "UUID" => Ok(PropertyType::Uuid),
            "OBJECT" => Ok(PropertyType::Object),
            "NULL" => Ok(PropertyType::Null),
            s if s.starts_with("ARRAY[") && s.ends_with(']') => {
                let inner_type = &s[6..s.len()-1];
                let inner = self.parse_property_type(inner_type)?;
                Ok(PropertyType::Array(Box::new(inner)))
            }
            _ => Err(KuzuReifiedError::SchemaError {
                field: "type".to_string(),
                issue: format!("Unknown property type: {}", type_str),
            })
        }
    }
    
    /// Validate property type matches expected type
    async fn validate_property_type(
        &self,
        value: &PropertyValue,
        expected_type: &PropertyType,
        field_name: &str,
    ) -> Result<()> {
        let matches = match (value, expected_type) {
            (PropertyValue::Null, PropertyType::Null) => true,
            (PropertyValue::Bool(_), PropertyType::Boolean) => true,
            (PropertyValue::Int(_), PropertyType::Integer) => true,
            (PropertyValue::Float(_), PropertyType::Float) => true,
            (PropertyValue::String(_), PropertyType::String) => true,
            (PropertyValue::Binary(_), PropertyType::Binary) => true,
            (PropertyValue::Timestamp(_), PropertyType::Timestamp) => true,
            (PropertyValue::Uuid(_), PropertyType::Uuid) => true,
            (PropertyValue::Object(_), PropertyType::Object) => true,
            (PropertyValue::Array(arr), PropertyType::Array(expected_inner)) => {
                // Validate all array elements
                for item in arr {
                    self.validate_property_type(item, expected_inner, field_name).await?;
                }
                true
            }
            (PropertyValue::Null, _) => true, // Null is compatible with any type
            _ => false,
        };
        
        if !matches {
            self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Err(KuzuReifiedError::ValidationError {
                entity: field_name.to_string(),
                constraint: format!("Type mismatch: expected {:?}, got {:?}", expected_type, value.type_name()),
            });
        }
        
        Ok(())
    }
    
    /// Validate property constraint
    async fn validate_constraint(
        &self,
        properties: &PropertyMap,
        constraint: &PropertyConstraint,
    ) -> Result<()> {
        match constraint {
            PropertyConstraint::MinValue(min_val) => {
                // Apply to all numeric properties
                for (name, value) in properties {
                    match value {
                        PropertyValue::Int(i) => {
                            if (*i as f64) < *min_val {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Value {} is less than minimum {}", i, min_val),
                                });
                            }
                        }
                        PropertyValue::Float(f) => {
                            if *f < *min_val {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Value {} is less than minimum {}", f, min_val),
                                });
                            }
                        }
                        _ => {} // Skip non-numeric values
                    }
                }
            }
            PropertyConstraint::MaxValue(max_val) => {
                // Similar validation for maximum value
                for (name, value) in properties {
                    match value {
                        PropertyValue::Int(i) => {
                            if (*i as f64) > *max_val {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Value {} exceeds maximum {}", i, max_val),
                                });
                            }
                        }
                        PropertyValue::Float(f) => {
                            if *f > *max_val {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Value {} exceeds maximum {}", f, max_val),
                                });
                            }
                        }
                        _ => {} // Skip non-numeric values
                    }
                }
            }
            PropertyConstraint::MinLength(min_len) => {
                for (name, value) in properties {
                    match value {
                        PropertyValue::String(s) => {
                            if s.len() < *min_len {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("String length {} is less than minimum {}", s.len(), min_len),
                                });
                            }
                        }
                        PropertyValue::Array(arr) => {
                            if arr.len() < *min_len {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Array length {} is less than minimum {}", arr.len(), min_len),
                                });
                            }
                        }
                        _ => {} // Skip other types
                    }
                }
            }
            PropertyConstraint::MaxLength(max_len) => {
                for (name, value) in properties {
                    match value {
                        PropertyValue::String(s) => {
                            if s.len() > *max_len {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("String length {} exceeds maximum {}", s.len(), max_len),
                                });
                            }
                        }
                        PropertyValue::Array(arr) => {
                            if arr.len() > *max_len {
                                self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                return Err(KuzuReifiedError::ValidationError {
                                    entity: name.clone(),
                                    constraint: format!("Array length {} exceeds maximum {}", arr.len(), max_len),
                                });
                            }
                        }
                        _ => {} // Skip other types
                    }
                }
            }
            PropertyConstraint::Pattern(pattern) => {
                let regex = regex::Regex::new(pattern)
                    .map_err(|e| KuzuReifiedError::ValidationError {
                        entity: "pattern".to_string(),
                        constraint: format!("Invalid regex pattern: {}", e),
                    })?;
                
                for (name, value) in properties {
                    if let PropertyValue::String(s) = value {
                        if !regex.is_match(s) {
                            self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return Err(KuzuReifiedError::ValidationError {
                                entity: name.clone(),
                                constraint: format!("String '{}' does not match pattern '{}'", s, pattern),
                            });
                        }
                    }
                }
            }
            PropertyConstraint::EnumValues(allowed_values) => {
                for (name, value) in properties {
                    if let PropertyValue::String(s) = value {
                        if !allowed_values.contains(s) {
                            self.stats.validation_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            return Err(KuzuReifiedError::ValidationError {
                                entity: name.clone(),
                                constraint: format!("Value '{}' is not in allowed values: {:?}", s, allowed_values),
                            });
                        }
                    }
                }
            }
            PropertyConstraint::Unique => {
                // Unique constraint would need database-level validation
                // This is a placeholder for now
                debug!("Unique constraint validation not implemented yet");
            }
            PropertyConstraint::CustomValidator(validator_name) => {
                // Custom validators would be pluggable
                debug!("Custom validator '{}' not implemented yet", validator_name);
            }
        }
        
        Ok(())
    }
}

impl SchemaStats {
    /// Get validation success rate
    pub fn validation_success_rate(&self) -> f64 {
        let total = self.validations_performed.load(std::sync::atomic::Ordering::Relaxed);
        let failures = self.validation_failures.load(std::sync::atomic::Ordering::Relaxed);
        
        if total == 0 {
            0.0
        } else {
            (total - failures) as f64 / total as f64
        }
    }
    
    /// Get total type registrations
    pub fn total_types_registered(&self) -> u64 {
        self.node_types_registered.load(std::sync::atomic::Ordering::Relaxed) +
        self.relationship_types_registered.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    async fn create_test_schema_manager() -> SchemaManager {
        let config = KuzuReifiedConfig::development();
        let manager = SchemaManager::new(&config).await.unwrap();
        manager.initialize().await.unwrap();
        manager
    }
    
    #[tokio::test]
    async fn test_schema_manager_creation() {
        let manager = create_test_schema_manager().await;
        
        assert!(manager.validation_enabled);
        assert_eq!(manager.get_schema_version().await, 1);
        
        let node_types = manager.list_node_types().await;
        assert!(node_types.contains(&"Person".to_string()));
        assert!(node_types.contains(&"Company".to_string()));
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_node_type_creation() {
        let manager = create_test_schema_manager().await;
        
        manager.create_node_type("Product", vec![
            ("name".to_string(), "STRING".to_string()),
            ("price".to_string(), "FLOAT".to_string()),
        ]).await.unwrap();
        
        let schema = manager.get_node_schema("Product").await.unwrap();
        assert_eq!(schema.name, "Product");
        assert!(schema.required_properties.contains_key("name"));
        assert!(schema.required_properties.contains_key("price"));
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_relationship_type_creation() {
        let manager = create_test_schema_manager().await;
        
        manager.create_relationship_type("BOUGHT", vec![
            ("quantity".to_string(), "INTEGER".to_string()),
            ("date".to_string(), "TIMESTAMP".to_string()),
        ]).await.unwrap();
        
        let schema = manager.get_relationship_schema("BOUGHT").await.unwrap();
        assert_eq!(schema.name, "BOUGHT");
        assert!(schema.required_properties.contains_key("quantity"));
        assert!(schema.required_properties.contains_key("date"));
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_node_validation() {
        let manager = create_test_schema_manager().await;
        
        // Valid node
        let valid_node = Node::new("Person", crate::properties!("name" => "Alice"));
        assert!(manager.validate_node(&valid_node).await.is_ok());
        
        // Invalid node (missing required property)
        let invalid_node = Node::new("Person", crate::properties!("age" => 30));
        assert!(manager.validate_node(&invalid_node).await.is_err());
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_relationship_validation() {
        let manager = create_test_schema_manager().await;
        
        let from = EntityId::new();
        let to = EntityId::new();
        
        // Valid relationship
        let valid_rel = Relationship::new(from, to, "KNOWS", crate::properties!());
        assert!(manager.validate_relationship(&valid_rel).await.is_ok());
        
        // Invalid relationship (unknown type)
        let invalid_rel = Relationship::new(from, to, "UNKNOWN_TYPE", crate::properties!());
        assert!(manager.validate_relationship(&invalid_rel).await.is_err());
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_schema_evolution() {
        let manager = create_test_schema_manager().await;
        
        let initial_version = manager.get_schema_version().await;
        
        // Add optional property
        manager.add_optional_property("Person", "email", "STRING", true).await.unwrap();
        
        let new_version = manager.get_schema_version().await;
        assert!(new_version > initial_version);
        
        let schema = manager.get_node_schema("Person").await.unwrap();
        assert!(schema.optional_properties.contains_key("email"));
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_property_constraints() {
        let manager = create_test_schema_manager().await;
        
        // Add constraint
        let constraint = PropertyConstraint::MinLength(3);
        manager.add_constraint("Person", constraint, true).await.unwrap();
        
        // Test validation with constraint
        let valid_node = Node::new("Person", crate::properties!("name" => "Alice"));
        assert!(manager.validate_node(&valid_node).await.is_ok());
        
        let invalid_node = Node::new("Person", crate::properties!("name" => "Al"));
        assert!(manager.validate_node(&invalid_node).await.is_err());
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_schema_export_import() {
        let manager = create_test_schema_manager().await;
        
        // Add custom type
        manager.create_node_type("CustomType", vec![
            ("field".to_string(), "STRING".to_string()),
        ]).await.unwrap();
        
        // Export schema
        let exported = manager.export_schema().await.unwrap();
        assert!(exported.contains("CustomType"));
        
        // Create new manager and import
        let new_manager = SchemaManager::new(&KuzuReifiedConfig::development()).await.unwrap();
        new_manager.import_schema(&exported).await.unwrap();
        
        // Verify import
        let schema = new_manager.get_node_schema("CustomType").await;
        assert!(schema.is_some());
        
        manager.shutdown().await.unwrap();
        new_manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_property_type_parsing() {
        let manager = create_test_schema_manager().await;
        
        // Test various property types
        assert!(matches!(manager.parse_property_type("STRING").unwrap(), PropertyType::String));
        assert!(matches!(manager.parse_property_type("INTEGER").unwrap(), PropertyType::Integer));
        assert!(matches!(manager.parse_property_type("FLOAT").unwrap(), PropertyType::Float));
        assert!(matches!(manager.parse_property_type("BOOLEAN").unwrap(), PropertyType::Boolean));
        
        // Test array type
        if let PropertyType::Array(inner) = manager.parse_property_type("ARRAY[STRING]").unwrap() {
            assert!(matches!(**inner, PropertyType::String));
        } else {
            panic!("Expected array type");
        }
        
        // Test invalid type
        assert!(manager.parse_property_type("INVALID_TYPE").is_err());
        
        manager.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_schema_stats() {
        let manager = create_test_schema_manager().await;
        
        let stats = manager.get_stats();
        let initial_validations = stats.validations_performed.load(std::sync::atomic::Ordering::Relaxed);
        
        // Perform some validations
        let node = Node::new("Person", crate::properties!("name" => "Test"));
        let _ = manager.validate_node(&node).await;
        
        assert!(stats.validations_performed.load(std::sync::atomic::Ordering::Relaxed) > initial_validations);
        assert!(stats.validation_success_rate() > 0.0);
        assert!(stats.total_types_registered() > 0);
        
        manager.shutdown().await.unwrap();
    }
}