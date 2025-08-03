//! Query Executor with Cypher Extensions for Reification
//!
//! This module provides an extended Cypher query executor that supports
//! reification-specific operations and optimizations.

use crate::reification::ReificationManager;
use crate::schema::SchemaManager;
use crate::types::*;
use crate::{KuzuReifiedConfig, KuzuReifiedError, Result};

use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use dashmap::DashMap;
use tracing::{debug, info, warn};
use regex::Regex;
use std::collections::HashMap;

/// Query executor with Cypher extensions for reification
pub struct QueryExecutor {
    /// Configuration
    config: KuzuReifiedConfig,
    /// Schema manager reference
    schema_manager: Arc<SchemaManager>,
    /// Reification manager reference
    reification_manager: Arc<ReificationManager>,
    /// Query plan cache
    plan_cache: Arc<DashMap<String, Arc<QueryPlan>>>,
    /// Query execution statistics
    stats: Arc<QueryStats>,
    /// Cypher extensions
    extensions: Arc<CypherExtensions>,
    /// Whether the executor is initialized
    is_initialized: Arc<RwLock<bool>>,
}

/// Extended Cypher operations for reification
pub struct CypherExtensions {
    /// Reification-specific regex patterns
    reify_pattern: Regex,
    unreify_pattern: Regex,
    reified_match_pattern: Regex,
    from_to_pattern: Regex,
}

/// Query execution plan for optimization
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Original query string
    pub query: String,
    /// Parsed query operations
    pub operations: Vec<QueryOperation>,
    /// Estimated cost
    pub estimated_cost: u64,
    /// Whether reification is involved
    pub involves_reification: bool,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
    /// Creation timestamp for cache invalidation
    pub created_at: Instant,
}

/// Individual query operations
#[derive(Debug, Clone)]
pub enum QueryOperation {
    Match {
        pattern: String,
        conditions: Vec<String>,
    },
    Create {
        pattern: String,
        properties: HashMap<String, String>,
    },
    Reify {
        from: String,
        to: String,
        rel_type: String,
        properties: HashMap<String, String>,
    },
    Unreify {
        node_pattern: String,
    },
    Return {
        fields: Vec<String>,
    },
    Where {
        condition: String,
    },
    Set {
        assignments: Vec<String>,
    },
    Delete {
        pattern: String,
        detach: bool,
    },
}

/// Query optimization hints
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    UseIndex(String),
    PreferReificationCache,
    ParallelExecution,
    MaterializeIntermediateResults,
    UseColumnStore,
}

/// Query execution result with extended metadata
pub type QueryResult = CypherResult;

impl QueryExecutor {
    /// Create a new query executor
    pub async fn new(
        config: &KuzuReifiedConfig,
        schema_manager: Arc<SchemaManager>,
        reification_manager: Arc<ReificationManager>,
    ) -> Result<Self> {
        let extensions = Arc::new(CypherExtensions::new()?);
        
        let executor = Self {
            config: config.clone(),
            schema_manager,
            reification_manager,
            plan_cache: Arc::new(DashMap::new()),
            stats: Arc::new(QueryStats::default()),
            extensions,
            is_initialized: Arc::new(RwLock::new(false)),
        };
        
        info!("QueryExecutor created with optimization enabled: {}", config.enable_optimization);
        Ok(executor)
    }
    
    /// Initialize the query executor
    pub async fn initialize(&self) -> Result<()> {
        debug!("Initializing query executor...");
        
        // Initialize any query-specific resources
        self.create_reification_views().await?;
        
        // Mark as initialized
        *self.is_initialized.write().await = true;
        
        info!("QueryExecutor initialized successfully");
        Ok(())
    }
    
    /// Execute a Cypher query with reification extensions
    pub async fn execute(&self, query: &str) -> Result<QueryResult> {
        self.ensure_initialized().await?;
        
        let start_time = Instant::now();
        
        // Parse and optimize the query
        let plan = self.get_or_create_plan(query).await?;
        
        // Execute the plan
        let result = self.execute_plan(&plan).await?;
        
        // Update statistics
        let execution_time = start_time.elapsed().as_micros() as u64;
        self.stats.execution_time_us.fetch_add(execution_time, std::sync::atomic::Ordering::Relaxed);
        self.stats.queries_executed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        debug!("Executed query in {}μs: {}", execution_time, query);
        
        Ok(result)
    }
    
    /// Execute a reification-specific query
    pub async fn execute_reify(&self, from: EntityId, to: EntityId, rel_type: &str, properties: PropertyMap) -> Result<QueryResult> {
        self.ensure_initialized().await?;
        
        let reified_id = self.reification_manager
            .reify_relationship(from, to, rel_type.to_string(), properties)
            .await?;
        
        // Return the reified node as result
        let mut result = QueryResult::empty();
        result.columns = vec!["reified_node_id".to_string()];
        result.rows = vec![vec![PropertyValue::Uuid(reified_id.as_uuid())]];
        result.stats.reified_relationships += 1;
        
        Ok(result)
    }
    
    /// Execute an unreification query
    pub async fn execute_unreify(&self, node_id: EntityId) -> Result<QueryResult> {
        self.ensure_initialized().await?;
        
        let rel_id = self.reification_manager
            .unreify_relationship(node_id)
            .await?;
        
        // Return the restored relationship as result
        let mut result = QueryResult::empty();
        result.columns = vec!["relationship_id".to_string()];
        result.rows = vec![vec![PropertyValue::Uuid(rel_id.as_uuid())]];
        
        Ok(result)
    }
    
    /// Execute a query to find reified relationships
    pub async fn execute_find_reified(&self, pattern: &str) -> Result<QueryResult> {
        self.ensure_initialized().await?;
        
        // This would be a complex query to find reified relationships matching a pattern
        let reified_rels = self.reification_manager.get_all_reified_relationships().await?;
        
        let mut result = QueryResult::empty();
        result.columns = vec![
            "node_id".to_string(),
            "original_from".to_string(),
            "original_to".to_string(),
            "original_type".to_string(),
        ];
        
        for reified in reified_rels {
            if pattern.is_empty() || reified.original_relationship.original_type.contains(pattern) {
                result.rows.push(vec![
                    PropertyValue::Uuid(reified.node.id.as_uuid()),
                    PropertyValue::Uuid(reified.original_relationship.original_from.as_uuid()),
                    PropertyValue::Uuid(reified.original_relationship.original_to.as_uuid()),
                    PropertyValue::String(reified.original_relationship.original_type.clone()),
                ]);
            }
        }
        
        result.stats.reified_relationships = result.rows.len() as u64;
        Ok(result)
    }
    
    /// Get query execution statistics
    pub fn get_stats(&self) -> &QueryStats {
        &self.stats
    }
    
    /// Clear query plan cache
    pub async fn clear_cache(&self) {
        self.plan_cache.clear();
        debug!("Query plan cache cleared");
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, f64) {
        let size = self.plan_cache.len();
        let hit_rate = if self.stats.queries_executed.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            self.stats.cache_hits.load(std::sync::atomic::Ordering::Relaxed) as f64 /
            self.stats.queries_executed.load(std::sync::atomic::Ordering::Relaxed) as f64
        } else {
            0.0
        };
        (size, hit_rate)
    }
    
    /// Shutdown the query executor
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down QueryExecutor...");
        
        // Clear caches
        self.plan_cache.clear();
        
        // Mark as not initialized
        *self.is_initialized.write().await = false;
        
        info!("QueryExecutor shutdown complete");
        Ok(())
    }
    
    // Private helper methods
    
    /// Ensure the executor is initialized
    async fn ensure_initialized(&self) -> Result<()> {
        if !*self.is_initialized.read().await {
            return Err(KuzuReifiedError::Internal {
                details: "QueryExecutor not initialized".to_string(),
            });
        }
        Ok(())
    }
    
    /// Get or create a query plan
    async fn get_or_create_plan(&self, query: &str) -> Result<Arc<QueryPlan>> {
        // Check cache first
        if let Some(cached_plan) = self.plan_cache.get(query) {
            self.stats.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(cached_plan.clone());
        }
        
        self.stats.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        // Parse and create new plan
        let plan = self.parse_and_optimize_query(query).await?;
        let plan_arc = Arc::new(plan);
        
        // Cache the plan
        self.plan_cache.insert(query.to_string(), plan_arc.clone());
        
        // Manage cache size
        self.manage_cache_size().await;
        
        Ok(plan_arc)
    }
    
    /// Parse and optimize a query
    async fn parse_and_optimize_query(&self, query: &str) -> Result<QueryPlan> {
        let mut operations = Vec::new();
        let mut involves_reification = false;
        let mut optimization_hints = Vec::new();
        
        // Check for reification extensions
        if self.extensions.reify_pattern.is_match(query) {
            involves_reification = true;
            optimization_hints.push(OptimizationHint::PreferReificationCache);
            operations.push(self.parse_reify_operation(query)?);
        } else if self.extensions.unreify_pattern.is_match(query) {
            involves_reification = true;
            operations.push(self.parse_unreify_operation(query)?);
        } else if self.extensions.reified_match_pattern.is_match(query) {
            involves_reification = true;
            optimization_hints.push(OptimizationHint::UseColumnStore);
            operations.extend(self.parse_standard_cypher(query)?);
        } else {
            // Standard Cypher query
            operations.extend(self.parse_standard_cypher(query)?);
        }
        
        // Add optimization hints based on query characteristics
        if query.contains("MATCH") && query.contains("WHERE") {
            optimization_hints.push(OptimizationHint::UseIndex("default".to_string()));
        }
        
        if query.split_whitespace().count() > 20 {
            optimization_hints.push(OptimizationHint::ParallelExecution);
        }
        
        // Estimate query cost
        let estimated_cost = self.estimate_query_cost(&operations).await;
        
        Ok(QueryPlan {
            query: query.to_string(),
            operations,
            estimated_cost,
            involves_reification,
            optimization_hints,
            created_at: Instant::now(),
        })
    }
    
    /// Execute a query plan
    async fn execute_plan(&self, plan: &QueryPlan) -> Result<QueryResult> {
        debug!("Executing plan with {} operations (cost: {})", 
               plan.operations.len(), plan.estimated_cost);
        
        let mut result = QueryResult::empty();
        let mut current_context = QueryContext::new();
        
        for operation in &plan.operations {
            match operation {
                QueryOperation::Match { pattern, conditions } => {
                    let match_result = self.execute_match(pattern, conditions, &current_context).await?;
                    self.merge_results(&mut result, match_result);
                }
                QueryOperation::Create { pattern, properties } => {
                    let create_result = self.execute_create(pattern, properties).await?;
                    self.merge_results(&mut result, create_result);
                }
                QueryOperation::Reify { from, to, rel_type, properties } => {
                    let reify_result = self.execute_reify_operation(from, to, rel_type, properties).await?;
                    self.merge_results(&mut result, reify_result);
                }
                QueryOperation::Unreify { node_pattern } => {
                    let unreify_result = self.execute_unreify_operation(node_pattern).await?;
                    self.merge_results(&mut result, unreify_result);
                }
                QueryOperation::Return { fields } => {
                    result = self.execute_return(fields, &current_context).await?;
                }
                QueryOperation::Where { condition } => {
                    result = self.execute_where(condition, result).await?;
                }
                QueryOperation::Set { assignments } => {
                    let set_result = self.execute_set(assignments, &current_context).await?;
                    self.merge_results(&mut result, set_result);
                }
                QueryOperation::Delete { pattern, detach } => {
                    let delete_result = self.execute_delete(pattern, *detach).await?;
                    self.merge_results(&mut result, delete_result);
                }
            }
        }
        
        // Apply optimizations if enabled
        if self.config.enable_optimization {
            self.apply_optimizations(&mut result, &plan.optimization_hints).await?;
        }
        
        Ok(result)
    }
    
    /// Parse reify operation
    fn parse_reify_operation(&self, query: &str) -> Result<QueryOperation> {
        // Example: REIFY (a)-[r:KNOWS]->(b) PROPERTIES {since: '2020'}
        if let Some(captures) = self.extensions.reify_pattern.captures(query) {
            let from = captures.get(1).unwrap().as_str().to_string();
            let to = captures.get(2).unwrap().as_str().to_string();
            let rel_type = captures.get(3).unwrap().as_str().to_string();
            
            // Parse properties if present
            let properties = HashMap::new(); // Simplified for now
            
            Ok(QueryOperation::Reify {
                from,
                to,
                rel_type,
                properties,
            })
        } else {
            Err(KuzuReifiedError::QueryError {
                query: query.to_string(),
                message: "Invalid REIFY syntax".to_string(),
            })
        }
    }
    
    /// Parse unreify operation
    fn parse_unreify_operation(&self, query: &str) -> Result<QueryOperation> {
        // Example: UNREIFY (n:ReifiedEdge)
        if let Some(captures) = self.extensions.unreify_pattern.captures(query) {
            let node_pattern = captures.get(1).unwrap().as_str().to_string();
            
            Ok(QueryOperation::Unreify { node_pattern })
        } else {
            Err(KuzuReifiedError::QueryError {
                query: query.to_string(),
                message: "Invalid UNREIFY syntax".to_string(),
            })
        }
    }
    
    /// Parse standard Cypher operations
    fn parse_standard_cypher(&self, query: &str) -> Result<Vec<QueryOperation>> {
        let mut operations = Vec::new();
        
        // This is a simplified parser - a real implementation would use a proper Cypher parser
        let upper_query = query.to_uppercase();
        
        if upper_query.contains("MATCH") {
            operations.push(QueryOperation::Match {
                pattern: "()".to_string(), // Simplified
                conditions: Vec::new(),
            });
        }
        
        if upper_query.contains("CREATE") {
            operations.push(QueryOperation::Create {
                pattern: "()".to_string(), // Simplified
                properties: HashMap::new(),
            });
        }
        
        if upper_query.contains("RETURN") {
            operations.push(QueryOperation::Return {
                fields: vec!["*".to_string()], // Simplified
            });
        }
        
        if upper_query.contains("WHERE") {
            operations.push(QueryOperation::Where {
                condition: "true".to_string(), // Simplified
            });
        }
        
        if upper_query.contains("SET") {
            operations.push(QueryOperation::Set {
                assignments: Vec::new(), // Simplified
            });
        }
        
        if upper_query.contains("DELETE") {
            operations.push(QueryOperation::Delete {
                pattern: "()".to_string(), // Simplified
                detach: upper_query.contains("DETACH"),
            });
        }
        
        Ok(operations)
    }
    
    /// Estimate query execution cost
    async fn estimate_query_cost(&self, operations: &[QueryOperation]) -> u64 {
        let mut cost = 0u64;
        
        for operation in operations {
            cost += match operation {
                QueryOperation::Match { .. } => 100,
                QueryOperation::Create { .. } => 50,
                QueryOperation::Reify { .. } => 200, // More expensive due to reification
                QueryOperation::Unreify { .. } => 150,
                QueryOperation::Return { .. } => 10,
                QueryOperation::Where { .. } => 75,
                QueryOperation::Set { .. } => 25,
                QueryOperation::Delete { .. } => 100,
            };
        }
        
        cost
    }
    
    /// Create reification-specific database views
    async fn create_reification_views(&self) -> Result<()> {
        debug!("Creating reification views...");
        
        // This would create database views for easier querying of reified relationships
        // For now, this is a placeholder
        
        Ok(())
    }
    
    /// Manage cache size
    async fn manage_cache_size(&self) {
        const MAX_CACHE_SIZE: usize = 1000;
        
        if self.plan_cache.len() > MAX_CACHE_SIZE {
            // Remove oldest entries
            let to_remove = self.plan_cache.len() - MAX_CACHE_SIZE;
            let mut removed = 0;
            
            // In a real implementation, you would track creation times
            let keys_to_remove: Vec<String> = self.plan_cache
                .iter()
                .take(to_remove)
                .map(|entry| entry.key().clone())
                .collect();
            
            for key in keys_to_remove {
                self.plan_cache.remove(&key);
                removed += 1;
                if removed >= to_remove {
                    break;
                }
            }
            
            debug!("Removed {} entries from query plan cache", removed);
        }
    }
    
    // Placeholder execution methods (would be implemented with actual Kuzu integration)
    
    async fn execute_match(&self, _pattern: &str, _conditions: &[String], _context: &QueryContext) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_create(&self, _pattern: &str, _properties: &HashMap<String, String>) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_reify_operation(&self, _from: &str, _to: &str, _rel_type: &str, _properties: &HashMap<String, String>) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_unreify_operation(&self, _node_pattern: &str) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_return(&self, _fields: &[String], _context: &QueryContext) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_where(&self, _condition: &str, result: QueryResult) -> Result<QueryResult> {
        Ok(result) // Pass through for now
    }
    
    async fn execute_set(&self, _assignments: &[String], _context: &QueryContext) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    async fn execute_delete(&self, _pattern: &str, _detach: bool) -> Result<QueryResult> {
        Ok(QueryResult::empty())
    }
    
    fn merge_results(&self, _target: &mut QueryResult, _source: QueryResult) {
        // Merge query results
    }
    
    async fn apply_optimizations(&self, _result: &mut QueryResult, _hints: &[OptimizationHint]) -> Result<()> {
        // Apply optimization hints
        Ok(())
    }
}

impl CypherExtensions {
    /// Create new Cypher extensions with reification patterns
    pub fn new() -> Result<Self> {
        let reify_pattern = Regex::new(r"REIFY\s+\((\w+)\)-\[.*?:(\w+)\]->\((\w+)\)")
            .map_err(|e| KuzuReifiedError::Internal {
                details: format!("Failed to compile reify regex: {}", e),
            })?;
        
        let unreify_pattern = Regex::new(r"UNREIFY\s+\((\w+):?.*?\)")
            .map_err(|e| KuzuReifiedError::Internal {
                details: format!("Failed to compile unreify regex: {}", e),
            })?;
        
        let reified_match_pattern = Regex::new(r"MATCH.*?ReifiedEdge")
            .map_err(|e| KuzuReifiedError::Internal {
                details: format!("Failed to compile reified match regex: {}", e),
            })?;
        
        let from_to_pattern = Regex::new(r"-\[:?(FROM|TO)\]-")
            .map_err(|e| KuzuReifiedError::Internal {
                details: format!("Failed to compile from-to regex: {}", e),
            })?;
        
        Ok(Self {
            reify_pattern,
            unreify_pattern,
            reified_match_pattern,
            from_to_pattern,
        })
    }
    
    /// Check if query contains reification operations
    pub fn has_reification(&self, query: &str) -> bool {
        self.reify_pattern.is_match(query) ||
        self.unreify_pattern.is_match(query) ||
        self.reified_match_pattern.is_match(query)
    }
    
    /// Check if query uses FROM/TO reification connections
    pub fn has_reification_connections(&self, query: &str) -> bool {
        self.from_to_pattern.is_match(query)
    }
}

/// Query execution context for maintaining state
#[derive(Debug, Default)]
pub struct QueryContext {
    /// Variables bound in the query
    pub variables: HashMap<String, PropertyValue>,
    /// Intermediate results
    pub intermediate_results: Vec<QueryResult>,
}

impl QueryContext {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SchemaManager;
    use crate::reification::ReificationManager;
    
    async fn create_test_query_executor() -> QueryExecutor {
        let config = KuzuReifiedConfig::development();
        let schema_manager = Arc::new(SchemaManager::new(&config).await.unwrap());
        let reification_manager = Arc::new(ReificationManager::new(&config, schema_manager.clone()).await.unwrap());
        
        let executor = QueryExecutor::new(&config, schema_manager, reification_manager).await.unwrap();
        executor.initialize().await.unwrap();
        executor
    }
    
    #[tokio::test]
    async fn test_query_executor_creation() {
        let executor = create_test_query_executor().await;
        
        assert!(executor.is_initialized.read().await.clone());
        assert_eq!(executor.plan_cache.len(), 0);
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_cypher_extensions() {
        let extensions = CypherExtensions::new().unwrap();
        
        // Test reification pattern
        assert!(extensions.has_reification("REIFY (a)-[r:KNOWS]->(b)"));
        assert!(extensions.has_reification("UNREIFY (n:ReifiedEdge)"));
        assert!(extensions.has_reification("MATCH (n:ReifiedEdge) RETURN n"));
        assert!(!extensions.has_reification("MATCH (n:Person) RETURN n"));
        
        // Test FROM/TO patterns
        assert!(extensions.has_reification_connections("MATCH (a)-[:FROM]->(r)"));
        assert!(extensions.has_reification_connections("MATCH (r)-[:TO]->(b)"));
        assert!(!extensions.has_reification_connections("MATCH (a)-[:KNOWS]->(b)"));
    }
    
    #[tokio::test]
    async fn test_query_plan_creation() {
        let executor = create_test_query_executor().await;
        
        let query = "MATCH (n:Person) RETURN n.name";
        let plan = executor.get_or_create_plan(query).await.unwrap();
        
        assert!(!plan.involves_reification);
        assert!(!plan.operations.is_empty());
        assert!(plan.estimated_cost > 0);
        
        // Test caching
        let plan2 = executor.get_or_create_plan(query).await.unwrap();
        assert_eq!(plan.query, plan2.query);
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reification_query_detection() {
        let executor = create_test_query_executor().await;
        
        let reify_query = "REIFY (a)-[r:KNOWS]->(b) PROPERTIES {since: '2020'}";
        let plan = executor.get_or_create_plan(reify_query).await.unwrap();
        
        assert!(plan.involves_reification);
        assert!(plan.optimization_hints.contains(&OptimizationHint::PreferReificationCache));
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_query_cost_estimation() {
        let executor = create_test_query_executor().await;
        
        let simple_query = "MATCH (n) RETURN n";
        let complex_query = "MATCH (a)-[r]->(b) WHERE a.name = 'test' CREATE (c:Node) SET c.prop = 'value' RETURN a, b, c";
        
        let simple_plan = executor.get_or_create_plan(simple_query).await.unwrap();
        let complex_plan = executor.get_or_create_plan(complex_query).await.unwrap();
        
        assert!(complex_plan.estimated_cost > simple_plan.estimated_cost);
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_cache_management() {
        let executor = create_test_query_executor().await;
        
        // Fill cache with queries
        for i in 0..10 {
            let query = format!("MATCH (n:Type{}) RETURN n", i);
            let _ = executor.get_or_create_plan(&query).await.unwrap();
        }
        
        assert_eq!(executor.plan_cache.len(), 10);
        
        let (cache_size, _hit_rate) = executor.get_cache_stats();
        assert_eq!(cache_size, 10);
        
        executor.clear_cache().await;
        assert_eq!(executor.plan_cache.len(), 0);
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_operation_parsing() {
        let executor = create_test_query_executor().await;
        
        // Test parsing different types of operations
        let queries = vec![
            "MATCH (n) RETURN n",
            "CREATE (n:Person {name: 'Alice'})",
            "MATCH (n) WHERE n.age > 21 RETURN n",
            "MATCH (n) SET n.updated = true",
            "MATCH (n) DELETE n",
        ];
        
        for query in queries {
            let plan = executor.get_or_create_plan(query).await.unwrap();
            assert!(!plan.operations.is_empty());
        }
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_optimization_hints() {
        let executor = create_test_query_executor().await;
        
        // Query with WHERE should suggest index usage
        let indexed_query = "MATCH (n) WHERE n.id = '123' RETURN n";
        let plan = executor.get_or_create_plan(indexed_query).await.unwrap();
        
        assert!(plan.optimization_hints.iter().any(|hint| matches!(hint, OptimizationHint::UseIndex(_))));
        
        // Long query should suggest parallel execution
        let long_query = "MATCH (a)-[r1]->(b)-[r2]->(c)-[r3]->(d) WHERE a.prop1 = 'value1' AND b.prop2 = 'value2' AND c.prop3 = 'value3' RETURN a, b, c, d";
        let plan = executor.get_or_create_plan(long_query).await.unwrap();
        
        assert!(plan.optimization_hints.contains(&OptimizationHint::ParallelExecution));
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_reify_unreify_operations() {
        let executor = create_test_query_executor().await;
        
        let from = EntityId::new();
        let to = EntityId::new();
        let properties = crate::properties!("weight" => 1.5);
        
        // Test reify operation
        let reify_result = executor.execute_reify(from, to, "CONNECTS", properties).await.unwrap();
        assert!(!reify_result.is_empty());
        assert_eq!(reify_result.stats.reified_relationships, 1);
        
        // Extract the reified node ID
        if let Some(row) = reify_result.get_row(0) {
            if let Some(PropertyValue::Uuid(uuid)) = row.get(0) {
                let reified_id = EntityId::from_uuid(*uuid);
                
                // Test unreify operation
                let unreify_result = executor.execute_unreify(reified_id).await.unwrap();
                assert!(!unreify_result.is_empty());
            }
        }
        
        executor.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_find_reified_query() {
        let executor = create_test_query_executor().await;
        
        // Test finding all reified relationships
        let result = executor.execute_find_reified("").await.unwrap();
        assert!(result.columns.contains(&"node_id".to_string()));
        assert!(result.columns.contains(&"original_type".to_string()));
        
        // Test pattern-based search
        let result = executor.execute_find_reified("KNOWS").await.unwrap();
        // Should return only relationships of type KNOWS (if any exist)
        
        executor.shutdown().await.unwrap();
    }
}