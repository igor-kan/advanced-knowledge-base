//! Query optimization and execution planning
//!
//! This module implements:
//! - Cost-based query optimization
//! - Execution strategy selection
//! - Join order optimization
//! - Index usage planning

use crate::{NodeId, EdgeId, GraphError, GraphResult};
use crate::graph::{CompressedSparseRow, Pattern, PatternNode, PatternEdge};
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};

/// Query planner for optimization
#[derive(Debug)]
pub struct QueryPlanner {
    /// Statistics collector for cost estimation
    stats_collector: StatisticsCollector,
    
    /// Index information for optimization
    index_registry: IndexRegistry,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new() -> Self {
        Self {
            stats_collector: StatisticsCollector::new(),
            index_registry: IndexRegistry::new(),
        }
    }

    /// Plan pattern query execution
    pub fn plan_pattern_query(
        &self,
        pattern: &Pattern,
        csr: &CompressedSparseRow,
    ) -> GraphResult<QueryPlan> {
        // Analyze pattern complexity
        let complexity = self.analyze_pattern_complexity(pattern)?;
        
        // Estimate costs for different strategies
        let cost_estimates = self.estimate_execution_costs(pattern, csr, &complexity)?;
        
        // Select optimal execution strategy
        let strategy = self.select_execution_strategy(&cost_estimates)?;
        
        // Generate optimized execution steps
        let steps = self.generate_execution_steps(pattern, &strategy, csr)?;
        
        Ok(QueryPlan {
            pattern: pattern.clone(),
            strategy,
            steps,
            estimated_cost: cost_estimates.get_cost_for_strategy(&strategy),
            complexity_score: complexity.total_score,
        })
    }

    /// Analyze pattern complexity for optimization
    fn analyze_pattern_complexity(&self, pattern: &Pattern) -> GraphResult<PatternComplexity> {
        let mut complexity = PatternComplexity::new();
        
        // Node complexity
        complexity.node_count = pattern.nodes.len();
        complexity.node_complexity_score = pattern.nodes.iter()
            .map(|node| self.calculate_node_complexity(node))
            .sum();
        
        // Edge complexity
        complexity.edge_count = pattern.edges.len();
        complexity.edge_complexity_score = pattern.edges.iter()
            .map(|edge| self.calculate_edge_complexity(edge))
            .sum();
        
        // Structural complexity
        complexity.is_acyclic = self.is_pattern_acyclic(pattern);
        complexity.max_degree = self.calculate_max_pattern_degree(pattern);
        complexity.has_constraints = !pattern.constraints.max_results.is_none() ||
                                   !pattern.constraints.timeout.is_none() ||
                                   !pattern.constraints.min_confidence.is_none();
        
        // Overall complexity score
        complexity.total_score = self.calculate_total_complexity_score(&complexity);
        
        Ok(complexity)
    }

    /// Estimate execution costs for different strategies
    fn estimate_execution_costs(
        &self,
        pattern: &Pattern,
        csr: &CompressedSparseRow,
        complexity: &PatternComplexity,
    ) -> GraphResult<CostEstimates> {
        let mut estimates = CostEstimates::new();
        
        // Sequential execution cost
        estimates.sequential_cost = self.estimate_sequential_cost(pattern, csr, complexity)?;
        
        // Parallel execution cost
        estimates.parallel_cost = self.estimate_parallel_cost(pattern, csr, complexity)?;
        
        // SIMD execution cost
        estimates.simd_cost = self.estimate_simd_cost(pattern, csr, complexity)?;
        
        // Memory usage estimates
        estimates.memory_usage = self.estimate_memory_usage(pattern, csr)?;
        
        Ok(estimates)
    }

    /// Select optimal execution strategy
    fn select_execution_strategy(&self, cost_estimates: &CostEstimates) -> GraphResult<ExecutionStrategy> {
        // Simple cost-based selection (real implementation would be more sophisticated)
        let min_cost = cost_estimates.sequential_cost
            .min(cost_estimates.parallel_cost)
            .min(cost_estimates.simd_cost);
        
        if min_cost == cost_estimates.simd_cost && cost_estimates.simd_cost > 0.0 {
            Ok(ExecutionStrategy::SIMD)
        } else if min_cost == cost_estimates.parallel_cost {
            Ok(ExecutionStrategy::Parallel)
        } else {
            Ok(ExecutionStrategy::Sequential)
        }
    }

    /// Generate optimized execution steps
    fn generate_execution_steps(
        &self,
        pattern: &Pattern,
        strategy: &ExecutionStrategy,
        csr: &CompressedSparseRow,
    ) -> GraphResult<Vec<QueryStep>> {
        let mut steps = Vec::new();
        
        match strategy {
            ExecutionStrategy::Sequential => {
                steps = self.generate_sequential_steps(pattern, csr)?;
            }
            ExecutionStrategy::Parallel => {
                steps = self.generate_parallel_steps(pattern, csr)?;
            }
            ExecutionStrategy::SIMD => {
                steps = self.generate_simd_steps(pattern, csr)?;
            }
        }
        
        // Optimize step order
        self.optimize_step_order(&mut steps, csr)?;
        
        Ok(steps)
    }

    /// Calculate node complexity score
    fn calculate_node_complexity(&self, node: &PatternNode) -> f64 {
        let mut score = 1.0;
        
        // Type filter reduces complexity
        if node.type_filter.is_some() {
            score *= 0.1;
        }
        
        // Property filters reduce complexity
        score *= 0.5_f64.powi(node.property_filters.len() as i32);
        
        score
    }

    /// Calculate edge complexity score
    fn calculate_edge_complexity(&self, edge: &PatternEdge) -> f64 {
        let mut score = 1.0;
        
        // Type filter reduces complexity
        if edge.type_filter.is_some() {
            score *= 0.2;
        }
        
        // Weight range filter reduces complexity
        if edge.weight_range.is_some() {
            score *= 0.3;
        }
        
        // Bidirectional edges are more complex
        if matches!(edge.direction, crate::graph::EdgeDirection::Both) {
            score *= 2.0;
        }
        
        score
    }

    /// Check if pattern is acyclic
    fn is_pattern_acyclic(&self, pattern: &Pattern) -> bool {
        // Simple cycle detection using DFS
        let mut graph = HashMap::new();
        
        // Build adjacency list
        for edge in &pattern.edges {
            graph.entry(edge.from.clone())
                .or_insert_with(Vec::new)
                .push(edge.to.clone());
        }
        
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for node in &pattern.nodes {
            if !visited.contains(&node.id) {
                if self.has_cycle_util(&graph, &node.id, &mut visited, &mut rec_stack) {
                    return false;
                }
            }
        }
        
        true
    }

    /// Utility function for cycle detection
    fn has_cycle_util(
        &self,
        graph: &HashMap<String, Vec<String>>,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        
        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.has_cycle_util(graph, neighbor, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(neighbor) {
                    return true;
                }
            }
        }
        
        rec_stack.remove(node);
        false
    }

    /// Calculate maximum degree in pattern
    fn calculate_max_pattern_degree(&self, pattern: &Pattern) -> usize {
        let mut degree_map = HashMap::new();
        
        for edge in &pattern.edges {
            *degree_map.entry(edge.from.clone()).or_insert(0) += 1;
            *degree_map.entry(edge.to.clone()).or_insert(0) += 1;
        }
        
        degree_map.values().max().copied().unwrap_or(0)
    }

    /// Calculate total complexity score
    fn calculate_total_complexity_score(&self, complexity: &PatternComplexity) -> f64 {
        let mut score = complexity.node_complexity_score + complexity.edge_complexity_score;
        
        // Cyclic patterns are more complex
        if !complexity.is_acyclic {
            score *= 2.0;
        }
        
        // High-degree patterns are more complex
        score *= (complexity.max_degree as f64).sqrt();
        
        // Constraints add complexity
        if complexity.has_constraints {
            score *= 1.5;
        }
        
        score
    }

    /// Estimate sequential execution cost
    fn estimate_sequential_cost(
        &self,
        pattern: &Pattern,
        csr: &CompressedSparseRow,
        complexity: &PatternComplexity,
    ) -> GraphResult<f64> {
        let node_count = csr.node_count() as f64;
        let edge_count = csr.edge_count() as f64;
        
        // Base cost: proportional to graph size and pattern complexity
        let base_cost = node_count * complexity.node_complexity_score +
                       edge_count * complexity.edge_complexity_score;
        
        // Multiply by pattern size factor
        let pattern_factor = (pattern.nodes.len() * pattern.edges.len()) as f64;
        
        Ok(base_cost * pattern_factor)
    }

    /// Estimate parallel execution cost
    fn estimate_parallel_cost(
        &self,
        pattern: &Pattern,
        csr: &CompressedSparseRow,
        complexity: &PatternComplexity,
    ) -> GraphResult<f64> {
        let sequential_cost = self.estimate_sequential_cost(pattern, csr, complexity)?;
        
        // Assume perfect parallelization with some overhead
        let thread_count = rayon::current_num_threads() as f64;
        let parallel_efficiency = 0.8; // 80% efficiency
        let overhead = 1.1; // 10% overhead
        
        Ok((sequential_cost / thread_count / parallel_efficiency) * overhead)
    }

    /// Estimate SIMD execution cost
    fn estimate_simd_cost(
        &self,
        pattern: &Pattern,
        csr: &CompressedSparseRow,
        complexity: &PatternComplexity,
    ) -> GraphResult<f64> {
        let parallel_cost = self.estimate_parallel_cost(pattern, csr, complexity)?;
        
        // SIMD provides additional speedup for vectorizable operations
        let simd_speedup = if self.is_simd_friendly(pattern) { 4.0 } else { 1.2 };
        let simd_overhead = 1.05; // 5% overhead for SIMD setup
        
        Ok((parallel_cost / simd_speedup) * simd_overhead)
    }

    /// Check if pattern is SIMD-friendly
    fn is_simd_friendly(&self, pattern: &Pattern) -> bool {
        // Simple heuristic: patterns with many similar nodes/edges benefit from SIMD
        pattern.nodes.len() >= 4 && pattern.edges.len() >= 4
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self, pattern: &Pattern, csr: &CompressedSparseRow) -> GraphResult<usize> {
        let node_memory = pattern.nodes.len() * std::mem::size_of::<NodeId>() * 1000; // Assume 1000 candidates per node
        let edge_memory = pattern.edges.len() * std::mem::size_of::<EdgeId>() * 100; // Assume 100 candidates per edge
        let working_memory = 1024 * 1024; // 1MB working memory
        
        Ok(node_memory + edge_memory + working_memory)
    }

    /// Generate sequential execution steps
    fn generate_sequential_steps(&self, pattern: &Pattern, csr: &CompressedSparseRow) -> GraphResult<Vec<QueryStep>> {
        let mut steps = Vec::new();
        
        // Step 1: Filter nodes
        for node in &pattern.nodes {
            steps.push(QueryStep {
                id: format!("filter_node_{}", node.id),
                operation: QueryOperation::NodeFilter(NodeFilter {
                    type_filter: node.type_filter.as_ref().map(|t| t.parse().unwrap_or(0)),
                    label_filter: None, // Would extract from properties
                    property_filters: node.property_filters.clone(),
                    confidence_threshold: None,
                }),
                combination: CombinationStrategy::Union,
                estimated_cost: 1.0,
                estimated_selectivity: 0.1,
            });
        }
        
        // Step 2: Traverse edges
        for edge in &pattern.edges {
            steps.push(QueryStep {
                id: format!("traverse_edge_{}_{}", edge.from, edge.to),
                operation: QueryOperation::EdgeTraversal(EdgeTraversal {
                    direction: match edge.direction {
                        crate::graph::EdgeDirection::Outgoing => TraversalDirection::Outgoing,
                        crate::graph::EdgeDirection::Incoming => TraversalDirection::Incoming,
                        crate::graph::EdgeDirection::Both => TraversalDirection::Both,
                    },
                    weight_range: edge.weight_range,
                    type_filter: edge.type_filter.clone(),
                }),
                combination: CombinationStrategy::Intersection,
                estimated_cost: 2.0,
                estimated_selectivity: 0.05,
            });
        }
        
        Ok(steps)
    }

    /// Generate parallel execution steps
    fn generate_parallel_steps(&self, pattern: &Pattern, csr: &CompressedSparseRow) -> GraphResult<Vec<QueryStep>> {
        let mut steps = self.generate_sequential_steps(pattern, csr)?;
        
        // Mark steps that can be parallelized
        for step in &mut steps {
            step.estimated_cost *= 0.3; // Assume 70% speedup from parallelization
        }
        
        Ok(steps)
    }

    /// Generate SIMD execution steps
    fn generate_simd_steps(&self, pattern: &Pattern, csr: &CompressedSparseRow) -> GraphResult<Vec<QueryStep>> {
        let mut steps = Vec::new();
        
        // Single SIMD pattern matching step
        steps.push(QueryStep {
            id: "simd_pattern_match".to_string(),
            operation: QueryOperation::PatternMatch(pattern.clone()),
            combination: CombinationStrategy::Union,
            estimated_cost: 0.5, // SIMD is faster
            estimated_selectivity: 0.01,
        });
        
        Ok(steps)
    }

    /// Optimize step execution order
    fn optimize_step_order(&self, steps: &mut Vec<QueryStep>, csr: &CompressedSparseRow) -> GraphResult<()> {
        // Sort by selectivity (most selective first) and cost
        steps.sort_by(|a, b| {
            let a_score = a.estimated_selectivity * a.estimated_cost;
            let b_score = b.estimated_selectivity * b.estimated_cost;
            a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(())
    }
}

/// Pattern complexity analysis
#[derive(Debug, Clone)]
pub struct PatternComplexity {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_complexity_score: f64,
    pub edge_complexity_score: f64,
    pub is_acyclic: bool,
    pub max_degree: usize,
    pub has_constraints: bool,
    pub total_score: f64,
}

impl PatternComplexity {
    fn new() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            node_complexity_score: 0.0,
            edge_complexity_score: 0.0,
            is_acyclic: true,
            max_degree: 0,
            has_constraints: false,
            total_score: 0.0,
        }
    }
}

/// Cost estimates for different execution strategies
#[derive(Debug, Clone)]
pub struct CostEstimates {
    pub sequential_cost: f64,
    pub parallel_cost: f64,
    pub simd_cost: f64,
    pub memory_usage: usize,
}

impl CostEstimates {
    fn new() -> Self {
        Self {
            sequential_cost: 0.0,
            parallel_cost: 0.0,
            simd_cost: 0.0,
            memory_usage: 0,
        }
    }

    fn get_cost_for_strategy(&self, strategy: &ExecutionStrategy) -> f64 {
        match strategy {
            ExecutionStrategy::Sequential => self.sequential_cost,
            ExecutionStrategy::Parallel => self.parallel_cost,
            ExecutionStrategy::SIMD => self.simd_cost,
        }
    }
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub pattern: Pattern,
    pub strategy: ExecutionStrategy,
    pub steps: Vec<QueryStep>,
    pub estimated_cost: f64,
    pub complexity_score: f64,
}

/// Execution strategy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    SIMD,
}

/// Individual query execution step
#[derive(Debug, Clone)]
pub struct QueryStep {
    pub id: String,
    pub operation: QueryOperation,
    pub combination: CombinationStrategy,
    pub estimated_cost: f64,
    pub estimated_selectivity: f64,
}

/// Query operation types
#[derive(Debug, Clone)]
pub enum QueryOperation {
    NodeFilter(NodeFilter),
    EdgeTraversal(EdgeTraversal),
    PatternMatch(Pattern),
    Aggregation(AggregationOperation),
}

/// Node filtering operation
#[derive(Debug, Clone)]
pub struct NodeFilter {
    pub type_filter: Option<u32>,
    pub label_filter: Option<String>,
    pub property_filters: HashMap<String, serde_json::Value>,
    pub confidence_threshold: Option<f64>,
}

/// Edge traversal operation
#[derive(Debug, Clone)]
pub struct EdgeTraversal {
    pub direction: TraversalDirection,
    pub weight_range: Option<(f32, f32)>,
    pub type_filter: Option<String>,
}

/// Traversal direction
#[derive(Debug, Clone, Copy)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    Both,
}

/// Aggregation operations
#[derive(Debug, Clone)]
pub enum AggregationOperation {
    Count,
    Sum(String),
    Average(String),
    GroupBy(String),
}

/// Result combination strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CombinationStrategy {
    Union,
    Intersection,
    Difference,
}

/// Statistics collector for cost estimation
#[derive(Debug)]
struct StatisticsCollector {
    // Would collect statistics about query execution
}

impl StatisticsCollector {
    fn new() -> Self {
        Self {}
    }
}

/// Index registry for optimization
#[derive(Debug)]
struct IndexRegistry {
    // Would track available indices
}

impl IndexRegistry {
    fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_planner_creation() {
        let planner = QueryPlanner::new();
        // Basic creation test
    }

    #[test]
    fn test_pattern_complexity_analysis() {
        let planner = QueryPlanner::new();
        
        let pattern = Pattern {
            nodes: vec![
                PatternNode {
                    id: "A".to_string(),
                    type_filter: Some("Person".to_string()),
                    property_filters: HashMap::new(),
                },
            ],
            edges: vec![],
            constraints: crate::graph::PatternConstraints::default(),
        };
        
        let complexity = planner.analyze_pattern_complexity(&pattern).unwrap();
        assert_eq!(complexity.node_count, 1);
        assert_eq!(complexity.edge_count, 0);
        assert!(complexity.is_acyclic);
    }

    #[test]
    fn test_execution_strategy_selection() {
        let estimates = CostEstimates {
            sequential_cost: 100.0,
            parallel_cost: 30.0,
            simd_cost: 20.0,
            memory_usage: 1024,
        };
        
        let planner = QueryPlanner::new();
        let strategy = planner.select_execution_strategy(&estimates).unwrap();
        
        assert_eq!(strategy, ExecutionStrategy::SIMD);
    }
}