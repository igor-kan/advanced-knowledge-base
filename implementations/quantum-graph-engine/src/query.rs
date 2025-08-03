//! High-performance query engine for graph pattern matching and traversals
//!
//! This module provides:
//! - Lightning-fast graph pattern matching
//! - Optimized path finding algorithms  
//! - Complex query optimization
//! - Parallel query execution

use crate::types::*;
use crate::storage::QuantumGraph;
use crate::{Error, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;
use std::sync::Arc;
use tokio::sync::RwLock;

/// High-performance query engine
pub struct QueryEngine {
    graph: Arc<QuantumGraph>,
    query_cache: Arc<RwLock<QueryCache>>,
    optimizer: QueryOptimizer,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new(graph: Arc<QuantumGraph>) -> Self {
        Self {
            graph,
            query_cache: Arc::new(RwLock::new(QueryCache::new())),
            optimizer: QueryOptimizer::new(),
        }
    }
    
    /// Find shortest path between two nodes
    pub async fn find_shortest_path(
        &self,
        from: NodeId,
        to: NodeId,
        config: PathConfig,
    ) -> Result<Option<Path>> {
        let start = std::time::Instant::now();
        
        // Check cache first
        let cache_key = format!("shortest_path_{}_{}", from, to);
        if let Some(cached_path) = self.query_cache.read().await.get(&cache_key) {
            return Ok(Some(cached_path.clone()));
        }
        
        let path = match config.algorithm {
            PathAlgorithm::Dijkstra => self.dijkstra(from, to, config.max_depth).await?,
            PathAlgorithm::AStar => self.a_star(from, to, config.max_depth).await?,
            PathAlgorithm::BidirectionalBFS => self.bidirectional_bfs(from, to, config.max_depth).await?,
        };
        
        // Cache the result
        if let Some(ref p) = path {
            self.query_cache.write().await.insert(cache_key, p.clone());
        }
        
        tracing::debug!("Shortest path query took {:?}", start.elapsed());
        Ok(path)
    }
    
    /// Find all paths between two nodes
    pub async fn find_all_paths(
        &self,
        from: NodeId,
        to: NodeId,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<Path>> {
        let start = std::time::Instant::now();
        let mut paths = Vec::new();
        
        self.dfs_all_paths(from, to, max_depth, max_paths, &mut paths, &mut Vec::new(), &mut HashSet::new()).await?;
        
        tracing::debug!("All paths query found {} paths in {:?}", paths.len(), start.elapsed());
        Ok(paths)
    }
    
    /// Find nodes matching a pattern
    pub async fn find_pattern(&self, query: &PatternQuery) -> Result<Vec<PatternMatch>> {
        let start = std::time::Instant::now();
        
        // Optimize query for better performance
        let optimized_query = self.optimizer.optimize(query);
        
        // Execute pattern matching
        let matches = self.execute_pattern_match(&optimized_query).await?;
        
        tracing::debug!("Pattern query found {} matches in {:?}", matches.len(), start.elapsed());
        Ok(matches)
    }
    
    /// Get k-hop neighbors of a node
    pub async fn get_k_hop_neighbors(
        &self,
        node_id: NodeId,
        k: usize,
        include_paths: bool,
    ) -> Result<KHopResult> {
        let start = std::time::Instant::now();
        
        let mut result = KHopResult::new();
        let mut visited = HashSet::new();
        let mut current_level = vec![node_id];
        
        for hop in 0..k {
            let mut next_level = Vec::new();
            
            // Process current level in parallel
            let neighbor_futures: Vec<_> = current_level.into_iter()
                .map(|node| async move {
                    self.graph.get_neighbors(node).await
                })
                .collect();
            
            let neighbor_results = futures::future::join_all(neighbor_futures).await;
            
            for neighbors_result in neighbor_results {
                let neighbors = neighbors_result?;
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        next_level.push(neighbor);
                        
                        if include_paths {
                            // Find path from source to this neighbor
                            if let Ok(Some(path)) = self.find_shortest_path(
                                node_id,
                                neighbor,
                                PathConfig::default().max_depth(hop + 1)
                            ).await {
                                result.add_neighbor_with_path(hop + 1, neighbor, path);
                            } else {
                                result.add_neighbor(hop + 1, neighbor);
                            }
                        } else {
                            result.add_neighbor(hop + 1, neighbor);
                        }
                    }
                }
            }
            
            current_level = next_level;
            if current_level.is_empty() {
                break;
            }
        }
        
        tracing::debug!("K-hop neighbors ({}) query took {:?}", k, start.elapsed());
        Ok(result)
    }
    
    /// Execute a complex graph traversal
    pub async fn traverse_graph(&self, config: TraversalConfig) -> Result<TraversalResult> {
        let start = std::time::Instant::now();
        
        let result = match config.strategy {
            TraversalStrategy::BreadthFirst => self.bfs_traversal(config).await?,
            TraversalStrategy::DepthFirst => self.dfs_traversal(config).await?,
            TraversalStrategy::Random => self.random_traversal(config).await?,
            TraversalStrategy::WeightedRandom => self.weighted_random_traversal(config).await?,
        };
        
        tracing::debug!("Graph traversal took {:?}", start.elapsed());
        Ok(result)
    }
    
    /// Dijkstra's algorithm for shortest path
    async fn dijkstra(&self, from: NodeId, to: NodeId, max_depth: usize) -> Result<Option<Path>> {
        let mut distances: HashMap<NodeId, f64> = HashMap::new();
        let mut previous: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        distances.insert(from, 0.0);
        heap.push(Reverse((0.0, from)));
        
        while let Some(Reverse((dist, current))) = heap.pop() {
            if current == to {
                return Ok(Some(self.reconstruct_path(to, &previous).await?));
            }
            
            if dist > distances.get(&current).unwrap_or(&f64::INFINITY) {
                continue;
            }
            
            if previous.len() >= max_depth {
                continue;
            }
            
            let outgoing_edges = self.graph.get_outgoing_edges(current).await?;
            
            for edge_id in outgoing_edges {
                if let Some(edge) = self.graph.get_edge(edge_id).await? {
                    let neighbor = edge.to;
                    let new_dist = dist + edge.weight();
                    
                    if new_dist < distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor, new_dist);
                        previous.insert(neighbor, (current, edge_id));
                        heap.push(Reverse((new_dist, neighbor)));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// A* algorithm for shortest path with heuristic
    async fn a_star(&self, from: NodeId, to: NodeId, max_depth: usize) -> Result<Option<Path>> {
        // For now, use Dijkstra as fallback (heuristic would require node coordinates)
        self.dijkstra(from, to, max_depth).await
    }
    
    /// Bidirectional BFS for shortest path
    async fn bidirectional_bfs(&self, from: NodeId, to: NodeId, max_depth: usize) -> Result<Option<Path>> {
        let mut forward_visited = HashMap::new();
        let mut backward_visited = HashMap::new();
        let mut forward_queue = VecDeque::new();
        let mut backward_queue = VecDeque::new();
        
        forward_visited.insert(from, (None, None)); // (previous_node, edge_id)
        backward_visited.insert(to, (None, None));
        forward_queue.push_back((from, 0));
        backward_queue.push_back((to, 0));
        
        while !forward_queue.is_empty() || !backward_queue.is_empty() {
            // Expand forward search
            if let Some((current, depth)) = forward_queue.pop_front() {
                if depth >= max_depth / 2 {
                    continue;
                }
                
                let neighbors = self.graph.get_neighbors(current).await?;
                for neighbor in neighbors {
                    if backward_visited.contains_key(&neighbor) {
                        // Found intersection - reconstruct path
                        return Ok(Some(self.reconstruct_bidirectional_path(
                            from, to, neighbor, &forward_visited, &backward_visited
                        ).await?));
                    }
                    
                    if !forward_visited.contains_key(&neighbor) {
                        forward_visited.insert(neighbor, (Some(current), None)); // TODO: Add edge_id
                        forward_queue.push_back((neighbor, depth + 1));
                    }
                }
            }
            
            // Expand backward search
            if let Some((current, depth)) = backward_queue.pop_front() {
                if depth >= max_depth / 2 {
                    continue;
                }
                
                let neighbors = self.graph.get_neighbors(current).await?;
                for neighbor in neighbors {
                    if forward_visited.contains_key(&neighbor) {
                        // Found intersection - reconstruct path
                        return Ok(Some(self.reconstruct_bidirectional_path(
                            from, to, neighbor, &forward_visited, &backward_visited
                        ).await?));
                    }
                    
                    if !backward_visited.contains_key(&neighbor) {
                        backward_visited.insert(neighbor, (Some(current), None)); // TODO: Add edge_id
                        backward_queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// DFS to find all paths
    async fn dfs_all_paths(
        &self,
        current: NodeId,
        target: NodeId,
        max_depth: usize,
        max_paths: usize,
        paths: &mut Vec<Path>,
        current_path: &mut Vec<NodeId>,
        visited: &mut HashSet<NodeId>,
    ) -> Result<()> {
        if paths.len() >= max_paths || current_path.len() >= max_depth {
            return Ok(());
        }
        
        current_path.push(current);
        visited.insert(current);
        
        if current == target {
            let mut path = Path::new();
            for &node in current_path {
                path.add_hop(node, None, 1.0); // TODO: Add actual edge weights
            }
            paths.push(path);
        } else {
            let neighbors = self.graph.get_neighbors(current).await?;
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    self.dfs_all_paths(target, neighbor, max_depth, max_paths, paths, current_path, visited).await?;
                }
            }
        }
        
        current_path.pop();
        visited.remove(&current);
        Ok(())
    }
    
    /// Execute pattern matching query
    async fn execute_pattern_match(&self, query: &PatternQuery) -> Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        
        // Start with the most selective node pattern
        let start_pattern = self.optimizer.find_most_selective_pattern(query);
        
        // Find candidate nodes for the start pattern
        let candidates = self.find_candidate_nodes(&start_pattern).await?;
        
        // Try to match the full pattern from each candidate
        for candidate in candidates {
            if let Some(pattern_match) = self.match_pattern_from_node(query, candidate).await? {
                matches.push(pattern_match);
            }
        }
        
        Ok(matches)
    }
    
    /// Find candidate nodes for a pattern
    async fn find_candidate_nodes(&self, pattern: &NodePattern) -> Result<Vec<NodeId>> {
        // This is a simplified implementation
        // In a real system, this would use indexes to find matching nodes efficiently
        let stats = self.graph.get_stats();
        let mut candidates = Vec::new();
        
        // For now, return a subset of nodes (this would be optimized with proper indexing)
        for i in 0..std::cmp::min(1000, stats.node_count) {
            candidates.push(NodeId::from_u64(i));
        }
        
        Ok(candidates)
    }
    
    /// Try to match pattern starting from a specific node
    async fn match_pattern_from_node(&self, _query: &PatternQuery, _start_node: NodeId) -> Result<Option<PatternMatch>> {
        // Simplified pattern matching implementation
        // Real implementation would use sophisticated pattern matching algorithms
        Ok(Some(PatternMatch {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            score: 1.0,
        }))
    }
    
    /// Reconstruct path from Dijkstra's algorithm result
    async fn reconstruct_path(&self, target: NodeId, previous: &HashMap<NodeId, (NodeId, EdgeId)>) -> Result<Path> {
        let mut path = Path::new();
        let mut current = target;
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Trace back through the path
        while let Some((prev_node, edge_id)) = previous.get(&current) {
            nodes.push(current);
            edges.push(*edge_id);
            current = *prev_node;
        }
        nodes.push(current); // Add the start node
        
        // Reverse to get forward path
        nodes.reverse();
        edges.reverse();
        
        // Build Path object
        let mut total_weight = 0.0;
        for (i, &node) in nodes.iter().enumerate() {
            let edge_id = if i < edges.len() { Some(edges[i]) } else { None };
            let weight = if let Some(eid) = edge_id {
                if let Some(edge) = self.graph.get_edge(eid).await? {
                    edge.weight()
                } else {
                    1.0
                }
            } else {
                0.0
            };
            
            path.add_hop(node, edge_id, weight);
            total_weight += weight;
        }
        
        path.weight = total_weight;
        Ok(path)
    }
    
    /// Reconstruct path from bidirectional search
    async fn reconstruct_bidirectional_path(
        &self,
        _start: NodeId,
        _end: NodeId,
        _meeting_point: NodeId,
        _forward_visited: &HashMap<NodeId, (Option<NodeId>, Option<EdgeId>)>,
        _backward_visited: &HashMap<NodeId, (Option<NodeId>, Option<EdgeId>)>,
    ) -> Result<Path> {
        // Simplified implementation - would reconstruct the actual path
        Ok(Path::new())
    }
    
    /// BFS traversal implementation
    async fn bfs_traversal(&self, config: TraversalConfig) -> Result<TraversalResult> {
        let mut result = TraversalResult::new();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        for start_node in config.start_nodes {
            queue.push_back((start_node, 0));
            visited.insert(start_node);
        }
        
        while let Some((current, depth)) = queue.pop_front() {
            if depth >= config.max_depth {
                continue;
            }
            
            result.add_visited_node(current, depth);
            
            if config.node_filter.as_ref().map_or(true, |f| f.matches_node_id(current)) {
                let neighbors = self.graph.get_neighbors(current).await?;
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
            
            if result.visited_nodes.len() >= config.max_results {
                break;
            }
        }
        
        Ok(result)
    }
    
    /// DFS traversal implementation
    async fn dfs_traversal(&self, config: TraversalConfig) -> Result<TraversalResult> {
        let mut result = TraversalResult::new();
        let mut stack = Vec::new();
        let mut visited = HashSet::new();
        
        for start_node in config.start_nodes {
            stack.push((start_node, 0));
        }
        
        while let Some((current, depth)) = stack.pop() {
            if depth >= config.max_depth || visited.contains(&current) {
                continue;
            }
            
            visited.insert(current);
            result.add_visited_node(current, depth);
            
            if config.node_filter.as_ref().map_or(true, |f| f.matches_node_id(current)) {
                let neighbors = self.graph.get_neighbors(current).await?;
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        stack.push((neighbor, depth + 1));
                    }
                }
            }
            
            if result.visited_nodes.len() >= config.max_results {
                break;
            }
        }
        
        Ok(result)
    }
    
    /// Random traversal implementation
    async fn random_traversal(&self, _config: TraversalConfig) -> Result<TraversalResult> {
        // TODO: Implement random walk algorithm
        Ok(TraversalResult::new())
    }
    
    /// Weighted random traversal implementation
    async fn weighted_random_traversal(&self, _config: TraversalConfig) -> Result<TraversalResult> {
        // TODO: Implement weighted random walk algorithm
        Ok(TraversalResult::new())
    }
}

/// Configuration for path finding
#[derive(Debug, Clone)]
pub struct PathConfig {
    pub algorithm: PathAlgorithm,
    pub max_depth: usize,
    pub weight_function: Option<WeightFunction>,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            algorithm: PathAlgorithm::Dijkstra,
            max_depth: 10,
            weight_function: None,
        }
    }
}

impl PathConfig {
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
    
    pub fn algorithm(mut self, algorithm: PathAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

/// Path finding algorithms
#[derive(Debug, Clone, Copy)]
pub enum PathAlgorithm {
    Dijkstra,
    AStar,
    BidirectionalBFS,
}

/// Weight function for path calculations
#[derive(Debug, Clone)]
pub enum WeightFunction {
    EdgeWeight,
    Uniform,
    Custom(fn(&Edge) -> f64),
}

/// Pattern query for graph pattern matching
#[derive(Debug, Clone)]
pub struct PatternQuery {
    pub nodes: HashMap<String, NodePattern>,
    pub edges: Vec<EdgePattern>,
    pub constraints: Vec<Constraint>,
}

impl PatternQuery {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            constraints: Vec::new(),
        }
    }
    
    pub fn add_node(mut self, alias: String, pattern: NodePattern) -> Self {
        self.nodes.insert(alias, pattern);
        self
    }
    
    pub fn add_edge(mut self, pattern: EdgePattern) -> Self {
        self.edges.push(pattern);
        self
    }
}

/// Node pattern for matching
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub node_type: Option<String>,
    pub properties: HashMap<String, PropertyPattern>,
}

/// Edge pattern for matching
#[derive(Debug, Clone)]
pub struct EdgePattern {
    pub from: String,
    pub to: String,
    pub edge_type: Option<String>,
    pub properties: HashMap<String, PropertyPattern>,
}

/// Property pattern for matching
#[derive(Debug, Clone)]
pub enum PropertyPattern {
    Exact(PropertyValue),
    Range(PropertyValue, PropertyValue),
    Contains(String),
    Regex(String),
}

/// Query constraint
#[derive(Debug, Clone)]
pub enum Constraint {
    PathLength(String, String, usize, usize), // from, to, min, max
    NodeCount(usize, usize), // min, max
    EdgeCount(usize, usize), // min, max
}

/// Pattern matching result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub nodes: HashMap<String, NodeId>,
    pub edges: HashMap<String, EdgeId>,
    pub score: f64,
}

/// K-hop neighborhood result
#[derive(Debug, Clone)]
pub struct KHopResult {
    pub neighbors_by_hop: HashMap<usize, Vec<NodeId>>,
    pub paths: HashMap<NodeId, Path>,
}

impl KHopResult {
    fn new() -> Self {
        Self {
            neighbors_by_hop: HashMap::new(),
            paths: HashMap::new(),
        }
    }
    
    fn add_neighbor(&mut self, hop: usize, node: NodeId) {
        self.neighbors_by_hop.entry(hop).or_insert_with(Vec::new).push(node);
    }
    
    fn add_neighbor_with_path(&mut self, hop: usize, node: NodeId, path: Path) {
        self.add_neighbor(hop, node);
        self.paths.insert(node, path);
    }
}

/// Traversal configuration
#[derive(Debug, Clone)]
pub struct TraversalConfig {
    pub start_nodes: Vec<NodeId>,
    pub strategy: TraversalStrategy,
    pub max_depth: usize,
    pub max_results: usize,
    pub node_filter: Option<NodeFilter>,
    pub edge_filter: Option<EdgeFilter>,
}

/// Traversal strategy
#[derive(Debug, Clone, Copy)]
pub enum TraversalStrategy {
    BreadthFirst,
    DepthFirst,
    Random,
    WeightedRandom,
}

/// Node filter for traversals
#[derive(Debug, Clone)]
pub struct NodeFilter {
    pub node_type: Option<String>,
    pub properties: HashMap<String, PropertyPattern>,
}

impl NodeFilter {
    fn matches_node_id(&self, _node_id: NodeId) -> bool {
        // Simplified implementation
        true
    }
}

/// Edge filter for traversals
#[derive(Debug, Clone)]
pub struct EdgeFilter {
    pub edge_type: Option<String>,
    pub weight_range: Option<(f64, f64)>,
}

/// Traversal result
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub visited_nodes: Vec<(NodeId, usize)>, // (node, depth)
    pub edges_traversed: Vec<EdgeId>,
    pub total_distance: f64,
}

impl TraversalResult {
    fn new() -> Self {
        Self {
            visited_nodes: Vec::new(),
            edges_traversed: Vec::new(),
            total_distance: 0.0,
        }
    }
    
    fn add_visited_node(&mut self, node: NodeId, depth: usize) {
        self.visited_nodes.push((node, depth));
    }
}

/// Query optimizer for better performance
pub struct QueryOptimizer {
    statistics: Arc<RwLock<QueryStatistics>>,
}

impl QueryOptimizer {
    fn new() -> Self {
        Self {
            statistics: Arc::new(RwLock::new(QueryStatistics::new())),
        }
    }
    
    fn optimize(&self, query: &PatternQuery) -> PatternQuery {
        // Simple optimization - in practice this would be much more sophisticated
        query.clone()
    }
    
    fn find_most_selective_pattern(&self, query: &PatternQuery) -> NodePattern {
        // Return the first pattern as a simple heuristic
        query.nodes.values().next().cloned().unwrap_or_else(|| NodePattern {
            node_type: None,
            properties: HashMap::new(),
        })
    }
}

/// Query statistics for optimization
#[derive(Debug, Default)]
pub struct QueryStatistics {
    pub query_count: u64,
    pub avg_execution_time: f64,
    pub node_selectivity: HashMap<String, f64>,
}

impl QueryStatistics {
    fn new() -> Self {
        Self::default()
    }
}

/// Query cache for performance
pub struct QueryCache {
    cache: HashMap<String, Path>,
    max_size: usize,
}

impl QueryCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_size: 10000,
        }
    }
    
    fn get(&self, key: &str) -> Option<&Path> {
        self.cache.get(key)
    }
    
    fn insert(&mut self, key: String, path: Path) {
        if self.cache.len() >= self.max_size {
            // Simple eviction - remove oldest entries
            let keys_to_remove: Vec<String> = self.cache.keys()
                .take(self.max_size / 10)
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
        
        self.cache.insert(key, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::QuantumGraph;
    use crate::GraphConfig;
    
    #[tokio::test]
    async fn test_query_engine_creation() {
        let config = GraphConfig::default();
        let graph = Arc::new(QuantumGraph::new(config).await.unwrap());
        let query_engine = QueryEngine::new(graph);
        
        // Basic test to ensure query engine can be created
        assert!(query_engine.query_cache.read().await.cache.is_empty());
    }
    
    #[tokio::test]
    async fn test_path_config() {
        let config = PathConfig::default()
            .max_depth(5)
            .algorithm(PathAlgorithm::BidirectionalBFS);
        
        assert_eq!(config.max_depth, 5);
        assert!(matches!(config.algorithm, PathAlgorithm::BidirectionalBFS));
    }
    
    #[tokio::test]
    async fn test_pattern_query_builder() {
        let query = PatternQuery::new()
            .add_node("person".to_string(), NodePattern {
                node_type: Some("Person".to_string()),
                properties: HashMap::new(),
            })
            .add_edge(EdgePattern {
                from: "person".to_string(),
                to: "company".to_string(),
                edge_type: Some("WORKS_AT".to_string()),
                properties: HashMap::new(),
            });
        
        assert_eq!(query.nodes.len(), 1);
        assert_eq!(query.edges.len(), 1);
    }
}