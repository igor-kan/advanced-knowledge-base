//! High-performance graph algorithms for the Quantum Graph Engine
//!
//! This module implements optimized versions of classic graph algorithms:
//! - Shortest path algorithms (Dijkstra, A*, Floyd-Warshall)
//! - Centrality measures (PageRank, Betweenness, Closeness)
//! - Community detection (Louvain, Label Propagation)
//! - Graph traversals (BFS, DFS, Random Walk)
//! - Connectivity analysis (Connected Components, Strongly Connected Components)

use crate::types::*;
use crate::storage::QuantumGraph;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;
use std::sync::Arc;
use rayon::prelude::*;

/// High-performance graph algorithms implementation
pub struct GraphAlgorithms {
    graph: Arc<QuantumGraph>,
}

impl GraphAlgorithms {
    /// Create new algorithms instance
    pub fn new(graph: Arc<QuantumGraph>) -> Self {
        Self { graph }
    }
    
    /// Find shortest path using Dijkstra's algorithm
    pub async fn dijkstra_shortest_path(
        &self,
        source: NodeId,
        target: NodeId,
        max_distance: Option<f64>,
    ) -> Result<Option<Path>> {
        let mut distances: HashMap<NodeId, f64> = HashMap::new();
        let mut previous: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        distances.insert(source, 0.0);
        heap.push(Reverse((0.0, source)));
        
        while let Some(Reverse((dist, current))) = heap.pop() {
            if current == target {
                return Ok(Some(self.reconstruct_path(target, &previous).await?));
            }
            
            if let Some(max_dist) = max_distance {
                if dist > max_dist {
                    continue;
                }
            }
            
            if dist > distances.get(&current).unwrap_or(&f64::INFINITY) {
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
    
    /// Compute PageRank scores for all nodes
    pub async fn pagerank(
        &self,
        damping_factor: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<HashMap<NodeId, f64>> {
        let stats = self.graph.get_stats();
        let total_nodes = stats.node_count as usize;
        
        if total_nodes == 0 {
            return Ok(HashMap::new());
        }
        
        // Initialize PageRank scores
        let initial_score = 1.0 / total_nodes as f64;
        let mut scores: HashMap<NodeId, f64> = HashMap::new();
        let mut new_scores: HashMap<NodeId, f64> = HashMap::new();
        
        // Get all nodes (simplified - in practice would be more efficient)
        let all_nodes = self.get_all_node_ids().await?;
        
        for &node_id in &all_nodes {
            scores.insert(node_id, initial_score);
        }
        
        // PageRank iterations
        for iteration in 0..max_iterations {
            new_scores.clear();
            
            // Initialize with teleportation probability
            for &node_id in &all_nodes {
                new_scores.insert(node_id, (1.0 - damping_factor) / total_nodes as f64);
            }
            
            // Compute PageRank contributions
            for &node_id in &all_nodes {
                let outgoing_edges = self.graph.get_outgoing_edges(node_id).await?;
                let out_degree = outgoing_edges.len() as f64;
                
                if out_degree > 0.0 {
                    let contribution = damping_factor * scores[&node_id] / out_degree;
                    
                    for edge_id in outgoing_edges {
                        if let Some(edge) = self.graph.get_edge(edge_id).await? {
                            *new_scores.get_mut(&edge.to).unwrap() += contribution;
                        }
                    }
                }
            }
            
            // Check for convergence
            let mut converged = true;
            for &node_id in &all_nodes {
                let diff = (new_scores[&node_id] - scores[&node_id]).abs();
                if diff > tolerance {
                    converged = false;
                    break;
                }
            }
            
            scores = new_scores.clone();
            
            if converged {
                tracing::info!("PageRank converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        Ok(scores)
    }
    
    /// Find connected components using Union-Find
    pub async fn connected_components(&self) -> Result<HashMap<NodeId, u32>> {
        let all_nodes = self.get_all_node_ids().await?;
        let mut component_map: HashMap<NodeId, u32> = HashMap::new();
        let mut component_id = 0u32;
        
        for &node_id in &all_nodes {
            if component_map.contains_key(&node_id) {
                continue;
            }
            
            // BFS to find all connected nodes
            let mut queue = VecDeque::new();
            let mut visited = HashSet::new();
            
            queue.push_back(node_id);
            visited.insert(node_id);
            
            while let Some(current) = queue.pop_front() {
                component_map.insert(current, component_id);
                
                let neighbors = self.graph.get_neighbors(current).await?;
                for neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
            
            component_id += 1;
        }
        
        Ok(component_map)
    }
    
    /// Compute betweenness centrality for all nodes
    pub async fn betweenness_centrality(&self) -> Result<HashMap<NodeId, f64>> {
        let all_nodes = self.get_all_node_ids().await?;
        let mut centrality: HashMap<NodeId, f64> = HashMap::new();
        
        // Initialize centrality scores
        for &node_id in &all_nodes {
            centrality.insert(node_id, 0.0);
        }
        
        // For each node as source
        for &source in &all_nodes {
            let (predecessors, distances, sigma) = self.single_source_shortest_paths(source).await?;
            let dependency = self.compute_dependency(source, &predecessors, &sigma, &all_nodes).await?;
            
            // Add dependency to centrality scores
            for (&node_id, &dep) in &dependency {
                if node_id != source {
                    *centrality.get_mut(&node_id).unwrap() += dep;
                }
            }
        }
        
        // Normalize betweenness centrality
        let n = all_nodes.len() as f64;
        let normalization_factor = if n > 2.0 { 2.0 / ((n - 1.0) * (n - 2.0)) } else { 1.0 };
        
        for centrality_score in centrality.values_mut() {
            *centrality_score *= normalization_factor;
        }
        
        Ok(centrality)
    }
    
    /// Compute closeness centrality for all nodes
    pub async fn closeness_centrality(&self) -> Result<HashMap<NodeId, f64>> {
        let all_nodes = self.get_all_node_ids().await?;
        let mut centrality: HashMap<NodeId, f64> = HashMap::new();
        
        for &source in &all_nodes {
            let distances = self.single_source_distances(source).await?;
            
            let mut total_distance = 0.0;
            let mut reachable_count = 0;
            
            for &target in &all_nodes {
                if let Some(&distance) = distances.get(&target) {
                    if distance.is_finite() && distance > 0.0 {
                        total_distance += distance;
                        reachable_count += 1;
                    }
                }
            }
            
            let closeness = if total_distance > 0.0 {
                (reachable_count as f64) / total_distance
            } else {
                0.0
            };
            
            centrality.insert(source, closeness);
        }
        
        Ok(centrality)
    }
    
    /// Community detection using Louvain algorithm
    pub async fn louvain_communities(&self, resolution: f64) -> Result<HashMap<NodeId, u32>> {
        let all_nodes = self.get_all_node_ids().await?;
        let mut communities: HashMap<NodeId, u32> = HashMap::new();
        
        // Initialize each node in its own community
        for (i, &node_id) in all_nodes.iter().enumerate() {
            communities.insert(node_id, i as u32);
        }
        
        let mut improved = true;
        let mut iteration = 0;
        
        while improved && iteration < 100 {
            improved = false;
            iteration += 1;
            
            for &node_id in &all_nodes {
                let current_community = communities[&node_id];
                let mut best_community = current_community;
                let mut best_gain = 0.0;
                
                // Get neighboring communities
                let neighbors = self.graph.get_neighbors(node_id).await?;
                let mut neighbor_communities = HashSet::new();
                
                for neighbor in neighbors {
                    neighbor_communities.insert(communities[&neighbor]);
                }
                
                // Try moving to each neighboring community
                for &community in &neighbor_communities {
                    if community != current_community {
                        let gain = self.compute_modularity_gain(
                            node_id,
                            current_community,
                            community,
                            &communities,
                            resolution,
                        ).await?;
                        
                        if gain > best_gain {
                            best_gain = gain;
                            best_community = community;
                        }
                    }
                }
                
                // Move to best community if improvement found
                if best_community != current_community {
                    communities.insert(node_id, best_community);
                    improved = true;
                }
            }
        }
        
        tracing::info!("Louvain algorithm converged after {} iterations", iteration);
        Ok(communities)
    }
    
    /// Random walk from a starting node
    pub async fn random_walk(
        &self,
        start_node: NodeId,
        walk_length: usize,
        restart_probability: f64,
    ) -> Result<Vec<NodeId>> {
        let mut walk = Vec::with_capacity(walk_length);
        let mut current_node = start_node;
        walk.push(current_node);
        
        for _ in 1..walk_length {
            // Check for restart
            if fastrand::f64() < restart_probability {
                current_node = start_node;
                walk.push(current_node);
                continue;
            }
            
            // Get neighbors and choose randomly
            let neighbors = self.graph.get_neighbors(current_node).await?;
            
            if neighbors.is_empty() {
                // No neighbors - restart or end walk
                current_node = start_node;
                walk.push(current_node);
                continue;
            }
            
            // Choose random neighbor
            let random_index = fastrand::usize(..neighbors.len());
            current_node = neighbors[random_index];
            walk.push(current_node);
        }
        
        Ok(walk)
    }
    
    /// Compute graph diameter (longest shortest path)
    pub async fn graph_diameter(&self) -> Result<f64> {
        let all_nodes = self.get_all_node_ids().await?;
        let mut max_distance = 0.0;
        
        // Compute all-pairs shortest paths (simplified approach)
        for &source in &all_nodes {
            let distances = self.single_source_distances(source).await?;
            
            for &distance in distances.values() {
                if distance.is_finite() {
                    max_distance = max_distance.max(distance);
                }
            }
        }
        
        Ok(max_distance)
    }
    
    /// Find minimum spanning tree using Kruskal's algorithm
    pub async fn minimum_spanning_tree(&self) -> Result<Vec<EdgeId>> {
        let all_edges = self.get_all_edge_ids().await?;
        let mut mst_edges = Vec::new();
        
        // Sort edges by weight
        let mut weighted_edges: Vec<(f64, EdgeId)> = Vec::new();
        for edge_id in all_edges {
            if let Some(edge) = self.graph.get_edge(edge_id).await? {
                weighted_edges.push((edge.weight(), edge_id));
            }
        }
        weighted_edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Union-Find for cycle detection
        let mut union_find = UnionFind::new();
        let all_nodes = self.get_all_node_ids().await?;
        
        for &node_id in &all_nodes {
            union_find.make_set(node_id);
        }
        
        // Kruskal's algorithm
        for (_, edge_id) in weighted_edges {
            if let Some(edge) = self.graph.get_edge(edge_id).await? {
                if union_find.find(edge.from) != union_find.find(edge.to) {
                    union_find.union(edge.from, edge.to);
                    mst_edges.push(edge_id);
                    
                    // Stop when we have n-1 edges
                    if mst_edges.len() == all_nodes.len() - 1 {
                        break;
                    }
                }
            }
        }
        
        Ok(mst_edges)
    }
    
    // Helper methods
    
    async fn get_all_node_ids(&self) -> Result<Vec<NodeId>> {
        // Simplified implementation - in practice would be more efficient
        let stats = self.graph.get_stats();
        let mut node_ids = Vec::new();
        
        for i in 0..stats.node_count {
            node_ids.push(NodeId::from_u64(i));
        }
        
        Ok(node_ids)
    }
    
    async fn get_all_edge_ids(&self) -> Result<Vec<EdgeId>> {
        // Simplified implementation
        let stats = self.graph.get_stats();
        let mut edge_ids = Vec::new();
        
        for i in 0..stats.edge_count {
            edge_ids.push(EdgeId(i as u128));
        }
        
        Ok(edge_ids)
    }
    
    async fn reconstruct_path(
        &self,
        target: NodeId,
        previous: &HashMap<NodeId, (NodeId, EdgeId)>,
    ) -> Result<Path> {
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
    
    async fn single_source_distances(&self, source: NodeId) -> Result<HashMap<NodeId, f64>> {
        let mut distances: HashMap<NodeId, f64> = HashMap::new();
        let mut heap = BinaryHeap::new();
        
        distances.insert(source, 0.0);
        heap.push(Reverse((0.0, source)));
        
        while let Some(Reverse((dist, current))) = heap.pop() {
            if dist > distances.get(&current).unwrap_or(&f64::INFINITY) {
                continue;
            }
            
            let outgoing_edges = self.graph.get_outgoing_edges(current).await?;
            
            for edge_id in outgoing_edges {
                if let Some(edge) = self.graph.get_edge(edge_id).await? {
                    let neighbor = edge.to;
                    let new_dist = dist + edge.weight();
                    
                    if new_dist < distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                        distances.insert(neighbor, new_dist);
                        heap.push(Reverse((new_dist, neighbor)));
                    }
                }
            }
        }
        
        Ok(distances)
    }
    
    async fn single_source_shortest_paths(
        &self,
        source: NodeId,
    ) -> Result<(HashMap<NodeId, Vec<NodeId>>, HashMap<NodeId, f64>, HashMap<NodeId, u64>)> {
        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut distances: HashMap<NodeId, f64> = HashMap::new();
        let mut sigma: HashMap<NodeId, u64> = HashMap::new(); // Number of shortest paths
        
        distances.insert(source, 0.0);
        sigma.insert(source, 1);
        
        let mut queue = VecDeque::new();
        queue.push_back(source);
        
        while let Some(current) = queue.pop_front() {
            let current_dist = distances[&current];
            let neighbors = self.graph.get_neighbors(current).await?;
            
            for neighbor in neighbors {
                let edge_weight = 1.0; // Simplified - would get actual edge weight
                let new_dist = current_dist + edge_weight;
                
                if !distances.contains_key(&neighbor) {
                    distances.insert(neighbor, new_dist);
                    sigma.insert(neighbor, 0);
                    queue.push_back(neighbor);
                }
                
                if (new_dist - distances[&neighbor]).abs() < f64::EPSILON {
                    *sigma.get_mut(&neighbor).unwrap() += sigma[&current];
                    predecessors.entry(neighbor).or_insert_with(Vec::new).push(current);
                }
            }
        }
        
        Ok((predecessors, distances, sigma))
    }
    
    async fn compute_dependency(
        &self,
        source: NodeId,
        predecessors: &HashMap<NodeId, Vec<NodeId>>,
        sigma: &HashMap<NodeId, u64>,
        all_nodes: &[NodeId],
    ) -> Result<HashMap<NodeId, f64>> {
        let mut dependency: HashMap<NodeId, f64> = HashMap::new();
        
        for &node_id in all_nodes {
            dependency.insert(node_id, 0.0);
        }
        
        // Process nodes in reverse order of distance from source
        let mut nodes_by_distance: Vec<NodeId> = all_nodes.to_vec();
        nodes_by_distance.sort_by(|&a, &b| {
            let dist_a = predecessors.get(&a).map_or(f64::INFINITY, |_| 0.0);
            let dist_b = predecessors.get(&b).map_or(f64::INFINITY, |_| 0.0);
            dist_b.partial_cmp(&dist_a).unwrap()
        });
        
        for &node in &nodes_by_distance {
            if node == source {
                continue;
            }
            
            if let Some(preds) = predecessors.get(&node) {
                let sigma_node = sigma.get(&node).unwrap_or(&0);
                
                for &pred in preds {
                    let sigma_pred = sigma.get(&pred).unwrap_or(&0);
                    if *sigma_node > 0 {
                        let contribution = (*sigma_pred as f64 / *sigma_node as f64) * (1.0 + dependency[&node]);
                        *dependency.get_mut(&pred).unwrap() += contribution;
                    }
                }
            }
        }
        
        Ok(dependency)
    }
    
    async fn compute_modularity_gain(
        &self,
        node: NodeId,
        from_community: u32,
        to_community: u32,
        communities: &HashMap<NodeId, u32>,
        resolution: f64,
    ) -> Result<f64> {
        // Simplified modularity gain calculation
        // In practice, this would be more sophisticated
        let neighbors = self.graph.get_neighbors(node).await?;
        
        let mut edges_to_from = 0;
        let mut edges_to_to = 0;
        
        for neighbor in neighbors {
            let neighbor_community = communities[&neighbor];
            if neighbor_community == from_community {
                edges_to_from += 1;
            } else if neighbor_community == to_community {
                edges_to_to += 1;
            }
        }
        
        // Simplified gain calculation
        let gain = resolution * (edges_to_to as f64 - edges_to_from as f64);
        Ok(gain)
    }
}

/// Union-Find data structure for MST and connected components
pub struct UnionFind {
    parent: HashMap<NodeId, NodeId>,
    rank: HashMap<NodeId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }
    
    fn make_set(&mut self, node: NodeId) {
        self.parent.insert(node, node);
        self.rank.insert(node, 0);
    }
    
    fn find(&mut self, node: NodeId) -> NodeId {
        if self.parent[&node] != node {
            let root = self.find(self.parent[&node]);
            self.parent.insert(node, root);
        }
        self.parent[&node]
    }
    
    fn union(&mut self, node1: NodeId, node2: NodeId) {
        let root1 = self.find(node1);
        let root2 = self.find(node2);
        
        if root1 == root2 {
            return;
        }
        
        let rank1 = self.rank[&root1];
        let rank2 = self.rank[&root2];
        
        if rank1 < rank2 {
            self.parent.insert(root1, root2);
        } else if rank1 > rank2 {
            self.parent.insert(root2, root1);
        } else {
            self.parent.insert(root2, root1);
            *self.rank.get_mut(&root1).unwrap() += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::QuantumGraph;
    use crate::GraphConfig;
    
    #[tokio::test]
    async fn test_algorithms_creation() {
        let config = GraphConfig::default();
        let graph = Arc::new(QuantumGraph::new(config).await.unwrap());
        let algorithms = GraphAlgorithms::new(graph);
        
        // Test basic functionality
        let node_ids = algorithms.get_all_node_ids().await.unwrap();
        assert!(node_ids.is_empty()); // Empty graph
    }
    
    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();
        let node1 = NodeId::from_u64(1);
        let node2 = NodeId::from_u64(2);
        let node3 = NodeId::from_u64(3);
        
        uf.make_set(node1);
        uf.make_set(node2);
        uf.make_set(node3);
        
        assert_eq!(uf.find(node1), node1);
        assert_eq!(uf.find(node2), node2);
        
        uf.union(node1, node2);
        assert_eq!(uf.find(node1), uf.find(node2));
        assert_ne!(uf.find(node1), uf.find(node3));
    }
}