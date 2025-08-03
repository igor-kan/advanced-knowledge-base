//! SIMD-optimized operations for ultra-fast graph processing
//!
//! This module provides hand-tuned SIMD implementations for critical graph operations:
//! - Vectorized edge traversals with AVX-512
//! - Parallel node processing
//! - Batch operations with SIMD acceleration
//! - Memory-efficient data layout optimizations

use crate::types::*;
use crate::Result;
use std::arch::x86_64::*;
use std::simd::prelude::*;

/// SIMD-optimized graph operations
pub struct SimdGraphOps;

impl SimdGraphOps {
    /// Vectorized batch processing of node IDs
    #[target_feature(enable = "avx512f")]
    pub unsafe fn batch_process_nodes(node_ids: &[u128]) -> Vec<u64> {
        let mut results = Vec::with_capacity(node_ids.len());
        
        // Process 4 u128 values at a time with AVX-512
        let chunks = node_ids.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Load 4 u128 values (512 bits total)
            let node0 = chunk[0];
            let node1 = chunk[1];
            let node2 = chunk[2];
            let node3 = chunk[3];
            
            // Extract lower 64 bits for hash computation
            let hash0 = node0 as u64;
            let hash1 = node1 as u64;
            let hash2 = node2 as u64;
            let hash3 = node3 as u64;
            
            // Pack into SIMD register
            let hashes = _mm256_set_epi64x(
                hash3 as i64,
                hash2 as i64,
                hash1 as i64,
                hash0 as i64,
            );
            
            // Vectorized hash computation (simplified)
            let multiplier = _mm256_set1_epi64x(0x9e3779b97f4a7c15i64);
            let result = _mm256_mullo_epi64(hashes, multiplier);
            
            // Extract results
            let mut extracted = [0i64; 4];
            _mm256_storeu_si256(extracted.as_mut_ptr() as *mut __m256i, result);
            
            results.extend_from_slice(&[
                extracted[0] as u64,
                extracted[1] as u64,
                extracted[2] as u64,
                extracted[3] as u64,
            ]);
        }
        
        // Process remainder without SIMD
        for &node_id in remainder {
            results.push((node_id as u64).wrapping_mul(0x9e3779b97f4a7c15));
        }
        
        results
    }
    
    /// AVX-512 optimized adjacency list traversal
    #[target_feature(enable = "avx512f")]
    pub unsafe fn vectorized_adjacency_scan(
        adjacency: &[u64],
        target: u64,
    ) -> Option<usize> {
        if adjacency.is_empty() {
            return None;
        }
        
        let target_vec = _mm512_set1_epi64(target as i64);
        
        // Process 8 u64 values at a time
        let chunks = adjacency.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            // Load 8 u64 values
            let data = _mm512_loadu_si512(chunk.as_ptr() as *const _);
            
            // Compare with target
            let mask = _mm512_cmpeq_epi64_mask(data, target_vec);
            
            if mask != 0 {
                // Found match - find exact position
                let pos = mask.trailing_zeros() as usize;
                return Some(chunk_idx * 8 + pos);
            }
        }
        
        // Check remainder
        for (i, &value) in remainder.iter().enumerate() {
            if value == target {
                return Some(chunks.len() * 8 + i);
            }
        }
        
        None
    }
    
    /// Vectorized edge weight computation
    #[target_feature(enable = "avx2")]
    pub unsafe fn compute_edge_weights(
        weights: &[f32],
        multipliers: &[f32],
    ) -> Vec<f32> {
        assert_eq!(weights.len(), multipliers.len());
        let mut results = Vec::with_capacity(weights.len());
        
        // Process 8 f32 values at a time with AVX2
        let chunks = weights.len() / 8;
        
        for i in 0..chunks {
            let base_idx = i * 8;
            
            // Load weights and multipliers
            let w = _mm256_loadu_ps(&weights[base_idx]);
            let m = _mm256_loadu_ps(&multipliers[base_idx]);
            
            // Vectorized multiplication
            let result = _mm256_mul_ps(w, m);
            
            // Store results
            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), result);
            results.extend_from_slice(&temp);
        }
        
        // Process remainder
        for i in (chunks * 8)..weights.len() {
            results.push(weights[i] * multipliers[i]);
        }
        
        results
    }
    
    /// SIMD-optimized graph coloring for parallel processing
    #[target_feature(enable = "avx512f")]
    pub unsafe fn vectorized_graph_coloring(
        adjacency_matrix: &[u64],
        num_nodes: usize,
    ) -> Vec<u32> {
        let mut colors = vec![0u32; num_nodes];
        let mut available_colors = vec![true; num_nodes];
        
        for node in 0..num_nodes {
            // Reset available colors using SIMD
            let chunks = available_colors.len() / 16;
            let true_vec = _mm512_set1_epi32(1);
            
            for i in 0..chunks {
                let base_idx = i * 16;
                _mm512_storeu_si512(
                    available_colors[base_idx..].as_mut_ptr() as *mut _,
                    true_vec,
                );
            }
            
            // Mark colors used by neighbors as unavailable
            for neighbor in 0..num_nodes {
                if adjacency_matrix[node * num_nodes + neighbor] != 0 && colors[neighbor] != 0 {
                    available_colors[colors[neighbor] as usize] = false;
                }
            }
            
            // Find first available color
            for (color, &available) in available_colors.iter().enumerate() {
                if available {
                    colors[node] = color as u32;
                    break;
                }
            }
        }
        
        colors
    }
    
    /// Memory-efficient batch node comparison
    pub fn batch_node_compare(nodes_a: &[NodeId], nodes_b: &[NodeId]) -> Vec<bool> {
        assert_eq!(nodes_a.len(), nodes_b.len());
        
        if is_x86_feature_detected!("avx512f") {
            unsafe { Self::avx512_node_compare(nodes_a, nodes_b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::avx2_node_compare(nodes_a, nodes_b) }
        } else {
            // Fallback to scalar comparison
            nodes_a.iter()
                .zip(nodes_b.iter())
                .map(|(&a, &b)| a == b)
                .collect()
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_node_compare(nodes_a: &[NodeId], nodes_b: &[NodeId]) -> Vec<bool> {
        let mut results = Vec::with_capacity(nodes_a.len());
        
        // Process pairs of u128 values (each NodeId is u128)
        let chunks = nodes_a.len() / 4; // 4 u128 values per 512-bit register
        
        for i in 0..chunks {
            let base_idx = i * 4;
            
            // Load node IDs (treating u128 as two u64 values each)
            let a_low = _mm256_set_epi64x(
                (nodes_a[base_idx + 3].0) as i64,
                (nodes_a[base_idx + 2].0) as i64,
                (nodes_a[base_idx + 1].0) as i64,
                (nodes_a[base_idx].0) as i64,
            );
            
            let b_low = _mm256_set_epi64x(
                (nodes_b[base_idx + 3].0) as i64,
                (nodes_b[base_idx + 2].0) as i64,
                (nodes_b[base_idx + 1].0) as i64,
                (nodes_b[base_idx].0) as i64,
            );
            
            let a_high = _mm256_set_epi64x(
                (nodes_a[base_idx + 3].0 >> 64) as i64,
                (nodes_a[base_idx + 2].0 >> 64) as i64,
                (nodes_a[base_idx + 1].0 >> 64) as i64,
                (nodes_a[base_idx].0 >> 64) as i64,
            );
            
            let b_high = _mm256_set_epi64x(
                (nodes_b[base_idx + 3].0 >> 64) as i64,
                (nodes_b[base_idx + 2].0 >> 64) as i64,
                (nodes_b[base_idx + 1].0 >> 64) as i64,
                (nodes_b[base_idx].0 >> 64) as i64,
            );
            
            // Compare low and high parts
            let cmp_low = _mm256_cmpeq_epi64(a_low, b_low);
            let cmp_high = _mm256_cmpeq_epi64(a_high, b_high);
            
            // Combine comparisons (both parts must match)
            let result = _mm256_and_si256(cmp_low, cmp_high);
            
            // Extract comparison results
            let mask = _mm256_movemask_epi8(result);
            
            // Convert mask to boolean results
            for j in 0..4 {
                let bit_pos = j * 8; // Each comparison result is 8 bytes
                let is_equal = (mask >> bit_pos) & 0xFF == 0xFF;
                results.push(is_equal);
            }
        }
        
        // Process remainder
        for i in (chunks * 4)..nodes_a.len() {
            results.push(nodes_a[i] == nodes_b[i]);
        }
        
        results
    }
    
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_node_compare(nodes_a: &[NodeId], nodes_b: &[NodeId]) -> Vec<bool> {
        let mut results = Vec::with_capacity(nodes_a.len());
        
        // Process 2 u128 values at a time with AVX2
        let chunks = nodes_a.len() / 2;
        
        for i in 0..chunks {
            let base_idx = i * 2;
            
            // Load and compare (simplified for AVX2)
            let eq0 = nodes_a[base_idx] == nodes_b[base_idx];
            let eq1 = nodes_a[base_idx + 1] == nodes_b[base_idx + 1];
            
            results.push(eq0);
            results.push(eq1);
        }
        
        // Process remainder
        for i in (chunks * 2)..nodes_a.len() {
            results.push(nodes_a[i] == nodes_b[i]);
        }
        
        results
    }
}

/// SIMD-optimized adjacency list representation
#[repr(align(64))] // Cache line alignment for optimal performance
pub struct SimdAdjacencyList {
    /// Node IDs stored in SIMD-friendly format
    pub nodes: Vec<u64>,
    /// Edge weights for vectorized computation
    pub weights: Vec<f32>,
    /// Compressed adjacency data
    pub compressed_data: Vec<u8>,
}

impl SimdAdjacencyList {
    /// Create a new SIMD-optimized adjacency list
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            weights: Vec::new(),
            compressed_data: Vec::new(),
        }
    }
    
    /// Add multiple edges with SIMD optimization
    pub fn batch_add_edges(&mut self, edges: &[(u64, u64, f32)]) {
        // Reserve capacity for better performance
        self.nodes.reserve(edges.len() * 2);
        self.weights.reserve(edges.len());
        
        // Process in chunks for SIMD efficiency
        for chunk in edges.chunks(8) {
            for &(from, to, weight) in chunk {
                self.nodes.push(from);
                self.nodes.push(to);
                self.weights.push(weight);
            }
        }
        
        // Sort for better cache locality
        self.sort_by_node_id();
    }
    
    /// Sort adjacency list by node ID for cache efficiency
    fn sort_by_node_id(&mut self) {
        // Implementation would sort the adjacency data
        // This is a placeholder for the actual sorting logic
    }
    
    /// Find neighbors using SIMD scan
    pub fn find_neighbors(&self, node_id: u64) -> Vec<u64> {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                self.simd_find_neighbors(node_id)
            }
        } else {
            // Fallback to linear search
            self.nodes.iter()
                .enumerate()
                .step_by(2)
                .filter_map(|(i, &from)| {
                    if from == node_id {
                        self.nodes.get(i + 1).copied()
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
    
    #[target_feature(enable = "avx512f")]
    unsafe fn simd_find_neighbors(&self, node_id: u64) -> Vec<u64> {
        let mut neighbors = Vec::new();
        let target = _mm512_set1_epi64(node_id as i64);
        
        // Process pairs of nodes (from, to)
        let pair_chunks = self.nodes.chunks_exact(16); // 8 pairs per chunk
        
        for chunk in pair_chunks {
            // Load 8 "from" nodes
            let from_nodes = _mm512_loadu_si512(chunk.as_ptr() as *const _);
            
            // Compare with target
            let mask = _mm512_cmpeq_epi64_mask(from_nodes, target);
            
            if mask != 0 {
                // Found matches - extract corresponding "to" nodes
                for i in 0..8 {
                    if (mask >> i) & 1 != 0 {
                        let to_idx = i * 2 + 1;
                        if let Some(&to_node) = chunk.get(to_idx) {
                            neighbors.push(to_node);
                        }
                    }
                }
            }
        }
        
        neighbors
    }
}

impl Default for SimdAdjacencyList {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance benchmarks for SIMD operations
pub mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    /// Benchmark vectorized vs scalar node processing
    pub fn benchmark_node_processing(node_count: usize) -> (std::time::Duration, std::time::Duration) {
        let node_ids: Vec<u128> = (0..node_count as u128).collect();
        
        // Scalar benchmark
        let start = Instant::now();
        let scalar_results: Vec<u64> = node_ids.iter()
            .map(|&id| (id as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();
        let scalar_time = start.elapsed();
        
        // SIMD benchmark
        let start = Instant::now();
        let simd_results = if is_x86_feature_detected!("avx512f") {
            unsafe { SimdGraphOps::batch_process_nodes(&node_ids) }
        } else {
            scalar_results.clone()
        };
        let simd_time = start.elapsed();
        
        println!("Scalar: {:?}, SIMD: {:?}, Speedup: {:.2}x", 
                scalar_time, simd_time, 
                scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
        
        (scalar_time, simd_time)
    }
    
    /// Benchmark adjacency list scanning
    pub fn benchmark_adjacency_scan(list_size: usize, target: u64) -> (std::time::Duration, std::time::Duration) {
        let adjacency: Vec<u64> = (0..list_size as u64).collect();
        
        // Scalar search
        let start = Instant::now();
        let scalar_result = adjacency.iter().position(|&x| x == target);
        let scalar_time = start.elapsed();
        
        // SIMD search
        let start = Instant::now();
        let simd_result = if is_x86_feature_detected!("avx512f") {
            unsafe { SimdGraphOps::vectorized_adjacency_scan(&adjacency, target) }
        } else {
            scalar_result
        };
        let simd_time = start.elapsed();
        
        println!("Adjacency scan - Scalar: {:?}, SIMD: {:?}", scalar_time, simd_time);
        assert_eq!(scalar_result, simd_result);
        
        (scalar_time, simd_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_node_processing() {
        let node_ids = vec![1u128, 2, 3, 4, 5, 6, 7, 8];
        
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let results = SimdGraphOps::batch_process_nodes(&node_ids);
                assert_eq!(results.len(), node_ids.len());
                
                // Verify results match scalar computation
                for (i, &node_id) in node_ids.iter().enumerate() {
                    let expected = (node_id as u64).wrapping_mul(0x9e3779b97f4a7c15);
                    assert_eq!(results[i], expected);
                }
            }
        }
    }
    
    #[test]
    fn test_simd_adjacency_list() {
        let mut adj_list = SimdAdjacencyList::new();
        let edges = vec![
            (1, 2, 1.0),
            (1, 3, 2.0),
            (2, 4, 1.5),
            (3, 4, 0.5),
        ];
        
        adj_list.batch_add_edges(&edges);
        
        let neighbors = adj_list.find_neighbors(1);
        assert!(neighbors.contains(&2));
        assert!(neighbors.contains(&3));
    }
    
    #[test] 
    fn test_node_comparison() {
        let nodes_a = vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)];
        let nodes_b = vec![NodeId(1), NodeId(5), NodeId(3), NodeId(6)];
        
        let results = SimdGraphOps::batch_node_compare(&nodes_a, &nodes_b);
        
        assert_eq!(results, vec![true, false, true, false]);
    }
    
    #[test]
    fn test_edge_weight_computation() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let multipliers = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let results = SimdGraphOps::compute_edge_weights(&weights, &multipliers);
                let expected: Vec<f32> = weights.iter().zip(multipliers.iter())
                    .map(|(&w, &m)| w * m)
                    .collect();
                
                assert_eq!(results, expected);
            }
        }
    }
}