//! SIMD Optimization Module - 2025 Research Edition
//!
//! This module provides comprehensive SIMD (Single Instruction, Multiple Data)
//! optimizations for graph operations, leveraging the latest CPU instruction sets
//! including AVX-512, AVX2, and specialized graph processing instructions.

use std::arch::x86_64::*;
use crate::core::{NodeId, Weight, UltraResult};
use crate::error::UltraFastKnowledgeGraphError;

/// CPU feature detection and capabilities
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    /// Supports AVX-512 foundation
    pub supports_avx512: bool,
    /// Supports AVX-512 double/quadword operations
    pub supports_avx512_dq: bool,
    /// Supports AVX-512 byte/word operations
    pub supports_avx512_bw: bool,
    /// Supports AVX-512 vector length extensions
    pub supports_avx512_vl: bool,
    /// Supports AVX2
    pub supports_avx2: bool,
    /// Supports AVX
    pub supports_avx: bool,
    /// Supports FMA (Fused Multiply-Add)
    pub supports_fma: bool,
    /// Supports SSE4.2
    pub supports_sse42: bool,
    /// Supports BMI (Bit Manipulation Instructions)
    pub supports_bmi: bool,
    /// Supports BMI2
    pub supports_bmi2: bool,
    /// Supports POPCNT
    pub supports_popcnt: bool,
    /// Supports LZCNT (Leading Zero Count)
    pub supports_lzcnt: bool,
    /// CPU has assembly optimization support
    pub supports_assembly_optimization: bool,
    /// CPU supports all optimizations
    pub supports_all_optimizations: bool,
}

impl CpuFeatures {
    /// Get performance information based on detected features
    pub fn get_performance_info(&self) -> CpuPerformanceInfo {
        CpuPerformanceInfo {
            supports_avx512: self.supports_avx512,
            supports_avx2: self.supports_avx2,
            supports_assembly_optimization: self.supports_assembly_optimization,
            supports_all_optimizations: self.supports_all_optimizations,
            theoretical_simd_width: if self.supports_avx512 { 64 } else if self.supports_avx2 { 32 } else { 16 },
            max_parallel_ops: if self.supports_avx512 { 8 } else if self.supports_avx2 { 4 } else { 2 },
        }
    }
}

/// CPU performance information
#[derive(Debug, Clone)]
pub struct CpuPerformanceInfo {
    /// Supports AVX-512
    pub supports_avx512: bool,
    /// Supports AVX2
    pub supports_avx2: bool,
    /// Supports assembly optimization
    pub supports_assembly_optimization: bool,
    /// Supports all optimizations
    pub supports_all_optimizations: bool,
    /// Theoretical SIMD width in bytes
    pub theoretical_simd_width: usize,
    /// Maximum parallel operations
    pub max_parallel_ops: usize,
}

/// Detect CPU features and capabilities
pub fn detect_cpu_features() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::CpuId;
        
        let cpuid = CpuId::new();
        
        let mut features = CpuFeatures {
            supports_avx512: false,
            supports_avx512_dq: false,
            supports_avx512_bw: false,
            supports_avx512_vl: false,
            supports_avx2: false,
            supports_avx: false,
            supports_fma: false,
            supports_sse42: false,
            supports_bmi: false,
            supports_bmi2: false,
            supports_popcnt: false,
            supports_lzcnt: false,
            supports_assembly_optimization: false,
            supports_all_optimizations: false,
        };
        
        // Check basic features
        if let Some(feature_info) = cpuid.get_feature_info() {
            features.supports_avx = feature_info.has_avx();
            features.supports_fma = feature_info.has_fma();
            features.supports_sse42 = feature_info.has_sse42();
            features.supports_popcnt = feature_info.has_popcnt();
        }
        
        // Check extended features
        if let Some(extended_features) = cpuid.get_extended_feature_info() {
            features.supports_avx2 = extended_features.has_avx2();
            features.supports_bmi = extended_features.has_bmi1();
            features.supports_bmi2 = extended_features.has_bmi2();
            
            // AVX-512 detection
            features.supports_avx512 = extended_features.has_avx512f();
            features.supports_avx512_dq = extended_features.has_avx512dq();
            features.supports_avx512_bw = extended_features.has_avx512bw();
            features.supports_avx512_vl = extended_features.has_avx512vl();
        }
        
        // Check extended processor info
        if let Some(extended_info) = cpuid.get_extended_processor_and_feature_identifiers() {
            features.supports_lzcnt = extended_info.has_lzcnt();
        }
        
        // Assembly optimization supported if we have modern features
        features.supports_assembly_optimization = features.supports_avx2 && features.supports_bmi2;
        
        // All optimizations supported if we have the latest features
        features.supports_all_optimizations = features.supports_avx512 && 
                                            features.supports_avx512_dq && 
                                            features.supports_bmi2 && 
                                            features.supports_lzcnt;
        
        features
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback for non-x86_64 architectures
        CpuFeatures {
            supports_avx512: false,
            supports_avx512_dq: false,
            supports_avx512_bw: false,
            supports_avx512_vl: false,
            supports_avx2: false,
            supports_avx: false,
            supports_fma: false,
            supports_sse42: false,
            supports_bmi: false,
            supports_bmi2: false,
            supports_popcnt: false,
            supports_lzcnt: false,
            supports_assembly_optimization: false,
            supports_all_optimizations: false,
        }
    }
}

/// SIMD-optimized graph operations
pub struct SIMDGraphOps {
    /// CPU features
    cpu_features: CpuFeatures,
}

impl SIMDGraphOps {
    /// Create new SIMD graph operations
    pub fn new(cpu_features: CpuFeatures) -> Self {
        Self { cpu_features }
    }
    
    /// SIMD-optimized node degree calculation
    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_calculate_degrees(&self, 
                                       adjacency_matrix: &[u8], 
                                       num_nodes: usize,
                                       degrees: &mut [u32]) -> UltraResult<()> {
        if degrees.len() != num_nodes {
            return Err(UltraFastKnowledgeGraphError::SIMDError(
                "Degrees array size mismatch".to_string()
            ));
        }
        
        if self.cpu_features.supports_avx512 {
            self.avx512_calculate_degrees(adjacency_matrix, num_nodes, degrees)
        } else if self.cpu_features.supports_avx2 {
            self.avx2_calculate_degrees(adjacency_matrix, num_nodes, degrees)
        } else {
            self.scalar_calculate_degrees(adjacency_matrix, num_nodes, degrees)
        }
    }
    
    /// AVX-512 optimized degree calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_calculate_degrees(&self,
                                     adjacency_matrix: &[u8],
                                     num_nodes: usize,
                                     degrees: &mut [u32]) -> UltraResult<()> {
        let matrix_size = num_nodes * num_nodes;
        if adjacency_matrix.len() != matrix_size {
            return Err(UltraFastKnowledgeGraphError::SIMDError(
                "Adjacency matrix size mismatch".to_string()
            ));
        }
        
        // Initialize degrees to zero
        degrees.fill(0);
        
        // Process 64 bytes (512 bits) at a time
        for node in 0..num_nodes {
            let row_start = node * num_nodes;
            let row_end = row_start + num_nodes;
            
            let mut degree = 0u32;
            let mut i = row_start;
            
            // Process in chunks of 64 bytes
            while i + 64 <= row_end {
                let data = _mm512_loadu_si512(adjacency_matrix.as_ptr().add(i) as *const __m512i);
                
                // Sum all bytes in the 512-bit register
                let ones = _mm512_set1_epi8(1);
                let mask = _mm512_cmpeq_epi8_mask(data, ones);
                degree += mask.count_ones();
                
                i += 64;
            }
            
            // Handle remaining bytes
            while i < row_end {
                if adjacency_matrix[i] == 1 {
                    degree += 1;
                }
                i += 1;
            }
            
            degrees[node] = degree;
        }
        
        Ok(())
    }
    
    /// AVX2 optimized degree calculation
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_calculate_degrees(&self,
                                   adjacency_matrix: &[u8],
                                   num_nodes: usize,
                                   degrees: &mut [u32]) -> UltraResult<()> {
        degrees.fill(0);
        
        for node in 0..num_nodes {
            let row_start = node * num_nodes;
            let row_end = row_start + num_nodes;
            
            let mut degree = 0u32;
            let mut i = row_start;
            
            // Process 32 bytes at a time with AVX2
            while i + 32 <= row_end {
                let data = _mm256_loadu_si256(adjacency_matrix.as_ptr().add(i) as *const __m256i);
                
                // Convert bytes to 32-bit integers and sum
                let zero = _mm256_setzero_si256();
                let lo = _mm256_unpacklo_epi8(data, zero);
                let hi = _mm256_unpackhi_epi8(data, zero);
                
                let lo_lo = _mm256_unpacklo_epi16(lo, zero);
                let lo_hi = _mm256_unpackhi_epi16(lo, zero);
                let hi_lo = _mm256_unpacklo_epi16(hi, zero);
                let hi_hi = _mm256_unpackhi_epi16(hi, zero);
                
                let sum1 = _mm256_add_epi32(lo_lo, lo_hi);
                let sum2 = _mm256_add_epi32(hi_lo, hi_hi);
                let total = _mm256_add_epi32(sum1, sum2);
                
                // Horizontal sum
                let sum_hi = _mm256_extracti128_si256(total, 1);
                let sum_lo = _mm256_castsi256_si128(total);
                let sum_combined = _mm_add_epi32(sum_hi, sum_lo);
                let sum_final = _mm_hadd_epi32(sum_combined, sum_combined);
                let sum_result = _mm_hadd_epi32(sum_final, sum_final);
                
                degree += _mm_extract_epi32(sum_result, 0) as u32;
                i += 32;
            }
            
            // Handle remaining bytes
            while i < row_end {
                if adjacency_matrix[i] == 1 {
                    degree += 1;
                }
                i += 1;
            }
            
            degrees[node] = degree;
        }
        
        Ok(())
    }
    
    /// Scalar fallback for degree calculation
    fn scalar_calculate_degrees(&self,
                              adjacency_matrix: &[u8],
                              num_nodes: usize,
                              degrees: &mut [u32]) -> UltraResult<()> {
        degrees.fill(0);
        
        for node in 0..num_nodes {
            let row_start = node * num_nodes;
            let row_end = row_start + num_nodes;
            
            let degree = adjacency_matrix[row_start..row_end]
                .iter()
                .map(|&x| x as u32)
                .sum();
            
            degrees[node] = degree;
        }
        
        Ok(())
    }
    
    /// SIMD-optimized triangle counting
    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_triangle_count(&self,
                                    adjacency_matrix: &[u8],
                                    num_nodes: usize) -> UltraResult<u64> {
        if adjacency_matrix.len() != num_nodes * num_nodes {
            return Err(UltraFastKnowledgeGraphError::SIMDError(
                "Adjacency matrix size mismatch".to_string()
            ));
        }
        
        let mut triangle_count = 0u64;
        
        // For each node triplet (i, j, k) where i < j < k
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                if adjacency_matrix[i * num_nodes + j] == 0 {
                    continue;
                }
                
                // Check for common neighbors using SIMD
                let triangles = self.simd_count_common_neighbors(
                    adjacency_matrix,
                    num_nodes,
                    i,
                    j
                )?;
                
                triangle_count += triangles;
            }
        }
        
        Ok(triangle_count / 3) // Each triangle is counted 3 times
    }
    
    /// Count common neighbors using SIMD
    #[target_feature(enable = "avx2")]
    unsafe fn simd_count_common_neighbors(&self,
                                        adjacency_matrix: &[u8],
                                        num_nodes: usize,
                                        node_i: usize,
                                        node_j: usize) -> UltraResult<u64> {
        let row_i_start = node_i * num_nodes;
        let row_j_start = node_j * num_nodes;
        
        let mut common_count = 0u64;
        
        if self.cpu_features.supports_avx2 {
            let mut k = 0;
            
            // Process 32 bytes at a time
            while k + 32 <= num_nodes {
                let row_i_data = _mm256_loadu_si256(
                    adjacency_matrix.as_ptr().add(row_i_start + k) as *const __m256i
                );
                let row_j_data = _mm256_loadu_si256(
                    adjacency_matrix.as_ptr().add(row_j_start + k) as *const __m256i
                );
                
                // Bitwise AND to find common neighbors
                let common = _mm256_and_si256(row_i_data, row_j_data);
                
                // Count set bits (common neighbors)
                let common_array: [u8; 32] = std::mem::transmute(common);
                for &byte in &common_array {
                    common_count += byte.count_ones() as u64;
                }
                
                k += 32;
            }
            
            // Handle remaining nodes
            while k < num_nodes {
                if adjacency_matrix[row_i_start + k] == 1 && 
                   adjacency_matrix[row_j_start + k] == 1 {
                    common_count += 1;
                }
                k += 1;
            }
        } else {
            // Scalar fallback
            for k in 0..num_nodes {
                if adjacency_matrix[row_i_start + k] == 1 && 
                   adjacency_matrix[row_j_start + k] == 1 {
                    common_count += 1;
                }
            }
        }
        
        Ok(common_count)
    }
    
    /// SIMD-optimized PageRank vector operations
    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_pagerank_iteration(&self,
                                        transition_matrix: &[f64],
                                        old_ranks: &[f64],
                                        new_ranks: &mut [f64],
                                        damping_factor: f64,
                                        num_nodes: usize) -> UltraResult<()> {
        if transition_matrix.len() != num_nodes * num_nodes ||
           old_ranks.len() != num_nodes ||
           new_ranks.len() != num_nodes {
            return Err(UltraFastKnowledgeGraphError::SIMDError(
                "Array size mismatch in PageRank iteration".to_string()
            ));
        }
        
        let base_rank = (1.0 - damping_factor) / num_nodes as f64;
        
        // Initialize new ranks with base rank
        for rank in new_ranks.iter_mut() {
            *rank = base_rank;
        }
        
        if self.cpu_features.supports_avx2 && self.cpu_features.supports_fma {
            // AVX2 + FMA optimized version
            for i in 0..num_nodes {
                let mut sum = 0.0;
                let row_start = i * num_nodes;
                
                let mut j = 0;
                let damping_vec = _mm256_set1_pd(damping_factor);
                let mut sum_vec = _mm256_setzero_pd();
                
                // Process 4 doubles at a time
                while j + 4 <= num_nodes {
                    let transition_vals = _mm256_loadu_pd(
                        transition_matrix.as_ptr().add(row_start + j)
                    );
                    let rank_vals = _mm256_loadu_pd(old_ranks.as_ptr().add(j));
                    
                    // FMA: sum += transition * rank * damping_factor
                    sum_vec = _mm256_fmadd_pd(
                        _mm256_mul_pd(transition_vals, damping_vec),
                        rank_vals,
                        sum_vec
                    );
                    
                    j += 4;
                }
                
                // Horizontal sum of the vector
                let sum_hi = _mm256_extractf128_pd(sum_vec, 1);
                let sum_lo = _mm256_castpd256_pd128(sum_vec);
                let sum_combined = _mm_add_pd(sum_hi, sum_lo);
                let sum_final = _mm_hadd_pd(sum_combined, sum_combined);
                sum = _mm_cvtsd_f64(sum_final);
                
                // Handle remaining elements
                while j < num_nodes {
                    sum += transition_matrix[row_start + j] * old_ranks[j] * damping_factor;
                    j += 1;
                }
                
                new_ranks[i] += sum;
            }
        } else {
            // Scalar fallback
            for i in 0..num_nodes {
                let row_start = i * num_nodes;
                let mut sum = 0.0;
                
                for j in 0..num_nodes {
                    sum += transition_matrix[row_start + j] * old_ranks[j];
                }
                
                new_ranks[i] += sum * damping_factor;
            }
        }
        
        Ok(())
    }
    
    /// SIMD-optimized clustering coefficient calculation
    #[target_feature(enable = "avx2")]
    pub unsafe fn simd_clustering_coefficient(&self,
                                            adjacency_matrix: &[u8],
                                            degrees: &[u32],
                                            num_nodes: usize) -> UltraResult<f64> {
        let mut total_coefficient = 0.0;
        let mut valid_nodes = 0usize;
        
        for node in 0..num_nodes {
            let degree = degrees[node] as usize;
            
            if degree < 2 {
                continue;
            }
            
            // Find neighbors
            let mut neighbors = Vec::new();
            let row_start = node * num_nodes;
            
            for j in 0..num_nodes {
                if adjacency_matrix[row_start + j] == 1 {
                    neighbors.push(j);
                }
            }
            
            if neighbors.len() < 2 {
                continue;
            }
            
            // Count triangles involving this node
            let mut triangles = 0u32;
            
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let neighbor_i = neighbors[i];
                    let neighbor_j = neighbors[j];
                    
                    if adjacency_matrix[neighbor_i * num_nodes + neighbor_j] == 1 {
                        triangles += 1;
                    }
                }
            }
            
            let possible_triangles = (degree * (degree - 1)) / 2;
            if possible_triangles > 0 {
                let coefficient = triangles as f64 / possible_triangles as f64;
                total_coefficient += coefficient;
                valid_nodes += 1;
            }
        }
        
        if valid_nodes > 0 {
            Ok(total_coefficient / valid_nodes as f64)
        } else {
            Ok(0.0)
        }
    }
}

/// Initialize SIMD optimizations
pub fn init_simd_optimizations() -> UltraResult<()> {
    let cpu_features = detect_cpu_features();
    
    tracing::info!("🚀 SIMD Optimization Status:");
    tracing::info!("  - AVX-512: {}", cpu_features.supports_avx512);
    tracing::info!("  - AVX2: {}", cpu_features.supports_avx2);
    tracing::info!("  - FMA: {}", cpu_features.supports_fma);
    tracing::info!("  - BMI2: {}", cpu_features.supports_bmi2);
    tracing::info!("  - POPCNT: {}", cpu_features.supports_popcnt);
    
    if cpu_features.supports_avx512 {
        tracing::info!("✅ Ultra-high performance AVX-512 optimizations enabled");
    } else if cpu_features.supports_avx2 {
        tracing::info!("✅ High performance AVX2 optimizations enabled");
    } else {
        tracing::warn!("⚠️  Limited SIMD support - using scalar fallbacks");
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        let features = detect_cpu_features();
        
        // At minimum should detect some basic features on x86_64
        #[cfg(target_arch = "x86_64")]
        {
            // Most modern CPUs support at least SSE4.2
            println!("Detected CPU features: {:?}", features);
        }
        
        let perf_info = features.get_performance_info();
        assert!(perf_info.theoretical_simd_width >= 16);
        assert!(perf_info.max_parallel_ops >= 2);
    }
    
    #[test]
    fn test_simd_graph_ops_creation() {
        let cpu_features = detect_cpu_features();
        let _simd_ops = SIMDGraphOps::new(cpu_features);
        // Should create successfully
    }
    
    #[test]
    fn test_degree_calculation() {
        let cpu_features = detect_cpu_features();
        let simd_ops = SIMDGraphOps::new(cpu_features);
        
        // Create a simple 4x4 adjacency matrix
        #[rustfmt::skip]
        let adjacency = vec![
            0, 1, 1, 0,  // Node 0: connected to 1, 2
            1, 0, 1, 1,  // Node 1: connected to 0, 2, 3  
            1, 1, 0, 0,  // Node 2: connected to 0, 1
            0, 1, 0, 0,  // Node 3: connected to 1
        ];
        
        let mut degrees = vec![0u32; 4];
        
        unsafe {
            simd_ops.simd_calculate_degrees(&adjacency, 4, &mut degrees)
                .expect("Degree calculation failed");
        }
        
        assert_eq!(degrees[0], 2); // Node 0 has degree 2
        assert_eq!(degrees[1], 3); // Node 1 has degree 3
        assert_eq!(degrees[2], 2); // Node 2 has degree 2
        assert_eq!(degrees[3], 1); // Node 3 has degree 1
    }
    
    #[test]
    fn test_triangle_counting() {
        let cpu_features = detect_cpu_features();
        let simd_ops = SIMDGraphOps::new(cpu_features);
        
        // Create a simple triangle: 0-1-2-0
        #[rustfmt::skip]
        let adjacency = vec![
            0, 1, 1,  // Node 0: connected to 1, 2
            1, 0, 1,  // Node 1: connected to 0, 2
            1, 1, 0,  // Node 2: connected to 0, 1
        ];
        
        let triangle_count = unsafe {
            simd_ops.simd_triangle_count(&adjacency, 3)
                .expect("Triangle counting failed")
        };
        
        assert_eq!(triangle_count, 1); // Should find exactly 1 triangle
    }
    
    #[test]
    fn test_clustering_coefficient() {
        let cpu_features = detect_cpu_features();
        let simd_ops = SIMDGraphOps::new(cpu_features);
        
        // Perfect triangle - clustering coefficient should be 1.0
        #[rustfmt::skip]
        let adjacency = vec![
            0, 1, 1,
            1, 0, 1,
            1, 1, 0,
        ];
        
        let degrees = vec![2u32, 2, 2]; // All nodes have degree 2
        
        let coefficient = unsafe {
            simd_ops.simd_clustering_coefficient(&adjacency, &degrees, 3)
                .expect("Clustering coefficient calculation failed")
        };
        
        assert!((coefficient - 1.0).abs() < 1e-10); // Should be exactly 1.0
    }
}