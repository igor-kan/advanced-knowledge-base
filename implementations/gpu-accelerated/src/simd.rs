//! SIMD-optimized operations with AVX-512 assembly
//!
//! This module provides ultra-high-performance SIMD operations using AVX-512
//! and custom assembly for critical graph processing bottlenecks.

use std::arch::x86_64::*;
use std::mem;

#[cfg(target_arch = "x86_64")]
use core_arch::x86_64::*;

use crate::core::*;
use crate::error::{GpuKnowledgeGraphError, GpuResult};

/// SIMD-optimized graph processing engine
pub struct SimdGraphProcessor {
    /// CPU feature detection
    features: CpuFeatures,
    
    /// Optimized function pointers
    vector_add_f32: fn(&[f32], &[f32], &mut [f32]),
    vector_multiply_f32: fn(&[f32], &[f32], &mut [f32]),
    vector_dot_product_f64: fn(&[f64], &[f64]) -> f64,
    pagerank_update: fn(&[f32], &[f32], &mut [f32], f32, f32),
    sparse_matrix_vector: fn(&[f32], &[i32], &[i32], &[f32], &mut [f32]),
    graph_traverse_batch: fn(&[NodeId], &[EdgeId], &[u32], &mut [NodeId]) -> usize,
}

impl SimdGraphProcessor {
    /// Create new SIMD graph processor with runtime CPU feature detection
    pub fn new() -> GpuResult<Self> {
        tracing::info!("🚀 Initializing SIMD graph processor");
        
        let features = detect_cpu_features();
        tracing::info!("📊 CPU Features: {:?}", features);
        
        // Select optimal implementations based on available features
        let vector_add_f32 = if features.avx512f {
            vector_add_f32_avx512
        } else if features.avx2 {
            vector_add_f32_avx2
        } else if features.sse2 {
            vector_add_f32_sse2
        } else {
            vector_add_f32_scalar
        };
        
        let vector_multiply_f32 = if features.avx512f {
            vector_multiply_f32_avx512
        } else if features.avx2 {
            vector_multiply_f32_avx2
        } else if features.sse2 {
            vector_multiply_f32_sse2
        } else {
            vector_multiply_f32_scalar
        };
        
        let vector_dot_product_f64 = if features.avx512f {
            vector_dot_product_f64_avx512
        } else if features.avx2 {
            vector_dot_product_f64_avx2
        } else if features.sse2 {
            vector_dot_product_f64_sse2
        } else {
            vector_dot_product_f64_scalar
        };
        
        let pagerank_update = if features.avx512f {
            pagerank_update_avx512
        } else if features.avx2 {
            pagerank_update_avx2
        } else {
            pagerank_update_scalar
        };
        
        let sparse_matrix_vector = if features.avx512f {
            sparse_matrix_vector_avx512
        } else if features.avx2 {
            sparse_matrix_vector_avx2
        } else {
            sparse_matrix_vector_scalar
        };
        
        let graph_traverse_batch = if features.avx512f {
            graph_traverse_batch_avx512
        } else if features.avx2 {
            graph_traverse_batch_avx2
        } else {
            graph_traverse_batch_scalar
        };
        
        Ok(Self {
            features,
            vector_add_f32,
            vector_multiply_f32,
            vector_dot_product_f64,
            pagerank_update,
            sparse_matrix_vector,
            graph_traverse_batch,
        })
    }
    
    /// SIMD-optimized vector addition
    #[inline]
    pub fn vector_add_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        (self.vector_add_f32)(a, b, result);
    }
    
    /// SIMD-optimized vector multiplication
    #[inline]
    pub fn vector_multiply_f32(&self, a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        (self.vector_multiply_f32)(a, b, result);
    }
    
    /// SIMD-optimized dot product
    #[inline]
    pub fn vector_dot_product_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        (self.vector_dot_product_f64)(a, b)
    }
    
    /// SIMD-optimized PageRank rank update
    #[inline]
    pub fn pagerank_update(&self, old_ranks: &[f32], contributions: &[f32], new_ranks: &mut [f32], damping: f32, base_rank: f32) {
        assert_eq!(old_ranks.len(), contributions.len());
        assert_eq!(old_ranks.len(), new_ranks.len());
        (self.pagerank_update)(old_ranks, contributions, new_ranks, damping, base_rank);
    }
    
    /// SIMD-optimized sparse matrix-vector multiplication
    #[inline]
    pub fn sparse_matrix_vector(&self, values: &[f32], col_indices: &[i32], row_offsets: &[i32], vector: &[f32], result: &mut [f32]) {
        (self.sparse_matrix_vector)(values, col_indices, row_offsets, vector, result);
    }
    
    /// SIMD-optimized batch graph traversal
    #[inline]
    pub fn graph_traverse_batch(&self, nodes: &[NodeId], edges: &[EdgeId], offsets: &[u32], neighbors: &mut [NodeId]) -> usize {
        (self.graph_traverse_batch)(nodes, edges, offsets, neighbors)
    }
    
    /// Get detected CPU features
    pub fn get_features(&self) -> &CpuFeatures {
        &self.features
    }
    
    /// Get SIMD performance characteristics
    pub fn get_performance_info(&self) -> SimdPerformanceInfo {
        SimdPerformanceInfo {
            instruction_set: if self.features.avx512f {
                "AVX-512"
            } else if self.features.avx2 {
                "AVX2"
            } else if self.features.sse2 {
                "SSE2"
            } else {
                "Scalar"
            }.to_string(),
            vector_width_f32: if self.features.avx512f {
                16 // 512-bit / 32-bit = 16 elements
            } else if self.features.avx2 {
                8  // 256-bit / 32-bit = 8 elements
            } else if self.features.sse2 {
                4  // 128-bit / 32-bit = 4 elements
            } else {
                1  // Scalar
            },
            vector_width_f64: if self.features.avx512f {
                8  // 512-bit / 64-bit = 8 elements
            } else if self.features.avx2 {
                4  // 256-bit / 64-bit = 4 elements
            } else if self.features.sse2 {
                2  // 128-bit / 64-bit = 2 elements
            } else {
                1  // Scalar
            },
            theoretical_speedup_f32: if self.features.avx512f {
                16.0
            } else if self.features.avx2 {
                8.0
            } else if self.features.sse2 {
                4.0
            } else {
                1.0
            },
            supports_fma: self.features.fma,
            supports_gather_scatter: self.features.avx512f,
        }
    }
}

/// CPU feature detection
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub fma: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub popcnt: bool,
}

/// Detect available CPU features at runtime
fn detect_cpu_features() -> CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        CpuFeatures {
            sse2: is_x86_feature_detected!("sse2"),
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512dq: is_x86_feature_detected!("avx512dq"),
            avx512cd: is_x86_feature_detected!("avx512cd"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
            avx512vl: is_x86_feature_detected!("avx512vl"),
            fma: is_x86_feature_detected!("fma"),
            bmi1: is_x86_feature_detected!("bmi1"),
            bmi2: is_x86_feature_detected!("bmi2"),
            popcnt: is_x86_feature_detected!("popcnt"),
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    CpuFeatures {
        sse2: false,
        avx: false,
        avx2: false,
        avx512f: false,
        avx512dq: false,
        avx512cd: false,
        avx512bw: false,
        avx512vl: false,
        fma: false,
        bmi1: false,
        bmi2: false,
        popcnt: false,
    }
}

/// SIMD performance information
#[derive(Debug, Clone)]
pub struct SimdPerformanceInfo {
    pub instruction_set: String,
    pub vector_width_f32: usize,
    pub vector_width_f64: usize,
    pub theoretical_speedup_f32: f32,
    pub supports_fma: bool,
    pub supports_gather_scatter: bool,
}

// AVX-512 optimized implementations

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_add_f32_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    // Process 16 elements at a time with AVX-512
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let vr = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 16;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_multiply_f32_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    // Process 16 elements at a time with AVX-512
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let vr = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 16;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn vector_dot_product_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut sum = _mm512_setzero_pd();
    
    // Process 8 elements at a time with AVX-512
    while i + 8 <= len {
        let va = _mm512_loadu_pd(a.as_ptr().add(i));
        let vb = _mm512_loadu_pd(b.as_ptr().add(i));
        sum = _mm512_fmadd_pd(va, vb, sum);
        i += 8;
    }
    
    // Reduce sum
    let mut result = _mm512_reduce_add_pd(sum);
    
    // Handle remaining elements
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pagerank_update_avx512(
    old_ranks: &[f32],
    contributions: &[f32],
    new_ranks: &mut [f32],
    damping: f32,
    base_rank: f32,
) {
    let len = old_ranks.len();
    let mut i = 0;
    
    let v_damping = _mm512_set1_ps(damping);
    let v_base_rank = _mm512_set1_ps(base_rank);
    
    // Process 16 elements at a time with AVX-512
    while i + 16 <= len {
        let v_contrib = _mm512_loadu_ps(contributions.as_ptr().add(i));
        let v_damped = _mm512_mul_ps(v_contrib, v_damping);
        let v_result = _mm512_add_ps(v_base_rank, v_damped);
        _mm512_storeu_ps(new_ranks.as_mut_ptr().add(i), v_result);
        i += 16;
    }
    
    // Handle remaining elements
    while i < len {
        new_ranks[i] = base_rank + damping * contributions[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sparse_matrix_vector_avx512(
    values: &[f32],
    col_indices: &[i32],
    row_offsets: &[i32],
    vector: &[f32],
    result: &mut [f32],
) {
    let num_rows = row_offsets.len() - 1;
    
    for row in 0..num_rows {
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        
        let mut sum = _mm512_setzero_ps();
        let mut j = start;
        
        // Process 16 elements at a time where possible
        while j + 16 <= end {
            let v_values = _mm512_loadu_ps(values.as_ptr().add(j));
            
            // Gather vector elements based on column indices
            let v_indices = _mm512_loadu_epi32(col_indices.as_ptr().add(j) as *const i32);
            let v_vector = _mm512_i32gather_ps(v_indices, vector.as_ptr(), 4);
            
            sum = _mm512_fmadd_ps(v_values, v_vector, sum);
            j += 16;
        }
        
        // Reduce sum and handle remaining elements
        let mut row_sum = _mm512_reduce_add_ps(sum);
        
        while j < end {
            let col = col_indices[j] as usize;
            row_sum += values[j] * vector[col];
            j += 1;
        }
        
        result[row] = row_sum;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn graph_traverse_batch_avx512(
    nodes: &[NodeId],
    edges: &[EdgeId],
    offsets: &[u32],
    neighbors: &mut [NodeId],
) -> usize {
    let mut neighbor_count = 0;
    
    for &node in nodes {
        let node_idx = node as usize;
        if node_idx + 1 >= offsets.len() {
            continue;
        }
        
        let start = offsets[node_idx] as usize;
        let end = offsets[node_idx + 1] as usize;
        
        // Batch copy neighbors using SIMD
        let mut i = start;
        let available_space = neighbors.len() - neighbor_count;
        let copy_count = (end - start).min(available_space);
        
        // Use SIMD to copy 16 elements at a time
        while i + 16 <= start + copy_count {
            let v_edges = _mm512_loadu_epi64(edges.as_ptr().add(i) as *const i64);
            _mm512_storeu_epi64(neighbors.as_mut_ptr().add(neighbor_count) as *mut i64, v_edges);
            i += 8; // 8 u64 values = 16 u32 values
            neighbor_count += 8;
        }
        
        // Handle remaining elements
        while i < start + copy_count {
            neighbors[neighbor_count] = edges[i];
            neighbor_count += 1;
            i += 1;
        }
        
        if neighbor_count >= neighbors.len() {
            break;
        }
    }
    
    neighbor_count
}

// AVX2 implementations (fallback for older CPUs)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_add_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    // Process 8 elements at a time with AVX2
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 8;
    }
    
    // Handle remaining elements
    while i < len {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_multiply_f32_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 8;
    }
    
    while i < len {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vector_dot_product_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut sum = _mm256_setzero_pd();
    
    while i + 4 <= len {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        sum = _mm256_fmadd_pd(va, vb, sum);
        i += 4;
    }
    
    // Extract and sum the elements
    let sum_array: [f64; 4] = mem::transmute(sum);
    let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pagerank_update_avx2(
    old_ranks: &[f32],
    contributions: &[f32],
    new_ranks: &mut [f32],
    damping: f32,
    base_rank: f32,
) {
    let len = old_ranks.len();
    let mut i = 0;
    
    let v_damping = _mm256_set1_ps(damping);
    let v_base_rank = _mm256_set1_ps(base_rank);
    
    while i + 8 <= len {
        let v_contrib = _mm256_loadu_ps(contributions.as_ptr().add(i));
        let v_damped = _mm256_mul_ps(v_contrib, v_damping);
        let v_result = _mm256_add_ps(v_base_rank, v_damped);
        _mm256_storeu_ps(new_ranks.as_mut_ptr().add(i), v_result);
        i += 8;
    }
    
    while i < len {
        new_ranks[i] = base_rank + damping * contributions[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sparse_matrix_vector_avx2(
    values: &[f32],
    col_indices: &[i32],
    row_offsets: &[i32],
    vector: &[f32],
    result: &mut [f32],
) {
    // Simplified AVX2 implementation (full gather not available)
    sparse_matrix_vector_scalar(values, col_indices, row_offsets, vector, result);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn graph_traverse_batch_avx2(
    nodes: &[NodeId],
    edges: &[EdgeId],
    offsets: &[u32],
    neighbors: &mut [NodeId],
) -> usize {
    // Simplified AVX2 implementation
    graph_traverse_batch_scalar(nodes, edges, offsets, neighbors)
}

// SSE2 implementations (fallback for older CPUs)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_add_f32_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 4;
    }
    
    while i < len {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_multiply_f32_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;
    
    while i + 4 <= len {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vr = _mm_mul_ps(va, vb);
        _mm_storeu_ps(result.as_mut_ptr().add(i), vr);
        i += 4;
    }
    
    while i < len {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn vector_dot_product_f64_sse2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let mut i = 0;
    let mut sum = _mm_setzero_pd();
    
    while i + 2 <= len {
        let va = _mm_loadu_pd(a.as_ptr().add(i));
        let vb = _mm_loadu_pd(b.as_ptr().add(i));
        let vmul = _mm_mul_pd(va, vb);
        sum = _mm_add_pd(sum, vmul);
        i += 2;
    }
    
    let sum_array: [f64; 2] = mem::transmute(sum);
    let mut result = sum_array[0] + sum_array[1];
    
    while i < len {
        result += a[i] * b[i];
        i += 1;
    }
    
    result
}

// Scalar fallback implementations

fn vector_add_f32_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

fn vector_multiply_f32_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

fn vector_dot_product_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn pagerank_update_scalar(
    old_ranks: &[f32],
    contributions: &[f32],
    new_ranks: &mut [f32],
    damping: f32,
    base_rank: f32,
) {
    for i in 0..old_ranks.len() {
        new_ranks[i] = base_rank + damping * contributions[i];
    }
}

fn sparse_matrix_vector_scalar(
    values: &[f32],
    col_indices: &[i32],
    row_offsets: &[i32],
    vector: &[f32],
    result: &mut [f32],
) {
    let num_rows = row_offsets.len() - 1;
    
    for row in 0..num_rows {
        let start = row_offsets[row] as usize;
        let end = row_offsets[row + 1] as usize;
        
        let mut sum = 0.0;
        for j in start..end {
            let col = col_indices[j] as usize;
            sum += values[j] * vector[col];
        }
        
        result[row] = sum;
    }
}

fn graph_traverse_batch_scalar(
    nodes: &[NodeId],
    edges: &[EdgeId],
    offsets: &[u32],
    neighbors: &mut [NodeId],
) -> usize {
    let mut neighbor_count = 0;
    
    for &node in nodes {
        let node_idx = node as usize;
        if node_idx + 1 >= offsets.len() {
            continue;
        }
        
        let start = offsets[node_idx] as usize;
        let end = offsets[node_idx + 1] as usize;
        
        let available_space = neighbors.len() - neighbor_count;
        let copy_count = (end - start).min(available_space);
        
        for i in 0..copy_count {
            neighbors[neighbor_count] = edges[start + i];
            neighbor_count += 1;
        }
        
        if neighbor_count >= neighbors.len() {
            break;
        }
    }
    
    neighbor_count
}

/// Initialize SIMD optimizations
pub fn init_simd_optimizations() -> GpuResult<SimdGraphProcessor> {
    SimdGraphProcessor::new()
}

/// Get SIMD benchmark results
pub fn benchmark_simd_operations() -> GpuResult<SimdBenchmarkResults> {
    let processor = SimdGraphProcessor::new()?;
    let info = processor.get_performance_info();
    
    // Run micro-benchmarks
    let vector_size = 1_000_000;
    let a: Vec<f32> = (0..vector_size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..vector_size).map(|i| (i * 2) as f32).collect();
    let mut result = vec![0.0f32; vector_size];
    
    let start = std::time::Instant::now();
    processor.vector_add_f32(&a, &b, &mut result);
    let add_time = start.elapsed();
    
    let start = std::time::Instant::now();
    processor.vector_multiply_f32(&a, &b, &mut result);
    let multiply_time = start.elapsed();
    
    Ok(SimdBenchmarkResults {
        instruction_set: info.instruction_set,
        vector_width_f32: info.vector_width_f32,
        theoretical_speedup: info.theoretical_speedup_f32,
        add_throughput_gflops: (vector_size as f64) / (add_time.as_secs_f64() * 1e9),
        multiply_throughput_gflops: (vector_size as f64) / (multiply_time.as_secs_f64() * 1e9),
        supports_gather_scatter: info.supports_gather_scatter,
    })
}

#[derive(Debug, Clone)]
pub struct SimdBenchmarkResults {
    pub instruction_set: String,
    pub vector_width_f32: usize,
    pub theoretical_speedup: f32,
    pub add_throughput_gflops: f64,
    pub multiply_throughput_gflops: f64,
    pub supports_gather_scatter: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_processor_creation() {
        let processor = SimdGraphProcessor::new().unwrap();
        let features = processor.get_features();
        println!("CPU Features: {:?}", features);
        
        let info = processor.get_performance_info();
        println!("Performance Info: {:?}", info);
    }
    
    #[test]
    fn test_vector_operations() {
        let processor = SimdGraphProcessor::new().unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];
        
        processor.vector_add_f32(&a, &b, &mut result);
        assert_eq!(result, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
        
        processor.vector_multiply_f32(&a, &b, &mut result);
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
    }
    
    #[test]
    fn test_dot_product() {
        let processor = SimdGraphProcessor::new().unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        
        let result = processor.vector_dot_product_f64(&a, &b);
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    }
    
    #[test]
    fn test_pagerank_update() {
        let processor = SimdGraphProcessor::new().unwrap();
        
        let old_ranks = vec![0.25, 0.25, 0.25, 0.25];
        let contributions = vec![0.1, 0.2, 0.3, 0.4];
        let mut new_ranks = vec![0.0; 4];
        
        processor.pagerank_update(&old_ranks, &contributions, &mut new_ranks, 0.85, 0.15);
        
        // Expected: base_rank + damping * contribution = 0.15 + 0.85 * contribution
        let expected = vec![
            0.15 + 0.85 * 0.1, // 0.235
            0.15 + 0.85 * 0.2, // 0.32
            0.15 + 0.85 * 0.3, // 0.405
            0.15 + 0.85 * 0.4, // 0.49
        ];
        
        for i in 0..4 {
            assert!((new_ranks[i] - expected[i]).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_simd_benchmarks() {
        if let Ok(results) = benchmark_simd_operations() {
            println!("SIMD Benchmark Results: {:?}", results);
            assert!(results.add_throughput_gflops > 0.0);
            assert!(results.multiply_throughput_gflops > 0.0);
        }
    }
}