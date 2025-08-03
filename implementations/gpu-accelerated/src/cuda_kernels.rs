//! CUDA kernel implementations for graph algorithms
//!
//! This module contains the actual CUDA kernel code for high-performance
//! graph operations including BFS, shortest path, PageRank, and matrix operations.

use std::ffi::CString;

/// CUDA kernel source code for BFS traversal
pub const BFS_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void bfs_frontier_expand(
    const int* __restrict__ nodes,
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    int* __restrict__ distances,
    bool* __restrict__ frontier,
    bool* __restrict__ next_frontier,
    const int num_nodes,
    const int current_level
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        if (!frontier[node]) continue;
        
        const int start = edge_offsets[node];
        const int end = edge_offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            const int neighbor = edges[i];
            
            if (distances[neighbor] == -1) {
                distances[neighbor] = current_level + 1;
                next_frontier[neighbor] = true;
            }
        }
        
        frontier[node] = false;
    }
}

extern "C" __global__ void bfs_init_distances(
    int* __restrict__ distances,
    const int num_nodes,
    const int start_node
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_nodes; i += stride) {
        distances[i] = (i == start_node) ? 0 : -1;
    }
}

extern "C" __global__ void bfs_check_frontier(
    const bool* __restrict__ frontier,
    bool* __restrict__ has_work,
    const int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    __shared__ bool local_has_work;
    if (threadIdx.x == 0) local_has_work = false;
    __syncthreads();
    
    for (int i = tid; i < num_nodes; i += stride) {
        if (frontier[i]) {
            local_has_work = true;
            break;
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && local_has_work) {
        *has_work = true;
    }
}
"#;

/// CUDA kernel source code for Dijkstra shortest path
pub const DIJKSTRA_KERNEL_SOURCE: &str = r#"
#include <limits.h>

extern "C" __global__ void dijkstra_relax_edges(
    const int* __restrict__ nodes,
    const int* __restrict__ edges,
    const float* __restrict__ weights,
    const int* __restrict__ edge_offsets,
    float* __restrict__ distances,
    bool* __restrict__ updated,
    const bool* __restrict__ in_queue,
    const int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        if (!in_queue[node]) continue;
        
        const float node_dist = distances[node];
        if (node_dist == INFINITY) continue;
        
        const int start = edge_offsets[node];
        const int end = edge_offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            const int neighbor = edges[i];
            const float edge_weight = weights[i];
            const float new_dist = node_dist + edge_weight;
            
            if (new_dist < distances[neighbor]) {
                atomicMinFloat(&distances[neighbor], new_dist);
                updated[neighbor] = true;
            }
        }
    }
}

extern "C" __global__ void dijkstra_init_distances(
    float* __restrict__ distances,
    bool* __restrict__ in_queue,
    const int num_nodes,
    const int start_node
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_nodes; i += stride) {
        distances[i] = (i == start_node) ? 0.0f : INFINITY;
        in_queue[i] = (i == start_node);
    }
}

extern "C" __global__ void dijkstra_update_queue(
    const bool* __restrict__ updated,
    bool* __restrict__ in_queue,
    bool* __restrict__ has_work,
    const int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    __shared__ bool local_has_work;
    if (threadIdx.x == 0) local_has_work = false;
    __syncthreads();
    
    for (int i = tid; i < num_nodes; i += stride) {
        if (updated[i]) {
            in_queue[i] = true;
            local_has_work = true;
        }
    }
    
    __syncthreads();
    if (threadIdx.x == 0 && local_has_work) {
        *has_work = true;
    }
}

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
"#;

/// CUDA kernel source code for PageRank algorithm
pub const PAGERANK_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void pagerank_compute_contributions(
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    const int* __restrict__ out_degrees,
    const float* __restrict__ old_ranks,
    float* __restrict__ contributions,
    const int num_nodes,
    const float damping_factor
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        const float rank = old_ranks[node];
        const int out_deg = out_degrees[node];
        
        if (out_deg == 0) continue;
        
        const float contribution = damping_factor * rank / out_deg;
        const int start = edge_offsets[node];
        const int end = edge_offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            const int neighbor = edges[i];
            atomicAdd(&contributions[neighbor], contribution);
        }
    }
}

extern "C" __global__ void pagerank_update_ranks(
    const float* __restrict__ contributions,
    float* __restrict__ new_ranks,
    const int num_nodes,
    const float damping_factor,
    const float base_rank
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        new_ranks[node] = base_rank + contributions[node];
    }
}

extern "C" __global__ void pagerank_init_ranks(
    float* __restrict__ ranks,
    const int num_nodes,
    const float initial_rank
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_nodes; i += stride) {
        ranks[i] = initial_rank;
    }
}

extern "C" __global__ void pagerank_clear_contributions(
    float* __restrict__ contributions,
    const int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_nodes; i += stride) {
        contributions[i] = 0.0f;
    }
}

extern "C" __global__ void pagerank_compute_convergence(
    const float* __restrict__ old_ranks,
    const float* __restrict__ new_ranks,
    float* __restrict__ diff_sum,
    const int num_nodes
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;
    
    for (int i = tid; i < num_nodes; i += stride) {
        const float diff = fabsf(new_ranks[i] - old_ranks[i]);
        local_sum += diff;
    }
    
    const int thread_id = threadIdx.x;
    shared_sum[thread_id] = local_sum;
    __syncthreads();
    
    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_id < s) {
            shared_sum[thread_id] += shared_sum[thread_id + s];
        }
        __syncthreads();
    }
    
    if (thread_id == 0) {
        atomicAdd(diff_sum, shared_sum[0]);
    }
}
"#;

/// CUDA kernel source code for matrix operations
pub const MATRIX_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void sparse_matrix_vector_multiply(
    const float* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_offsets,
    const float* __restrict__ vector,
    float* __restrict__ result,
    const int num_rows
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= num_rows) return;
    
    const int start = row_offsets[row];
    const int end = row_offsets[row + 1];
    
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += values[i] * vector[col_indices[i]];
    }
    
    result[row] = sum;
}

extern "C" __global__ void dense_matrix_multiply(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

extern "C" __global__ void matrix_transpose(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows, const int cols
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= rows || col >= cols) return;
    
    output[col * rows + row] = input[row * cols + col];
}
"#;

/// CUDA kernel source code for pattern matching
pub const PATTERN_MATCHING_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void pattern_match_nodes(
    const int* __restrict__ node_types,
    const int* __restrict__ pattern,
    bool* __restrict__ matches,
    const int num_nodes,
    const int pattern_length
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        bool match = true;
        
        for (int i = 0; i < pattern_length; i++) {
            // Simple type matching - can be extended for complex patterns
            if (node_types[node] != pattern[i]) {
                match = false;
                break;
            }
        }
        
        matches[node] = match;
    }
}

extern "C" __global__ void pattern_match_subgraphs(
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    const int* __restrict__ node_types,
    const int* __restrict__ pattern_nodes,
    const int* __restrict__ pattern_edges,
    bool* __restrict__ matches,
    const int num_nodes,
    const int pattern_size
) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= num_nodes) return;
    
    // Check if this node can be the starting point of the pattern
    if (node_types[node] != pattern_nodes[0]) {
        matches[node] = false;
        return;
    }
    
    // Simple pattern matching - can be extended for more complex patterns
    matches[node] = true; // Placeholder implementation
}
"#;

/// CUDA kernel source code for community detection
pub const COMMUNITY_DETECTION_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void louvain_modularity_gain(
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    const float* __restrict__ edge_weights,
    const int* __restrict__ communities,
    const float* __restrict__ community_weights,
    float* __restrict__ modularity_gains,
    const int num_nodes,
    const int target_community,
    const float total_weight
) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= num_nodes) return;
    
    const int current_community = communities[node];
    if (current_community == target_community) {
        modularity_gains[node] = 0.0f;
        return;
    }
    
    float ki_in = 0.0f; // Weight of edges from node to target community
    float ki = 0.0f;    // Total weight of edges from node
    
    const int start = edge_offsets[node];
    const int end = edge_offsets[node + 1];
    
    for (int i = start; i < end; i++) {
        const int neighbor = edges[i];
        const float weight = edge_weights[i];
        
        ki += weight;
        
        if (communities[neighbor] == target_community) {
            ki_in += weight;
        }
    }
    
    const float sigma_tot = community_weights[target_community];
    const float two_m = 2.0f * total_weight;
    
    // Modularity gain formula
    modularity_gains[node] = (ki_in / total_weight) - ((sigma_tot * ki) / (two_m * two_m));
}

extern "C" __global__ void update_communities(
    const float* __restrict__ modularity_gains,
    int* __restrict__ communities,
    bool* __restrict__ changed,
    const int num_nodes,
    const int target_community
) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= num_nodes) return;
    
    if (modularity_gains[node] > 0.0f) {
        const int old_community = communities[node];
        communities[node] = target_community;
        
        if (old_community != target_community) {
            *changed = true;
        }
    }
}
"#;

/// CUDA kernel source code for triangle counting
pub const TRIANGLE_COUNTING_KERNEL_SOURCE: &str = r#"
extern "C" __global__ void count_triangles_node_parallel(
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    unsigned long long* __restrict__ triangle_counts,
    const int num_nodes
) {
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node >= num_nodes) return;
    
    const int start_u = edge_offsets[node];
    const int end_u = edge_offsets[node + 1];
    
    unsigned long long local_count = 0;
    
    // For each neighbor v of u
    for (int i = start_u; i < end_u; i++) {
        const int neighbor_v = edges[i];
        
        if (neighbor_v <= node) continue; // Avoid double counting
        
        const int start_v = edge_offsets[neighbor_v];
        const int end_v = edge_offsets[neighbor_v + 1];
        
        // Count common neighbors of u and v
        int ptr_u = start_u;
        int ptr_v = start_v;
        
        while (ptr_u < end_u && ptr_v < end_v) {
            const int neighbor_u = edges[ptr_u];
            const int neighbor_v_inner = edges[ptr_v];
            
            if (neighbor_u == neighbor_v_inner && neighbor_u > neighbor_v) {
                local_count++;
                ptr_u++;
                ptr_v++;
            } else if (neighbor_u < neighbor_v_inner) {
                ptr_u++;
            } else {
                ptr_v++;
            }
        }
    }
    
    triangle_counts[node] = local_count;
}

extern "C" __global__ void count_triangles_edge_parallel(
    const int* __restrict__ edges,
    const int* __restrict__ edge_offsets,
    unsigned long long* __restrict__ triangle_count,
    const int num_edges
) {
    const int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_id >= num_edges) return;
    
    // Find the source node for this edge
    // This requires binary search or precomputed edge-to-node mapping
    // Simplified implementation assumes sequential edge IDs
    
    unsigned long long local_count = 0;
    
    // Triangle counting logic for this specific edge
    // Implementation details depend on graph representation
    
    atomicAdd(triangle_count, local_count);
}
"#;

/// Kernel compilation configuration
pub struct KernelConfig {
    pub compute_capability: (i32, i32),
    pub max_threads_per_block: u32,
    pub shared_memory_per_block: usize,
    pub optimization_level: i32,
    pub debug_symbols: bool,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            compute_capability: (8, 0), // Default to Ampere architecture
            max_threads_per_block: 1024,
            shared_memory_per_block: 48 * 1024, // 48KB
            optimization_level: 3, // -O3
            debug_symbols: false,
        }
    }
}

/// Kernel launch parameters
#[derive(Debug, Clone)]
pub struct LaunchParams {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_memory_bytes: usize,
}

impl LaunchParams {
    /// Create optimal launch parameters for 1D problem
    pub fn for_1d_problem(num_elements: usize, max_threads_per_block: u32) -> Self {
        let block_size = max_threads_per_block.min(256); // Common optimal block size
        let grid_size = ((num_elements as u32 + block_size - 1) / block_size).max(1);
        
        Self {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_memory_bytes: 0,
        }
    }
    
    /// Create optimal launch parameters for 2D problem
    pub fn for_2d_problem(width: usize, height: usize, max_threads_per_block: u32) -> Self {
        let block_dim_x = 16u32.min(max_threads_per_block);
        let block_dim_y = (max_threads_per_block / block_dim_x).min(16);
        
        let grid_dim_x = ((width as u32 + block_dim_x - 1) / block_dim_x).max(1);
        let grid_dim_y = ((height as u32 + block_dim_y - 1) / block_dim_y).max(1);
        
        Self {
            grid_dim: (grid_dim_x, grid_dim_y, 1),
            block_dim: (block_dim_x, block_dim_y, 1),
            shared_memory_bytes: 0,
        }
    }
    
    /// Create launch parameters with shared memory
    pub fn with_shared_memory(mut self, shared_memory_bytes: usize) -> Self {
        self.shared_memory_bytes = shared_memory_bytes;
        self
    }
}

/// CUDA kernel compilation utilities
pub struct KernelCompiler;

impl KernelCompiler {
    /// Compile kernel source code to PTX
    pub fn compile_to_ptx(
        source: &str,
        kernel_name: &str,
        config: &KernelConfig,
    ) -> Result<String, String> {
        // TODO: Implement actual NVRTC compilation
        // This would use the NVIDIA Runtime Compilation library
        
        tracing::debug!("Compiling CUDA kernel: {}", kernel_name);
        
        // Placeholder PTX generation
        let ptx_header = format!(
            ".version 7.0\n.target sm_{}{}\n.address_size 64\n\n",
            config.compute_capability.0, config.compute_capability.1
        );
        
        let ptx_kernel = format!(
            ".visible .entry {}(\n    .param .u64 param_0\n)\n{{\n    ret;\n}}\n",
            kernel_name
        );
        
        Ok(format!("{}{}", ptx_header, ptx_kernel))
    }
    
    /// Get compilation flags for given configuration
    pub fn get_compilation_flags(config: &KernelConfig) -> Vec<String> {
        let mut flags = Vec::new();
        
        flags.push(format!("--gpu-architecture=sm_{}{}", 
                          config.compute_capability.0, config.compute_capability.1));
        
        flags.push(format!("-O{}", config.optimization_level));
        
        if config.debug_symbols {
            flags.push("--device-debug".to_string());
            flags.push("--generate-line-info".to_string());
        }
        
        flags.push("--use_fast_math".to_string());
        flags.push("--restrict".to_string());
        flags.push("--ftz=true".to_string()); // Flush denormals to zero
        flags.push("--prec-div=false".to_string()); // Use fast division
        flags.push("--prec-sqrt=false".to_string()); // Use fast square root
        
        flags
    }
    
    /// Validate kernel source code
    pub fn validate_kernel_source(source: &str) -> Result<(), String> {
        // Basic validation checks
        if source.is_empty() {
            return Err("Kernel source is empty".to_string());
        }
        
        if !source.contains("extern \"C\"") {
            return Err("Kernel must have extern \"C\" linkage".to_string());
        }
        
        if !source.contains("__global__") {
            return Err("Kernel must contain at least one __global__ function".to_string());
        }
        
        Ok(())
    }
}

/// Kernel parameter types for type-safe kernel launches
pub enum KernelParam {
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    Pointer(*const std::ffi::c_void),
    MutablePointer(*mut std::ffi::c_void),
}

impl KernelParam {
    /// Convert to raw pointer for kernel launch
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        match self {
            KernelParam::Int32(v) => v as *const i32 as *const std::ffi::c_void,
            KernelParam::UInt32(v) => v as *const u32 as *const std::ffi::c_void,
            KernelParam::Int64(v) => v as *const i64 as *const std::ffi::c_void,
            KernelParam::UInt64(v) => v as *const u64 as *const std::ffi::c_void,
            KernelParam::Float32(v) => v as *const f32 as *const std::ffi::c_void,
            KernelParam::Float64(v) => v as *const f64 as *const std::ffi::c_void,
            KernelParam::Pointer(p) => *p,
            KernelParam::MutablePointer(p) => *p as *const std::ffi::c_void,
        }
    }
}

/// High-level kernel launch interface
pub struct KernelLauncher;

impl KernelLauncher {
    /// Launch BFS kernel
    pub fn launch_bfs_frontier_expand(
        nodes: *const i32,
        edges: *const i32,
        edge_offsets: *const i32,
        distances: *mut i32,
        frontier: *mut bool,
        next_frontier: *mut bool,
        num_nodes: i32,
        current_level: i32,
        launch_params: &LaunchParams,
    ) -> Vec<KernelParam> {
        vec![
            KernelParam::Pointer(nodes as *const std::ffi::c_void),
            KernelParam::Pointer(edges as *const std::ffi::c_void),
            KernelParam::Pointer(edge_offsets as *const std::ffi::c_void),
            KernelParam::MutablePointer(distances as *mut std::ffi::c_void),
            KernelParam::MutablePointer(frontier as *mut std::ffi::c_void),
            KernelParam::MutablePointer(next_frontier as *mut std::ffi::c_void),
            KernelParam::Int32(num_nodes),
            KernelParam::Int32(current_level),
        ]
    }
    
    /// Launch PageRank kernel
    pub fn launch_pagerank_compute_contributions(
        edges: *const i32,
        edge_offsets: *const i32,
        out_degrees: *const i32,
        old_ranks: *const f32,
        contributions: *mut f32,
        num_nodes: i32,
        damping_factor: f32,
        launch_params: &LaunchParams,
    ) -> Vec<KernelParam> {
        vec![
            KernelParam::Pointer(edges as *const std::ffi::c_void),
            KernelParam::Pointer(edge_offsets as *const std::ffi::c_void),
            KernelParam::Pointer(out_degrees as *const std::ffi::c_void),
            KernelParam::Pointer(old_ranks as *const std::ffi::c_void),
            KernelParam::MutablePointer(contributions as *mut std::ffi::c_void),
            KernelParam::Int32(num_nodes),
            KernelParam::Float32(damping_factor),
        ]
    }
    
    /// Launch sparse matrix-vector multiplication kernel
    pub fn launch_sparse_matvec(
        values: *const f32,
        col_indices: *const i32,
        row_offsets: *const i32,
        vector: *const f32,
        result: *mut f32,
        num_rows: i32,
        launch_params: &LaunchParams,
    ) -> Vec<KernelParam> {
        vec![
            KernelParam::Pointer(values as *const std::ffi::c_void),
            KernelParam::Pointer(col_indices as *const std::ffi::c_void),
            KernelParam::Pointer(row_offsets as *const std::ffi::c_void),
            KernelParam::Pointer(vector as *const std::ffi::c_void),
            KernelParam::MutablePointer(result as *mut std::ffi::c_void),
            KernelParam::Int32(num_rows),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kernel_source_validation() {
        assert!(KernelCompiler::validate_kernel_source(BFS_KERNEL_SOURCE).is_ok());
        assert!(KernelCompiler::validate_kernel_source(PAGERANK_KERNEL_SOURCE).is_ok());
        assert!(KernelCompiler::validate_kernel_source("").is_err());
        assert!(KernelCompiler::validate_kernel_source("void test() {}").is_err());
    }
    
    #[test]
    fn test_launch_params_1d() {
        let params = LaunchParams::for_1d_problem(10000, 256);
        assert_eq!(params.block_dim.0, 256);
        assert_eq!(params.grid_dim.0, (10000 + 255) / 256);
    }
    
    #[test]
    fn test_launch_params_2d() {
        let params = LaunchParams::for_2d_problem(1024, 1024, 256);
        assert!(params.block_dim.0 <= 16);
        assert!(params.block_dim.1 <= 16);
        assert!(params.block_dim.0 * params.block_dim.1 <= 256);
    }
    
    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert_eq!(config.compute_capability, (8, 0));
        assert_eq!(config.max_threads_per_block, 1024);
        assert_eq!(config.optimization_level, 3);
    }
    
    #[test]
    fn test_compilation_flags() {
        let config = KernelConfig::default();
        let flags = KernelCompiler::get_compilation_flags(&config);
        
        assert!(flags.contains(&"--gpu-architecture=sm_80".to_string()));
        assert!(flags.contains(&"-O3".to_string()));
        assert!(flags.contains(&"--use_fast_math".to_string()));
    }
    
    #[test]
    fn test_kernel_param_types() {
        let int_param = KernelParam::Int32(42);
        let float_param = KernelParam::Float32(3.14);
        let ptr_param = KernelParam::Pointer(std::ptr::null());
        
        // Ensure we can get pointers from all parameter types
        assert!(!int_param.as_ptr().is_null());
        assert!(!float_param.as_ptr().is_null());
        assert!(ptr_param.as_ptr().is_null());
    }
}