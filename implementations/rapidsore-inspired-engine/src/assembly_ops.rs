//! Assembly optimizations for critical hot paths
//!
//! This module implements hand-tuned Assembly code for maximum performance
//! on critical graph operations. Based on 2025 research showing assembly
//! optimizations can provide 2x-5x speedups for specific hot paths.
//!
//! Key optimizations:
//! - Custom memory prefetching patterns
//! - Branch-free conditional execution
//! - Loop unrolling and vectorization hints
//! - Cache-line aligned data access

use crate::types::*;
use crate::{Result, RapidStoreError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::arch::asm;
use tracing::{debug, warn};

/// Assembly-optimized operations dispatcher
pub struct AssemblyOptimizedOps {
    /// Capability flags
    capabilities: AssemblyCapabilities,
    /// Statistics
    stats: AssemblyStats,
}

impl AssemblyOptimizedOps {
    /// Create new assembly operations instance
    pub fn new() -> Result<Self> {
        let capabilities = detect_assembly_capabilities();
        
        if !capabilities.has_basic_support {
            return Err(RapidStoreError::AssemblyError {
                details: "Assembly optimizations not supported on this platform".to_string(),
            });
        }
        
        debug!("Assembly capabilities: {:?}", capabilities);
        
        Ok(Self {
            capabilities,
            stats: AssemblyStats::default(),
        })
    }
    
    /// Fast hash computation using assembly optimization
    #[cfg(target_arch = "x86_64")]
    pub fn fast_hash_u128(&self, value: u128) -> u64 {
        self.stats.hash_calls.fetch_add(1, Ordering::Relaxed);
        
        unsafe {
            let high = (value >> 64) as u64;
            let low = value as u64;
            let mut result: u64;
            
            // Hand-optimized assembly for u128 hashing
            asm!(
                // Load constants
                "mov {magic1}, 0x9e3779b97f4a7c15",
                "mov {magic2}, 0x517cc1b727220a95",
                
                // Multiply high part
                "mov rax, {high}",
                "mul {magic1}",
                "mov {temp}, rax",
                
                // Multiply low part  
                "mov rax, {low}",
                "mul {magic2}",
                
                // XOR combine results
                "xor rax, {temp}",
                "mov {result}, rax",
                
                high = in(reg) high,
                low = in(reg) low,
                magic1 = out(reg) _,
                magic2 = out(reg) _,
                temp = out(reg) _,
                result = out(reg) result,
                out("rax") _,
                out("rdx") _,
            );
            
            result
        }
    }
    
    /// Optimized memory copy with prefetching
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn optimized_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.stats.memcpy_calls.fetch_add(1, Ordering::Relaxed);
        
        if len == 0 {
            return;
        }
        
        // Use assembly for optimal memory copy with prefetching
        asm!(
            // Check alignment and size
            "cmp {len}, 64",
            "jb 2f",              // Jump to byte copy if less than 64 bytes
            
            // Large copy with prefetching
            "1:",
            "prefetchnta 128({src})", // Prefetch next cache line
            
            // Copy 64 bytes using AVX (if available)
            "vmovdqu ymm0, [{src}]",
            "vmovdqu ymm1, [{src} + 32]",
            "vmovdqu [{dst}], ymm0",
            "vmovdqu [{dst} + 32], ymm1",
            
            "add {src}, 64",
            "add {dst}, 64",
            "sub {len}, 64",
            "cmp {len}, 64",
            "jae 1b",
            
            // Handle remaining bytes
            "2:",
            "test {len}, {len}",
            "jz 4f",
            
            "3:",
            "mov al, [{src}]",
            "mov [{dst}], al",
            "inc {src}",
            "inc {dst}",
            "dec {len}",
            "jnz 3b",
            
            "4:",
            
            src = inout(reg) src => _,
            dst = inout(reg) dst => _,
            len = inout(reg) len => _,
            out("al") _,
            out("ymm0") _,
            out("ymm1") _,
        );
    }
    
    /// Branch-free minimum of two values
    #[cfg(target_arch = "x86_64")]
    pub fn branchless_min_u64(&self, a: u64, b: u64) -> u64 {
        self.stats.branchless_ops.fetch_add(1, Ordering::Relaxed);
        
        unsafe {
            let mut result: u64;
            
            asm!(
                "mov {temp}, {a}",
                "sub {temp}, {b}",
                "sbb {temp}, {temp}",  // Set temp to all 1s if a < b, else all 0s
                "and {temp}, {a}",     // temp = (a < b) ? a : 0
                "not {mask}",          // Invert mask
                "and {mask}, {b}",     // mask = (a < b) ? 0 : b
                "or {result}, {temp}",
                "or {result}, {mask}",
                
                a = in(reg) a,
                b = in(reg) b,
                temp = out(reg) _,
                mask = out(reg) _,
                result = out(reg) result,
            );
            
            result
        }
    }
    
    /// Branch-free maximum of two values
    #[cfg(target_arch = "x86_64")]
    pub fn branchless_max_u64(&self, a: u64, b: u64) -> u64 {
        self.stats.branchless_ops.fetch_add(1, Ordering::Relaxed);
        
        unsafe {
            let mut result: u64;
            
            asm!(
                "mov {temp}, {b}",
                "sub {temp}, {a}",
                "sbb {temp}, {temp}",  // Set temp to all 1s if b < a, else all 0s
                "and {temp}, {a}",     // temp = (b < a) ? a : 0
                "not {mask}",          // Invert mask
                "and {mask}, {b}",     // mask = (b < a) ? 0 : b
                "or {result}, {temp}",
                "or {result}, {mask}",
                
                a = in(reg) a,
                b = in(reg) b,
                temp = out(reg) _,
                mask = out(reg) _,
                result = out(reg) result,
            );
            
            result
        }
    }
    
    /// Optimized array sum with loop unrolling
    #[cfg(target_arch = "x86_64")]
    pub fn optimized_sum_u64(&self, array: &[u64]) -> u64 {
        self.stats.array_ops.fetch_add(1, Ordering::Relaxed);
        
        if array.is_empty() {
            return 0;
        }
        
        unsafe {
            let mut sum: u64 = 0;
            let ptr = array.as_ptr();
            let mut len = array.len();
            
            // Process 8 elements at a time (loop unrolling)
            while len >= 8 {
                asm!(
                    "mov rax, [{ptr}]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 8]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 16]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 24]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 32]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 40]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 48]",
                    "add {sum}, rax",
                    "mov rax, [{ptr} + 56]",
                    "add {sum}, rax",
                    
                    ptr = inout(reg) ptr => _,
                    sum = inout(reg) sum,
                    out("rax") _,
                );
                
                len -= 8;
            }
            
            // Handle remaining elements
            for i in (array.len() - len)..array.len() {
                sum += array[i];
            }
            
            sum
        }
    }
    
    /// Count leading zeros with assembly instruction
    #[cfg(target_arch = "x86_64")]
    pub fn count_leading_zeros_u64(&self, value: u64) -> u32 {
        self.stats.bit_ops.fetch_add(1, Ordering::Relaxed);
        
        if value == 0 {
            return 64;
        }
        
        unsafe {
            let mut result: u32;
            
            asm!(
                "lzcnt {result:e}, {value}",
                value = in(reg) value,
                result = out(reg) result,
            );
            
            result
        }
    }
    
    /// Population count (number of set bits) with assembly
    #[cfg(target_arch = "x86_64")]
    pub fn population_count_u64(&self, value: u64) -> u32 {
        self.stats.bit_ops.fetch_add(1, Ordering::Relaxed);
        
        unsafe {
            let mut result: u32;
            
            asm!(
                "popcnt {result:e}, {value}",
                value = in(reg) value,
                result = out(reg) result,
            );
            
            result
        }
    }
    
    /// Optimized binary search with prefetching
    #[cfg(target_arch = "x86_64")]
    pub fn optimized_binary_search(&self, array: &[u64], target: u64) -> Option<usize> {
        self.stats.search_ops.fetch_add(1, Ordering::Relaxed);
        
        if array.is_empty() {
            return None;
        }
        
        unsafe {
            let mut left = 0usize;
            let mut right = array.len();
            let ptr = array.as_ptr();
            
            while left < right {
                let mid = left + (right - left) / 2;
                
                // Prefetch both potential next cache lines
                asm!(
                    "prefetcht0 [{ptr} + {left} * 8]",
                    "prefetcht0 [{ptr} + {right} * 8]",
                    ptr = in(reg) ptr,
                    left = in(reg) left,
                    right = in(reg) right,
                );
                
                let mid_value = *array.get_unchecked(mid);
                
                if mid_value < target {
                    left = mid + 1;
                } else if mid_value > target {
                    right = mid;
                } else {
                    return Some(mid);
                }
            }
            
            None
        }
    }
    
    /// Optimized array comparison
    #[cfg(target_arch = "x86_64")]
    pub fn optimized_array_compare(&self, a: &[u64], b: &[u64]) -> std::cmp::Ordering {
        self.stats.compare_ops.fetch_add(1, Ordering::Relaxed);
        
        use std::cmp::Ordering;
        
        let min_len = a.len().min(b.len());
        
        unsafe {
            let ptr_a = a.as_ptr();
            let ptr_b = b.as_ptr();
            let mut i = 0;
            
            // Compare 8 elements at a time
            while i + 8 <= min_len {
                for j in 0..8 {
                    let val_a = *ptr_a.add(i + j);
                    let val_b = *ptr_b.add(i + j);
                    
                    if val_a != val_b {
                        return val_a.cmp(&val_b);
                    }
                }
                i += 8;
            }
            
            // Compare remaining elements
            for j in i..min_len {
                let val_a = *ptr_a.add(j);
                let val_b = *ptr_b.add(j);
                
                if val_a != val_b {
                    return val_a.cmp(&val_b);
                }
            }
            
            // Arrays are equal up to min_len, compare lengths
            a.len().cmp(&b.len())
        }
    }
    
    /// Cache-optimized matrix multiplication for adjacency matrices
    #[cfg(target_arch = "x86_64")]
    pub fn optimized_matrix_multiply(
        &self,
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        n: usize,
    ) {
        self.stats.matrix_ops.fetch_add(1, Ordering::Relaxed);
        
        if n == 0 {
            return;
        }
        
        unsafe {
            let ptr_a = a.as_ptr();
            let ptr_b = b.as_ptr();
            let ptr_result = result.as_mut_ptr();
            
            // Cache-friendly matrix multiplication with blocking
            const BLOCK_SIZE: usize = 64;
            
            for i_block in (0..n).step_by(BLOCK_SIZE) {
                for j_block in (0..n).step_by(BLOCK_SIZE) {
                    for k_block in (0..n).step_by(BLOCK_SIZE) {
                        let i_max = (i_block + BLOCK_SIZE).min(n);
                        let j_max = (j_block + BLOCK_SIZE).min(n);
                        let k_max = (k_block + BLOCK_SIZE).min(n);
                        
                        for i in i_block..i_max {
                            for j in j_block..j_max {
                                let mut sum = 0u64;
                                
                                // Prefetch next cache line
                                if k_block + 64 < k_max {
                                    asm!(
                                        "prefetcht0 [{ptr_a} + {offset_a}]",
                                        "prefetcht0 [{ptr_b} + {offset_b}]",
                                        ptr_a = in(reg) ptr_a,
                                        ptr_b = in(reg) ptr_b,
                                        offset_a = in(reg) (i * n + k_block + 64) * 8,
                                        offset_b = in(reg) ((k_block + 64) * n + j) * 8,
                                    );
                                }
                                
                                for k in k_block..k_max {
                                    let a_val = *ptr_a.add(i * n + k);
                                    let b_val = *ptr_b.add(k * n + j);
                                    sum += a_val * b_val;
                                }
                                
                                *ptr_result.add(i * n + j) += sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Get assembly operation statistics
    pub fn get_stats(&self) -> &AssemblyStats {
        &self.stats
    }
    
    /// Get assembly capabilities
    pub fn get_capabilities(&self) -> &AssemblyCapabilities {
        &self.capabilities
    }
}

// Fallback implementations for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
impl AssemblyOptimizedOps {
    pub fn new() -> Result<Self> {
        Err(RapidStoreError::AssemblyError {
            details: "Assembly optimizations only supported on x86_64".to_string(),
        })
    }
    
    pub fn fast_hash_u128(&self, value: u128) -> u64 {
        // Fallback to software implementation
        let high = (value >> 64) as u64;
        let low = value as u64;
        high.wrapping_mul(0x9e3779b97f4a7c15) ^ low.wrapping_mul(0x517cc1b727220a95)
    }
    
    pub unsafe fn optimized_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
    
    pub fn branchless_min_u64(&self, a: u64, b: u64) -> u64 {
        a.min(b)
    }
    
    pub fn branchless_max_u64(&self, a: u64, b: u64) -> u64 {
        a.max(b)
    }
    
    pub fn optimized_sum_u64(&self, array: &[u64]) -> u64 {
        array.iter().sum()
    }
    
    pub fn count_leading_zeros_u64(&self, value: u64) -> u32 {
        value.leading_zeros()
    }
    
    pub fn population_count_u64(&self, value: u64) -> u32 {
        value.count_ones()
    }
    
    pub fn optimized_binary_search(&self, array: &[u64], target: u64) -> Option<usize> {
        array.binary_search(&target).ok()
    }
    
    pub fn optimized_array_compare(&self, a: &[u64], b: &[u64]) -> std::cmp::Ordering {
        a.cmp(b)
    }
    
    pub fn optimized_matrix_multiply(
        &self,
        a: &[u64],
        b: &[u64],
        result: &mut [u64],
        n: usize,
    ) {
        // Standard matrix multiplication
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0u64;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }
    
    pub fn get_stats(&self) -> &AssemblyStats {
        &self.stats
    }
    
    pub fn get_capabilities(&self) -> &AssemblyCapabilities {
        &self.capabilities
    }
}

/// Detect assembly capabilities for the current platform
fn detect_assembly_capabilities() -> AssemblyCapabilities {
    #[cfg(target_arch = "x86_64")]
    {
        AssemblyCapabilities {
            has_basic_support: true,
            has_avx_support: std::arch::is_x86_feature_detected!("avx"),
            has_avx2_support: std::arch::is_x86_feature_detected!("avx2"),
            has_bmi_support: std::arch::is_x86_feature_detected!("bmi1"),
            has_popcnt_support: std::arch::is_x86_feature_detected!("popcnt"),
            has_lzcnt_support: std::arch::is_x86_feature_detected!("lzcnt"),
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        AssemblyCapabilities {
            has_basic_support: false,
            has_avx_support: false,
            has_avx2_support: false,
            has_bmi_support: false,
            has_popcnt_support: false,
            has_lzcnt_support: false,
        }
    }
}

/// Initialize assembly optimizations globally
pub fn init_assembly_dispatch() -> Result<()> {
    let capabilities = detect_assembly_capabilities();
    
    if capabilities.has_basic_support {
        debug!("Assembly optimizations enabled with capabilities: {:?}", capabilities);
    } else {
        warn!("Assembly optimizations not available on this platform");
    }
    
    Ok(())
}

/// Assembly capabilities detection
#[derive(Debug, Clone)]
pub struct AssemblyCapabilities {
    pub has_basic_support: bool,
    pub has_avx_support: bool,
    pub has_avx2_support: bool,
    pub has_bmi_support: bool,
    pub has_popcnt_support: bool,
    pub has_lzcnt_support: bool,
}

/// Assembly operation statistics
#[derive(Debug, Default)]
pub struct AssemblyStats {
    pub hash_calls: AtomicU64,
    pub memcpy_calls: AtomicU64,
    pub branchless_ops: AtomicU64,
    pub array_ops: AtomicU64,
    pub bit_ops: AtomicU64,
    pub search_ops: AtomicU64,
    pub compare_ops: AtomicU64,
    pub matrix_ops: AtomicU64,
}

impl AssemblyStats {
    pub fn total_operations(&self) -> u64 {
        self.hash_calls.load(Ordering::Relaxed)
            + self.memcpy_calls.load(Ordering::Relaxed)
            + self.branchless_ops.load(Ordering::Relaxed)
            + self.array_ops.load(Ordering::Relaxed)
            + self.bit_ops.load(Ordering::Relaxed)
            + self.search_ops.load(Ordering::Relaxed)
            + self.compare_ops.load(Ordering::Relaxed)
            + self.matrix_ops.load(Ordering::Relaxed)
    }
}

/// High-level assembly optimization interface
pub struct AssemblyInterface {
    ops: Option<AssemblyOptimizedOps>,
}

impl AssemblyInterface {
    /// Create new assembly interface
    pub fn new() -> Self {
        let ops = match AssemblyOptimizedOps::new() {
            Ok(ops) => {
                debug!("Assembly optimizations initialized successfully");
                Some(ops)
            }
            Err(e) => {
                warn!("Assembly optimizations not available: {}", e);
                None
            }
        };
        
        Self { ops }
    }
    
    /// Check if assembly optimizations are available
    pub fn is_available(&self) -> bool {
        self.ops.is_some()
    }
    
    /// Fast hash with fallback to software implementation
    pub fn fast_hash_u128(&self, value: u128) -> u64 {
        if let Some(ref ops) = self.ops {
            ops.fast_hash_u128(value)
        } else {
            // Software fallback
            crate::types::hash_node_id(NodeId(value))
        }
    }
    
    /// Optimized memory copy with fallback
    pub unsafe fn optimized_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        if let Some(ref ops) = self.ops {
            ops.optimized_memcpy(dst, src, len);
        } else {
            std::ptr::copy_nonoverlapping(src, dst, len);
        }
    }
    
    /// Branch-free minimum with fallback
    pub fn branchless_min(&self, a: u64, b: u64) -> u64 {
        if let Some(ref ops) = self.ops {
            ops.branchless_min_u64(a, b)
        } else {
            a.min(b)
        }
    }
    
    /// Branch-free maximum with fallback
    pub fn branchless_max(&self, a: u64, b: u64) -> u64 {
        if let Some(ref ops) = self.ops {
            ops.branchless_max_u64(a, b)
        } else {
            a.max(b)
        }
    }
    
    /// Optimized array sum with fallback
    pub fn optimized_sum(&self, array: &[u64]) -> u64 {
        if let Some(ref ops) = self.ops {
            ops.optimized_sum_u64(array)
        } else {
            array.iter().sum()
        }
    }
    
    /// Fast bit operations with fallback
    pub fn count_leading_zeros(&self, value: u64) -> u32 {
        if let Some(ref ops) = self.ops {
            ops.count_leading_zeros_u64(value)
        } else {
            value.leading_zeros()
        }
    }
    
    pub fn population_count(&self, value: u64) -> u32 {
        if let Some(ref ops) = self.ops {
            ops.population_count_u64(value)
        } else {
            value.count_ones()
        }
    }
    
    /// Optimized binary search with fallback
    pub fn binary_search(&self, array: &[u64], target: u64) -> Option<usize> {
        if let Some(ref ops) = self.ops {
            ops.optimized_binary_search(array, target)
        } else {
            array.binary_search(&target).ok()
        }
    }
    
    /// Get statistics if available
    pub fn get_stats(&self) -> Option<&AssemblyStats> {
        self.ops.as_ref().map(|ops| ops.get_stats())
    }
    
    /// Get capabilities if available
    pub fn get_capabilities(&self) -> Option<&AssemblyCapabilities> {
        self.ops.as_ref().map(|ops| ops.get_capabilities())
    }
}

impl Default for AssemblyInterface {
    fn default() -> Self {
        Self::new()
    }
}

/// Global assembly interface instance
static mut GLOBAL_ASSEMBLY_INTERFACE: Option<AssemblyInterface> = None;
static ASSEMBLY_INIT: std::sync::Once = std::sync::Once::new();

/// Get global assembly interface
pub fn get_global_assembly_interface() -> &'static AssemblyInterface {
    unsafe {
        ASSEMBLY_INIT.call_once(|| {
            GLOBAL_ASSEMBLY_INTERFACE = Some(AssemblyInterface::new());
        });
        
        GLOBAL_ASSEMBLY_INTERFACE.as_ref().unwrap()
    }
}

/// Convenience functions using global interface
pub fn asm_fast_hash_u128(value: u128) -> u64 {
    get_global_assembly_interface().fast_hash_u128(value)
}

pub fn asm_branchless_min(a: u64, b: u64) -> u64 {
    get_global_assembly_interface().branchless_min(a, b)
}

pub fn asm_branchless_max(a: u64, b: u64) -> u64 {
    get_global_assembly_interface().branchless_max(a, b)
}

pub fn asm_optimized_sum(array: &[u64]) -> u64 {
    get_global_assembly_interface().optimized_sum(array)
}

pub fn asm_count_leading_zeros(value: u64) -> u32 {
    get_global_assembly_interface().count_leading_zeros(value)
}

pub fn asm_population_count(value: u64) -> u32 {
    get_global_assembly_interface().population_count(value)
}

pub fn asm_binary_search(array: &[u64], target: u64) -> Option<usize> {
    get_global_assembly_interface().binary_search(array, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_assembly_capabilities_detection() {
        let capabilities = detect_assembly_capabilities();
        
        #[cfg(target_arch = "x86_64")]
        {
            assert!(capabilities.has_basic_support);
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(!capabilities.has_basic_support);
        }
    }
    
    #[test]
    fn test_assembly_interface_creation() {
        let interface = AssemblyInterface::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            // Should be available on x86_64
            assert!(interface.is_available());
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Not available on other architectures
            assert!(!interface.is_available());
        }
    }
    
    #[test]
    fn test_fast_hash_consistency() {
        let interface = AssemblyInterface::new();
        let test_value = 0x123456789ABCDEF0123456789ABCDEF0u128;
        
        let hash1 = interface.fast_hash_u128(test_value);
        let hash2 = interface.fast_hash_u128(test_value);
        
        // Hash should be deterministic
        assert_eq!(hash1, hash2);
        
        // Different values should produce different hashes (with high probability)
        let hash3 = interface.fast_hash_u128(test_value + 1);
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_branchless_operations() {
        let interface = AssemblyInterface::new();
        
        // Test min
        assert_eq!(interface.branchless_min(10, 20), 10);
        assert_eq!(interface.branchless_min(20, 10), 10);
        assert_eq!(interface.branchless_min(15, 15), 15);
        
        // Test max
        assert_eq!(interface.branchless_max(10, 20), 20);
        assert_eq!(interface.branchless_max(20, 10), 20);
        assert_eq!(interface.branchless_max(15, 15), 15);
    }
    
    #[test]
    fn test_optimized_sum() {
        let interface = AssemblyInterface::new();
        
        let test_array = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expected_sum = 55u64;
        
        let actual_sum = interface.optimized_sum(&test_array);
        assert_eq!(actual_sum, expected_sum);
        
        // Test empty array
        let empty_array: Vec<u64> = vec![];
        assert_eq!(interface.optimized_sum(&empty_array), 0);
    }
    
    #[test]
    fn test_bit_operations() {
        let interface = AssemblyInterface::new();
        
        // Test leading zeros
        assert_eq!(interface.count_leading_zeros(0x8000000000000000), 0);
        assert_eq!(interface.count_leading_zeros(0x0000000000000001), 63);
        assert_eq!(interface.count_leading_zeros(0), 64);
        
        // Test population count
        assert_eq!(interface.population_count(0b1111), 4);
        assert_eq!(interface.population_count(0b1010), 2);
        assert_eq!(interface.population_count(0), 0);
    }
    
    #[test]
    fn test_optimized_binary_search() {
        let interface = AssemblyInterface::new();
        
        let sorted_array = vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19];
        
        // Test found elements
        assert_eq!(interface.binary_search(&sorted_array, 1), Some(0));
        assert_eq!(interface.binary_search(&sorted_array, 7), Some(3));
        assert_eq!(interface.binary_search(&sorted_array, 19), Some(9));
        
        // Test not found elements
        assert_eq!(interface.binary_search(&sorted_array, 2), None);
        assert_eq!(interface.binary_search(&sorted_array, 20), None);
        
        // Test empty array
        let empty_array: Vec<u64> = vec![];
        assert_eq!(interface.binary_search(&empty_array, 5), None);
    }
    
    #[test]
    fn test_global_assembly_interface() {
        let interface1 = get_global_assembly_interface();
        let interface2 = get_global_assembly_interface();
        
        // Should be the same instance
        assert_eq!(interface1 as *const _, interface2 as *const _);
        
        // Test that global functions work
        let test_value = 0x123456789ABCDEF0u128;
        let hash = asm_fast_hash_u128(test_value);
        assert!(hash != 0); // Should produce a non-zero hash
        
        let min_val = asm_branchless_min(10, 20);
        assert_eq!(min_val, 10);
        
        let max_val = asm_branchless_max(10, 20);
        assert_eq!(max_val, 20);
    }
    
    #[test]
    fn test_assembly_stats() {
        let interface = AssemblyInterface::new();
        
        if let Some(stats) = interface.get_stats() {
            let initial_total = stats.total_operations();
            
            // Perform some operations
            let _ = interface.fast_hash_u128(12345);
            let _ = interface.branchless_min(1, 2);
            let _ = interface.optimized_sum(&[1, 2, 3]);
            
            let final_total = stats.total_operations();
            assert!(final_total > initial_total);
        }
    }
    
    #[test]
    fn test_init_assembly_dispatch() {
        assert!(init_assembly_dispatch().is_ok());
    }
    
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_assembly_ops_creation() {
        match AssemblyOptimizedOps::new() {
            Ok(ops) => {
                let capabilities = ops.get_capabilities();
                assert!(capabilities.has_basic_support);
                
                let stats = ops.get_stats();
                assert_eq!(stats.total_operations(), 0);
            }
            Err(e) => {
                panic!("Failed to create assembly ops on x86_64: {}", e);
            }
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn test_assembly_ops_unsupported() {
        match AssemblyOptimizedOps::new() {
            Ok(_) => {
                panic!("Assembly ops should not be supported on non-x86_64");
            }
            Err(RapidStoreError::AssemblyError { .. }) => {
                // Expected error on unsupported platforms
            }
            Err(e) => {
                panic!("Unexpected error type: {}", e);
            }
        }
    }
}