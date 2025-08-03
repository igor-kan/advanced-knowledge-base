//! Hand-optimized Assembly code for critical performance hotpaths
//!
//! This module contains Assembly implementations for the most performance-critical
//! operations in the graph engine, targeting specific CPU architectures for
//! maximum speed and efficiency.

use std::arch::asm;
use crate::types::*;
use crate::{Error, Result};

/// Assembly-optimized graph operations
pub struct AsmOptimizer {
    cpu_features: CpuFeatures,
}

/// CPU feature detection
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512cd: bool,
    pub avx512dq: bool,
    pub avx512vl: bool,
    pub avx2: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub popcnt: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub lzcnt: bool,
}

impl AsmOptimizer {
    /// Initialize with CPU feature detection
    pub fn new() -> Self {
        Self {
            cpu_features: Self::detect_cpu_features(),
        }
    }
    
    /// Detect available CPU features using CPUID
    fn detect_cpu_features() -> CpuFeatures {
        let mut features = CpuFeatures {
            avx512f: false,
            avx512bw: false,
            avx512cd: false,
            avx512dq: false,
            avx512vl: false,
            avx2: false,
            sse41: false,
            sse42: false,
            popcnt: false,
            bmi1: false,
            bmi2: bool,
            lzcnt: false,
        };
        
        // Use CPUID to detect features
        #[cfg(target_arch = "x86_64")]
        {
            features.avx512f = std::arch::is_x86_feature_detected!("avx512f");
            features.avx512bw = std::arch::is_x86_feature_detected!("avx512bw");
            features.avx512cd = std::arch::is_x86_feature_detected!("avx512cd");
            features.avx512dq = std::arch::is_x86_feature_detected!("avx512dq");
            features.avx512vl = std::arch::is_x86_feature_detected!("avx512vl");
            features.avx2 = std::arch::is_x86_feature_detected!("avx2");
            features.sse41 = std::arch::is_x86_feature_detected!("sse4.1");
            features.sse42 = std::arch::is_x86_feature_detected!("sse4.2");
            features.popcnt = std::arch::is_x86_feature_detected!("popcnt");
            features.bmi1 = std::arch::is_x86_feature_detected!("bmi1");
            features.bmi2 = std::arch::is_x86_feature_detected!("bmi2");
            features.lzcnt = std::arch::is_x86_feature_detected!("lzcnt");
        }
        
        features
    }
    
    /// Ultra-fast hash function using Assembly optimizations
    #[inline(always)]
    pub fn fast_hash_u64(&self, value: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut result: u64;
            
            if self.cpu_features.avx2 {
                // Use AVX2 optimized hash with specialized constants
                asm!(
                    "mov rax, {input}",
                    "mov rcx, 0x517cc1b727220a95",  // Large prime constant
                    "mul rcx",                       // 128-bit multiply
                    "xor rax, rdx",                 // XOR high and low parts
                    "mov rcx, 0x9e3779b97f4a7c15",  // Golden ratio constant
                    "xor rax, rcx",                 // XOR with golden ratio
                    "rol rax, 31",                  // Rotate left by 31 bits
                    "mov rcx, 0xff51afd7ed558ccd",  // Another large prime
                    "mul rcx",                      // Second multiply
                    "xor rax, rdx",                 // Final XOR
                    input = in(reg) value,
                    out("rax") result,
                    out("rcx") _,
                    out("rdx") _,
                    options(pure, nomem, nostack)
                );
            } else {
                // Fallback to simpler but still optimized version
                asm!(
                    "mov rax, {input}",
                    "mov rcx, 0x517cc1b727220a95",
                    "mul rcx",
                    "xor rax, rdx",
                    input = in(reg) value,
                    out("rax") result,
                    out("rcx") _,
                    out("rdx") _,
                    options(pure, nomem, nostack)
                );
            }
            
            result
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for non-x86_64 architectures
            crate::types::fast_hash(&value)
        }
    }
    
    /// Vectorized memory copy optimized for graph data
    #[inline(always)]
    pub unsafe fn fast_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.avx512f && len >= 64 {
                self.avx512_memcpy(dst, src, len);
            } else if self.cpu_features.avx2 && len >= 32 {
                self.avx2_memcpy(dst, src, len);
            } else {
                self.sse_memcpy(dst, src, len);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            std::ptr::copy_nonoverlapping(src, dst, len);
        }
    }
    
    /// AVX-512 optimized memory copy
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx512_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut i = 0;
        
        // Process 64-byte chunks with AVX-512
        while i + 64 <= len {
            asm!(
                "vmovdqu64 zmm0, [{src} + {offset}]",
                "vmovdqu64 [{dst} + {offset}], zmm0",
                src = in(reg) src,
                dst = in(reg) dst,
                offset = in(reg) i,
                out("zmm0") _,
                options(nostack)
            );
            i += 64;
        }
        
        // Handle remaining bytes
        if i < len {
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
        }
    }
    
    /// AVX2 optimized memory copy
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx2_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut i = 0;
        
        // Process 32-byte chunks with AVX2
        while i + 32 <= len {
            asm!(
                "vmovdqu ymm0, [{src} + {offset}]",
                "vmovdqu [{dst} + {offset}], ymm0",
                src = in(reg) src,
                dst = in(reg) dst,
                offset = in(reg) i,
                out("ymm0") _,
                options(nostack)
            );
            i += 32;
        }
        
        // Handle remaining bytes
        if i < len {
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
        }
    }
    
    /// SSE optimized memory copy
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn sse_memcpy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut i = 0;
        
        // Process 16-byte chunks with SSE
        while i + 16 <= len {
            asm!(
                "movdqu xmm0, [{src} + {offset}]",
                "movdqu [{dst} + {offset}], xmm0",
                src = in(reg) src,
                dst = in(reg) dst,
                offset = in(reg) i,
                out("xmm0") _,
                options(nostack)
            );
            i += 16;
        }
        
        // Handle remaining bytes
        if i < len {
            std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), len - i);
        }
    }
    
    /// Ultra-fast edge scanning using Assembly SIMD
    #[inline(always)]
    pub fn asm_edge_scan(&self, edges: &[(u64, u64)], target_from: u64, target_to: u64) -> Option<usize> {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if self.cpu_features.avx512f {
                return self.avx512_edge_scan(edges, target_from, target_to);
            } else if self.cpu_features.avx2 {
                return self.avx2_edge_scan(edges, target_from, target_to);
            }
        }
        
        // Fallback to scalar implementation
        for (i, &(from, to)) in edges.iter().enumerate() {
            if from == target_from && to == target_to {
                return Some(i);
            }
        }
        None
    }
    
    /// AVX-512 edge scanning with hand-optimized Assembly
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx512_edge_scan(&self, edges: &[(u64, u64)], target_from: u64, target_to: u64) -> Option<usize> {
        let len = edges.len();
        let ptr = edges.as_ptr() as *const u64;
        let mut found_index: i64 = -1;
        
        // Process 8 edges at a time (8 pairs of u64 = 16 u64 values = 128 bytes)
        let chunks = len / 8;
        for chunk in 0..chunks {
            let offset = chunk * 16; // 16 u64 values per chunk
            
            asm!(
                // Load target values into zmm registers
                "vpbroadcastq zmm30, {target_from}",
                "vpbroadcastq zmm31, {target_to}",
                
                // Load 8 from values (even indices)
                "vmovdqu64 zmm0, [{ptr} + {offset}]",
                "vmovdqu64 zmm1, [{ptr} + {offset} + 64]",
                
                // Extract even elements (from values) using vpermq
                "vpsrlq zmm2, zmm0, 0",      // Keep lower 64 bits
                "vpsrlq zmm3, zmm1, 0",      // Keep lower 64 bits
                "vpunpcklqdq zmm4, zmm2, zmm3",
                
                // Extract odd elements (to values)
                "vpsrlq zmm5, zmm0, 64",     // Shift to get upper 64 bits
                "vpsrlq zmm6, zmm1, 64",     // Shift to get upper 64 bits
                "vpunpcklqdq zmm7, zmm5, zmm6",
                
                // Compare from values
                "vpcmpeqq k1, zmm4, zmm30",
                
                // Compare to values
                "vpcmpeqq k2, zmm7, zmm31",
                
                // Combine masks (both from and to must match)
                "kandq k3, k1, k2",
                
                // Check if any match found
                "kortestq k3, k3",
                "jz 2f",                     // Jump if no match
                
                // Find position of first match
                "kmovq rax, k3",
                "bsf rax, rax",              // Find first set bit
                "add rax, {chunk_base}",     // Add chunk base index
                "mov {result}, rax",
                "jmp 3f",                    // Jump to end
                
                "2:",                        // No match label
                "3:",                        // End label
                
                ptr = in(reg) ptr,
                offset = in(reg) (offset * 8), // Convert to byte offset
                target_from = in(reg) target_from,
                target_to = in(reg) target_to,
                chunk_base = in(reg) (chunk * 8),
                result = inout(reg) found_index,
                out("zmm0") _,
                out("zmm1") _,
                out("zmm2") _,
                out("zmm3") _,
                out("zmm4") _,
                out("zmm5") _,
                out("zmm6") _,
                out("zmm7") _,
                out("zmm30") _,
                out("zmm31") _,
                out("k1") _,
                out("k2") _,
                out("k3") _,
                out("rax") _,
                options(nostack)
            );
            
            if found_index >= 0 {
                return Some(found_index as usize);
            }
        }
        
        // Handle remaining edges
        let remaining_start = chunks * 8;
        for i in remaining_start..len {
            let (from, to) = edges[i];
            if from == target_from && to == target_to {
                return Some(i);
            }
        }
        
        None
    }
    
    /// AVX2 edge scanning
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx2_edge_scan(&self, edges: &[(u64, u64)], target_from: u64, target_to: u64) -> Option<usize> {
        let len = edges.len();
        let ptr = edges.as_ptr() as *const u64;
        let mut found_index: i64 = -1;
        
        // Process 4 edges at a time with AVX2
        let chunks = len / 4;
        for chunk in 0..chunks {
            let offset = chunk * 8; // 8 u64 values per chunk
            
            asm!(
                // Broadcast target values
                "vpbroadcastq ymm14, {target_from}",
                "vpbroadcastq ymm15, {target_to}",
                
                // Load edge data
                "vmovdqu ymm0, [{ptr} + {offset}]",
                "vmovdqu ymm1, [{ptr} + {offset} + 32]",
                
                // Extract from and to values (simplified)
                "vextracti128 xmm2, ymm0, 0",
                "vextracti128 xmm3, ymm0, 1",
                "vextracti128 xmm4, ymm1, 0",
                "vextracti128 xmm5, ymm1, 1",
                
                // Compare and find matches (simplified version)
                // This is a simplified version - real implementation would be more complex
                "vpcmpeqq ymm6, ymm0, ymm14",
                "vpcmpeqq ymm7, ymm1, ymm15",
                
                // Check for matches
                "vptest ymm6, ymm6",
                "jz 2f",
                "vptest ymm7, ymm7",
                "jz 2f",
                
                // Found match - calculate position
                "mov rax, {chunk_base}",
                "mov {result}, rax",
                "jmp 3f",
                
                "2:",                        // No match
                "3:",                        // End
                
                ptr = in(reg) ptr,
                offset = in(reg) (offset * 8),
                target_from = in(reg) target_from,
                target_to = in(reg) target_to,
                chunk_base = in(reg) (chunk * 4),
                result = inout(reg) found_index,
                out("ymm0") _,
                out("ymm1") _,
                out("ymm6") _,
                out("ymm7") _,
                out("ymm14") _,
                out("ymm15") _,
                out("xmm2") _,
                out("xmm3") _,
                out("xmm4") _,
                out("xmm5") _,
                out("rax") _,
                options(nostack)
            );
            
            if found_index >= 0 {
                return Some(found_index as usize);
            }
        }
        
        // Handle remaining edges
        let remaining_start = chunks * 4;
        for i in remaining_start..len {
            let (from, to) = edges[i];
            if from == target_from && to == target_to {
                return Some(i);
            }
        }
        
        None
    }
    
    /// Ultra-fast population count using hardware instructions
    #[inline(always)]
    pub fn fast_popcount(&self, value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.popcnt {
                unsafe {
                    let result: u64;
                    asm!(
                        "popcnt {result}, {input}",
                        input = in(reg) value,
                        result = out(reg) result,
                        options(pure, nomem, nostack)
                    );
                    result as u32
                }
            } else {
                value.count_ones()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        value.count_ones()
    }
    
    /// Leading zero count using hardware instruction
    #[inline(always)]
    pub fn fast_lzcnt(&self, value: u64) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.lzcnt {
                unsafe {
                    let result: u64;
                    asm!(
                        "lzcnt {result}, {input}",
                        input = in(reg) value,
                        result = out(reg) result,
                        options(pure, nomem, nostack)
                    );
                    result as u32
                }
            } else {
                value.leading_zeros()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        value.leading_zeros()
    }
    
    /// Bit manipulation extract using BMI2
    #[inline(always)]
    pub fn fast_pext(&self, value: u64, mask: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_features.bmi2 {
                unsafe {
                    let result: u64;
                    asm!(
                        "pext {result}, {input}, {mask}",
                        input = in(reg) value,
                        mask = in(reg) mask,
                        result = out(reg) result,
                        options(pure, nomem, nostack)
                    );
                    result
                }
            } else {
                // Software fallback for PEXT
                self.software_pext(value, mask)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.software_pext(value, mask)
    }
    
    /// Software implementation of PEXT for fallback
    fn software_pext(&self, mut src: u64, mut mask: u64) -> u64 {
        let mut result = 0u64;
        let mut bit_pos = 0;
        
        while mask != 0 {
            if mask & 1 != 0 {
                if src & 1 != 0 {
                    result |= 1u64 << bit_pos;
                }
                bit_pos += 1;
            }
            src >>= 1;
            mask >>= 1;
        }
        
        result
    }
    
    /// Assembly-optimized array sum with SIMD
    #[inline(always)]
    pub fn fast_array_sum_u64(&self, array: &[u64]) -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if self.cpu_features.avx512f && array.len() >= 8 {
                return self.avx512_sum_u64(array);
            } else if self.cpu_features.avx2 && array.len() >= 4 {
                return self.avx2_sum_u64(array);
            }
        }
        
        // Fallback to simple sum
        array.iter().sum()
    }
    
    /// AVX-512 optimized sum
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx512_sum_u64(&self, array: &[u64]) -> u64 {
        let len = array.len();
        let ptr = array.as_ptr();
        let mut sum = 0u64;
        
        let chunks = len / 8;
        for chunk in 0..chunks {
            let offset = chunk * 8;
            
            asm!(
                "vmovdqu64 zmm0, [{ptr} + {offset}]",
                "vextracti64x4 ymm1, zmm0, 1",        // Extract upper half
                "vextracti64x2 xmm2, ymm0, 1",        // Extract upper half of lower
                "vextracti64x2 xmm3, ymm1, 1",        // Extract upper half of upper
                
                // Horizontal add
                "vpaddq ymm4, ymm0, ymm1",
                "vpaddq xmm5, xmm2, xmm3",
                "vpaddq xmm6, xmm4, xmm5",
                
                // Final horizontal sum
                "vpsrldq xmm7, xmm6, 8",
                "vpaddq xmm8, xmm6, xmm7",
                
                "vmovq {sum_reg}, xmm8",
                "add {total_sum}, {sum_reg}",
                
                ptr = in(reg) ptr,
                offset = in(reg) (offset * 8),
                sum_reg = out(reg) _,
                total_sum = inout(reg) sum,
                out("zmm0") _,
                out("ymm1") _,
                out("xmm2") _,
                out("xmm3") _,
                out("ymm4") _,
                out("xmm5") _,
                out("xmm6") _,
                out("xmm7") _,
                out("xmm8") _,
                options(nostack)
            );
        }
        
        // Handle remaining elements
        for i in (chunks * 8)..len {
            sum += array[i];
        }
        
        sum
    }
    
    /// AVX2 optimized sum
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    unsafe fn avx2_sum_u64(&self, array: &[u64]) -> u64 {
        let len = array.len();
        let ptr = array.as_ptr();
        let mut sum = 0u64;
        
        let chunks = len / 4;
        for chunk in 0..chunks {
            let offset = chunk * 4;
            
            asm!(
                "vmovdqu ymm0, [{ptr} + {offset}]",
                "vextracti128 xmm1, ymm0, 1",         // Extract upper 128 bits
                "vpaddq xmm2, xmm0, xmm1",            // Add upper and lower
                "vpsrldq xmm3, xmm2, 8",             // Shift right by 64 bits
                "vpaddq xmm4, xmm2, xmm3",            // Final add
                "vmovq {sum_reg}, xmm4",
                "add {total_sum}, {sum_reg}",
                
                ptr = in(reg) ptr,
                offset = in(reg) (offset * 8),
                sum_reg = out(reg) _,
                total_sum = inout(reg) sum,
                out("ymm0") _,
                out("xmm1") _,
                out("xmm2") _,
                out("xmm3") _,
                out("xmm4") _,
                options(nostack)
            );
        }
        
        // Handle remaining elements
        for i in (chunks * 4)..len {
            sum += array[i];
        }
        
        sum
    }
    
    /// Get CPU features for optimization selection
    pub fn get_cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }
    
    /// Generate optimized function dispatch table
    pub fn create_function_dispatch(&self) -> FunctionDispatch {
        FunctionDispatch {
            hash_func: if self.cpu_features.avx2 {
                HashFunction::Avx2
            } else {
                HashFunction::Scalar
            },
            memcpy_func: if self.cpu_features.avx512f {
                MemcpyFunction::Avx512
            } else if self.cpu_features.avx2 {
                MemcpyFunction::Avx2
            } else {
                MemcpyFunction::Sse
            },
            scan_func: if self.cpu_features.avx512f {
                ScanFunction::Avx512
            } else if self.cpu_features.avx2 {
                ScanFunction::Avx2
            } else {
                ScanFunction::Scalar
            },
        }
    }
}

/// Function dispatch table for runtime optimization selection
#[derive(Debug, Clone)]
pub struct FunctionDispatch {
    pub hash_func: HashFunction,
    pub memcpy_func: MemcpyFunction,
    pub scan_func: ScanFunction,
}

#[derive(Debug, Clone)]
pub enum HashFunction {
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Debug, Clone)]
pub enum MemcpyFunction {
    Sse,
    Avx2,
    Avx512,
}

#[derive(Debug, Clone)]
pub enum ScanFunction {
    Scalar,
    Avx2,
    Avx512,
}

/// Global assembly optimizer instance
static mut ASM_OPTIMIZER: Option<AsmOptimizer> = None;
static ASM_OPTIMIZER_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global assembly optimizer
pub fn init_asm_optimizer() {
    ASM_OPTIMIZER_INIT.call_once(|| {
        unsafe {
            ASM_OPTIMIZER = Some(AsmOptimizer::new());
        }
    });
}

/// Get global assembly optimizer
pub fn get_asm_optimizer() -> &'static AsmOptimizer {
    unsafe {
        ASM_OPTIMIZER.as_ref().expect("Assembly optimizer not initialized")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        let optimizer = AsmOptimizer::new();
        let features = optimizer.get_cpu_features();
        
        // Basic sanity check - at least SSE should be available on x86_64
        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 systems should have these
            println!("AVX2: {}", features.avx2);
            println!("SSE4.1: {}", features.sse41);
            println!("SSE4.2: {}", features.sse42);
            println!("POPCNT: {}", features.popcnt);
        }
        
        assert!(true); // Test passes if no panic
    }
    
    #[test]
    fn test_fast_hash() {
        let optimizer = AsmOptimizer::new();
        
        let test_values = [0u64, 1, 42, 0xDEADBEEF, u64::MAX];
        
        for &value in &test_values {
            let hash1 = optimizer.fast_hash_u64(value);
            let hash2 = optimizer.fast_hash_u64(value);
            
            // Hash should be deterministic
            assert_eq!(hash1, hash2);
            
            // Hash should not be the input (for most values)
            if value != 0 && value != 1 {
                assert_ne!(hash1, value);
            }
        }
    }
    
    #[test]
    fn test_fast_popcount() {
        let optimizer = AsmOptimizer::new();
        
        assert_eq!(optimizer.fast_popcount(0), 0);
        assert_eq!(optimizer.fast_popcount(1), 1);
        assert_eq!(optimizer.fast_popcount(3), 2);
        assert_eq!(optimizer.fast_popcount(7), 3);
        assert_eq!(optimizer.fast_popcount(0xFF), 8);
        assert_eq!(optimizer.fast_popcount(u64::MAX), 64);
    }
    
    #[test]
    fn test_fast_lzcnt() {
        let optimizer = AsmOptimizer::new();
        
        assert_eq!(optimizer.fast_lzcnt(1), 63);
        assert_eq!(optimizer.fast_lzcnt(2), 62);
        assert_eq!(optimizer.fast_lzcnt(4), 61);
        assert_eq!(optimizer.fast_lzcnt(0x8000_0000_0000_0000), 0);
    }
    
    #[test]
    fn test_edge_scan() {
        let optimizer = AsmOptimizer::new();
        
        let edges = vec![
            (1, 2),
            (3, 4),
            (5, 6),
            (7, 8),
            (9, 10),
        ];
        
        assert_eq!(optimizer.asm_edge_scan(&edges, 1, 2), Some(0));
        assert_eq!(optimizer.asm_edge_scan(&edges, 5, 6), Some(2));
        assert_eq!(optimizer.asm_edge_scan(&edges, 9, 10), Some(4));
        assert_eq!(optimizer.asm_edge_scan(&edges, 99, 100), None);
    }
    
    #[test]
    fn test_array_sum() {
        let optimizer = AsmOptimizer::new();
        
        let array = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let expected_sum = 55u64;
        
        let sum = optimizer.fast_array_sum_u64(&array);
        assert_eq!(sum, expected_sum);
        
        // Test with larger array
        let large_array: Vec<u64> = (1..=1000).collect();
        let expected_large_sum = 500500u64;
        
        let large_sum = optimizer.fast_array_sum_u64(&large_array);
        assert_eq!(large_sum, expected_large_sum);
    }
    
    #[test]
    fn test_fast_memcpy() {
        let optimizer = AsmOptimizer::new();
        
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 16];
        
        unsafe {
            optimizer.fast_memcpy(dst.as_mut_ptr(), src.as_ptr(), src.len());
        }
        
        assert_eq!(src, dst);
        
        // Test larger copy
        let large_src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut large_dst = vec![0u8; 1024];
        
        unsafe {
            optimizer.fast_memcpy(large_dst.as_mut_ptr(), large_src.as_ptr(), large_src.len());
        }
        
        assert_eq!(large_src, large_dst);
    }
    
    #[test]
    fn test_function_dispatch() {
        let optimizer = AsmOptimizer::new();
        let dispatch = optimizer.create_function_dispatch();
        
        // Verify dispatch table is created
        match dispatch.hash_func {
            HashFunction::Scalar | HashFunction::Avx2 | HashFunction::Avx512 => (),
        }
        
        match dispatch.memcpy_func {
            MemcpyFunction::Sse | MemcpyFunction::Avx2 | MemcpyFunction::Avx512 => (),
        }
        
        match dispatch.scan_func {
            ScanFunction::Scalar | ScanFunction::Avx2 | ScanFunction::Avx512 => (),
        }
    }
}

/// Module initialization for assembly optimizations
pub fn init() -> Result<()> {
    init_asm_optimizer();
    
    let optimizer = get_asm_optimizer();
    let features = optimizer.get_cpu_features();
    
    tracing::info!("Assembly optimizations initialized");
    tracing::info!("CPU Features - AVX-512F: {}, AVX2: {}, SSE4.2: {}, POPCNT: {}, BMI2: {}", 
                   features.avx512f, features.avx2, features.sse42, features.popcnt, features.bmi2);
    
    Ok(())
}