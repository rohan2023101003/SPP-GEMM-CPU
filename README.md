# GEMM Optimization

This document explains the optimization techniques applied to transform the original naive matrix multiplication implementation into a high-performance General Matrix Multiplication (GEMM) implementation.

## System Specifications

The optimized code was developed and tested on:
- **CPU**: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz (64 cores)
- **Architecture**: x86_64
- **Memory**: 32 MiB L2 cache, 44 MiB L3 cache
- **NUMA**: 2 NUMA nodes with 32 CPUs each
- **AVX Support**: AVX, AVX2, AVX512 instruction sets available

## Performance Improvement

The optimized code achieved a runtime of **312ms** for GEMM operations, which represents a significant speedup compared to the original naive implementation.

## Key Optimization Techniques

### 1. Memory Alignment

```cpp
float* m1_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * n * k));
float* m2_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * k * m));
float* result_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * n * m));
```

- **What**: Replaced standard memory allocation with 64-byte aligned memory allocation
- **Why**: Proper alignment ensures optimal AVX-512 vector load/store operations and prevents cache-line splits
- **Improvement**: Reduces memory access latency and enables efficient SIMD operations

### 2. Matrix Transposition

```cpp
float* m2t_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * k * m));
// Transpose code with parallel block implementation
```

- **What**: Transposed the second matrix for better memory access patterns
- **Why**: Converting from row-major to column-major access for the second matrix enables sequential memory access during multiplication
- **Improvement**: Better cache utilization and reduced TLB misses

### 3. Cache Blocking (Tiling)

```cpp
const int BN = 48, BM = 48, BK = 960;
```

- **What**: Implemented block-based matrix multiplication with carefully tuned block sizes
- **Why**: Ensures that blocks of data fit within L1/L2 caches
- **Improvement**: Drastically reduces cache misses and memory bandwidth requirements
- **Block Size Selection**: BN and BM were chosen to fit multiplication blocks within L1 cache, while BK allows for effective register reuse

### 4. Vectorization with AVX-512

```cpp
__m512 sum0 = _mm512_setzero_ps();
__m512 sum1 = _mm512_setzero_ps();
__m512 sum2 = _mm512_setzero_ps();
__m512 sum3 = _mm512_setzero_ps();

// Using AVX-512 intrinsics for SIMD operations
__m512 a0 = _mm512_loadu_ps(&m1[i * k + l]);
__m512 b0 = _mm512_loadu_ps(&m2t[j * k + l]);
sum0 = _mm512_fmadd_ps(a0, b0, sum0);
```

- **What**: Utilized AVX-512 SIMD instructions to process 16 floating-point operations simultaneously
- **Why**: AVX-512 allows processing 16 single-precision floats in parallel
- **Improvement**: Theoretical 16x throughput improvement per core
- **Technique**: Using multiple accumulators (sum0-sum3) to hide FMA latency and improve instruction-level parallelism

### 5. Multi-threading with OpenMP

```cpp
omp_set_num_threads(64);
#pragma omp parallel for schedule(static) proc_bind(close)
for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
    // Block processing
}
```

- **What**: Parallelized the computation across all 64 cores
- **Why**: Modern server CPUs have high core counts that must be utilized for maximum performance
- **Improvement**: Near-linear speedup with number of cores
- **NUMA Awareness**: Used `proc_bind(close)` to maintain thread-to-core locality and minimize NUMA effects

### 6. Loop Unrolling

```cpp
for (; l + 63 < k_end; l += 64) {
    // Process 4 chunks of 16 elements each
}
```

- **What**: Manual loop unrolling to process multiple data chunks per iteration
- **Why**: Reduces branch prediction misses and increases instruction-level parallelism
- **Improvement**: Better instruction pipelining and CPU utilization

### 7. Compiler Directives

```cpp
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512vl,avx512dq,avx512bw,avx512cd,avx512vbmi,avx512vnni,avx512bitalg,avx512vpopcntdq")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("inline")
```

- **What**: Advanced compiler optimizations and target-specific instruction set enablement
- **Why**: Ensures the compiler generates the most efficient code for the target architecture
- **Improvement**: Better auto-vectorization, function inlining, and math optimizations

## Comparison with Original Code

The original code performed a naive matrix multiplication with these limitations:

```cpp
for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
        result[i*m + j] = 0;
        for (int l = 0; l < k; ++l) 
            result[i*m + j] += m1[i*k + l] * m2[l*m + j];
    }
```

1. **No Parallelism**: Single-threaded execution despite having 64 available cores
2. **Poor Locality**: The inner loop accesses `m2` in a strided pattern, causing cache misses
3. **No Vectorization**: Doesn't leverage SIMD instructions available on the CPU
4. **No Cache Optimization**: Does not account for cache hierarchies, leading to unnecessary memory traffic
5. **Non-aligned Memory**: No guarantee that memory is properly aligned for vector operations

## Conclusion

The optimized GEMM implementation leverages multiple layers of parallelism (thread-level, instruction-level, and data-level) along with cache-conscious algorithms to achieve substantial performance improvements. The combination of cache blocking, vectorization, and multi-threading effectively utilizes the modern CPU architecture with its advanced instruction sets and multi-level cache hierarchy.