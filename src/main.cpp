#pragma GCC optimize("Ofast,unroll-loops")
#pragma GCC target("avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx2,bmi,bmi2,fma,pclmul,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <algorithm>
#include <omp.h>

namespace solution {
    // Cache-optimized tile sizes based on Intel Xeon Gold 6226R
    // L1: 1 MiB, L2: 32 MiB, L3: 44 MiB per socket
    constexpr int BLOCK_SIZE_M = 128; // Adjusted for L2 cache
    constexpr int BLOCK_SIZE_N = 128;
    constexpr int BLOCK_SIZE_K = 256;
    
    // Memory alignment for AVX-512
    constexpr int ALIGN_BYTES = 64; // 512 bits = 64 bytes
    
    // AVX-512 processes 16 floats at a time
    constexpr int SIMD_WIDTH = 16;
    
    // Compute a block using AVX-512 with careful accumulation
    static void compute_block_avx512(const float* __restrict__ m1, 
                                  const float* __restrict__ m2, 
                                  float* __restrict__ result,
                                  const int n, const int k, const int m,
                                  const int i_start, const int i_end,
                                  const int j_start, const int j_end) {
        // Process each result element with AVX-512
        for (int i = i_start; i < i_end; ++i) {
            // Zero out this row of the result
            std::fill_n(&result[i*m + j_start], j_end - j_start, 0.0f);
            
            // Process blocks of columns for better cache utilization
            for (int jj = j_start; jj < j_end; jj += BLOCK_SIZE_M) {
                int j_block_end = std::min(jj + BLOCK_SIZE_M, j_end);
                
                // Align j_block_end down to SIMD boundary for vectorized part
                int j_simd_end = jj + ((j_block_end - jj) / SIMD_WIDTH) * SIMD_WIDTH;
                
                // Process in blocks for better cache usage
                for (int kk = 0; kk < k; kk += BLOCK_SIZE_K) {
                    int k_end = std::min(kk + BLOCK_SIZE_K, k);
                    
                    // For elements that can use AVX-512
                    for (int j = jj; j < j_simd_end; j += SIMD_WIDTH) {
                        // Load current result values
                        __m512 sum = _mm512_loadu_ps(&result[i*m + j]);
                        
                        // Inner loop over current k-block
                        for (int l = kk; l < k_end; ++l) {
                            // Broadcast single value from m1
                            __m512 m1_val = _mm512_set1_ps(m1[i*k + l]);
                            
                            // Load 16 values from m2
                            __m512 m2_vals = _mm512_loadu_ps(&m2[l*m + j]);
                            
                            // Multiply and accumulate using FMA
                            #ifdef __FMA__
                            sum = _mm512_fmadd_ps(m1_val, m2_vals, sum);
                            #else
                            sum = _mm512_add_ps(sum, _mm512_mul_ps(m1_val, m2_vals));
                            #endif
                        }
                        
                        // Store result back
                        _mm512_storeu_ps(&result[i*m + j], sum);
                    }
                    
                    // Handle remaining elements (less than SIMD_WIDTH) with scalar code
                    for (int j = j_simd_end; j < j_block_end; ++j) {
                        float* result_ptr = &result[i*m + j];
                        
                        // Process current k-block for this element
                        for (int l = kk; l < k_end; ++l) {
                            *result_ptr += m1[i*k + l] * m2[l*m + j];
                        }
                    }
                }
            }
        }
    }
    
    // Fallback AVX2 implementation for cases where AVX-512 might not be optimal
    static void compute_block_avx2(const float* __restrict__ m1, 
                                  const float* __restrict__ m2, 
                                  float* __restrict__ result,
                                  const int n, const int k, const int m,
                                  const int i_start, const int i_end,
                                  const int j_start, const int j_end) {
        constexpr int AVX2_WIDTH = 8; // AVX2 processes 8 floats at a time
        
        // Process each result element with AVX2
        for (int i = i_start; i < i_end; ++i) {
            // Zero out this row of the result
            std::fill_n(&result[i*m + j_start], j_end - j_start, 0.0f);
            
            // Process blocks of columns for better cache utilization
            for (int jj = j_start; jj < j_end; jj += BLOCK_SIZE_M) {
                int j_block_end = std::min(jj + BLOCK_SIZE_M, j_end);
                
                // Align j_block_end down to SIMD boundary for vectorized part
                int j_simd_end = jj + ((j_block_end - jj) / AVX2_WIDTH) * AVX2_WIDTH;
                
                // Process in blocks for better cache usage
                for (int kk = 0; kk < k; kk += BLOCK_SIZE_K) {
                    int k_end = std::min(kk + BLOCK_SIZE_K, k);
                    
                    // For elements that can use SIMD
                    for (int j = jj; j < j_simd_end; j += AVX2_WIDTH) {
                        // Load current result values
                        __m256 sum = _mm256_loadu_ps(&result[i*m + j]);
                        
                        // Process row of first matrix against column of second matrix
                        for (int l = kk; l < k_end; ++l) {
                            // Broadcast single value from m1
                            __m256 m1_val = _mm256_set1_ps(m1[i*k + l]);
                            
                            // Load 8 values from m2
                            __m256 m2_vals = _mm256_loadu_ps(&m2[l*m + j]);
                            
                            // Multiply and accumulate using FMA if available
                            #ifdef __FMA__
                            sum = _mm256_fmadd_ps(m1_val, m2_vals, sum);
                            #else
                            sum = _mm256_add_ps(sum, _mm256_mul_ps(m1_val, m2_vals));
                            #endif
                        }
                        
                        // Store result
                        _mm256_storeu_ps(&result[i*m + j], sum);
                    }
                    
                    // Handle remaining elements (less than AVX2_WIDTH)
                    for (int j = j_simd_end; j < j_block_end; ++j) {
                        float* result_ptr = &result[i*m + j];
                        
                        // Process current k-block for this element
                        for (int l = kk; l < k_end; ++l) {
                            *result_ptr += m1[i*k + l] * m2[l*m + j];
                        }
                    }
                }
            }
        }
    }
    
    // Helper for aligned memory allocation
    static float* aligned_alloc_float(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGN_BYTES, count * sizeof(float)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<float*>(ptr);
    }

    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        
        // Open input files with optimized buffering
        std::ifstream m1_fs(m1_path, std::ios::binary | std::ios::in);
        std::ifstream m2_fs(m2_path, std::ios::binary | std::ios::in);
        
        // Set larger buffer size for better I/O performance
        const int BUFFER_SIZE = 16 * 1024 * 1024; // 16MB buffer
        std::unique_ptr<char[]> m1_buffer(new char[BUFFER_SIZE]);
        std::unique_ptr<char[]> m2_buffer(new char[BUFFER_SIZE]);
        m1_fs.rdbuf()->pubsetbuf(m1_buffer.get(), BUFFER_SIZE);
        m2_fs.rdbuf()->pubsetbuf(m2_buffer.get(), BUFFER_SIZE);
        
        // Allocate memory for matrices with alignment for AVX-512
        float* m1 = aligned_alloc_float(n*k);
        float* m2 = aligned_alloc_float(k*m);
        float* result = aligned_alloc_float(n*m);
        
        // Read input data efficiently
        m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * k * m);
        m1_fs.close();
        m2_fs.close();
        
        // Zero out result matrix
        std::memset(result, 0, n * m * sizeof(float));
        
        // Auto-tuned thread count based on hardware and matrix dimensions
        int max_threads = omp_get_max_threads();
        int num_threads;
        
        // Determine optimal thread count based on matrix size
        if (n >= 4096) {
            num_threads = max_threads;
        } else if (n >= 2048) {
            num_threads = std::max(32, max_threads * 3 / 4);
        } else if (n >= 1024) {
            num_threads = std::max(16, max_threads / 2);
        } else if (n >= 512) {
            num_threads = std::max(8, max_threads / 4);
        } else if (n >= 256) {
            num_threads = std::min(4, max_threads);
        } else {
            num_threads = 1; // Single thread for small matrices
        }
        
        // Set thread count for OpenMP
        omp_set_num_threads(num_threads);
        
        // Divide matrix into 2D blocks for better cache utilization and OpenMP parallelism
        const int block_rows = 128; // Adjust based on cache size
        const int block_cols = m <= 512 ? m : 512; // Use full width for small matrices
        
        // Choose implementation based on matrix size and alignment
        bool use_avx512 = m >= 32 && (m % 16 <= 8);
        
        // Process matrix in blocks with OpenMP parallelism
        #pragma omp parallel
        {
            // Calculate NUMA node for this thread for potential optimization
            int thread_id = omp_get_thread_num();
            
            // Create better task distribution across NUMA nodes with guided scheduling
            #pragma omp for schedule(guided)
            for (int i_block = 0; i_block < n; i_block += block_rows) {
                int i_end = std::min(i_block + block_rows, n);
                
                for (int j_block = 0; j_block < m; j_block += block_cols) {
                    int j_end = std::min(j_block + block_cols, m);
                    
                    // Use appropriate SIMD implementation
                    if (use_avx512) {
                        compute_block_avx512(m1, m2, result, n, k, m, 
                                           i_block, i_end, j_block, j_end);
                    } else {
                        compute_block_avx2(m1, m2, result, n, k, m, 
                                          i_block, i_end, j_block, j_end);
                    }
                }
            }
        }
        
        // Write results with optimized buffering
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::unique_ptr<char[]> result_buffer(new char[BUFFER_SIZE]);
        sol_fs.rdbuf()->pubsetbuf(result_buffer.get(), BUFFER_SIZE);
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();
        
        // Free aligned memory
        free(m1);
        free(m2);
        free(result);
        
        return sol_path;
    }
};