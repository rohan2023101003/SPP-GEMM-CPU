#pragma GCC optimize("O3,unroll-loops,tree-vectorize")
#pragma GCC target("avx2,fma,bmi,bmi2,popcnt")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace solution {
    // Align allocations to 32-byte boundary for AVX
    static constexpr size_t ALIGNMENT = 32;
    
    // Cache-optimized tile sizes
    // L1 cache is 1MiB per core, so we want to fit data in L1 cache
    static constexpr int BLOCK_SIZE_M = 128;
    static constexpr int BLOCK_SIZE_N = 128;
    static constexpr int BLOCK_SIZE_K = 128;
    
    // AVX2 processes 8 floats at once
    static constexpr int SIMD_WIDTH = 8;
    
    // Allocate aligned memory
    static float* allocate_aligned(size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, ALIGNMENT, size * sizeof(float)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<float*>(ptr);
    }

    // Optimized kernel for multiplying a micro-block using AVX2
    static inline void kernel_8x8(const float* A, const float* B, float* C, 
                                 int lda, int ldb, int ldc, int k) {
        // Register blocking: 8x8 output block using 8 AVX registers
        __m256 c00 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps();
        __m256 c40 = _mm256_setzero_ps();
        __m256 c50 = _mm256_setzero_ps();
        __m256 c60 = _mm256_setzero_ps();
        __m256 c70 = _mm256_setzero_ps();

        // Process k dimension in chunks with unrolling
        for (int l = 0; l < k; ++l) {
            // Load 8 values from B once, reuse for 8 rows of A
            __m256 b0 = _mm256_loadu_ps(&B[l * ldb]);
            
            // Process 8 rows of A against the same column of B
            __m256 a0 = _mm256_broadcast_ss(&A[0 * lda + l]);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            
            __m256 a1 = _mm256_broadcast_ss(&A[1 * lda + l]);
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            
            __m256 a2 = _mm256_broadcast_ss(&A[2 * lda + l]);
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            
            __m256 a3 = _mm256_broadcast_ss(&A[3 * lda + l]);
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            
            __m256 a4 = _mm256_broadcast_ss(&A[4 * lda + l]);
            c40 = _mm256_fmadd_ps(a4, b0, c40);
            
            __m256 a5 = _mm256_broadcast_ss(&A[5 * lda + l]);
            c50 = _mm256_fmadd_ps(a5, b0, c50);
            
            __m256 a6 = _mm256_broadcast_ss(&A[6 * lda + l]);
            c60 = _mm256_fmadd_ps(a6, b0, c60);
            
            __m256 a7 = _mm256_broadcast_ss(&A[7 * lda + l]);
            c70 = _mm256_fmadd_ps(a7, b0, c70);
        }
        
        // Store results with streaming store for better performance
        _mm256_storeu_ps(&C[0 * ldc], c00);
        _mm256_storeu_ps(&C[1 * ldc], c10);
        _mm256_storeu_ps(&C[2 * ldc], c20);
        _mm256_storeu_ps(&C[3 * ldc], c30);
        _mm256_storeu_ps(&C[4 * ldc], c40);
        _mm256_storeu_ps(&C[5 * ldc], c50);
        _mm256_storeu_ps(&C[6 * ldc], c60);
        _mm256_storeu_ps(&C[7 * ldc], c70);
    }

    // Compute a block with cache-friendly access pattern
    static void compute_block(const float* __restrict__ A, 
                             const float* __restrict__ B, 
                             float* __restrict__ C,
                             int n, int k, int m) {
        // Declare packed buffers first before marking them threadprivate
        static thread_local float* A_packed = nullptr;
        static thread_local float* B_packed = nullptr;
        
        // Lazy allocation of packed buffers
        if (!A_packed) A_packed = allocate_aligned(BLOCK_SIZE_N * BLOCK_SIZE_K);
        if (!B_packed) B_packed = allocate_aligned(BLOCK_SIZE_K * BLOCK_SIZE_M);
        
        // Zero out result matrix
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::fill_n(&C[i*m], m, 0.0f);
        }
        
        // 3-level blocking for L1/L2/L3 cache optimization
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < n; i += BLOCK_SIZE_N) {
            for (int j = 0; j < m; j += BLOCK_SIZE_M) {
                int ib = std::min(BLOCK_SIZE_N, n - i);
                int jb = std::min(BLOCK_SIZE_M, m - j);
                
                // Process k dimension in blocks
                for (int p = 0; p < k; p += BLOCK_SIZE_K) {
                    int pb = std::min(BLOCK_SIZE_K, k - p);
                    
                    // Pack a block of A into contiguous memory for better cache behavior
                    for (int ii = 0; ii < ib; ++ii) {
                        for (int kk = 0; kk < pb; ++kk) {
                            A_packed[ii * pb + kk] = A[(i + ii) * k + (p + kk)];
                        }
                    }
                    
                    // Pack a block of B into contiguous memory
                    for (int kk = 0; kk < pb; ++kk) {
                        for (int jj = 0; jj < jb; ++jj) {
                            B_packed[kk * jb + jj] = B[(p + kk) * m + (j + jj)];
                        }
                    }
                    
                    // Compute micro-blocks using vectorized kernel
                    for (int ii = 0; ii < ib; ii += 8) {
                        for (int jj = 0; jj < jb; jj += 8) {
                            if (ii + 8 <= ib && jj + 8 <= jb) {
                                // Full 8x8 block fits
                                kernel_8x8(&A_packed[ii * pb], &B_packed[0 * jb + jj], 
                                          &C[(i + ii) * m + (j + jj)], 
                                          pb, jb, m, pb);
                            } else {
                                // Handle edge cases
                                int iLimit = std::min(8, ib - ii);
                                int jLimit = std::min(8, jb - jj);
                                
                                // Scalar implementation for edges
                                for (int iii = 0; iii < iLimit; ++iii) {
                                    for (int jjj = 0; jjj < jLimit; ++jjj) {
                                        float sum = 0.0f;
                                        for (int kk = 0; kk < pb; ++kk) {
                                            sum += A_packed[(ii + iii) * pb + kk] * 
                                                   B_packed[kk * jb + (jj + jjj)];
                                        }
                                        C[(i + ii + iii) * m + (j + jj + jjj)] += sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";

        // Get optimal number of threads based on system
        int num_threads = std::min(64, n); // Use up to 64 threads, but not more than n rows
        omp_set_num_threads(num_threads);
        
        // Use direct I/O with larger buffer for better performance
        const int BUFFER_SIZE = 16 * 1024 * 1024; // 16MB buffer
        
        // Allocate aligned memory for matrices
        float* m1 = allocate_aligned(n * k);
        float* m2 = allocate_aligned(k * m);
        float* result = allocate_aligned(n * m);
        
        // Read input files efficiently
        {
            std::ifstream m1_fs(m1_path, std::ios::binary);
            if (!m1_fs) {
                throw std::runtime_error("Failed to open file: " + m1_path);
            }
            
            // Use buffer for better I/O performance
            std::vector<char> buffer(BUFFER_SIZE);
            m1_fs.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
            
            // Read matrix data
            m1_fs.read(reinterpret_cast<char*>(m1), n * k * sizeof(float));
            m1_fs.close();
        }
        
        {
            std::ifstream m2_fs(m2_path, std::ios::binary);
            if (!m2_fs) {
                throw std::runtime_error("Failed to open file: " + m2_path);
            }
            
            // Use buffer for better I/O performance
            std::vector<char> buffer(BUFFER_SIZE);
            m2_fs.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
            
            // Read matrix data
            m2_fs.read(reinterpret_cast<char*>(m2), k * m * sizeof(float));
            m2_fs.close();
        }
        
        // Start computation
        compute_block(m1, m2, result, n, k, m);
        
        // Write results efficiently
        {
            std::ofstream sol_fs(sol_path, std::ios::binary);
            if (!sol_fs) {
                throw std::runtime_error("Failed to open output file: " + sol_path);
            }
            
            // Use buffer for better I/O performance
            std::vector<char> buffer(BUFFER_SIZE);
            sol_fs.rdbuf()->pubsetbuf(buffer.data(), buffer.size());
            
            // Write matrix data
            sol_fs.write(reinterpret_cast<char*>(result), n * m * sizeof(float));
            sol_fs.close();
        }
        
        // Free allocated memory
        free(m1);
        free(m2);
        free(result);
        
        return sol_path;
    }
}