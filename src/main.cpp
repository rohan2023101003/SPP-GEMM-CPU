#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC target("avx512f", "avx512dq", "avx512bw", "avx512vl", "bmi", "bmi2", "fma")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <cstring>

// Define optimized block sizes based on cache sizes
// L1: 1MiB, L2: 32MiB, L3: 44MiB
#define MC 96    // Thread-local block size
#define KC 384   // Optimize for L2 cache
#define NC 2048  // Optimize for L3 cache

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        
        // Efficient file I/O with buffering
        constexpr size_t BUFFER_SIZE = 64 * 1024 * 1024; // 64MB buffer
        std::vector<char> buffer(BUFFER_SIZE);
        
        std::ifstream m1_fs(m1_path, std::ios::binary);
        std::ifstream m2_fs(m2_path, std::ios::binary);
        std::ofstream sol_fs(sol_path, std::ios::binary);
        
        m1_fs.rdbuf()->pubsetbuf(buffer.data(), BUFFER_SIZE/2);
        m2_fs.rdbuf()->pubsetbuf(buffer.data() + BUFFER_SIZE/2, BUFFER_SIZE/2);

        const int alignment = 64; // AVX-512 alignment
        
        // Use aligned memory allocation
        float* m1 = static_cast<float*>(_mm_malloc(sizeof(float) * n * k, alignment));
        float* m2 = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, alignment));
        float* result = static_cast<float*>(_mm_malloc(sizeof(float) * n * m, alignment));

        if (!m1 || !m2 || !result) {
            std::cerr << "Memory allocation failed" << std::endl;
            if (m1) _mm_free(m1);
            if (m2) _mm_free(m2);
            if (result) _mm_free(result);
            return "";
        }

        // Read input matrices
        m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();

        // Initialize result matrix to zeros - parallel initialization
        #pragma omp parallel for simd
        for (int i = 0; i < n * m; ++i) {
            result[i] = 0.0f;
        }

        // Determine optimal thread count based on NUMA architecture
        // Use 32 threads maximum (server has 64 logical cores, but 2 NUMA nodes)
        int num_threads = std::min(32, omp_get_max_threads());
        omp_set_num_threads(num_threads);
        
        // Set thread affinity for NUMA awareness
        #ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_schedule(omp_sched_static, 1);
        #endif

        // Pre-transposed copy of matrix B for better memory access
        float* m2_trans = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, alignment));
        if (!m2_trans) {
            std::cerr << "Transposed matrix allocation failed" << std::endl;
            _mm_free(m1); _mm_free(m2); _mm_free(result);
            return "";
        }
        
        // Transpose m2 for better cache locality during multiplication
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; j += 16) {
                int j_limit = std::min(j + 16, m);
                for (int jj = j; jj < j_limit; ++jj) {
                    m2_trans[jj * k + i] = m2[i * m + jj];
                }
            }
        }

        #pragma omp parallel
        {
            // Allocate thread-local storage for block computations
            float* local_C = static_cast<float*>(_mm_malloc(sizeof(float) * MC * NC, alignment));
            
            if (local_C) {
                // Static scheduling with chunk size for better load balancing
                #pragma omp for schedule(static, 1) collapse(2)
                for (int i_block = 0; i_block < n; i_block += MC) {
                    for (int j_block = 0; j_block < m; j_block += NC) {
                        int i_limit = std::min(i_block + MC, n);
                        int j_limit = std::min(j_block + NC, m);
                        
                        // Clear local result buffer
                        std::memset(local_C, 0, sizeof(float) * MC * NC);
                        
                        // Process k blocks
                        for (int k_block = 0; k_block < k; k_block += KC) {
                            int k_limit = std::min(k_block + KC, k);
                            
                            // Matrix multiplication with AVX-512 for each block
                            for (int i = i_block; i < i_limit; ++i) {
                                int local_i = i - i_block;
                                
                                for (int j = j_block; j < j_limit; j += 16) {
                                    // Load current C values
                                    __m512 c_vals = _mm512_load_ps(&local_C[local_i * NC + (j - j_block)]);
                                    
                                    // Process all values in this k block
                                    for (int kk = k_block; kk < k_limit; ++kk) {
                                        // Broadcast A value
                                        __m512 a_val = _mm512_set1_ps(m1[i * k + kk]);
                                        
                                        // Load B values (using transposed matrix for better locality)
                                        int end_j = std::min(j + 16, j_limit);
                                        if (end_j - j == 16) {
                                            // Full vector load
                                            __m512 b_vals = _mm512_load_ps(&m2_trans[j * k + kk]);
                                            
                                            // FMA operation
                                            c_vals = _mm512_fmadd_ps(a_val, b_vals, c_vals);
                                        } else {
                                            // Handle edge case with mask
                                            __mmask16 mask = _cvtu32_mask16((1 << (end_j - j)) - 1);
                                            __m512 b_vals = _mm512_maskz_load_ps(mask, &m2_trans[j * k + kk]);
                                            c_vals = _mm512_mask_fmadd_ps(c_vals, mask, a_val, b_vals);
                                        }
                                    }
                                    
                                    // Store back the result
                                    int end_j = std::min(j + 16, j_limit);
                                    if (end_j - j == 16) {
                                        _mm512_store_ps(&local_C[local_i * NC + (j - j_block)], c_vals);
                                    } else {
                                        __mmask16 mask = _cvtu32_mask16((1 << (end_j - j)) - 1);
                                        _mm512_mask_store_ps(&local_C[local_i * NC + (j - j_block)], mask, c_vals);
                                    }
                                }
                            }
                        }
                        
                        // Write local results back to global result matrix
                        for (int i = i_block; i < i_limit; ++i) {
                            int local_i = i - i_block;
                            int j = j_block;
                            
                            // Use vector operations to write back results in chunks
                            for (; j + 15 < j_limit; j += 16) {
                                __m512 c_vals = _mm512_load_ps(&local_C[local_i * NC + (j - j_block)]);
                                __m512 r_vals = _mm512_load_ps(&result[i * m + j]);
                                _mm512_store_ps(&result[i * m + j], _mm512_add_ps(r_vals, c_vals));
                            }
                            
                            // Handle remaining elements
                            for (; j < j_limit; ++j) {
                                result[i * m + j] += local_C[local_i * NC + (j - j_block)];
                            }
                        }
                    }
                }
                
                _mm_free(local_C);
            }
        }

        // Write result to output file
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();

        // Clean up
        _mm_free(m1);
        _mm_free(m2);
        _mm_free(m2_trans);
        _mm_free(result);

        return sol_path;
    }
}