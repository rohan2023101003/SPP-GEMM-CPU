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

// Block sizes optimized for numerical stability and cache efficiency
#define MC 64    // Thread-local block size - smaller for better numerical stability
#define KC 256   // Slightly reduced for better balance
#define NC 1024  // Still large enough for vectorization benefits

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

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

        // Initialize result matrix to zeros
        #pragma omp parallel for
        for (int i = 0; i < n * m; ++i) {
            result[i] = 0.0f;
        }

        // Configure OpenMP
        int num_threads = std::min(32, omp_get_max_threads());
        omp_set_num_threads(num_threads);
        
        #ifdef _OPENMP
        omp_set_dynamic(0);
        #endif

        std::atomic<bool> allocation_failed(false);

        #pragma omp parallel
        {
            // Thread-local storage for block computations
            float* local_C = static_cast<float*>(_mm_malloc(sizeof(float) * MC * NC, alignment));
            if (!local_C) {
                allocation_failed.store(true);
            }

            #pragma omp barrier
            if (!allocation_failed.load()) {
                #pragma omp for schedule(static, 1)
                for (int i_block = 0; i_block < n; i_block += MC) {
                    for (int j_block = 0; j_block < m; j_block += NC) {
                        int i_limit = std::min(i_block + MC, n);
                        int j_limit = std::min(j_block + NC, m);
                        
                        // Initialize local result matrix
                        for (int i = 0; i < (i_limit - i_block) * NC; ++i) {
                            local_C[i] = 0.0f;
                        }
                        
                        // Block matrix multiplication
                        for (int k_block = 0; k_block < k; k_block += KC) {
                            int k_limit = std::min(k_block + KC, k);
                            
                            for (int i = i_block; i < i_limit; ++i) {
                                int local_i = i - i_block;
                                
                                for (int kk = k_block; kk < k_limit; ++kk) {
                                    float a_val = m1[i * k + kk];
                                    __m512 a_vec = _mm512_set1_ps(a_val);
                                    
                                    // Use proper alignment for vectorized operations
                                    int j = j_block;
                                    for (; j + 15 < j_limit; j += 16) {
                                        __m512 b_vec = _mm512_loadu_ps(&m2[kk * m + j]);
                                        __m512 c_vec = _mm512_loadu_ps(&local_C[local_i * NC + (j - j_block)]);
                                        
                                        // Use FMA for better precision with a single instruction
                                        c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                        _mm512_storeu_ps(&local_C[local_i * NC + (j - j_block)], c_vec);
                                    }
                                    
                                    // Handle edge cases precisely
                                    for (; j < j_limit; ++j) {
                                        local_C[local_i * NC + (j - j_block)] += a_val * m2[kk * m + j];
                                    }
                                }
                            }
                        }
                        
                        // Write local results back to global result with double precision accumulation
                        #pragma omp critical
                        {
                            for (int i = i_block; i < i_limit; ++i) {
                                int local_i = i - i_block;
                                for (int j = j_block; j < j_limit; ++j) {
                                    // This critical section ensures no race conditions when updating result
                                    result[i * m + j] += local_C[local_i * NC + (j - j_block)];
                                }
                            }
                        }
                    }
                }
            }

            if (local_C) _mm_free(local_C);
        }

        if (allocation_failed.load()) {
            std::cerr << "One or more threads failed memory allocation." << std::endl;
            _mm_free(m1); _mm_free(m2); _mm_free(result);
            return "";
        }

        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();

        // Clean up
        _mm_free(m1);
        _mm_free(m2);
        _mm_free(result);

        return sol_path;
    }
}