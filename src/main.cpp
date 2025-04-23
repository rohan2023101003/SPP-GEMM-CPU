#pragma GCC optimize("O3","unroll-loops","inline","fast-math")
#pragma GCC option("arch=native","tune=native")
#pragma GCC target("avx512f","avx512dq","avx512bw","avx512vl","bmi","bmi2","fma","popcnt")

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
#include <cstring>
#include <numa.h>

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        // Initialize NUMA policy
        if (numa_available() >= 0) {
            numa_set_localalloc(); // Prefer local memory allocation
        }

        // Alignment for AVX-512
        const int alignment = 64;
        
        // Allocate aligned memory
        float* m1 = static_cast<float*>(_mm_malloc(sizeof(float) * n * k, alignment));
        float* m2 = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, alignment));
        float* result = static_cast<float*>(_mm_malloc(sizeof(float) * n * m, alignment));

        if (!m1 || !m2 || !result) {
            std::cerr << "Memory allocation failed" << std::endl;
            if (m1) _mm_free(m1);
            if (m2) _mm_free(m2);
            if (result) _mm_free(result);
            return sol_path;
        }

        // Read input matrices
        m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();

        // Zero out result matrix using AVX-512 instructions
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n * m; i += 16) {
            if (i + 16 <= n * m) {
                _mm512_storeu_ps(&result[i], _mm512_setzero_ps());
            } else {
                // Handle remaining elements
                for (int j = i; j < n * m; j++) {
                    result[j] = 0.0f;
                }
            }
        }

        // Transpose matrix B for better memory access patterns
        float* m2_transposed = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, alignment));
        if (!m2_transposed) {
            std::cerr << "Failed to allocate memory for transposed matrix" << std::endl;
            _mm_free(m1);
            _mm_free(m2);
            _mm_free(result);
            return sol_path;
        }
        
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < m; j++) {
                m2_transposed[j * k + i] = m2[i * m + j];
            }
        }

        // Get number of NUMA nodes
        int num_numa_nodes = 1;
        if (numa_available() >= 0) {
            num_numa_nodes = numa_num_configured_nodes();
        }

        // Set number of threads to utilize all physical cores (32 per NUMA node)
        int num_threads = 64; // Total cores from specs
        omp_set_num_threads(num_threads);

        // Optimized blocking parameters for Intel Xeon Gold 6226R
        const int MC = 96;   // Block size for M dimension (fits in L1)
        const int KC = 512;  // Block size for K dimension (fits in L2)
        const int NC = 4096; // Block size for N dimension (fits in L3)
        
        // Micro-kernel tile sizes
        const int MR = 6;    // 6 rows at a time
        const int NR = 16;   // Process 16 columns at a time (one AVX-512 register)

        // Flag to track allocation failures
        bool allocation_failed = false;

        // Main computation with multi-level blocking
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % num_numa_nodes;
            
            // Try to bind thread to correct NUMA node
            if (numa_available() >= 0) {
                numa_run_on_node(numa_node);
            }
            
            // Thread-local accumulation buffer (aligned)
            float* local_C = static_cast<float*>(_mm_malloc(sizeof(float) * MR * NR, alignment));
            
            // Check allocation success
            if (!local_C) {
                #pragma omp critical
                {
                    allocation_failed = true;
                    std::cerr << "Thread local buffer allocation failed" << std::endl;
                }
            } else {
                // Distribute blocks to threads in a NUMA-aware fashion
                #pragma omp for schedule(dynamic, 1) collapse(2)
                for (int i_block = 0; i_block < n; i_block += MC) {
                    for (int j_block = 0; j_block < m; j_block += NC) {
                        const int i_limit = std::min(i_block + MC, n);
                        const int j_limit = std::min(j_block + NC, m);
                        
                        // Process MC×NC block
                        for (int k_block = 0; k_block < k; k_block += KC) {
                            const int k_limit = std::min(k_block + KC, k);

                            // Process micro-blocks within the MC×KC×NC block
                            for (int i = i_block; i < i_limit; i += MR) {
                                const int i_micro_limit = std::min(i + MR, i_limit);
                                
                                for (int j = j_block; j < j_limit; j += NR) {
                                    const int j_micro_limit = std::min(j + NR, j_limit);
                                    
                                    // Initialize local accumulation buffer to zero
                                    for (int local_i = 0; local_i < MR; local_i++) {
                                        for (int local_j = 0; local_j < NR; local_j++) {
                                            local_C[local_i * NR + local_j] = 0.0f;
                                        }
                                    }
                                    
                                    // Process k dimension for this micro-block
                                    for (int kk = k_block; kk < k_limit; kk++) {
                                        // Compute MR×NR micro-kernel
                                        for (int ii = 0; ii < i_micro_limit - i; ii++) {
                                            const float* a_ptr = &m1[(i + ii) * k + kk];
                                            const float* b_ptr = &m2_transposed[j * k + kk];
                                            
                                            // Prefetch next A and B data
                                            if (kk + 8 < k_limit) {
                                                _mm_prefetch(reinterpret_cast<const char*>(&m1[(i + ii) * k + kk + 8]), _MM_HINT_T0);
                                                _mm_prefetch(reinterpret_cast<const char*>(&m2_transposed[j * k + kk + 8]), _MM_HINT_T1);
                                            }
                                            
                                            // Broadcast A element to vector register
                                            __m512 a_vec = _mm512_set1_ps(a_ptr[0]);
                                            
                                            // Process full NR-element chunks with AVX-512
                                            if (j + NR <= j_limit) {
                                                __m512 b_vec = _mm512_loadu_ps(b_ptr);
                                                __m512 c_vec = _mm512_loadu_ps(&local_C[ii * NR]);
                                                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                                _mm512_storeu_ps(&local_C[ii * NR], c_vec);
                                            } 
                                            // Handle edge case
                                            else {
                                                for (int jj = 0; jj < j_micro_limit - j; jj++) {
                                                    local_C[ii * NR + jj] += m1[(i + ii) * k + kk] * m2_transposed[(j + jj) * k + kk];
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Write micro-block result back to the global result matrix
                                    for (int ii = 0; ii < i_micro_limit - i; ii++) {
                                        for (int jj = 0; jj < j_micro_limit - j; jj++) {
                                            #pragma omp atomic update
                                            result[(i + ii) * m + (j + jj)] += local_C[ii * NR + jj];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Free thread-local buffer
                _mm_free(local_C);
            }
        }

        // Free transposed matrix
        _mm_free(m2_transposed);

        // Write result to output file
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();
        
        // Free allocated memory
        _mm_free(m1);
        _mm_free(m2);
        _mm_free(result);
        
        return sol_path;
    }
}

// Implementation of the required function signature for dense GEMM
void dense_gemm(const double *A, const double *B, double *C, int m, int k, int n) {
    // Set number of threads
    omp_set_num_threads(std::min(32, omp_get_max_threads()));
    
    // Zero out the C matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
        }
    }
    
    // Block sizes optimal for cache
    const int BLK_M = 64;
    const int BLK_N = 64;
    const int BLK_K = 64;
    
    // Blocked matrix multiplication with parallelism
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int i_blk = 0; i_blk < m; i_blk += BLK_M) {
        for (int j_blk = 0; j_blk < n; j_blk += BLK_N) {
            const int i_end = std::min(i_blk + BLK_M, m);
            const int j_end = std::min(j_blk + BLK_N, n);
            
            for (int k_blk = 0; k_blk < k; k_blk += BLK_K) {
                const int k_end = std::min(k_blk + BLK_K, k);
                
                for (int i = i_blk; i < i_end; i++) {
                    for (int kk = k_blk; kk < k_end; kk++) {
                        double a_val = A[i*k + kk];
                        
                        for (int j = j_blk; j < j_end; j++) {
                            C[i*n + j] += a_val * B[kk*n + j];
                        }
                    }
                }
            }
        }
    }
}

// Implementation of the required function signature for sparse SpMM
void sparse_spmm(
    const double *A_values, const int *A_col_ind, const int *A_row_ptr,
    const double *B_values, const int *B_col_ind, const int *B_row_ptr,
    double **C_values, int **C_col_ind, int **C_row_ptr,
    int m, int k, int n) {
    
    // Set optimal number of threads
    omp_set_num_threads(std::min(32, omp_get_max_threads()));
    
    // Parallel-friendly implementation
    std::vector<std::vector<std::pair<int, double>>> temp_rows(m);
    
    *C_row_ptr = new int[m + 1];
    (*C_row_ptr)[0] = 0;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        // Pre-allocate memory to reduce reallocations
        temp_rows[i].reserve(std::min(n, 32));  // Reasonable estimate based on sparsity
        
        std::vector<double> values(n, 0.0);
        std::vector<bool> is_nonzero(n, false);
        
        for (int j_ptr = A_row_ptr[i]; j_ptr < A_row_ptr[i + 1]; j_ptr++) {
            int j = A_col_ind[j_ptr];
            double val_A = A_values[j_ptr];
            
            // Prefetch next row of B for better cache utilization
            if (j_ptr + 1 < A_row_ptr[i + 1]) {
                int next_j = A_col_ind[j_ptr + 1];
                __builtin_prefetch(&B_row_ptr[next_j], 0, 3);
            }
            
            for (int k_ptr = B_row_ptr[j]; k_ptr < B_row_ptr[j + 1]; k_ptr++) {
                int k = B_col_ind[k_ptr];
                double val_B = B_values[k_ptr];
                
                values[k] += val_A * val_B;
                is_nonzero[k] = true;
            }
        }
        
        // Store non-zero entries
        for (int j = 0; j < n; j++) {
            if (is_nonzero[j]) {
                temp_rows[i].emplace_back(j, values[j]);
            }
        }
        
        // Sort by column index
        std::sort(temp_rows[i].begin(), temp_rows[i].end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    
    // Sequential part - calculate row pointers
    for (int i = 0; i < m; i++) {
        (*C_row_ptr)[i + 1] = (*C_row_ptr)[i] + temp_rows[i].size();
    }
    
    // Allocate memory for values and column indices
    int total_nnz = (*C_row_ptr)[m];
    *C_values = new double[total_nnz];
    *C_col_ind = new int[total_nnz];
    
    // Fill values and column indices in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        int idx = (*C_row_ptr)[i];
        for (const auto& entry : temp_rows[i]) {
            (*C_col_ind)[idx] = entry.first;
            (*C_values)[idx] = entry.second;
            idx++;
        }
    }
}