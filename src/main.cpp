#pragma GCC optimize("O3","unroll-loops","inline")
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

        // Zero out result matrix
        std::memset(result, 0, sizeof(float) * n * m);

        // Set number of threads based on NUMA topology
        int num_numa_nodes = 1;
        if (numa_available() >= 0) {
            num_numa_nodes = numa_num_configured_nodes();
        }
        
        // We have 32 cores per socket (64 total)
        int num_threads = 64;
        omp_set_num_threads(num_threads);

        // Optimized blocking parameters
        // Tuned for Intel Xeon Gold cache hierarchy
        const int MC = 128;  // Block size for M dimension
        const int KC = 384;  // Block size for K dimension
        const int NC = 3072; // Block size for N dimension

        // Smaller thread-local blocks for better cache utilization
        const int MR = 4;    // 4 rows at a time
        const int NR = 16;   // 16 columns at a time (one AVX-512 register)

        // Main computation with multi-level blocking
        #pragma omp parallel
        {
            // Bind thread to appropriate NUMA node
            int thread_id = omp_get_thread_num();
            int numa_node = thread_id % num_numa_nodes;
            
            if (numa_available() >= 0) {
                numa_run_on_node(numa_node);
            }
            
            // Thread-private result matrix
            float* local_result = static_cast<float*>(_mm_malloc(sizeof(float) * n * m, alignment));
            if (!local_result) {
                std::cerr << "Thread-local result allocation failed" << std::endl;
                return;
            }
            
            // Initialize local result to zero
            std::memset(local_result, 0, sizeof(float) * n * m);
            
            // Small tile buffer for inner computation
            float local_C[MR * NR];
            
            // Process blocks in parallel
            #pragma omp for schedule(dynamic, 1)
            for (int i_block = 0; i_block < n; i_block += MC) {
                const int i_limit = std::min(i_block + MC, n);
                
                for (int j_block = 0; j_block < m; j_block += NC) {
                    const int j_limit = std::min(j_block + NC, m);
                    
                    // Process this panel
                    for (int k_block = 0; k_block < k; k_block += KC) {
                        const int k_limit = std::min(k_block + KC, k);

                        // Process micro-blocks
                        for (int i = i_block; i < i_limit; i += MR) {
                            const int i_micro_limit = std::min(i + MR, i_limit);
                            
                            for (int j = j_block; j < j_limit; j += NR) {
                                const int j_micro_limit = std::min(j + NR, j_limit);
                                
                                // Zero out local tile
                                std::memset(local_C, 0, sizeof(float) * MR * NR);
                                
                                // Compute small tile
                                for (int kk = k_block; kk < k_limit; kk++) {
                                    for (int ii = 0; ii < i_micro_limit - i; ii++) {
                                        // Load row of A into registers
                                        const float a_val = m1[(i + ii) * k + kk];
                                        const __m512 a_vec = _mm512_set1_ps(a_val);
                                        
                                        // Process full vector chunks
                                        int jj = 0;
                                        for (; jj < ((j_micro_limit - j) / 16) * 16; jj += 16) {
                                            // Load, multiply and accumulate
                                            const __m512 b_vec = _mm512_loadu_ps(&m2[kk * m + j + jj]);
                                            const __m512 c_vec = _mm512_loadu_ps(&local_C[ii * NR + jj]);
                                            const __m512 result_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                            _mm512_storeu_ps(&local_C[ii * NR + jj], result_vec);
                                        }
                                        
                                        // Handle remainder elements
                                        for (; jj < j_micro_limit - j; jj++) {
                                            local_C[ii * NR + jj] += a_val * m2[kk * m + j + jj];
                                        }
                                    }
                                }
                                
                                // Store results in thread-local result matrix
                                for (int ii = 0; ii < i_micro_limit - i; ii++) {
                                    for (int jj = 0; jj < j_micro_limit - j; jj++) {
                                        local_result[(i + ii) * m + (j + jj)] += local_C[ii * NR + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Reduce thread-local results into the global result
            #pragma omp critical
            {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        result[i * m + j] += local_result[i * m + j];
                    }
                }
            }
            
            // Free thread-local result
            if (local_result) {
                _mm_free(local_result);
            }
        }

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

// Basic implementation of the required function signatures for compatibility
void dense_gemm(const double *A, const double *B, double *C, int m, int k, int n) {
    // Zero out C matrix
    std::memset(C, 0, sizeof(double) * m * n);
    
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            double a_val = A[i*k + kk];
            for (int j = 0; j < n; j++) {
                C[i*n + j] += a_val * B[kk*n + j];
            }
        }
    }
}

void sparse_spmm(
    const double *A_values, const int *A_col_ind, const int *A_row_ptr,
    const double *B_values, const int *B_col_ind, const int *B_row_ptr,
    double **C_values, int **C_col_ind, int **C_row_ptr,
    int m, int k, int n) {
    
    // Parallel-friendly implementation
    std::vector<std::vector<std::pair<int, double>>> temp_rows(m);
    
    *C_row_ptr = new int[m + 1];
    (*C_row_ptr)[0] = 0;
    
    #pragma omp parallel
    {
        // Each thread processes some rows
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < m; i++) {
            std::vector<double> values(n, 0.0);
            std::vector<bool> is_nonzero(n, false);
            
            for (int j_ptr = A_row_ptr[i]; j_ptr < A_row_ptr[i + 1]; j_ptr++) {
                int j = A_col_ind[j_ptr];
                double val_A = A_values[j_ptr];
                
                for (int k_ptr = B_row_ptr[j]; k_ptr < B_row_ptr[j + 1]; k_ptr++) {
                    int k = B_col_ind[k_ptr];
                    double val_B = B_values[k_ptr];
                    
                    values[k] += val_A * val_B;
                    is_nonzero[k] = true;
                }
            }
            
            // Store non-zero entries
            std::vector<std::pair<int, double>> row_entries;
            for (int j = 0; j < n; j++) {
                if (is_nonzero[j]) {
                    row_entries.emplace_back(j, values[j]);
                }
            }
            
            // Sort by column index
            std::sort(row_entries.begin(), row_entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            
            temp_rows[i] = std::move(row_entries);
        }
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