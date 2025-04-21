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

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        // Allocate aligned memory for better vectorization
        const int alignment = 64; // For AVX-512
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

        // Set number of threads based on system capabilities
        int num_threads = std::min(32, omp_get_max_threads()); // Limit to 32 threads (number of physical cores)
        omp_set_num_threads(num_threads);

        // Initialize result matrix to zero
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n * m; ++i) {
            result[i] = 0.0f;
        }

        // Define block sizes for better cache utilization
        const int MC = 64;  // Block size for L1 cache
        const int KC = 256; // Block size for inner loop - k dimension
        const int NC = 2048; // Block size for outer loop - n dimension

        // Flag to indicate memory allocation failure
        bool allocation_failed = false;

        // Multi-level blocking for better cache utilization
        #pragma omp parallel shared(allocation_failed)
        {
            // Thread-local buffer for accumulation
            float* local_C = static_cast<float*>(_mm_malloc(sizeof(float) * MC * NC, alignment));
            
            if (!local_C) {
                allocation_failed = true;
                #pragma omp barrier
            }
            
            if (!allocation_failed) {
                #pragma omp for schedule(dynamic, 1)
                for (int i_block = 0; i_block < n; i_block += MC) {
                    const int i_limit = std::min(i_block + MC, n);
                    
                    for (int j_block = 0; j_block < m; j_block += NC) {
                        const int j_limit = std::min(j_block + NC, m);
                        
                        // Zero the local accumulation buffer
                        for (int i = 0; i < (i_limit - i_block); ++i) {
                            for (int j = 0; j < (j_limit - j_block); ++j) {
                                local_C[i * NC + j] = 0.0f;
                            }
                        }
                        
                        // Compute block of C
                        for (int k_block = 0; k_block < k; k_block += KC) {
                            const int k_limit = std::min(k_block + KC, k);
                            
                            // Compute micro-kernel
                            for (int i = i_block; i < i_limit; ++i) {
                                const int local_i = i - i_block;
                                
                                for (int kk = k_block; kk < k_limit; ++kk) {
                                    const float a_val = m1[i * k + kk];
                                    const __m512 a_vec = _mm512_set1_ps(a_val);
                                    
                                    for (int j = j_block; j < j_limit; j += 16) {
                                        if (j + 16 <= j_limit) {
                                            // Load, multiply and accumulate with AVX-512
                                            const __m512 b_vec = _mm512_loadu_ps(&m2[kk * m + j]);
                                            const __m512 c_vec = _mm512_loadu_ps(&local_C[local_i * NC + (j - j_block)]);
                                            const __m512 result_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                            _mm512_storeu_ps(&local_C[local_i * NC + (j - j_block)], result_vec);
                                        } else {
                                            // Handle edge case with remainder
                                            for (int jj = j; jj < j_limit; ++jj) {
                                                local_C[local_i * NC + (jj - j_block)] += a_val * m2[kk * m + jj];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Write the block of local_C back to result matrix
                        #pragma omp critical
                        {
                            for (int i = i_block; i < i_limit; ++i) {
                                const int local_i = i - i_block;
                                for (int j = j_block; j < j_limit; ++j) {
                                    result[i * m + j] = local_C[local_i * NC + (j - j_block)];
                                }
                            }
                        }
                    }
                }
            }
            
            if (local_C) {
                _mm_free(local_C);
            }
        }

        if (allocation_failed) {
            std::cerr << "Thread-local memory allocation failed" << std::endl;
        }

        // Write the result to output file
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();
        
        // Free allocated memory
        _mm_free(m1);
        _mm_free(m2);
        _mm_free(result);
        
        return sol_path;
    }
}

// These function signatures are implemented to match the homework requirements
// But they're not actually used in the main compute function above

void dense_gemm(const double *A, const double *B, double *C, int m, int k, int n) {
    // Zero out the C matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = 0.0;
        }
    }
    
    // Basic matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                C[i*n + j] += A[i*k + l] * B[l*n + j];
            }
        }
    }
}

void sparse_spmm(
    const double *A_values, const int *A_col_ind, const int *A_row_ptr,
    const double *B_values, const int *B_col_ind, const int *B_row_ptr,
    double **C_values, int **C_col_ind, int **C_row_ptr,
    int m, int k, int n) {
    
    // Create temporary storage for result
    std::vector<std::vector<std::pair<int, double>>> temp_rows(m);
    
    // We'll store the row pointers as we go
    *C_row_ptr = new int[m + 1];
    (*C_row_ptr)[0] = 0;
    
    // For each row in A
    for (int i = 0; i < m; i++) {
        std::vector<std::pair<int, double>> row_entries;
        std::vector<bool> is_nonzero(n, false);
        std::vector<double> values(n, 0.0);
        
        // For each non-zero element in row i of A
        for (int j_ptr = A_row_ptr[i]; j_ptr < A_row_ptr[i + 1]; j_ptr++) {
            int j = A_col_ind[j_ptr];
            double val_A = A_values[j_ptr];
            
            // For each non-zero element in row j of B
            for (int k_ptr = B_row_ptr[j]; k_ptr < B_row_ptr[j + 1]; k_ptr++) {
                int k = B_col_ind[k_ptr];
                double val_B = B_values[k_ptr];
                
                // Accumulate result in C(i,k)
                values[k] += val_A * val_B;
                is_nonzero[k] = true;
            }
        }
        
        // Store non-zero entries for this row
        for (int j = 0; j < n; j++) {
            if (is_nonzero[j]) {
                row_entries.emplace_back(j, values[j]);
            }
        }
        
        // Sort by column index
        std::sort(row_entries.begin(), row_entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        
        temp_rows[i] = std::move(row_entries);
        (*C_row_ptr)[i + 1] = (*C_row_ptr)[i] + temp_rows[i].size();
    }
    
    // Allocate memory for values and column indices
    int total_nnz = (*C_row_ptr)[m];
    *C_values = new double[total_nnz];
    *C_col_ind = new int[total_nnz];
    
    // Fill in the values and column indices
    int idx = 0;
    for (int i = 0; i < m; i++) {
        for (const auto& entry : temp_rows[i]) {
            (*C_col_ind)[idx] = entry.first;
            (*C_values)[idx] = entry.second;
            idx++;
        }
    }
}