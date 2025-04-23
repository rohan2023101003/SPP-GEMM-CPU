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

        // Initialize NUMA policy if available
        if (numa_available() >= 0) {
            numa_set_localalloc();
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

        // Initialize result to zero
        std::memset(result, 0, sizeof(float) * n * m);

        // Get number of NUMA nodes
        int num_numa_nodes = 1;
        if (numa_available() >= 0) {
            num_numa_nodes = numa_num_configured_nodes();
        }

        // Set number of threads (adjusted to be more conservative for better numerical stability)
        int num_threads = 32; // Using half the cores for better stability
        omp_set_num_threads(num_threads);

        // Block sizes optimized for numerical stability and cache efficiency
        const int MC = 64;  // Block size for M dimension
        const int KC = 256; // Block size for K dimension
        const int NC = 64;  // Block size for N dimension (reduced for better stability)

        // Perform matrix multiplication with blocking for cache efficiency
        #pragma omp parallel
        {
            // Try to bind thread to NUMA node for better memory locality
            if (numa_available() >= 0) {
                int thread_id = omp_get_thread_num();
                int numa_node = thread_id % num_numa_nodes;
                numa_run_on_node(numa_node);
            }

            // Local buffer for accumulation to reduce roundoff errors
            float* local_result = static_cast<float*>(_mm_malloc(sizeof(float) * MC * NC, alignment));
            
            if (local_result) {
                // Process blocks of the matrices
                #pragma omp for schedule(dynamic)
                for (int i0 = 0; i0 < n; i0 += MC) {
                    const int iLimit = std::min(i0 + MC, n);
                    
                    for (int j0 = 0; j0 < m; j0 += NC) {
                        const int jLimit = std::min(j0 + NC, m);
                        
                        // Initialize local result block to zero
                        for (int i = 0; i < (iLimit - i0); ++i) {
                            std::memset(&local_result[i * NC], 0, sizeof(float) * (jLimit - j0));
                        }
                        
                        // Compute block C(i0:iLimit, j0:jLimit) = A(i0:iLimit, :) * B(:, j0:jLimit)
                        for (int l0 = 0; l0 < k; l0 += KC) {
                            const int lLimit = std::min(l0 + KC, k);
                            
                            // Process this block with careful accumulation
                            for (int i = i0; i < iLimit; ++i) {
                                for (int l = l0; l < lLimit; ++l) {
                                    const float a_il = m1[i * k + l];
                                    
                                    // Skip if A element is zero (helps with accuracy for sparse-ish data)
                                    if (std::abs(a_il) < 1e-10) continue;
                                    
                                    // Use AVX-512 for vectorized computation when possible
                                    __m512 a_vec = _mm512_set1_ps(a_il);
                                    
                                    for (int j = j0; j < jLimit; j += 16) {
                                        if (j + 16 <= jLimit) {
                                            // Load B row and current C values
                                            __m512 b_vec = _mm512_loadu_ps(&m2[l * m + j]);
                                            __m512 c_vec = _mm512_loadu_ps(&local_result[(i - i0) * NC + (j - j0)]);
                                            
                                            // Use FMA for better accuracy (does a*b + c with just one rounding)
                                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                                            _mm512_storeu_ps(&local_result[(i - i0) * NC + (j - j0)], c_vec);
                                        } else {
                                            // Handle the edge case
                                            for (int jj = j; jj < jLimit; ++jj) {
                                                local_result[(i - i0) * NC + (jj - j0)] += a_il * m2[l * m + jj];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Copy local result to global result matrix
                        #pragma omp critical
                        {
                            for (int i = i0; i < iLimit; ++i) {
                                for (int j = j0; j < jLimit; ++j) {
                                    result[i * m + j] = local_result[(i - i0) * NC + (j - j0)];
                                }
                            }
                        }
                    }
                }
                
                // Free local buffer
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

// Basic implementation of dense matrix multiplication
void dense_gemm(const double *A, const double *B, double *C, int m, int k, int n) {
    // Zero out C matrix
    std::memset(C, 0, sizeof(double) * m * n);
    
    // Basic triple loop with better accumulation order
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < m; i++) {
        for (int l = 0; l < k; l++) {
            const double a_il = A[i*k + l];
            if (std::abs(a_il) < 1e-15) continue;  // Skip if A element is close to zero
            
            for (int j = 0; j < n; j++) {
                C[i*n + j] += a_il * B[l*n + j];
            }
        }
    }
}

// Sparse matrix multiplication implementation
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
                
                // Skip very small values for better numerical stability
                if (std::abs(val_A) < 1e-15) continue;
                
                for (int k_ptr = B_row_ptr[j]; k_ptr < B_row_ptr[j + 1]; k_ptr++) {
                    int k = B_col_ind[k_ptr];
                    double val_B = B_values[k_ptr];
                    
                    if (std::abs(val_B) < 1e-15) continue;
                    
                    values[k] += val_A * val_B;
                    is_nonzero[k] = true;
                }
            }
            
            // Store non-zero entries with a threshold to filter numerical noise
            std::vector<std::pair<int, double>> row_entries;
            for (int j = 0; j < n; j++) {
                if (is_nonzero[j] && std::abs(values[j]) > 1e-15) {
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
    
    // Fill values and column indices
    for (int i = 0; i < m; i++) {
        int idx = (*C_row_ptr)[i];
        for (const auto& entry : temp_rows[i]) {
            (*C_col_ind)[idx] = entry.first;
            (*C_values)[idx] = entry.second;
            idx++;
        }
    }
}