#pragma GCC optimize("O3", "unroll-loops", "fast-math", "tree-vectorize")
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

// Constants for block sizes - tuned for the given architecture
constexpr int L1_BLOCK_SIZE = 64;    // L1 cache optimization
constexpr int L2_BLOCK_SIZE = 512;   // L2 cache optimization
constexpr int L3_BLOCK_SIZE = 2048;  // L3 cache optimization

// Implementation of dense_gemm as required in the homework
void dense_gemm(const double *A, const double *B, double *C, int m, int k, int n) {
    #pragma omp parallel
    {
        // Zero out C matrix - each thread handles its own portion
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i*n + j] = 0.0;
            }
        }

        // Cache-blocked matrix multiplication using NUMA-aware scheduling
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int i_outer = 0; i_outer < m; i_outer += L3_BLOCK_SIZE) {
            for (int j_outer = 0; j_outer < n; j_outer += L3_BLOCK_SIZE) {
                for (int k_outer = 0; k_outer < k; k_outer += L3_BLOCK_SIZE) {
                    // L2 cache blocking
                    const int i_end = std::min(i_outer + L3_BLOCK_SIZE, m);
                    const int j_end = std::min(j_outer + L3_BLOCK_SIZE, n);
                    const int k_end = std::min(k_outer + L3_BLOCK_SIZE, k);

                    for (int i_mid = i_outer; i_mid < i_end; i_mid += L2_BLOCK_SIZE) {
                        for (int j_mid = j_outer; j_mid < j_end; j_mid += L2_BLOCK_SIZE) {
                            for (int k_mid = k_outer; k_mid < k_end; k_mid += L2_BLOCK_SIZE) {
                                // L1 cache blocking
                                const int i_mid_end = std::min(i_mid + L2_BLOCK_SIZE, i_end);
                                const int j_mid_end = std::min(j_mid + L2_BLOCK_SIZE, j_end);
                                const int k_mid_end = std::min(k_mid + L2_BLOCK_SIZE, k_end);

                                for (int i = i_mid; i < i_mid_end; i += L1_BLOCK_SIZE) {
                                    for (int j = j_mid; j < j_mid_end; j += L1_BLOCK_SIZE) {
                                        // For each small block
                                        const int i_inner_end = std::min(i + L1_BLOCK_SIZE, i_mid_end);
                                        const int j_inner_end = std::min(j + L1_BLOCK_SIZE, j_mid_end);

                                        for (int k_inner = k_mid; k_inner < k_mid_end; k_inner += L1_BLOCK_SIZE) {
                                            const int k_inner_end = std::min(k_inner + L1_BLOCK_SIZE, k_mid_end);

                                            // Process inner blocks with AVX512 vectorization
                                            for (int ii = i; ii < i_inner_end; ii++) {
                                                for (int kk = k_inner; kk < k_inner_end; kk++) {
                                                    const __m512d a_val = _mm512_set1_pd(A[ii*k + kk]);
                                                    
                                                    // Process 8 elements at once with AVX512
                                                    for (int jj = j; jj < j_inner_end; jj += 8) {
                                                        if (jj + 8 <= j_inner_end) {
                                                            // Load C values
                                                            __m512d c_vals = _mm512_loadu_pd(&C[ii*n + jj]);
                                                            
                                                            // Load B values
                                                            __m512d b_vals = _mm512_loadu_pd(&B[kk*n + jj]);
                                                            
                                                            // Perform FMA: c += a * b
                                                            c_vals = _mm512_fmadd_pd(a_val, b_vals, c_vals);
                                                            
                                                            // Store back to C
                                                            _mm512_storeu_pd(&C[ii*n + jj], c_vals);
                                                        } else {
                                                            // Handle remaining elements (less than vector width)
                                                            for (int j_rem = jj; j_rem < j_inner_end; j_rem++) {
                                                                C[ii*n + j_rem] += A[ii*k + kk] * B[kk*n + j_rem];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Implementation of sparse_spmm as required in the homework
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
    #pragma omp parallel
    {
        // Each thread needs its own hash table for accumulating results
        std::vector<std::pair<int, double>> local_result(n, {-1, 0.0});
        
        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < m; i++) {
            // Reset temporary storage
            for (auto& p : local_result) {
                p.first = -1;
                p.second = 0.0;
            }
            int nnz_in_row = 0;
            
            // For each non-zero element in row i of A
            for (int j_ptr = A_row_ptr[i]; j_ptr < A_row_ptr[i + 1]; j_ptr++) {
                int j = A_col_ind[j_ptr];
                double val_A = A_values[j_ptr];
                
                // For each non-zero element in row j of B
                for (int k_ptr = B_row_ptr[j]; k_ptr < B_row_ptr[j + 1]; k_ptr++) {
                    int k = B_col_ind[k_ptr];
                    double val_B = B_values[k_ptr];
                    
                    // Accumulate result in C(i,k)
                    if (local_result[k].first == -1) {
                        local_result[k].first = k;
                        local_result[k].second = val_A * val_B;
                        nnz_in_row++;
                    } else {
                        local_result[k].second += val_A * val_B;
                    }
                }
            }
            
            // Copy the results to the temporary storage
            if (nnz_in_row > 0) {
                std::vector<std::pair<int, double>> row_entries;
                row_entries.reserve(nnz_in_row);
                
                for (const auto& p : local_result) {
                    if (p.first != -1) {
                        row_entries.emplace_back(p.first, p.second);
                    }
                }
                
                // Sort by column index
                std::sort(row_entries.begin(), row_entries.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                
                #pragma omp critical
                {
                    temp_rows[i] = std::move(row_entries);
                }
            }
        }
    }
    
    // Calculate total number of non-zeros
    int total_nnz = 0;
    for (int i = 0; i < m; i++) {
        (*C_row_ptr)[i] = total_nnz;
        total_nnz += temp_rows[i].size();
    }
    (*C_row_ptr)[m] = total_nnz;
    
    // Allocate memory for values and column indices
    *C_values = new double[total_nnz];
    *C_col_ind = new int[total_nnz];
    
    // Fill in the values and column indices
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
        int start_idx = (*C_row_ptr)[i];
        for (size_t j = 0; j < temp_rows[i].size(); j++) {
            (*C_col_ind)[start_idx + j] = temp_rows[i][j].first;
            (*C_values)[start_idx + j] = temp_rows[i][j].second;
        }
    }
}

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);
        
        // Set thread count for OpenMP
        int num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
        
        // Memory alignment for better vectorization
        const auto m1 = std::make_unique<float[]>(n*k);
        const auto m2 = std::make_unique<float[]>(k*m);
        auto result = std::make_unique<float[]>(n*m);
        
        // Read input matrices
        m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();
        
        // Determine whether to use dense or sparse implementation based on matrix size
        // For this example, we're using the dense GEMM converted to float
        // In a real implementation, you would check sparsity and choose accordingly
        
        // Convert float matrices to double for our optimized GEMM implementation
        auto m1_double = std::make_unique<double[]>(n*k);
        auto m2_double = std::make_unique<double[]>(k*m);
        auto result_double = std::make_unique<double[]>(n*m);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n*k; i++) {
            m1_double[i] = static_cast<double>(m1[i]);
        }
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < k*m; i++) {
            m2_double[i] = static_cast<double>(m2[i]);
        }
        
        // Perform matrix multiplication using our optimized GEMM
        dense_gemm(m1_double.get(), m2_double.get(), result_double.get(), n, k, m);
        
        // Convert back to float for output
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n*m; i++) {
            result[i] = static_cast<float>(result_double[i]);
        }
        
        // Write result to output file
        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
}