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
#include <chrono>

// Alignment for AVX-512
#define ALIGNMENT 64

// Optimized block sizes for L1, L2 and L3 cache
#define MC 96   // Block size for L1 cache (smaller for better L1 utilization)
#define KC 256  // Block size for inner loop
#define NC 4096 // Block size for outer loop, targeting full matrix width

namespace solution {
    // Kernel function for computing a block of the output matrix
    inline void kernel_block(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        const int m, const int n, const int k,
        const int i_start, const int j_start, const int k_start,
        const int i_end, const int j_end, const int k_end
    ) {
        // Process 16 elements at a time using AVX-512
        for (int i = i_start; i < i_end; i++) {
            for (int kk = k_start; kk < k_end; kk++) {
                const float a_val = A[i * k + kk];
                const __m512 a_vec = _mm512_set1_ps(a_val);
                
                // Process 16 elements at a time
                for (int j = j_start; j < j_end; j += 16) {
                    if (j + 16 <= j_end) {
                        // Load, multiply and accumulate
                        const __m512 b_vec = _mm512_loadu_ps(&B[kk * n + j]);
                        const __m512 c_vec = _mm512_loadu_ps(&C[i * n + j]);
                        const __m512 result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm512_storeu_ps(&C[i * n + j], result);
                    } else {
                        // Handle edge case with remainder
                        for (int jj = j; jj < j_end; jj++) {
                            C[i * n + jj] += a_val * B[kk * n + jj];
                        }
                    }
                }
            }
        }
    }

    // Pack a block of matrix A into a memory-aligned buffer
    inline void pack_A(
        float* __restrict__ A_packed,
        const float* __restrict__ A,
        const int lda, const int m, const int k,
        const int i_start, const int k_start
    ) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                A_packed[i * k + j] = A[(i + i_start) * lda + (j + k_start)];
            }
        }
    }

    // Pack a block of matrix B into a memory-aligned buffer (with transpose)
    inline void pack_B(
        float* __restrict__ B_packed,
        const float* __restrict__ B,
        const int ldb, const int k, const int n,
        const int k_start, const int j_start
    ) {
        for (int i = 0; i < k; i++) {
            // Use AVX-512 for faster copying
            int j = 0;
            for (; j + 16 <= n; j += 16) {
                _mm512_storeu_ps(&B_packed[i * n + j], 
                                 _mm512_loadu_ps(&B[(i + k_start) * ldb + (j + j_start)]));
            }
            // Handle remainder
            for (; j < n; j++) {
                B_packed[i * n + j] = B[(i + k_start) * ldb + (j + j_start)];
            }
        }
    }

    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        // Allocate aligned memory
        float* m1 = static_cast<float*>(_mm_malloc(sizeof(float) * n * k, ALIGNMENT));
        float* m2 = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, ALIGNMENT));
        float* result = static_cast<float*>(_mm_malloc(sizeof(float) * n * m, ALIGNMENT));

        // Read input matrices with error checking
        if (!m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * n * k)) {
            std::cerr << "Error reading m1 file" << std::endl;
        }
        if (!m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * k * m)) {
            std::cerr << "Error reading m2 file" << std::endl;
        }
        m1_fs.close(); m2_fs.close();

        // Set optimal thread count for this system
        // For Intel Xeon Gold 6226R, we have 32 physical cores
        // Setting to a slightly lower number to avoid OS contention
        int max_threads = 32;
        omp_set_num_threads(max_threads);

        // Initialize result to zero
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n * m; ++i) {
            result[i] = 0.0f;
        }

        // Calculate thread-specific buffer sizes
        const int A_block_size = MC * KC;
        const int B_block_size = KC * NC;

        // Perform the matrix multiplication with hierarchical blocking
        #pragma omp parallel
        {
            // Thread-local buffers for packing
            float* A_packed = static_cast<float*>(_mm_malloc(sizeof(float) * A_block_size, ALIGNMENT));
            float* B_packed = static_cast<float*>(_mm_malloc(sizeof(float) * B_block_size, ALIGNMENT));

            // First level blocking for L3 cache
            #pragma omp for schedule(dynamic)
            for (int i_block = 0; i_block < n; i_block += MC) {
                const int i_limit = std::min(i_block + MC, n);
                const int i_size = i_limit - i_block;
                
                for (int k_block = 0; k_block < k; k_block += KC) {
                    const int k_limit = std::min(k_block + KC, k);
                    const int k_size = k_limit - k_block;
                    
                    // Pack block of A into contiguous memory for better cache locality
                    pack_A(A_packed, m1, k, i_size, k_size, i_block, k_block);

                    // Process blocks of matrix B against the current block of A
                    for (int j_block = 0; j_block < m; j_block += NC) {
                        const int j_limit = std::min(j_block + NC, m);
                        const int j_size = j_limit - j_block;
                        
                        // Pack block of B into contiguous memory
                        pack_B(B_packed, m2, m, k_size, j_size, k_block, j_block);
                        
                        // Compute kernel block
                        kernel_block(
                            A_packed, B_packed, result,
                            n, m, k_size,
                            i_block, j_block, 0,
                            i_limit, j_limit, k_size
                        );
                    }
                }
            }

            _mm_free(A_packed);
            _mm_free(B_packed);
        }

        // Write result to output file
        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();
        
        // Free memory
        _mm_free(m1);
        _mm_free(m2);
        _mm_free(result);
        
        return sol_path;
    }
}

// These functions are kept for compatibility with requirements
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