#pragma GCC optimize("O3,unroll-loops")
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
    // Tile sizes optimized for cache utilization
    constexpr int BLOCK_SIZE_M = 64;
    constexpr int BLOCK_SIZE_N = 64;
    constexpr int BLOCK_SIZE_K = 64;
    
    // Process an 8x8 block using AVX2 instructions
    static inline void kernel_avx_8x8(
        const float* A, const float* B, float* C,
        int lda, int ldb, int ldc, int k) {
        
        // Initialize accumulator registers to zero
        __m256 c0 = _mm256_setzero_ps();
        __m256 c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps();
        __m256 c3 = _mm256_setzero_ps();
        __m256 c4 = _mm256_setzero_ps();
        __m256 c5 = _mm256_setzero_ps();
        __m256 c6 = _mm256_setzero_ps();
        __m256 c7 = _mm256_setzero_ps();
        
        // Compute matrix multiplication for this micro-block
        for (int p = 0; p < k; p++) {
            // Load 8 elements from B
            __m256 b = _mm256_loadu_ps(&B[p * ldb]);
            
            // Broadcast A elements and multiply-accumulate
            __m256 a0 = _mm256_broadcast_ss(&A[0 * lda + p]);
            c0 = _mm256_fmadd_ps(a0, b, c0);
            
            __m256 a1 = _mm256_broadcast_ss(&A[1 * lda + p]);
            c1 = _mm256_fmadd_ps(a1, b, c1);
            
            __m256 a2 = _mm256_broadcast_ss(&A[2 * lda + p]);
            c2 = _mm256_fmadd_ps(a2, b, c2);
            
            __m256 a3 = _mm256_broadcast_ss(&A[3 * lda + p]);
            c3 = _mm256_fmadd_ps(a3, b, c3);
            
            __m256 a4 = _mm256_broadcast_ss(&A[4 * lda + p]);
            c4 = _mm256_fmadd_ps(a4, b, c4);
            
            __m256 a5 = _mm256_broadcast_ss(&A[5 * lda + p]);
            c5 = _mm256_fmadd_ps(a5, b, c5);
            
            __m256 a6 = _mm256_broadcast_ss(&A[6 * lda + p]);
            c6 = _mm256_fmadd_ps(a6, b, c6);
            
            __m256 a7 = _mm256_broadcast_ss(&A[7 * lda + p]);
            c7 = _mm256_fmadd_ps(a7, b, c7);
        }
        
        // Store the results
        _mm256_storeu_ps(&C[0 * ldc], c0);
        _mm256_storeu_ps(&C[1 * ldc], c1);
        _mm256_storeu_ps(&C[2 * ldc], c2);
        _mm256_storeu_ps(&C[3 * ldc], c3);
        _mm256_storeu_ps(&C[4 * ldc], c4);
        _mm256_storeu_ps(&C[5 * ldc], c5);
        _mm256_storeu_ps(&C[6 * ldc], c6);
        _mm256_storeu_ps(&C[7 * ldc], c7);
    }
    
    // Compute matrix multiplication with blocking and parallelism
    static void compute_matrix_multiply(
        const float* __restrict__ A,
        const float* __restrict__ B,
        float* __restrict__ C,
        int n, int k, int m,
        int start_row, int end_row) {
        
        // Initialize result matrix to zeros
        for (int i = start_row; i < end_row; i++) {
            std::memset(&C[i * m], 0, sizeof(float) * m);
        }
        
        // Block-based matrix multiplication for better cache utilization
        for (int i = start_row; i < end_row; i += BLOCK_SIZE_N) {
            int iLimit = std::min(i + BLOCK_SIZE_N, end_row);
            
            for (int j = 0; j < m; j += BLOCK_SIZE_M) {
                int jLimit = std::min(j + BLOCK_SIZE_M, m);
                
                for (int l = 0; l < k; l += BLOCK_SIZE_K) {
                    int lLimit = std::min(l + BLOCK_SIZE_K, k);
                    
                    // Process current block
                    for (int ii = i; ii < iLimit; ii += 8) {
                        int iiLimit = std::min(ii + 8, iLimit);
                        
                        for (int jj = j; jj < jLimit; jj += 8) {
                            int jjLimit = std::min(jj + 8, jLimit);
                            
                            // Full 8x8 block
                            if (iiLimit - ii == 8 && jjLimit - jj == 8) {
                                kernel_avx_8x8(
                                    &A[ii * k + l], &B[l * m + jj], &C[ii * m + jj],
                                    k, m, m, lLimit - l
                                );
                            } else {
                                // Handle edge cases with scalar code
                                for (int iii = ii; iii < iiLimit; iii++) {
                                    for (int jjj = jj; jjj < jjLimit; jjj++) {
                                        float sum = 0.0f;
                                        for (int ll = l; ll < lLimit; ll++) {
                                            sum += A[iii * k + ll] * B[ll * m + jjj];
                                        }
                                        C[iii * m + jjj] += sum;
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
        // Create a temporary file for the result
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        
        // Determine number of threads to use based on hardware
        int num_threads = std::min(64, n);
        omp_set_num_threads(num_threads);
        
        // Allocate memory for matrices
        std::unique_ptr<float[]> m1(new float[n * k]);
        std::unique_ptr<float[]> m2(new float[k * m]);
        std::unique_ptr<float[]> result(new float[n * m]);
        
        // Read input matrices
        {
            std::ifstream m1_file(m1_path, std::ios::binary);
            if (!m1_file) {
                std::cerr << "Error opening file: " << m1_path << std::endl;
                return sol_path;
            }
            m1_file.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
            m1_file.close();
        }
        
        {
            std::ifstream m2_file(m2_path, std::ios::binary);
            if (!m2_file) {
                std::cerr << "Error opening file: " << m2_path << std::endl;
                return sol_path;
            }
            m2_file.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
            m2_file.close();
        }
        
        // Compute matrix multiplication in parallel
        if (num_threads > 1) {
            // Use OpenMP to parallelize computation
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            // Calculate rows per thread with balanced distribution
            int rows_per_thread = n / num_threads;
            int extra_rows = n % num_threads;
            
            int start_row = 0;
            for (int t = 0; t < num_threads; t++) {
                int thread_rows = rows_per_thread + (t < extra_rows ? 1 : 0);
                int end_row = start_row + thread_rows;
                
                threads.emplace_back(
                    compute_matrix_multiply,
                    m1.get(), m2.get(), result.get(),
                    n, k, m,
                    start_row, end_row
                );
                
                start_row = end_row;
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Single-threaded computation
            compute_matrix_multiply(m1.get(), m2.get(), result.get(), n, k, m, 0, n);
        }
        
        // Write result to output file
        {
            std::ofstream sol_file(sol_path, std::ios::binary);
            if (!sol_file) {
                std::cerr << "Error opening file for writing: " << sol_path << std::endl;
                return sol_path;
            }
            sol_file.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
            sol_file.close();
        }
        
        return sol_path;
    }
}