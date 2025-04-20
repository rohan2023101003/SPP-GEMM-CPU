#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt,fma")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

namespace solution {
    // Tile sizes optimized for cache utilization
    constexpr int BLOCK_SIZE_M = 64;
    constexpr int BLOCK_SIZE_N = 64;
    constexpr int BLOCK_SIZE_K = 64;
    
    // Compute a block with careful accumulation to maintain precision
    static void compute_block(const float* __restrict__ m1, 
                              const float* __restrict__ m2, 
                              float* __restrict__ result,
                              int n, int k, int m,
                              int start_row, int end_row) {
        // Zero out result matrix
        for (int i = start_row; i < end_row; ++i) {
            std::fill_n(&result[i*m], m, 0.0f);
        }
        
        // Block-based matrix multiplication for better cache utilization
        for (int kk = 0; kk < k; kk += BLOCK_SIZE_K) {
            int k_end = std::min(kk + BLOCK_SIZE_K, k);
            
            for (int i = start_row; i < end_row; ++i) {
                // Pack a portion of m1 row into contiguous memory for better cache performance
                float m1_cache[BLOCK_SIZE_K];
                for (int l = kk; l < k_end; ++l) {
                    m1_cache[l - kk] = m1[i*k + l];
                }
                
                for (int jj = 0; jj < m; jj += BLOCK_SIZE_M) {
                    int j_end = std::min(jj + BLOCK_SIZE_M, m);
                    
                    // Process a block of the result matrix
                    for (int j = jj; j < j_end; ++j) {
                        // Local accumulator to maintain precision
                        float sum = result[i*m + j];
                        
                        // Process current k-block
                        for (int l = kk; l < k_end; ++l) {
                            sum += m1_cache[l - kk] * m2[l*m + j];
                        }
                        
                        result[i*m + j] = sum;
                    }
                }
            }
        }
    }

    // Vectorized version that maintains numerical accuracy
    static void compute_block_simd(const float* __restrict__ m1, 
                                  const float* __restrict__ m2, 
                                  float* __restrict__ result,
                                  int n, int k, int m,
                                  int start_row, int end_row) {
        constexpr int SIMD_WIDTH = 8; // AVX2 processes 8 floats at a time
        
        // Zero out result matrix
        for (int i = start_row; i < end_row; ++i) {
            std::fill_n(&result[i*m], m, 0.0f);
        }
        
        // Process each result element with careful accumulation
        for (int i = start_row; i < end_row; ++i) {
            // Process blocks of columns for better cache utilization
            for (int jj = 0; jj < m; jj += BLOCK_SIZE_M) {
                int j_end = std::min(jj + BLOCK_SIZE_M, m);
                
                // Align j_end down to SIMD boundary for vectorized part
                int j_simd_end = jj + ((j_end - jj) / SIMD_WIDTH) * SIMD_WIDTH;
                
                // For elements that can use SIMD
                for (int j = jj; j < j_simd_end; j += SIMD_WIDTH) {
                    // Initialize accumulators with zeros
                    __m256 sum = _mm256_setzero_ps();
                    
                    // Process row of first matrix against column of second matrix
                    for (int l = 0; l < k; ++l) {
                        // Broadcast single value from m1
                        __m256 m1_val = _mm256_set1_ps(m1[i*k + l]);
                        
                        // Load 8 values from m2
                        __m256 m2_vals = _mm256_loadu_ps(&m2[l*m + j]);
                        
                        // Multiply and accumulate
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(m1_val, m2_vals));
                    }
                    
                    // Store result
                    _mm256_storeu_ps(&result[i*m + j], sum);
                }
                
                // Handle remaining elements (less than SIMD_WIDTH)
                for (int j = j_simd_end; j < j_end; ++j) {
                    float sum = 0.0f;
                    
                    // Explicit loop for remaining calculations
                    for (int l = 0; l < k; ++l) {
                        sum += m1[i*k + l] * m2[l*m + j];
                    }
                    
                    result[i*m + j] = sum;
                }
            }
        }
    }
    
    // Hybrid approach that automatically selects best method based on matrix size
    static void compute_chunk(const float* __restrict__ m1, 
                             const float* __restrict__ m2, 
                             float* __restrict__ result,
                             int n, int k, int m,
                             int start_row, int end_row) {
        // Choose implementation based on matrix size
        if (k > 512 && m > 512) {
            // For large matrices, use SIMD implementation
            compute_block_simd(m1, m2, result, n, k, m, start_row, end_row);
        } else {
            // For smaller matrices or when k is small, use blocked implementation
            compute_block(m1, m2, result, n, k, m, start_row, end_row);
        }
    }

    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        
        // Open input files with optimized buffering
        std::ifstream m1_fs(m1_path, std::ios::binary | std::ios::in);
        std::ifstream m2_fs(m2_path, std::ios::binary | std::ios::in);
        
        // Set larger buffer size for better I/O performance
        const int BUFFER_SIZE = 8 * 1024 * 1024; // 8MB buffer
        char* m1_buffer = new char[BUFFER_SIZE];
        char* m2_buffer = new char[BUFFER_SIZE];
        m1_fs.rdbuf()->pubsetbuf(m1_buffer, BUFFER_SIZE);
        m2_fs.rdbuf()->pubsetbuf(m2_buffer, BUFFER_SIZE);
        
        // Allocate memory for matrices with alignment for SIMD
        const auto m1 = std::make_unique<float[]>(n*k);
        const auto m2 = std::make_unique<float[]>(k*m);
        auto result = std::make_unique<float[]>(n*m);
        
        // Read input data efficiently
        m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
        m1_fs.close(); 
        m2_fs.close();
        
        // Clean up buffers
        delete[] m1_buffer;
        delete[] m2_buffer;

        // Auto-tuned thread count based on hardware and matrix dimensions
        unsigned int max_threads = std::thread::hardware_concurrency();
        unsigned int num_threads;
        
        // Determine optimal thread count based on matrix size
        if (n >= 2048) {
            num_threads = max_threads;
        } else if (n >= 1024) {
            num_threads = std::max(1u, max_threads * 3 / 4);
        } else if (n >= 512) {
            num_threads = std::max(1u, max_threads / 2);
        } else if (n >= 256) {
            num_threads = std::min(4u, max_threads);
        } else {
            num_threads = 1; // Single thread for small matrices
        }
        
        // Make sure we don't use more threads than rows
        num_threads = std::min(num_threads, static_cast<unsigned int>(n));
        
        if (num_threads > 1) {
            // Calculate rows per thread with improved load balancing
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            // Use more sophisticated work distribution for better load balance
            int base_rows_per_thread = n / num_threads;
            int extra_rows = n % num_threads;
            
            int start_row = 0;
            for (unsigned int t = 0; t < num_threads; ++t) {
                // Give some threads one extra row for balanced distribution
                int thread_rows = base_rows_per_thread + (t < extra_rows ? 1 : 0);
                int end_row = start_row + thread_rows;
                
                if (start_row < end_row) {
                    threads.emplace_back(
                        compute_chunk,
                        m1.get(), m2.get(), result.get(), 
                        n, k, m, 
                        start_row, end_row
                    );
                }
                
                start_row = end_row;
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                if (thread.joinable()) {
                    thread.join();
                }
            }
        } else {
            // Single-threaded computation
            compute_chunk(m1.get(), m2.get(), result.get(), n, k, m, 0, n);
        }
        
        // Write results with optimized buffering
        std::ofstream sol_fs(sol_path, std::ios::binary);
        char* result_buffer = new char[BUFFER_SIZE];
        sol_fs.rdbuf()->pubsetbuf(result_buffer, BUFFER_SIZE);
        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        delete[] result_buffer;
        
        return sol_path;
    }
};