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

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        const int alignment = 64;
        float* m1 = static_cast<float*>(_mm_malloc(sizeof(float) * n * k, alignment));
        float* m2 = static_cast<float*>(_mm_malloc(sizeof(float) * k * m, alignment));
        float* result = static_cast<float*>(_mm_malloc(sizeof(float) * n * m, alignment));

        if (!m1 || !m2 || !result) {
            std::cerr << "Memory allocation failed" << std::endl;
            if (m1) _mm_free(m1);
            if (m2) _mm_free(m2);
            if (result) _mm_free(result);
            return std::string(); // Return empty string instead of bare return
        }

        m1_fs.read(reinterpret_cast<char*>(m1), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();

        int num_threads = std::min(32, omp_get_max_threads());
        omp_set_num_threads(num_threads);

        #pragma omp parallel for simd
        for (int i = 0; i < n * m; ++i) result[i] = 0.0f;

        const int MC = 128;
        const int KC = 512;
        const int NC = 1024;

        #pragma omp parallel
        {
            float* local_C = static_cast<float*>(_mm_malloc(sizeof(float) * MC * NC, alignment));
            if (!local_C) {
                std::cerr << "Thread-local memory allocation failed" << std::endl;
                return; // Exit parallel block, but avoids compilation error
            }

            #pragma omp for collapse(2) schedule(dynamic)
            for (int i_block = 0; i_block < n; i_block += MC) {
                for (int j_block = 0; j_block < m; j_block += NC) {
                    int i_limit = std::min(i_block + MC, n);
                    int j_limit = std::min(j_block + NC, m);

                    for (int i = 0; i < MC * NC; ++i) local_C[i] = 0.0f;

                    for (int k_block = 0; k_block < k; k_block += KC) {
                        int k_limit = std::min(k_block + KC, k);

                        for (int i = i_block; i < i_limit; ++i) {
                            int local_i = i - i_block;

                            for (int kk = k_block; kk < k_limit; ++kk) {
                                float a_val = m1[i * k + kk];
                                __m512 a_vec = _mm512_set1_ps(a_val);

                                int j = j_block;
                                for (; j + 31 < j_limit; j += 32) {
                                    __m512 b0 = _mm512_load_ps(&m2[kk * m + j]);
                                    __m512 b1 = _mm512_load_ps(&m2[kk * m + j + 16]);
                                    __m512 c0 = _mm512_load_ps(&local_C[local_i * NC + (j - j_block)]);
                                    __m512 c1 = _mm512_load_ps(&local_C[local_i * NC + (j - j_block + 16)]);
                                    c0 = _mm512_fmadd_ps(a_vec, b0, c0);
                                    c1 = _mm512_fmadd_ps(a_vec, b1, c1);
                                    _mm512_store_ps(&local_C[local_i * NC + (j - j_block)], c0);
                                    _mm512_store_ps(&local_C[local_i * NC + (j - j_block + 16)], c1);
                                }

                                for (; j < j_limit; ++j) {
                                    local_C[local_i * NC + (j - j_block)] += a_val * m2[kk * m + j];
                                }
                            }
                        }
                    }

                    for (int i = i_block; i < i_limit; ++i) {
                        int local_i = i - i_block;
                        for (int j = j_block; j < j_limit; ++j) {
                            result[i * m + j] += local_C[local_i * NC + (j - j_block)];
                        }
                    }
                }
            }

            _mm_free(local_C);
        }

        sol_fs.write(reinterpret_cast<const char*>(result), sizeof(float) * n * m);
        sol_fs.close();

        _mm_free(m1);
        _mm_free(m2);
        _mm_free(result);

        return sol_path;
    }
}
