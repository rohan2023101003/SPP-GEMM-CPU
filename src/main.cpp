#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx512f,avx512vl,avx512dq,avx512bw,avx512cd,avx512vbmi,avx512vnni,avx512bitalg,avx512vpopcntdq")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("fast-math")
#pragma GCC optimize("inline")

#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <omp.h>
#include <algorithm>

namespace solution {
    std::string compute(const std::string &m1_path, const std::string &m2_path, int n, int k, int m) {
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.dat";
        std::ofstream sol_fs(sol_path, std::ios::binary);
        std::ifstream m1_fs(m1_path, std::ios::binary), m2_fs(m2_path, std::ios::binary);

        float* m1_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * n * k));
        float* m2_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * k * m));
        float* result_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * n * m));

        std::unique_ptr<float[], decltype(&free)> m1(m1_ptr, &free);
        std::unique_ptr<float[], decltype(&free)> m2(m2_ptr, &free);
        std::unique_ptr<float[], decltype(&free)> result(result_ptr, &free);

        m1_fs.read(reinterpret_cast<char*>(m1.get()), sizeof(float) * n * k);
        m2_fs.read(reinterpret_cast<char*>(m2.get()), sizeof(float) * k * m);
        m1_fs.close(); m2_fs.close();

        std::memset(result.get(), 0, sizeof(float) * n * m);

        float* m2t_ptr = static_cast<float*>(aligned_alloc(64, sizeof(float) * k * m));
        std::unique_ptr<float[], decltype(&free)> m2t(m2t_ptr, &free);

        omp_set_num_threads(64);
        const int TRANS_BLOCK = 64;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int j0 = 0; j0 < m; j0 += TRANS_BLOCK) {
            for (int i0 = 0; i0 < k; i0 += TRANS_BLOCK) {
                int j_end = std::min(j0 + TRANS_BLOCK, m);
                int i_end = std::min(i0 + TRANS_BLOCK, k);
                for (int j = j0; j < j_end; j++) {
                    for (int i = i0; i < i_end; i++) {
                        m2t[j * k + i] = m2[i * m + j];
                    }
                }
            }
        }

        const int BN = 48, BM = 48, BK = 960;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i_block = 0; i_block < n; i_block += BN) {
            for (int j_block = 0; j_block < m; j_block += BM) {
                float local_block[BN][BM] = {0};

                for (int k_block = 0; k_block < k; k_block += BK) {
                    int i_end = std::min(i_block + BN, n);
                    int j_end = std::min(j_block + BM, m);
                    int k_end = std::min(k_block + BK, k);

                    for (int i = i_block; i < i_end; i++) {
                        int i_rel = i - i_block;
                        for (int j = j_block; j < j_end; j++) {
                            int j_rel = j - j_block;

                            __m512 sum0 = _mm512_setzero_ps();
                            __m512 sum1 = _mm512_setzero_ps();
                            __m512 sum2 = _mm512_setzero_ps();
                            __m512 sum3 = _mm512_setzero_ps();

                            int l = k_block;
                            for (; l + 63 < k_end; l += 64) {
                                __m512 a0 = _mm512_loadu_ps(&m1[i * k + l]);
                                __m512 b0 = _mm512_loadu_ps(&m2t[j * k + l]);
                                sum0 = _mm512_fmadd_ps(a0, b0, sum0);

                                __m512 a1 = _mm512_loadu_ps(&m1[i * k + l + 16]);
                                __m512 b1 = _mm512_loadu_ps(&m2t[j * k + l + 16]);
                                sum1 = _mm512_fmadd_ps(a1, b1, sum1);

                                __m512 a2 = _mm512_loadu_ps(&m1[i * k + l + 32]);
                                __m512 b2 = _mm512_loadu_ps(&m2t[j * k + l + 32]);
                                sum2 = _mm512_fmadd_ps(a2, b2, sum2);

                                __m512 a3 = _mm512_loadu_ps(&m1[i * k + l + 48]);
                                __m512 b3 = _mm512_loadu_ps(&m2t[j * k + l + 48]);
                                sum3 = _mm512_fmadd_ps(a3, b3, sum3);
                            }

                            for (; l + 15 < k_end; l += 16) {
                                __m512 a = _mm512_loadu_ps(&m1[i * k + l]);
                                __m512 b = _mm512_loadu_ps(&m2t[j * k + l]);
                                sum0 = _mm512_fmadd_ps(a, b, sum0);
                            }

                            float sum = _mm512_reduce_add_ps(_mm512_add_ps(
                                _mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3)));

                            for (; l < k_end; l++) {
                                sum += m1[i * k + l] * m2t[j * k + l];
                            }

                            local_block[i_rel][j_rel] += sum;
                        }
                    }
                }

                for (int i = i_block; i < std::min(i_block + BN, n); i++) {
                    for (int j = j_block; j < std::min(j_block + BM, m); j++) {
                        result[i * m + j] = local_block[i - i_block][j - j_block];
                    }
                }
            }
        }

        sol_fs.write(reinterpret_cast<const char*>(result.get()), sizeof(float) * n * m);
        sol_fs.close();
        return sol_path;
    }
};
