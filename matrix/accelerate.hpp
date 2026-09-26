#pragma once
#include <iostream>
#include <thread>
#include <vector>
#include <immintrin.h>

void add_avx2(const float* a, const float* b, float* target, size_t lo, size_t hi){
    size_t i = lo;
    while (i < hi && (((uintptr_t)&a[i] | (uintptr_t)&b[i] | (uintptr_t)&target[i]) & 31) != 0){
        // 31D -> 0001 1111B
        // 如果是32字节对齐的内存，则低5位应该是0
        target[i] = a[i] + b[i];
        ++i;
    }
    for (; i + 8 <= hi; i += 8){
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_store_ps(&target[i], _mm256_add_ps(va, vb));
    }
    for (; i < hi; i++) target[i] = a[i] + b[i];
}

void sub_avx2(const float* a, const float* b, float* target, size_t lo, size_t hi){
    size_t i = lo;
    while (i < hi && (((uintptr_t)&a[i] | (uintptr_t)&b[i] | (uintptr_t)&target[i]) & 31)){
        target[i] = a[i] - b[i];
        ++i;
    }
    for (; i + 8 <= hi; i += 8){
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_store_ps(&target[i], _mm256_sub_ps(va, vb));
    }
    for (; i < hi; i++) target[i] = a[i] - b[i];
}

void vec_add(const float* a, const float* b, float* target, size_t n, int nthreads, bool isadd = true){
    if (n == 0) return;
    nthreads = std::min<size_t>(n, nthreads);
    if (nthreads <= 0) nthreads = 1;

    std::vector<std::thread> pool;
    size_t chuck = (n + nthreads - 1) / nthreads;

    for (int t = 0; t < nthreads; t++){
        size_t lo = t * chuck;
        size_t hi = std::min(lo + chuck, n);
        if (lo >= hi) break;

        if (isadd == true){
            pool.emplace_back(add_avx2, a, b, target, lo, hi);
        }else{
            pool.emplace_back(sub_avx2, a, b, target, lo, hi);
        }
    }
    for (auto& thread : pool) thread.join();
}

void matmul_avx2(const float* a, const float* b, float* c, int M, int K, int N){
    for (int i = 0; i < M * N; i++) c[i] = 0.0f;
    for (int i = 0; i < M; i++){
        for (int k = 0; k < K; k++){
            __m256 va = _mm256_set1_ps(a[i * K + k]);
            int j = 0;
            for (; j + 8 <= N; j += 8){
                __m256 vb = _mm256_loadu_ps(&b[k*N + j]);
                __m256 vc = _mm256_loadu_ps(&c[i*N + j]);
                vc = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(&c[i * N + j], vc);
            }
            for (; j < N; j++) c[i * N + j] += a[i * K + k] * b[k * N + j];
        }
    }
}