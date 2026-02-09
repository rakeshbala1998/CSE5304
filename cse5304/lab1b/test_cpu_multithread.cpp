// CPU Multi-Threading Benchmark
// Tests different thread counts to find optimal

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

struct ThreadArgs {
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
    uint32_t start_row;
    uint32_t end_row;
};

void* mandelbrot_thread_worker(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    uint32_t img_size = args->img_size;
    uint32_t max_iters = args->max_iters;
    uint32_t* out = args->out;
    uint32_t start_row = args->start_row;
    uint32_t end_row = args->end_row;
    
    for (uint32_t i = start_row; i < end_row; ++i) {
        for (uint32_t j = 0; j < img_size; j += 8) {
            __m256 j_vec = _mm256_set_ps(
                float(j + 7), float(j + 6), float(j + 5), float(j + 4),
                float(j + 3), float(j + 2), float(j + 1), float(j + 0)
            );
            __m256 img_size_vec = _mm256_set1_ps(float(img_size));
            __m256 window_zoom_vec = _mm256_set1_ps(window_zoom);
            __m256 window_x_vec = _mm256_set1_ps(window_x);
            __m256 window_y_vec = _mm256_set1_ps(window_y);
            
            __m256 cx = _mm256_add_ps(
                _mm256_mul_ps(_mm256_div_ps(j_vec, img_size_vec), window_zoom_vec),
                window_x_vec
            );
            
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;
            __m256 cy_vec = _mm256_set1_ps(cy);
            
            __m256 x2 = _mm256_setzero_ps();
            __m256 y2 = _mm256_setzero_ps();
            __m256 w = _mm256_setzero_ps();
            __m256i iters_vec = _mm256_setzero_si256();
            
            __m256 four = _mm256_set1_ps(4.0f);
            __m256i one = _mm256_set1_epi32(1);
            
            for (uint32_t iter = 0; iter < max_iters; ++iter) {
                __m256 sum = _mm256_add_ps(x2, y2);
                __m256 mask = _mm256_cmp_ps(sum, four, _CMP_LE_OQ);
                if (_mm256_movemask_ps(mask) == 0) break;
                
                __m256 x = _mm256_add_ps(_mm256_sub_ps(x2, y2), cx);
                __m256 y = _mm256_add_ps(_mm256_sub_ps(w, sum), cy_vec);
                x2 = _mm256_mul_ps(x, x);
                y2 = _mm256_mul_ps(y, y);
                __m256 z = _mm256_add_ps(x, y);
                w = _mm256_mul_ps(z, z);
                
                __m256i mask_int = _mm256_castps_si256(mask);
                iters_vec = _mm256_sub_epi32(iters_vec, _mm256_and_si256(mask_int, one));
            }
            
            uint32_t iters[8];
            _mm256_storeu_si256((__m256i*)iters, iters_vec);
            for (int k = 0; k < 8; ++k) {
                out[i * img_size + j + k] = (uint32_t)(-iters[k]);
            }
        }
    }
    return nullptr;
}

double benchmark_cpu(uint32_t img_size, uint32_t max_iters, uint32_t* out, uint32_t num_threads) {
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArgs> thread_args(num_threads);
    
    uint32_t rows_per_thread = img_size / num_threads;
    
    const int warmup = 1;
    const int iterations = 3;
    
    // Warmup
    for (int w = 0; w < warmup; w++) {
        for (uint32_t t = 0; t < num_threads; ++t) {
            thread_args[t].img_size = img_size;
            thread_args[t].max_iters = max_iters;
            thread_args[t].out = out;
            thread_args[t].start_row = t * rows_per_thread;
            thread_args[t].end_row = (t == num_threads - 1) ? img_size : (t + 1) * rows_per_thread;
            pthread_create(&threads[t], nullptr, mandelbrot_thread_worker, &thread_args[t]);
        }
        for (uint32_t t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        for (uint32_t t = 0; t < num_threads; ++t) {
            thread_args[t].img_size = img_size;
            thread_args[t].max_iters = max_iters;
            thread_args[t].out = out;
            thread_args[t].start_row = t * rows_per_thread;
            thread_args[t].end_row = (t == num_threads - 1) ? img_size : (t + 1) * rows_per_thread;
            pthread_create(&threads[t], nullptr, mandelbrot_thread_worker, &thread_args[t]);
        }
        for (uint32_t t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() 
           / (double)iterations / 1e6;
}

int main() {
    const uint32_t img_size = 2048;
    const uint32_t max_iters = 2000;
    
    std::vector<uint32_t> out(img_size * img_size);
    
    std::cout << "CPU Multi-Threading Benchmark (2048x2048 image)\n";
    std::cout << "CPU has 64 cores\n";
    std::cout << "============================================================\n\n";
    
    // Test various thread counts
    std::vector<uint32_t> thread_counts = {32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512};
    
    double baseline_time = 0;
    
    for (uint32_t threads : thread_counts) {
        double time = benchmark_cpu(img_size, max_iters, out.data(), threads);
        double threads_per_core = threads / 64.0;
        
        if (threads == 64) baseline_time = time;
        double speedup = (baseline_time > 0) ? (baseline_time / time) : 1.0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Threads: " << std::setw(3) << threads 
                  << " (" << threads_per_core << "× cores)"
                  << "  Runtime: " << std::setw(7) << time << " ms";
        if (threads == 64) {
            std::cout << "  [baseline]";
        } else {
            std::cout << "  Speedup: " << std::setw(5) << speedup << "×";
        }
        std::cout << "\n";
    }
    
    return 0;
}
