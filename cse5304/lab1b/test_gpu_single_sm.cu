// GPU Single SM Multi-Threading Benchmark
// Tests different warp counts on a single SM

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <iomanip>

const uint32_t warp_size = 32;
const float window_zoom = 3.0f;
const float window_x = -2.0f;
const float window_y = -1.5f;

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void mandelbrot_single_sm(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out,
    uint32_t num_warps
) {
    uint32_t warp_id = threadIdx.x / warp_size;
    uint32_t lane_id = threadIdx.x % warp_size;
    
    uint64_t total_pixels = (uint64_t)img_size * img_size;
    uint64_t pixels_per_warp = (total_pixels + num_warps - 1) / num_warps;
    uint64_t start_pixel = warp_id * pixels_per_warp;
    uint64_t end_pixel = start_pixel + pixels_per_warp;
    if (end_pixel > total_pixels) end_pixel = total_pixels;
    
    for (uint64_t pixel = start_pixel + lane_id; pixel < end_pixel; pixel += warp_size) {
        uint32_t i = pixel / img_size;
        uint32_t j = pixel % img_size;
        
        float cx = (float(j) / float(img_size)) * window_zoom + window_x;
        float cy = (float(i) / float(img_size)) * window_zoom + window_y;
        
        float x2 = 0.0f;
        float y2 = 0.0f;
        float w = 0.0f;
        uint32_t iters = 0;
        while (x2 + y2 <= 4.0f && iters < max_iters) {
            float x = x2 - y2 + cx;
            float y = w - (x2 + y2) + cy;
            x2 = x * x;
            y2 = y * y;
            float z = x + y;
            w = z * z;
            ++iters;
        }
        
        out[pixel] = iters;
    }
}

double benchmark_single_sm(uint32_t img_size, uint32_t max_iters, uint32_t *out_device, uint32_t num_warps) {
    const int warmup = 2;
    const int iterations = 5;
    
    uint32_t threads = num_warps * warp_size;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        mandelbrot_single_sm<<<1, threads>>>(img_size, max_iters, out_device, num_warps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        mandelbrot_single_sm<<<1, threads>>>(img_size, max_iters, out_device, num_warps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() 
           / (double)iterations / 1e6;
}

int main() {
    const uint32_t img_size = 2048;
    const uint32_t max_iters = 2000;
    
    uint32_t *out_device;
    CUDA_CHECK(cudaMalloc(&out_device, img_size * img_size * sizeof(uint32_t)));
    
    std::cout << "GPU Single SM Multi-Threading Benchmark (2048x2048 image)\n";
    std::cout << "Each SM has 4 warp schedulers\n";
    std::cout << "Max 32 warps per block (hardware limit = 1024 threads)\n";
    std::cout << "============================================================\n\n";
    
    // Test various warp counts
    std::vector<uint32_t> warp_counts = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    
    double baseline_time = 0;
    
    for (uint32_t warps : warp_counts) {
        double time = benchmark_single_sm(img_size, max_iters, out_device, warps);
        double warps_per_scheduler = warps / 4.0;
        
        if (warps == 4) baseline_time = time;
        double speedup = (baseline_time > 0) ? (baseline_time / time) : 1.0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Warps: " << std::setw(2) << warps 
                  << " (" << std::setw(5) << warps_per_scheduler << " per scheduler)"
                  << "  Runtime: " << std::setw(7) << time << " ms";
        
        if (warps == 4) {
            std::cout << "  [baseline: 1 warp/scheduler]";
        } else {
            std::cout << "  Speedup: " << std::setw(5) << speedup << "×";
        }
        std::cout << "\n";
    }
    
    CUDA_CHECK(cudaFree(out_device));
    return 0;
}
