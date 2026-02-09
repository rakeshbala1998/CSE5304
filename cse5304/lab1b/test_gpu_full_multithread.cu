// GPU Full Multi-Threading Benchmark
// Tests different warp counts across all SMs

#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <iomanip>

const uint32_t warp_size = 32;
const uint32_t num_sms = 84;
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

__global__ void mandelbrot_full(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    uint32_t warps_per_block = blockDim.x / warp_size;
    uint32_t warp_id = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    uint32_t lane_id = threadIdx.x % warp_size;
    
    uint32_t total_warps = gridDim.x * warps_per_block;
    uint64_t total_pixels = (uint64_t)img_size * img_size;
    
    uint64_t pixels_per_warp = (total_pixels + total_warps - 1) / total_warps;
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

double benchmark_full(uint32_t img_size, uint32_t max_iters, uint32_t *out_device, 
                      uint32_t warps_per_sm) {
    const int warmup = 2;
    const int iterations = 5;
    
    uint32_t threads_per_block = warps_per_sm * warp_size;
    uint32_t num_blocks = num_sms;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        mandelbrot_full<<<num_blocks, threads_per_block>>>(img_size, max_iters, out_device);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        mandelbrot_full<<<num_blocks, threads_per_block>>>(img_size, max_iters, out_device);
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
    
    std::cout << "GPU Full Multi-Threading Benchmark (2048x2048 image)\n";
    std::cout << "GPU has 84 SMs with 4 warp schedulers each = 336 total schedulers\n";
    std::cout << "============================================================\n\n";
    
    // Test various warps per SM
    std::vector<uint32_t> warps_per_sm_list = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    
    double baseline_time = 0;
    
    for (uint32_t warps_per_sm : warps_per_sm_list) {
        uint32_t total_warps = num_sms * warps_per_sm;
        uint32_t total_threads = total_warps * warp_size;
        double warps_per_scheduler = warps_per_sm / 4.0;
        
        double time = benchmark_full(img_size, max_iters, out_device, warps_per_sm);
        
        if (warps_per_sm == 4) baseline_time = time;
        double speedup = (baseline_time > 0) ? (baseline_time / time) : 1.0;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Warps/SM: " << std::setw(2) << warps_per_sm 
                  << " (" << std::setw(5) << warps_per_scheduler << " per scheduler)"
                  << "  Total warps: " << std::setw(4) << total_warps
                  << "  Runtime: " << std::setw(6) << time << " ms";
        
        if (warps_per_sm == 4) {
            std::cout << "  [baseline]";
        } else {
            std::cout << "  Speedup: " << std::setw(5) << speedup << "×";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n";
    std::cout << "Note: Baseline is 4 warps/SM = 1 warp per scheduler (336 total warps)\n";
    
    CUDA_CHECK(cudaFree(out_device));
    return 0;
}
