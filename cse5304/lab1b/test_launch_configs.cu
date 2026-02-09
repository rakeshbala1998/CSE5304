#include <iostream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <cmath>

// Constants
const uint32_t warp_size = 32;
const uint32_t num_warp_schedulers = 4;
const uint32_t num_sms = 84;

// Mandelbrot window
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

// The kernel - same as mandelbrot_gpu_vector_multicore
__global__ void mandelbrot_kernel(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out,
    uint32_t num_blocks,
    uint32_t threads_per_block
) {
    // Calculate warp ID based on the actual grid dimensions
    uint32_t warps_per_block = threads_per_block / warp_size;
    uint32_t warp_id = blockIdx.x * warps_per_block + threadIdx.x / warp_size;
    
    // Calculate thread ID within warp (lane ID: 0-31)
    uint32_t lane_id = threadIdx.x % warp_size;
    
    // Total number of warps in this configuration
    uint32_t total_warps = num_blocks * warps_per_block;
    
    // Total pixels to compute
    uint64_t total_pixels = (uint64_t)img_size * img_size;
    
    // Partition work across warps
    uint64_t pixels_per_warp = (total_pixels + total_warps - 1) / total_warps;
    uint64_t start_pixel = warp_id * pixels_per_warp;
    uint64_t end_pixel = start_pixel + pixels_per_warp;
    if (end_pixel > total_pixels) end_pixel = total_pixels;
    
    // Each thread in the warp processes every 32nd pixel
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

double benchmark_config(uint32_t img_size, uint32_t max_iters, uint32_t *out_device,
                        uint32_t num_blocks, uint32_t threads_per_block) {
    const int warmup = 3;
    const int iterations = 5;
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        mandelbrot_kernel<<<num_blocks, threads_per_block>>>(
            img_size, max_iters, out_device, num_blocks, threads_per_block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        mandelbrot_kernel<<<num_blocks, threads_per_block>>>(
            img_size, max_iters, out_device, num_blocks, threads_per_block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() 
                      / (double)iterations / 1e6;  // Convert to ms
    
    return avg_time;
}

int main() {
    const uint32_t img_size = 2048;
    const uint32_t max_iters = 2000;
    
    // Allocate device memory
    uint32_t *out_device;
    CUDA_CHECK(cudaMalloc(&out_device, img_size * img_size * sizeof(uint32_t)));
    
    std::cout << "Testing different launch configurations on " << img_size << "x" << img_size 
              << " image\n";
    std::cout << "============================================================\n\n";
    
    // Configuration 1: <<<84, 128>>> = 336 warps (1 warp per scheduler)
    uint32_t blocks1 = 84;
    uint32_t threads1 = 128;
    uint32_t warps1 = blocks1 * (threads1 / warp_size);
    double time1 = benchmark_config(img_size, max_iters, out_device, blocks1, threads1);
    std::cout << "Config 1: <<<" << blocks1 << ", " << threads1 << ">>>\n";
    std::cout << "  Total warps: " << warps1 << " (1 per scheduler)\n";
    std::cout << "  Runtime: " << time1 << " ms\n\n";
    
    // Configuration 2: <<<168, 64>>> = 336 warps (1 warp per scheduler)
    uint32_t blocks2 = 168;
    uint32_t threads2 = 64;
    uint32_t warps2 = blocks2 * (threads2 / warp_size);
    double time2 = benchmark_config(img_size, max_iters, out_device, blocks2, threads2);
    std::cout << "Config 2: <<<" << blocks2 << ", " << threads2 << ">>>\n";
    std::cout << "  Total warps: " << warps2 << " (1 per scheduler)\n";
    std::cout << "  Runtime: " << time2 << " ms\n\n";
    
    // Configuration 3: <<<42, 256>>> = 336 warps (1 warp per scheduler)
    uint32_t blocks3 = 42;
    uint32_t threads3 = 256;
    uint32_t warps3 = blocks3 * (threads3 / warp_size);
    double time3 = benchmark_config(img_size, max_iters, out_device, blocks3, threads3);
    std::cout << "Config 3: <<<" << blocks3 << ", " << threads3 << ">>>\n";
    std::cout << "  Total warps: " << warps3 << " (1 per scheduler)\n";
    std::cout << "  Runtime: " << time3 << " ms\n\n";
    
    // Additional config: <<<84, 256>>> = 672 warps (2 per scheduler)
    uint32_t blocks4 = 84;
    uint32_t threads4 = 256;
    uint32_t warps4 = blocks4 * (threads4 / warp_size);
    double time4 = benchmark_config(img_size, max_iters, out_device, blocks4, threads4);
    std::cout << "Config 4: <<<" << blocks4 << ", " << threads4 << ">>>\n";
    std::cout << "  Total warps: " << warps4 << " (2 per scheduler)\n";
    std::cout << "  Runtime: " << time4 << " ms\n\n";
    
    std::cout << "============================================================\n";
    std::cout << "Summary:\n";
    std::cout << "  Config 1 (<<<84, 128>>>):  " << time1 << " ms\n";
    std::cout << "  Config 2 (<<<168, 64>>>):  " << time2 << " ms\n";
    std::cout << "  Config 3 (<<<42, 256>>>):  " << time3 << " ms\n";
    std::cout << "  Config 4 (<<<84, 256>>>):  " << time4 << " ms (for comparison)\n";
    
    CUDA_CHECK(cudaFree(out_device));
    return 0;
}
