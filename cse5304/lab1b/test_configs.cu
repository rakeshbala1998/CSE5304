#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

const float window_zoom = 1.0 / 10000.0f;
const float window_x = -0.743643887 - 0.5 * window_zoom;
const float window_y = 0.131825904 - 0.5 * window_zoom;
const uint32_t warp_size = 32;

__global__ void mandelbrot_kernel(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
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
        float x2 = 0.0f, y2 = 0.0f, w = 0.0f;
        uint32_t iters = 0;
        while (x2 + y2 <= 4.0f && iters < max_iters) {
            float x = x2 - y2 + cx;
            float y = w - (x2 + y2) + cy;
            x2 = x * x; y2 = y * y;
            float z = x + y; w = z * z;
            ++iters;
        }
        out[pixel] = iters;
    }
}

float test_config(int blocks, int threads, uint32_t img_size, uint32_t max_iters, uint32_t *d_out) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    mandelbrot_kernel<<<blocks, threads>>>(img_size, max_iters, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

int main() {
    uint32_t img_size = 2048, max_iters = 2000;
    uint32_t *d_out;
    cudaMalloc(&d_out, img_size * img_size * sizeof(uint32_t));
    
    printf("Testing launch configurations (2048x2048, 2000 iterations):\n");
    printf("All configs produce 336 warps (1 per scheduler)\n\n");
    printf("Config <<<84, 128>>>:  %.2f ms\n", test_config(84, 128, img_size, max_iters, d_out));
    printf("Config <<<168, 64>>>:  %.2f ms\n", test_config(168, 64, img_size, max_iters, d_out));
    printf("Config <<<42, 256>>>:  %.2f ms\n", test_config(42, 256, img_size, max_iters, d_out));
    
    cudaFree(d_out);
    return 0;
}
