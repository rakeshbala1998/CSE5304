#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 2048
#define HEIGHT 2048
#define MAX_ITERS 2000

__global__ void mandelbrot_kernel(int *out, int img_size, int max_iters) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= img_size || col >= img_size) return;
    
    float scale = 4.0f / img_size;
    float cr = col * scale - 2.0f;
    float ci = row * scale - 2.0f;
    
    float zr = 0.0f, zi = 0.0f;
    int iter = 0;
    
    while (iter < max_iters && (zr * zr + zi * zi) < 4.0f) {
        float temp = zr * zr - zi * zi + cr;
        zi = 2.0f * zr * zi + ci;
        zr = temp;
        iter++;
    }
    
    out[row * img_size + col] = iter;
}

int main() {
    int *d_out;
    size_t size = WIDTH * HEIGHT * sizeof(int);
    
    cudaMalloc(&d_out, size);
    
    // Test different warp counts on single SM
    // Single SM = 1 block with varying threads
    int warp_counts[] = {4, 8, 16, 24, 32};
    int num_tests = 5;
    
    printf("Testing Single SM Performance with Different Warp Counts\n");
    printf("Image: %dx%d, Max Iterations: %d\n\n", WIDTH, HEIGHT, MAX_ITERS);
    printf("Warps  Threads  Runtime(ms)\n");
    printf("-----  -------  -----------\n");
    
    for (int i = 0; i < num_tests; i++) {
        int warps = warp_counts[i];
        int threads = warps * 32; // 32 threads per warp
        
        // Single block with 'threads' threads
        dim3 block(threads, 1);
        dim3 grid(1, 1);
        
        // Warm-up
        mandelbrot_kernel<<<grid, block>>>(d_out, WIDTH, MAX_ITERS);
        cudaDeviceSynchronize();
        
        // Measure
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        mandelbrot_kernel<<<grid, block>>>(d_out, WIDTH, MAX_ITERS);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("%5d  %7d  %11.2f\n", warps, threads, milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    cudaFree(d_out);
    
    return 0;
}
