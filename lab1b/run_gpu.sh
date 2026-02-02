#!/bin/bash

cuFILE=mandelbrot_gpu_2_s26

nvcc -O3 --use_fast_math -gencode arch=compute_86,code=[sm_86,compute_86] --expt-relaxed-constexpr --std=c++17 $cuFILE.cu -o $cuFILE.out

# ./$cuFILE.out -i vector
# ./$cuFILE.out -i vector_multicore
# ./$cuFILE.out -i vector_multicore_multithread_single_sm
# ./$cuFILE.out -i vector_multicore_multithread_full
./$cuFILE.out -i all