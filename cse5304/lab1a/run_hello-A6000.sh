#!/bin/bash

cuFILE=hello-A6000

nvcc -O3 --use_fast_math -gencode arch=compute_86,code=[sm_86,compute_86] --expt-relaxed-constexpr --std=c++17 $cuFILE.cu -o $cuFILE.out

./$cuFILE.out