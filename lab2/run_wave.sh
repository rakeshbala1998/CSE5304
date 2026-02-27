#!/bin/bash

cuFILE=wave

nvcc -O3 --use_fast_math -gencode arch=compute_86,code=[sm_86,compute_86] --expt-relaxed-constexpr --std=c++17 --compiler-bindir /usr/bin $cuFILE.cu -o $cuFILE.out

./$cuFILE.out