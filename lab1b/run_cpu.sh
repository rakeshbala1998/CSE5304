#!/bin/bash

cuFILE=mandelbrot_cpu_2_s26

g++ -march=native -O3 -Wall -Wextra -o $cuFILE.out $cuFILE.cpp

# ./$cuFILE.out -i vector
# ./$cuFILE.out -i vector_multicore
# ./$cuFILE.out -i vector_multicore_multithread
./$cuFILE.out -i all