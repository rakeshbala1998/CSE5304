#!/bin/bash

echo "=== Testing CPU Multicore Mandelbrot Performance ==="
echo ""

for size in 512 1024 2048 4096; do
    echo "Image size: ${size}x${size}"
    ./mandelbrot_cpu_2_s26.out -i scalar -r $size 2>&1 | grep "Runtime"
    ./mandelbrot_cpu_2_s26.out -i vector_multicore -r $size 2>&1 | grep "Runtime"
    echo ""
done
