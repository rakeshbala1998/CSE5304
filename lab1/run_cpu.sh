#!/bin/bash

cuFILE=mandelbrot_cpu.cpp

g++ -march=native -O3 -Wall -Wextra -o mandelbrot $cuFILE

./mandelbrot
