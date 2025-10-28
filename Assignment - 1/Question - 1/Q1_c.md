
# Performance Benchmarking Suite

This repository includes two benchmarking tools designed to evaluate the performance of CPU and memory systems: `CPU_Benchmark` and `STREAM`.

## 1. CPU Benchmark
This program measures the floating-point and integer operations per second (FLOPS and IOPS) of CPUs. It tests performance across different compiler optimization levels, highlighting the impact of compiler optimizations on computational efficiency.

### Results Overview
- **Without Optimization (`-O0`):**
  - FLOPS Throughput: 13.061 GFLOPS
  - IOPS Throughput: 17.535 GIOPS
- **With Optimization (`-O2`):**
  - FLOPS Throughput: 575.316 GFLOPS
  - IOPS Throughput: 1157.582 GIOPS

### Compilation and Execution
```bash
gcc -O2 -pthread CPU_Benchmark.c -o CPU_Benchmark
./CPU_Benchmark [operation count] [thread count]
```
- `[operation count]`: Total number of operations.
- `[thread count]`: Number of threads used.

## 2. STREAM Benchmark
Developed by John D. McCalpin, STREAM measures memory bandwidth for simple computational kernels and reflects the system's memory bandwidth capabilities under various compiler optimizations.

### Results Overview
- **Without Optimization (`-O0`):**
  - Copy: 5442.2 MB/s
  - Scale: 4735.0 MB/s
  - Add: 8147.2 MB/s
  - Triad: 8314.5 MB/s
- **With Optimization (`-O2`):**
  - Copy: 26359.6 MB/s
  - Scale: 14379.4 MB/s
  - Add: 16117.2 MB/s
  - Triad: 16145.1 MB/s

### Compilation and Execution
```bash
gcc -O -fopenmp -DSTREAM_ARRAY_SIZE=10000000 -DNTIMES=10 stream.c -o stream
./stream
```

## Analysis
The results demonstrate significant improvements in both CPU operations and memory transfer rates with higher optimization levels, attributed to more efficient code execution paths, reduced overhead, and better utilization of hardware capabilities.

## Conclusion
These benchmarks are critical tools for system architects and developers, providing insights into the potential performance gains achievable through compiler optimizations and hardware tuning.

## Licensing
- `CPU_Benchmark` is provided under an open source license.
- `STREAM` is licensed under terms that allow modifications and distribution, subject to certain conditions outlined by its authors.

### Additional Information
For detailed licensing information and restrictions, especially for publishing benchmark results, please refer to the source files and the official [STREAM website](http://www.cs.virginia.edu/stream/).

