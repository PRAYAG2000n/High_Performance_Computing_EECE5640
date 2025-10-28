
# Benchmark Experiment README

## Overview
This document describes how to execute a benchmark experiment involving three single-threaded programs on two specific Linux-based systems (COE linux and Discovery Cluster). The benchmarks will cover a diverse set of workloads including floating point operations, integer calculations, and memory-intensive tasks.

## Benchmark Selection
1. **Floating Point Benchmark:** A program that primarily performs floating point operations.
2. **Integer Benchmark:** A program that primarily performs integer calculations.
3. **Memory Intensive Benchmark:** A program that is demanding on memory usage.

## Systems Specification
To obtain detailed system specifications, use the `lscpu` command:
```bash
lscpu
```
This command provides information about the CPU architecture, such as number of cores, threads per core, and CPU frequencies.

### System 1: COE Linux System
- **CPU Model:** Intel® Xeon® CPU E5-2698 v4 @ 2.20GHz
- **Cores:** 2 sockets, 20 cores/socket, 2 threads/core
- **Operating System:** Rocky Linux Release 8.9 (Green Obsidian)

### System 2: Discovery Explorer
- **CPU Model:** Intel® Xeon® Gold 5318Y CPU @ 2.10 GHz
- **Cores:** 2 sockets, 24 cores/socket, 2 threads/core
- **Operating System:** Rocky Linux 9.3 (Blue Onyx)

## File Creation
Create the necessary source files using the nano editor:
```bash
nano cpu_benchmark.c        # C program for CPU benchmarking
nano gpu_benchmark.cu       # CUDA C program for GPU benchmarking
nano matrix_multiply.c      # C program for matrix multiplication
```

## Compilation
Compile each benchmark using gcc and nvcc for CUDA files. Here's how to compile them simultaneously using a Makefile or shell script:
```bash
gcc cpu_benchmark -o cpu_benchmark.c -pthread
nvcc -o gpu_benchmark gpu_benchmark.cu -lpthread
gcc -o matrix_multiply matrix_multiply.c -pthread
```

### Run the executable:
After compiling, you can run the program by specifying the number of operations and the number of threads as command line arguments. For example, to run the benchmark with 1000000 operations and 1 threads:

```bash
./cpu_benchmark 1000000 1
```
This command executes the cpu_benchmark program where 1000000 is the total number of operations, and 1 is the number of threads.

```bash
./gpu_benchmark 1000000
```
This command executes the gpu_benchmark program where 1000000 is the total number of operations.

```bash
./matrix_multiply 1000000 1
```


## Results
All 3 sets of program was executed in both COE linux and Discovery Cluster and this is the result we got:

Execution Time in COE Linux System:

## CPU Benchmark:
Run #	Time (sec)
1	    0.001973
2	    0.001956
3	    0.001958
4	    0.001930
5	    0.001941
6	    0.001852
7	    0.001864
8	    0.001934
9	    0.001881
10	    0.001935
Average CPU Benchmark FLOPS time: 0.001923 sec

## GPU Benchmark:
Run #	Time (sec)
1	    0.0000003
2-10	0.0000000

Average GPU Benchmark FLOPS time: 0.000000 sec

## DGEMM Benchmark:
Run #	Time (sec)
1	    0.011176
2	    0.008573
3	    0.008576
4	    0.008547
5	    0.008574
6	    0.008576
7	    0.008566
8	    0.008577
9	    0.008568
10	    0.008575
Average time: 0.008833 s

# Execution Time in Discovery Explorer

## CPU Benchmark:
Run #	Time (sec)
1	    0.003118
2	    0.002313
3	    0.002096
4	    0.001709
5	    0.001624
6	    0.001810
7	    0.002379
8	    0.002739
9	    0.002720
10	    0.002756
perl
Copy
Average FLOPS time: 0.002326 sec

## GPU Benchmark:
Run #	Time (sec)
1-10	0.000000
perl
Copy
Average time: 0.000000 sec

## DGEMM Benchmark:
Run #	Time (sec)
1	    0.004983
2	    0.004064
3	    0.004059
4	    0.004050
5	    0.004059
6	    0.004059
7	    0.004050
8	    0.004050
9	    0.004057
10	    0.004057
css
Copy
Average time: 0.004152 s

## Source Code
The source code for these benchmarks can be obtained from:
(1) https://github.com/arihant15/Performance-Evaluation-Benchmark/commit/b5c38c4f4d12efc84ab9ae1827a8984d71a870ca
(2) https://github.com/arihant15/Performance-Evaluation-Benchmark/tree/master/GPU
(3) https://github.com/cappachu/dgemm/blob/master/benchmark.c

## Conclusion
All of the observations are written in pdf document

## License
This project and documentation are licensed under the MIT License.
