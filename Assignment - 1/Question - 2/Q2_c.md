
# Parallel QuickSort and MergeSort Benchmark - Scaling Analysis

This README provides a comprehensive overview of strong and weak scaling analysis for a parallel sorting program implemented in C. 
The program utilizes multithreading to perform quicksort and mergesort on an array, designed to run on Linux using pthreads.

## Strong Scaling
Strong scaling measures the performance of the program as the number of threads increases while keeping the total problem size (array size) constant.

### Strong Scaling Efficiency Calculation
The strong scaling efficiency is calculated using the formula:
\[ 	ext{Efficiency} = \left(rac{	ext{Execution Time at 1 Thread} 	imes 100}{	ext{Execution Time at N Threads} 	imes N}ight) \% \]

### Expected Strong Scaling Output
```
Threads  Array Size (elements)  Execution Time (s)  Efficiency
1        10000                 0.0020              100%
2        10000                 0.0012              83.5%
4        10000                 0.0009              55.5%
8        10000                 0.0008              31.25%
32       10000                 0.0031              2.03%
```

## Weak Scaling
Weak scaling measures how the execution time varies as the number of threads increases when the problem size per thread remains constant.

### Weak Scaling Efficiency Calculation
Weak scaling efficiency is calculated using:
\[ 	ext{Efficiency} = \left(rac{	ext{Execution Time at 1 Thread}}{	ext{Execution Time at N Threads}}ight) 	imes 100 \% \]

### Expected Weak Scaling Output
```
Threads  Array Size (elements)  Execution Time (s)  Efficiency
1        10000                 0.0020              100%
2        20000                 0.0013              153.8%
4        40000                 0.0010              200%
8        80000                 0.0009              222%
32       320000                0.0034              58.8%
```

## Compilation and Execution
To compile the program with GCC and pthreads support:
```bash
gcc -o parallel_sort parallel_sort.c -pthread
```

To run the program, specify the maximum number of threads:
```bash
./parallel_sort <max_threads>
```

## Modifying the Code
You can modify the source code using `nano` or any other editor to adjust parameters like array size or thread count to explore different scaling scenarios.

```bash
nano parallel_sort.c
```

## Conclusion
This program serves as a practical tool to study the scalability of parallel sorting algorithms on multicore processors. It helps in understanding both the strong and weak scaling characteristics essential for optimizing parallel applications.

