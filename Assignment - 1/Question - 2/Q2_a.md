
# Parallel QuickSort and MergeSort Benchmark

This repository contains a C program that utilizes multithreading to perform quicksort and mergesort on an array. 
The program is designed to run on Linux and utilizes pthreads to demonstrate the performance gains from using multiple threads.

## Program Description
The program divides an array into chunks, each sorted in parallel using quicksort, followed by a parallel merge step to 
combine the sorted chunks. It measures the time taken for sorting with different numbers of threads and displays the performance.

## Editing the Code
You can edit the source code using `nano`, a simple text editor available on most Linux systems. If `nano` is not installed, you can install it using:

```bash
sudo apt-get install nano  # Debian/Ubuntu
# or
sudo yum install nano      # CentOS/RHEL
```

To edit the code, simply run:

```bash
nano parallel_sort.c
```

Make any modifications you need and save the file by pressing `Ctrl+O` and then exit using `Ctrl+X`.

## Compilation
To compile the program, you need GCC and pthreads support on your Linux machine. Use the following command to compile:

```bash
gcc -o parallel_sort parallel_sort.c -pthread
```

## Execution
Run the program by specifying the maximum number of threads as a command-line argument:

```bash
./parallel_sort <max_threads>
```

Where `<max_threads>` is the maximum number of threads you want the program to use.

## Expected Output
The program will output the execution times for sorting the array using different numbers of threads. 
Here is an example of the expected output format:

```
Threads  Execution time (in seconds)
1        0.0020
2        0.0012
4        0.0009
8        0.0008
32       0.0031
```

This output shows how the execution time decreases as the number of threads increases up to a certain point, 
after which the overhead of managing more threads outweighs the performance benefits.

## Sample Run
For example, if you run the program with 32 as the maximum number of threads, it will display the execution times for 1, 2, 4, 8, 
and 32 threads, allowing you to observe the scale of performance changes.

## Conclusion
This benchmark is useful for understanding the impact of multithreading on sorting algorithms and can help in 
optimizing applications that require efficient data sorting and processing.

