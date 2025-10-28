
# README: Running Leibniz and Monte Carlo Simulations

This README provides instructions on how to compile and run the Leibniz and Monte Carlo π calculation programs using pthreads and OpenMP on a Linux system.

## Directory Structure

The project directory should contain the following files with nano as a prefix:
- `leibniz.cpp`: Leibniz method using pthreads.
- `monte_carlo.cpp`: Monte Carlo method using pthreads.
- `leibniz_openmp.cpp`: Leibniz method using OpenMP.
- `monte_carlo_openmp.cpp`: Monte Carlo method using OpenMP.

## Compilation Instructions

### Using pthreads

1. **Leibniz Method with pthreads:**
   ```bash
   g++ -o leibniz leibniz.cpp -lpthread
   ```

2. **Monte Carlo Method with pthreads:**
   ```bash
   g++ -o monte_carlo monte_carlo.cpp -lpthread
   ```

### Using OpenMP

1. **Leibniz Method with OpenMP:**
   ```bash
   g++ -fopenmp -o leibniz_openmp leibniz_openmp.cpp
   ```

2. **Monte Carlo Method with OpenMP:**
   ```bash
   g++ -fopenmp -o monte_carlo_openmp monte_carlo_openmp.cpp
   ```

## Execution Instructions

After compilation, you can run the programs as follows:

### Using pthreads

1. **Leibniz Method with pthreads:**
   ```bash
   ./leibniz <number of threads> <number of terms>
   ```

2. **Monte Carlo Method with pthreads:**
   ```bash
   ./monte_carlo <number of threads> <number of darts>
   ```

### Using OpenMP

1. **Leibniz Method with OpenMP:**
   ```bash
   ./leibniz_openmp <number of threads> <number of terms>
   ```

2. **Monte Carlo Method with OpenMP:**
   ```bash
   ./monte_carlo_openmp <number of threads> <number of darts>
   ```

## Example Usage

- Run the Leibniz method with 10 threads and 1,000,000 terms:
  ```bash
  ./leibniz_openmp 10 1000000
  ```

- Run the Monte Carlo simulation with 10 threads and 1,000,000 darts:
  ```bash
  ./monte_carlo_openmp 10 1000000
  ```

## Strong Scaling

Strong scaling is evaluated based on how the execution time decreases when increasing the number of threads while keeping the total workload constant. It measures the efficiency of parallelization by showing how well the added computing power reduces execution time.

## Weak Scaling

Weak scaling is measured by increasing the workload in proportion to the number of threads, aiming to maintain a constant workload per thread. This scaling metric helps in understanding how well the system can handle larger problems as the computing resources are scaled up.

## Sources:
https://ideone.com/en2U9d
https://www.geeksforgeeks.org/estimating-the-value-of-pi-using-monte-carlo-parallel-computing-method/
https://github.com/michaelballantyne/montecarlo-pi/blob/master/pi.c
https://github.com/FelemixX/Parallel-Programming-Open-MP-MPI-/blob/main/Pi%20calculation%20using%20Leibniz%20formula%20OpenMP.cpp
