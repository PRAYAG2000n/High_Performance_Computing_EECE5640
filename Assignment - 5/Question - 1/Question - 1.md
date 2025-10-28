
# Project Overview
This project includes two implementations of a histogram program:
1. **GPU version using CUDA** - Optimized for parallel execution on NVIDIA GPUs.
2. **CPU version using OpenMP** - Designed to run on multi-core processors to utilize CPU parallelism.

## Prerequisites
- **CUDA Toolkit**: Ensure you have CUDA installed if you plan to run the GPU version. Check with `nvcc --version`.
- **Compiler for OpenMP**: GCC or another compiler that supports OpenMP for the CPU version.

## Directory Structure
```plaintext
/Histogram
│
├── src/
│   ├── histogram_gpu.cu  - CUDA implementation for the GPU.
│   └── histogram_cpu.cpp - OpenMP implementation for the CPU.
│
└── README.md
```
## Create a C++ and CUDA file
```bash
nano histogram_gpu.cu // For computing the GPU program

nano histogram_cpu.cpp // For computing the CPU program
```
## Calling the GPU module
we can submit the job for getting the gpu module, if the gpu is available then it would show job submitted.
```bash

   srun --partition=courses-gpu --nodes=1 --pty --gres=gpu:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
```
If the GPU resource is available then it will be displayed as 'Job submitted'.

## Compiling and Running the Code

### GPU Version (CUDA)
1. **Compile the CUDA Program**
   Navigate to the `src/` directory and run:
   ```bash
   nvcc -O2 -arch=sm_60 -o histogram_gpu histogram_gpu.cu
   ```
   Replace `-arch=sm_60` with the appropriate compute capability for your GPU (e.g., `sm_70` for V100).

2. **Execute the Program**
   ```bash
   ./histogram_gpu
   ```

### CPU Version (OpenMP)
1. **Compile the OpenMP Program**
   Navigate to the `src/` directory and run:
   ```bash
   g++ -O2 -fopenmp histogram_cpu.cpp -o histogram_cpu
   ```

2. **Execute the Program**
   ```bash
   ./histogram_cpu
   ```

## Expected Output
Both programs will output the execution time and a sample value from each histogram class for varying input sizes (from 2^12 to 2^23).
```bash
--- Histogram for N = 2^14 = 16384 ---
CUDA histogram completed in 0.000115 ms
Example input value from each class:
Class 0: 7414
Class 1: 19708
Class 2: 20230
Class 3: 36333
Class 4: 48568
Class 5: 50429
Class 6: 62755
Class 7: 71435
Class 8: 87443
Class 9: 96678

```
## Sources
https://github.com/e-/MulticoreComputing/blob/master/OpenMP/histogram.c
https://github.com/epielemeier/oc-openmp-histogram/blob/master/histogram.c
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/histogram/main.cpp
https://github.com/shu65/cuda-histogram/blob/main/src/histogram_gpu.cu


## Troubleshooting
- **CUDA Compilation Issues**: Ensure that the CUDA path is correctly set and that host is using a compatible GCC version.
- **OpenMP Compilation Issues**: Make sure that OpenMP is enabled in the compiler settings.


