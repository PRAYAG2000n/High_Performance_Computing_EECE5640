
# 3D Stencil Computation in CUDA

This repository contains CUDA implementations of a 3D stencil computation. The stencil computation is a common algorithm used in image processing, simulations, and other areas requiring the processing of multi-dimensional data. This README provides a brief overview of the two implementations included: a naive approach and a tiled approach using shared memory.

## Files

- `naive_stencil.cu`: Naive implementation of the stencil computation without using shared memory.
- `tiled_stencil.cu`: Optimized implementation using shared memory to enhance performance by exploiting data locality.

## Requirements

- NVIDIA CUDA Toolkit (tested with version 10.2, but other versions should work)
- An NVIDIA GPU with CUDA Compute Capability 3.0 or higher

## Loading the CUDA application
CUDA application should be preloaded for GPU programming compilation
```bash
module load cuda/12.3.0
```
## Creating a new CUDA file
First the CUDA file is created so as to write our code.
```bash
nano naive_stencil.cu 
nano titled_stencil.cu
```

## Building and Running

Before running the compiled executables, request a GPU resource on a cluster managed with SLURM:

```bash
srun --partition=courses-gpu --nodes=1 --pty --gres=gpu:1 --ntasks=1 --mem=4GB --time=01:00:00 /bin/bash
```
when the GPU resource is available then it would be printed as 
```bash
srun: job xx.... has been allocated resources
```
Type nvidia-smi in bash to see which GPU resource has been allocated
```bash
nvidia-smi
```

1. **Compile the Code**
Use the NVIDIA CUDA Compiler (`nvcc`) to compile each `.cu` file.

```bash
nvcc naive_stencil.cu -o naive_stencil -lcudart
nvcc tiled_stencil.cu -o tiled_stencil -lcudart
```

2. **Run the Executables**

After compilation, you can run the generated binaries.
 ```bash
./naive_stencil
./tiled_stencil
```

## Example Output

Running the binaries will output the computed stencil values for a few positions in the resulting array. Rhe code can be modified to print additional data or to handle different input sizes.
```bash
N-value: ....
Native execution time: ........ms // when naive_stencil.cu is executed
Titled execution time: .......ms // when titled_stencil.cu is executed
```

## Performance Notes

- The naive implementation may perform adequately for small data sizes but will likely be slower for larger inputs due to high global memory access.
- The tiled version should provide better performance on larger data sets by reducing global memory access and exploiting the fast shared memory on CUDA cores.

The value of N is changed to 64 and 96 but not 128 because the CUDA version and the explorer cluster GPU is not capable of computing large numbers. 

## Sources
https://github.com/Hazard-Nico/1dStencilCUDA/blob/master/1d_stencil.cu
https://github.com/blueyi/CUDA-Stencil-Code-Optimization/blob/master/cuda/no-optimization/noop_jacobi_27pt.cu
https://github.com/Hazard-Nico/1dStencilCUDA/blob/master/1d_stencil_optim.cu
https://github.com/andrej/stencil-performance/blob/main/experiments/coalescing.cu


