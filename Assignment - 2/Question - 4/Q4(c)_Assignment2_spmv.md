
# SpMV RCM Implementation

This project implements the Sparse Matrix-Vector Multiplication (SpMV) using the Reverse Cuthill–McKee (RCM) algorithm for matrix reordering to optimize memory access patterns and improve performance.

## Features

- **Matrix Reading:** Parses `.mtx` files in the Matrix Market format.
- **RCM Ordering:** Applies the RCM algorithm to reorder the matrix.
- **SpMV Computation:** Performs sparse matrix-vector multiplication using both the original and reordered matrices.
- **Performance Comparison:** Compares performance between the original and reordered matrices.

## Prerequisites

- C++ Compiler (GCC recommended)
- OpenMP for parallel computation
- Linux or macOS environment (Windows users can use WSL)

## File Creation
Create the necessary source files using the nano editor (copy paste the code from cpp file attached):
```bash
nano spmv.cpp
```

## Compilation and Execution
Use the provided Makefile or the following GCC command to compile the project:
 ```bash
   g++ -O3 -fopenmp spmv_rcm_example.cpp -o spmv_rcm
 ```

## Usage

Run the program with a Matrix Market format file `.mtx` as an argument:

```bash
./spmv_rcm toy.mtx
```

Where `toy.mtx` is your input matrix file stored in the Matrix Market format.

## Input File Format

The program expects the input file in Matrix Market format which starts with a header (optional), followed by matrix dimensions and non-zero count, then lists each non-zero element in the matrix:

```
%%MatrixMarket matrix coordinate real general
% Rows Columns Non-zeros
4 4 5
1 1 1.0
2 2 2.0
3 3 3.0
4 4 4.0
2 1 0.5
```

## Output

The program outputs the timing results for both the original and RCM reordered SpMV operations, and it shows the speedup achieved with reordering.

## Example

Run:

```bash
./spmv_rcm example.mtx
```

Output:

```
Matrix loaded: 4 x 4, nnz=5
RCM ordering took: 0.000015 seconds.
Applying permutation took: 0.000010 seconds.
CSR original SpMV time (avg): 0.000005 s
CSR RCM SpMV time (avg): 0.000003 s
Speedup vs original CSR: 1.67
```

## Contributing

Contributions to the project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
