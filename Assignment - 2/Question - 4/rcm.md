
# Reverse Cuthill-McKee Algorithm Implementation

This repository contains the C++ implementation of the Reverse Cuthill-McKee (RCM) algorithm. The RCM algorithm is used to reduce the bandwidth of a sparse matrix, which can optimize the performance of certain numerical methods used for solving systems of linear equations.

## Prerequisites

To compile and run this program, you will need:
- A C++ compiler that supports C++11 or later (e.g., GCC, Clang, MSVC)

## Code writing
Create a file named rcm.cpp which is the RCM C++ code using nano command
```bash
nano rcm.cpp
```

## Compilation

Compile the program using a C++ compiler. For example, if you are using GCC, you can compile the program as follows:
```bash
g++ -O2 -o rcm rcm.cpp 
```

## Usage

After compilation, run the program:
```bash
./rcm
```

### Output
When you run the program, it outputs the RCM ordering for a predefined sparse matrix. Example output:
```
RCM order: 4 2 1 3 0
```

## Code Explanation

- `reverseCuthillMcKee` function: Implements the RCM algorithm. It takes the number of vertices `n`, row pointers `rowPtr`, column indices `colIdx`, and a reference to an output vector `rcmOrder` that will contain the RCM ordering.
- The `main` function defines a small graph represented as a sparse matrix through its row pointers and column indices and invokes the RCM algorithm.

## Contributing

Contributions to improve the algorithm or extend the functionality are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Sources
https://crd.lbl.gov/assets/Uploads/RCM-ipdps17.pdf
https://github.com/SourangshuGhosh/Reverse-Cuthill-McKee-Ordering-in-C/blob/master/rcm.cpp
