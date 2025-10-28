
# Nested Dissection Algorithm Implementation

This repository contains the C++ implementation of the Nested Dissection algorithm for ordering the nodes of a graph. This algorithm is primarily used to reduce the fill-in during the factorization of sparse matrices in numerical linear algebra.

## Prerequisites

To compile and run this program, you will need:
- A modern C++ compiler that supports C++11 or later (e.g., GCC, Clang)

## Code writing
first create a file np.cpp which is the nested dissection C++ code using nano command
```bash
nano np.cpp
```

## Compilation

Use the following command to compile the program with GCC or an equivalent compiler:
```bash
g++ -O2 -o nd nd.cpp -std=c++11
```

## Usage

Run the program by executing:
```bash
./nd
```

Input the number of nodes and edges, followed by each pair of connected nodes.

### Example Input
```
6 6
0 1
1 2
2 3
2 4
4 5
1 5
```

### Output
The output will display the Nested Dissection ordering of the graph nodes:
```
Nested Dissection ordering:
0 2 3 5 1 4
```

## Code Explanation

- The `nestedDissection` function recursively finds separators and orders the graph nodes to minimize matrix bandwidth.
- The `main` function provides an interface for inputting a graph and outputs the nested dissection order of its nodes.

## Contributing

Contributions are welcome. Please fork this repository and submit a pull request with your improvements.

## License

This project is distributed under the MIT License. See the LICENSE file in the repository for more details.
