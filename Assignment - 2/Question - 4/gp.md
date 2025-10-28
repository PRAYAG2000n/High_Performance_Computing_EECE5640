
# Kernighan-Lin Graph Partitioning Algorithm Implementation

This repository contains the C++ implementation of the Kernighan-Lin graph partitioning algorithm, which is designed to partition a graph into two subsets while minimizing the edge cut between them. This technique is useful in applications such as network design, circuit layout, and data clustering.

## Prerequisites

To compile and run this program, you will need:
- A modern C++ compiler that supports C++11 or later (e.g., GCC, Clang)

## Code writing
first create a file gp.cpp which is the nested dissection C++ code using nano command
```bash
nano gp.cpp
```

## Compilation

Use the following command to compile the program with GCC or an equivalent compiler:
```bash
g++ -O2 -o gp gp.cpp
```

## Usage

Run the program by executing:
```bash
./gp
```

Input the number of vertices and edges, followed by each pair of connected vertices.

### Example Input
```
5 6
0 1
1 2
2 3
2 4
4 5
1 5
```

### Output
The output will display the final edge cut and the partition of the vertices into subsets A and B:
```
Final cut cost: 2
Partition (vertex -> A/B):
0 B
1 A
2 A
3 B
4 B
```

## Code Explanation

- The `compute_cut` function calculates the number of edges between the two partitions.
- The `KL_pass` function performs one pass of the Kernighan-Lin graph partitioning algorithm, trying to find the best swaps that minimize the cut.
- The `main` function initializes the graph and runs the Kernighan-Lin algorithm, printing the results.

## Contributing

Contributions to improve the efficiency or functionality are welcome. Please fork this repository and submit a pull request.

## License

This project is distributed under the MIT License. See the LICENSE file in the repository for more details.
