
# Hypergraph Partitioning Using Fiduccia-Mattheyses Algorithm

This repository contains the C++ implementation of the Fiduccia-Mattheyses algorithm for hypergraph partitioning. The algorithm is designed to partition the hypergraph into two subsets while minimizing the cut cost, which is the number of hyperedges connecting vertices across the two subsets.

## Prerequisites

To compile and run this program, you will need:
- A modern C++ compiler that supports C++11 or later (e.g., GCC, Clang)

## Code writing
first create a file gp.cpp which is the nested dissection C++ code using nano command
```bash
nano hypgraph.cpp
```
## Compilation

Use the following command to compile the program with GCC or an equivalent compiler:
```bash
g++ -O2 -o hypgraph hypgraph.cpp
```

## Usage

Run the program by executing:
```bash
./hypgraph
```

Input the number of vertices and hyperedges, followed by the connectivity details of each hyperedge.

### Example Input
```
6 3
3 0 1 2
3 2 3 5
3 1 4 5
```

### Output
The output will display the final cut cost and the partition of the vertices into subsets A and B:
```
Final cut cost: 0
Partition:
0 B
1 B
2 B
3 B
4 B
```

## Code Explanation

- The `readHypergraph` function initializes the hypergraph structure from the input.
- The `fm_pass` function performs one pass of the Fiduccia-Mattheyses algorithm, adjusting vertex placement to minimize the cut cost.
- The `main` function provides an interface for inputting a hypergraph and outputs the final cut cost and partitions after performing the algorithm.

## Contributing

Contributions to improve the efficiency or functionality are welcome. Please fork this repository and submit a pull request.

## License

This project is distributed under the MIT License. See the LICENSE file in the repository for more details.
