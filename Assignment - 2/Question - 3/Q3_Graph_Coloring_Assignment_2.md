
# README for Graph Coloring Program

## Overview

This C++ program performs graph coloring using a parallel priority-based coloring algorithm. It generates a random undirected graph with a specified number of vertices and a hardcoded probability for edge creation (p = 0.6). The program uses OpenMP to leverage parallel computing for the graph coloring process. The main objective is to color the graph such that no two adjacent vertices share the same color.

## Features

- **Parallel Graph Coloring:** Utilizes OpenMP to speed up the coloring process by handling vertices in parallel.
- **Adjacency List Display:** For graphs with 20 or fewer vertices, the adjacency list and final color assignments are displayed. For larger graphs, the program notes that the graph is too large to display detailed information.
- **Conflict Resolution:** Includes a sequential fix-up step to ensure no two adjacent vertices share the same color, regardless of the initial parallel coloring outcome.

## Requirements

- C++ Compiler (GCC recommended)
- OpenMP support in the compiler
- Standard C++ libraries

## Compilation

To compile the program, ensure that OpenMP is enabled in your compiler settings. Here is how you can compile the program using GCC:

```bash
g++ -fopenmp -o graph_color graph_color.cpp
```

This command will compile the `graph_color.cpp` file and output an executable named `graph_color`.

## Running the Program

To run the program, simply execute the compiled binary from the command line:

```bash
./graph_coloring
```

You will be prompted to enter:

1. **Number of vertices (V):** The total number of vertices in the graph.
2. **Number of threads:** The number of parallel threads to use for graph coloring.

Based on the input number of vertices, the program will:

- Display the graph's adjacency list and final coloring if V is 20 or fewer.
- Note that the graph and color assignments are too large to display if V is greater than 20.

## Example Output

For a small graph (V <= 20):

```
Generating graph with 20 vertices and edge probability p = 0.6 (hardcoded)
Adjacency List:
Vertex 0 neighbors: 1 2 3 ...
...
Assigned colors:
Vertex 0 -> Color 1
Vertex 1 -> Color 2
...
Final check - no two adjacent vertices share the same color: YES
Number of distinct colors used: 5
Time taken (initial coloring only): 2.456 ms
```

For a large graph (V > 20):

```
Generating a large graph (V = 25, p = 0.6)
Graph is too large (25 vertices) to display final color assignment.
Graph is too large (25 vertices) to display adjacency list.
Final check - no two adjacent vertices share the same color: YES
Number of distinct colors used: 9
Time taken (initial coloring only): 3.123 ms
```

## Notes

- The conflict resolution phase ensures that no two adjacent vertices have the same color after the initial parallel coloring might have produced conflicts.
- The performance and effectiveness of the parallel coloring can vary based on the number of threads and the specific structure of the graph.
